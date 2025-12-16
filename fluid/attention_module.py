# Copyright (c) 2024, FluidMoE Contributors.

"""
完全自定义的 Fluid Attention 模块
支持前向计算优化和通信重叠
"""

import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

from fluid.attention_layers import FluidColumnParallelLinear, FluidRowParallelLinear
from fluid.attention_core import FluidDotProductAttention
from fluid.communication import fluid_all_to_all_sp2hp, fluid_all_to_all_hp2sp


class FluidSelfAttention(MegatronModule):
    """
    完全自定义的 Self-Attention 层,支持:
    1. 计算-通信重叠优化
    2. 自定义前向计算流程
    3. Fluid dW scheduling

    与 Megatron SelfAttention 的区别:
    - 前向计算可以自定义优化
    - 通信使用 Fluid AllToAll (不需要 patch)
    - 支持前向计算与通信的流水线重叠

    维护策略:
    - 参考 Megatron SelfAttention 的接口设计
    - 手动同步必要的功能更新
    - 优先保持接口兼容性
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        attention_type: str = "self",
        cp_comm_type: str = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        submodules=None,  # 可选参数，保持向后兼容
    ):
        """
        完全自定义的 FluidSelfAttention
        内部直接创建所有子模块，不依赖 submodules 参数
        """
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.cp_comm_type = cp_comm_type
        self.pg_collection = pg_collection or get_default_pg_collection()

        # Attention 参数计算
        self.kv_channels = config.kv_channels
        self.num_query_groups_per_partition = config.num_query_groups
        self.num_attention_heads_per_partition = config.num_attention_heads
        self.hidden_size_per_attention_head = self.kv_channels
        self.query_projection_size = (
            self.num_attention_heads_per_partition * self.hidden_size_per_attention_head
        )
        self.kv_projection_size = (
            self.num_query_groups_per_partition * self.hidden_size_per_attention_head
        )

        # Context Parallel 配置
        self.cp_size = config.context_parallel_size
        self.cp_group = self.pg_collection.cp if self.pg_collection else None

        # 1. QKV 投影层 (直接创建 Fluid 版本)
        self.linear_qkv = FluidColumnParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            layer_name=f"layer_{layer_number}_attn_qkv",
            layer_id=layer_number,
            tp_group=self.pg_collection.tp,
        )

        # 2. Core Attention - 使用 FluidDotProductAttention + 手动 Ulysses AllToAll
        # FluidDotProductAttention 移除了 CP 限制，让我们可以手动控制 AllToAll
        # 这样 Fluid scheduler 可以完全控制通信时机，实现通信-计算重叠
        self.core_attention = FluidDotProductAttention(
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            pg_collection=self.pg_collection,
        )

        # 3. 输出投影层 (直接创建 Fluid 版本)
        self.linear_proj = FluidRowParallelLinear(
            input_size=self.query_projection_size,
            output_size=self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            layer_name=f"layer_{layer_number}_attn_proj",
            layer_id=layer_number,
            tp_group=self.pg_collection.tp,
        )

        # Q/K LayerNorm (暂不支持)
        self.q_layernorm = None
        self.k_layernorm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_params=None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        packed_seq_params=None,
        inference_context=None,  # Added for compatibility
        **kwargs  # Accept any additional arguments
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 - 真正的 Ulysses Sequence Parallel 实现

        手动控制 AllToAll 通信，让 Fluid scheduler 实现通信-计算重叠：
        1. QKV 投影：在本地序列片段上进行
        2. AllToAll sp2hp：序列并行 -> 头并行（每个GPU获得完整序列，部分头）
        3. 注意力计算：在完整序列上计算（head parallel）
        4. AllToAll hp2sp：头并行 -> 序列并行（恢复序列分布）
        5. 输出投影

        关键：手动 AllToAll 允许 scheduler 在通信时调度 dW 计算
        """

        # ===== 步骤 1: QKV 投影 =====
        # [seq_len/CP, batch, hidden_size] -> [seq_len/CP, batch, 3*hidden_size]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # 分割 QKV
        # [seq_len/CP, batch, num_heads, head_dim]
        query, key, value = self._split_qkv(mixed_qkv)

        # 应用 Q/K LayerNorm (如果启用)
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)
        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        # ===== 步骤 2: AllToAll sp2hp (手动控制) =====
        # 只有在使用 Context Parallel 时才执行 AllToAll
        if self.cp_size > 1:
            # [seq_len/CP, batch, num_heads, head_dim] -> [seq_len, batch, num_heads/CP, head_dim]
            query = fluid_all_to_all_sp2hp(query, self.cp_group)
            key = fluid_all_to_all_sp2hp(key, self.cp_group)
            value = fluid_all_to_all_sp2hp(value, self.cp_group)

        # ===== 步骤 3: 注意力计算（在完整序列上） =====
        # DotProductAttention 期望 [seq_len, batch, num_heads/CP, head_dim]

        # 在 Ulysses SP 模式下，attention_mask 需要重新生成
        # 因为输入的 mask 是按照局部序列长度生成的，但现在序列是完整的
        if self.cp_size > 1:
            # Ulysses 模式：使用 causal mask（自动生成），忽略输入的 attention_mask
            # 因为 attn_mask_type=causal，FusedScaleMaskSoftmax 会自动生成正确大小的 mask
            attention_mask_for_attn = None
        else:
            # 非 CP 模式：使用输入的 attention_mask
            attention_mask_for_attn = attention_mask

        context = self.core_attention(
            query, key, value,
            attention_mask=attention_mask_for_attn,
            attn_mask_type=self.attn_mask_type,
            packed_seq_params=packed_seq_params,
        )

        # ===== 步骤 4: AllToAll hp2sp (手动控制) =====
        # 只有在使用 Context Parallel 时才执行 AllToAll
        if self.cp_size > 1:
            # [seq_len, batch, num_heads/CP, head_dim] -> [seq_len/CP, batch, num_heads, head_dim]
            # 但 DotProductAttention 输出是 [seq_len, batch, hidden_size/CP]
            # 需要先 reshape 回 4D 张量
            seq_len, batch, hidden_size_per_cp = context.shape
            num_heads_per_cp = self.num_attention_heads_per_partition // self.cp_size
            context_4d = context.view(seq_len, batch, num_heads_per_cp, self.hidden_size_per_attention_head)

            # 执行 hp2sp AllToAll
            context = fluid_all_to_all_hp2sp(context_4d, self.cp_group)

            # 恢复 3D 形状 [seq_len/CP, batch, hidden_size]
            context = context.view(
                seq_len // self.cp_size,
                batch,
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head
            )

        # ===== 步骤 5: 输出投影 =====
        # [seq_len/CP, batch, hidden_size] -> [seq_len/CP, batch, hidden_size]
        output, output_bias = self.linear_proj(context)

        return output, output_bias

    def _split_qkv(self, mixed_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        分割 QKV
        参考 Megatron SelfAttention.get_query_key_value_tensors
        """
        # [seq_len, batch, 3*hidden] -> [seq_len, batch, num_groups, 3*group_size]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # Split Q, K, V
        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # Reshape query: [seq, batch, num_groups, q_size] -> [seq, batch, num_heads, head_dim]
        query = query.reshape(
            query.size(0), query.size(1), -1, self.hidden_size_per_attention_head
        )

        return query, key, value

# Copyright (c) 2024, FluidMoE Contributors.

"""
完全自定义的 Fluid Attention 模块
支持前向计算优化和通信重叠
"""

import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

import os

from fluid.attention_layers import (
    FluidColumnParallelLinear,
    FluidRowParallelLinear,
)
from fluid.attention_core import FluidDotProductAttention
from fluid.communication import (
    fluid_all_to_all_sp2hp,
    fluid_all_to_all_hp2sp,
    fluid_all_to_all_qkv_sp2hp_batched,  # QKV合并通信
    fluid_all_to_all_qkv_hp2sp_batched,  # QKV合并通信
    fluid_all_to_all_mixed_qkv_sp2hp,    # 直接对mixed_qkv做AllToAll（Baseline优化）
    fluid_fused_all_to_all_sp2hp,
    fluid_fused_all_to_all_hp2sp,
    get_dx_num_chunks,
    # Fused backward: Linear GEMM + AllToAll pipeline
    fluid_fused_hp2sp_linear_proj,
    fluid_fused_sp2hp_core_attention,
)

# 前向重叠功能开关
def get_forward_overlap_enabled():
    """Check if forward overlap mode is enabled (动态读取环境变量)"""
    return os.environ.get('FLUID_FORWARD_OVERLAP', '0') == '1'


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
        # 优先从 pg_collection 获取 cp_size，确保与实际进程组一致
        # 这与 FluidMoELayer 的行为一致（从 pg_collection.ep.size() 获取 ep_size）
        self.cp_group = self.pg_collection.cp if self.pg_collection else None
        if self.cp_group is not None:
            self.cp_size = self.cp_group.size()
        else:
            self.cp_size = config.context_parallel_size

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

        # Selective recompute for core attention (saves O(seq²) memory)
        # Compatible with Megatron's --recompute-activations flag
        self.checkpoint_core_attention = (
            getattr(self.config, 'recompute_granularity', None) == 'selective'
            and "core_attn" in getattr(self.config, 'recompute_modules', [])
        )

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

        # ===== Forward Overlap Path =====
        # 使用 overlap_forward.py 的新实现
        # Enable with: FLUID_FORWARD_OVERLAP=1
        if get_forward_overlap_enabled() and self.cp_size > 1:
            if os.environ.get('FLUID_DEBUG_FORWARD_TIMING', '0') == '1':
                print(f"[Forward Overlap Attn] Using overlap path, cp_size={self.cp_size}", flush=True)
            return self._forward_with_overlap(
                hidden_states, attention_mask, rotary_pos_emb, packed_seq_params
            )

        # ===== 步骤 1: QKV 投影 =====
        # [seq_len/CP, batch, hidden_size] -> [seq_len/CP, batch, 3*hidden_size]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # ===== 步骤 2: AllToAll sp2hp (手动控制) =====
        # 只有在使用 Context Parallel 时才执行 AllToAll
        # Note: Pipelined mode (FLUID_PIPELINED_SP2HP=1) uses _forward_pipelined_qk_overlap path
        debug_timing = os.environ.get('FLUID_DEBUG_FORWARD_TIMING', '0') == '1'
        if debug_timing and self.cp_size > 1:
            if not hasattr(self, '_baseline_timing_count'):
                self._baseline_timing_count = 0
            self._baseline_timing_count += 1
            if self._baseline_timing_count <= 20:
                import torch.cuda
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_start.record()

        if self.cp_size > 1:
            # [seq_len/CP, batch, num_heads, head_dim] -> [seq_len, batch, num_heads/CP, head_dim]
            num_chunks = get_dx_num_chunks()
            if num_chunks > 1:
                # Chunked mode: need to split first, then do separate AllToAll
                query, key, value = self._split_qkv(mixed_qkv)
                # Apply Q/K LayerNorm if enabled
                if self.q_layernorm is not None:
                    query = self.q_layernorm(query)
                if self.k_layernorm is not None:
                    key = self.k_layernorm(key)
                # TODO: Add fused batched version for chunked mode
                query = fluid_fused_all_to_all_sp2hp(query, self.cp_group)
                key = fluid_fused_all_to_all_sp2hp(key, self.cp_group)
                value = fluid_fused_all_to_all_sp2hp(value, self.cp_group)
            else:
                # Optimized Baseline path: Direct AllToAll on mixed_qkv
                # Avoids redundant split->concat->split operations
                query, key, value = fluid_all_to_all_mixed_qkv_sp2hp(
                    mixed_qkv,
                    self.num_attention_heads_per_partition,
                    self.num_query_groups_per_partition,  # num_kv_heads
                    self.hidden_size_per_attention_head,
                    self.cp_group
                )
                # Apply Q/K LayerNorm after AllToAll (if enabled)
                if self.q_layernorm is not None:
                    query = self.q_layernorm(query)
                if self.k_layernorm is not None:
                    key = self.k_layernorm(key)
        else:
            # No CP: Just split locally
            query, key, value = self._split_qkv(mixed_qkv)
            # Apply Q/K LayerNorm if enabled
            if self.q_layernorm is not None:
                query = self.q_layernorm(query)
            if self.k_layernorm is not None:
                key = self.k_layernorm(key)

        # Baseline timing - sp2hp
        if debug_timing and self.cp_size > 1 and hasattr(self, '_baseline_timing_count') and self._baseline_timing_count <= 20:
            ev_sp2hp_end = torch.cuda.Event(enable_timing=True)
            ev_sp2hp_end.record()
            ev_attn_end = None  # will be set after attention

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

        # Use checkpointed core attention if enabled (saves O(seq²) memory)
        if self.checkpoint_core_attention:
            context = self._checkpointed_attention_forward(
                query, key, value,
                attention_mask=attention_mask_for_attn,
                attn_mask_type=self.attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            context = self.core_attention(
                query, key, value,
                attention_mask=attention_mask_for_attn,
                attn_mask_type=self.attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        # Baseline timing - after attention
        if debug_timing and self.cp_size > 1 and hasattr(self, '_baseline_timing_count') and self._baseline_timing_count <= 20:
            ev_attn_end = torch.cuda.Event(enable_timing=True)
            ev_attn_end.record()
            torch.cuda.synchronize()
            t_sp2hp = ev_start.elapsed_time(ev_sp2hp_end)
            t_attn = ev_sp2hp_end.elapsed_time(ev_attn_end)
            print(f"[Baseline Attn] sp2hp: {t_sp2hp:.2f}ms, core_attn: {t_attn:.2f}ms, total: {t_sp2hp+t_attn:.2f}ms")

        # ===== 步骤 4+5: AllToAll hp2sp + 输出投影 =====
        # 只有在使用 Context Parallel 时才执行 AllToAll
        if self.cp_size > 1:
            # [seq_len, batch, num_heads/CP, head_dim] -> [seq_len/CP, batch, num_heads, head_dim]
            # 但 DotProductAttention 输出是 [seq_len, batch, hidden_size/CP]
            # 需要先 reshape 回 4D 张量
            seq_len, batch, hidden_size_per_cp = context.shape
            num_heads_per_cp = self.num_attention_heads_per_partition // self.cp_size
            context_4d = context.view(seq_len, batch, num_heads_per_cp, self.hidden_size_per_attention_head)

            num_chunks = get_dx_num_chunks()
            if num_chunks > 1:
                # ============================================================
                # Fused hp2sp AllToAll + Linear Projection
                # Backward: chunked Linear GEMM + sp2hp AllToAll pipeline
                # ============================================================
                output = fluid_fused_hp2sp_linear_proj(
                    context_4d,
                    self.linear_proj.weight,
                    self.linear_proj.bias,  # For dBias computation
                    self.cp_group,
                    f"layer_{self.layer_number}_attn_proj",
                    self.layer_number,
                )
                # Bias is returned separately (skip_bias_add=True pattern)
                output_bias = self.linear_proj.bias
            else:
                # Non-fused path: separate hp2sp AllToAll + linear_proj
                context = fluid_all_to_all_hp2sp(context_4d, self.cp_group)

                # 恢复 3D 形状 [seq_len/CP, batch, hidden_size]
                context = context.view(
                    seq_len // self.cp_size,
                    batch,
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head
                )

                # 输出投影
                output, output_bias = self.linear_proj(context)
        else:
            # ===== 非 CP 模式: 直接输出投影 =====
            # [seq_len, batch, hidden_size] -> [seq_len, batch, hidden_size]
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

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        packed_seq_params=None,
    ):
        """
        Forward with selective activation checkpointing for core attention.

        This saves O(seq²) memory by not storing attention scores during forward,
        and recomputing them during backward.

        Key insight: This is compatible with Fluid's dX/dW scheduling because:
        1. We only checkpoint core_attention (which has no trainable parameters)
        2. The Linear layers (QKV, proj) are NOT checkpointed - their dW still goes to scheduler
        3. During backward, core_attention is recomputed, then the recomputed output
           flows back through the normal backward path of Linear layers
        """

        def custom_forward(*inputs):
            """Custom forward function for checkpointing."""
            query_ = inputs[0]
            key_ = inputs[1]
            value_ = inputs[2]
            attention_mask_ = inputs[3]
            attn_mask_type_ = inputs[4]

            # Convert attn_mask_type back from tensor
            if isinstance(attn_mask_type_, torch.Tensor):
                attn_mask_type_ = AttnMaskType(attn_mask_type_.item())

            output_ = self.core_attention(
                query_,
                key_,
                value_,
                attention_mask=attention_mask_,
                attn_mask_type=attn_mask_type_,
                packed_seq_params=packed_seq_params,
            )
            return output_

        # Convert attn_mask_type to tensor for checkpointing
        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type_tensor = torch.tensor([attn_mask_type.value], dtype=torch.int)

        # Use Megatron's checkpoint function
        # The False argument means: don't distribute saved activations across TP group
        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,  # distribute_saved_activations
            query,
            key,
            value,
            attention_mask,
            attn_mask_type_tensor,
        )

        return hidden_states

    def _forward_with_overlap(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        packed_seq_params=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用多轮 P2P 重叠实现的前向传播（统一实现）

        使用 Round-Robin Tournament 调度：
        - 将"每个rank和所有其他rank交换数据"拆成多轮
        - 若卡数P为偶数：总轮数 R = P-1
        - 若卡数P为奇数：添加dummy，总轮数 R = P
        - 每轮每张卡只和一个peer通信（避免冲突）
        - 通信流在跑第r轮的同时，计算流在处理第r-1轮的数据

        2卡场景自然退化为1轮通信（P=2, R=1）

        注意力层通信是规则的（Ulysses SP），每个rank发送/接收相同大小的数据

        Timeline:
        [QKV Round 0] → [P2P Round 1] || [Process Round 0] → ... → [Attention] → [hp2sp+proj]
        """
        from fluid.multicard_p2p import (
            AttentionMultiCardOverlapContext,
            attention_multicard_qkv_sp2hp_with_grad,
            attention_multicard_hp2sp_proj,
        )

        seq_local, batch, hidden_size = hidden_states.shape
        cp = self.cp_size
        my_rank = self.cp_group.rank()

        # 获取 QKV 权重
        qkv_weight = self.linear_qkv.weight
        heads = self.num_attention_heads_per_partition
        num_kv_heads = self.num_query_groups_per_partition
        dim = self.hidden_size_per_attention_head

        # 创建或获取多卡 overlap context（统一管理所有卡数场景）
        if not hasattr(self, '_multicard_overlap_ctx'):
            self._multicard_overlap_ctx = AttentionMultiCardOverlapContext(
                hidden_states.device, cp
            )

        debug_timing = os.environ.get('FLUID_DEBUG_FORWARD_TIMING', '0') == '1'
        if debug_timing:
            print(f"[Forward Overlap Attn Rank {my_rank}] Using P2P overlap with {self._multicard_overlap_ctx.num_rounds} rounds", flush=True)

        # ===== Step 1: QKV + sp2hp with multicard P2P overlap =====
        query, key, value = attention_multicard_qkv_sp2hp_with_grad(
            hidden_states,
            qkv_weight,
            heads,
            num_kv_heads,
            dim,
            self.cp_group,
            self._multicard_overlap_ctx,
            layer_name=f"layer_{self.layer_number}_attn_qkv",
            layer_id=self.layer_number,
        )

        # ===== Step 2: Core Attention =====
        attention_mask_for_attn = None  # Ulysses mode: use causal mask

        if self.checkpoint_core_attention:
            context = self._checkpointed_attention_forward(
                query, key, value,
                attention_mask=attention_mask_for_attn,
                attn_mask_type=self.attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            context = self.core_attention(
                query, key, value,
                attention_mask=attention_mask_for_attn,
                attn_mask_type=self.attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        # ===== Step 3: hp2sp + Output Projection with multicard overlap =====
        seq_full = seq_local * cp
        heads_local = heads // cp
        context_4d = context.view(seq_full, batch, heads_local, dim)

        output = attention_multicard_hp2sp_proj(
            context_4d,
            self.linear_proj.weight,
            self.linear_proj.bias,
            self.cp_group,
            self._multicard_overlap_ctx,
        )

        # Bias is returned separately (skip_bias_add=True pattern)
        output_bias = self.linear_proj.bias

        return output, output_bias

# Copyright (c) 2024, FluidMoE Contributors.

"""
完全自定义的 Fluid MoE 模块
支持前向计算优化和通信重叠
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    get_default_pg_collection,
    permute,
    unpermute,
)
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

from fluid.moe_layers import FluidGroupedMLP
from fluid.communication import (
    fluid_all_to_all_moe_dispatch,
    fluid_all_to_all_moe_combine,
)


class FluidMoELayer(MegatronModule):
    """
    完全自定义的 MoE 层

    与 Megatron MoELayer 的区别:
    - 直接使用 fluid_all_to_all_moe_dispatch/combine (不需要 FluidTokenDispatcher)
    - 使用 FluidGroupedMLP (支持 dW scheduling)
    - 支持前向计算优化

    组件:
    1. Router: 计算 routing scores (使用 Megatron 原版)
    2. Token Dispatch/Combine: 使用 fluid communication 函数
    3. Experts: Expert 计算 (FluidGroupedMLP)
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        submodules=None,  # 可选参数，保持向后兼容
    ):
        """
        完全自定义的 FluidMoELayer
        内部直接创建所有子模块，不依赖 submodules 参数
        """
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.pg_collection = pg_collection or get_default_pg_collection()

        # EP/TP 配置
        self.ep_size = self.pg_collection.ep.size()
        self.ep_group = self.pg_collection.ep
        self.tp_size = self.pg_collection.expt_tp.size()
        self.tp_group = self.pg_collection.expt_tp
        self.tp_ep_group = self.pg_collection.tp_ep

        # Local experts
        self.num_local_experts = config.num_moe_experts // self.ep_size
        self.local_expert_indices = list(
            range(
                self.pg_collection.ep.rank() * self.num_local_experts,
                (self.pg_collection.ep.rank() + 1) * self.num_local_experts,
            )
        )

        # 1. Router (使用 Fluid 版本支持 dW overlap)
        from fluid.moe_layers import FluidRouter
        self.router = FluidRouter(config=config, pg_collection=self.pg_collection, layer_number=layer_number)

        # 2. Experts (直接创建 FluidGroupedMLP)
        self.experts = FluidGroupedMLP(
            num_local_experts=self.num_local_experts,
            config=config,
            pg_collection=self.pg_collection,
        )

        # 保存状态用于 combine
        self.permutation_map = None
        self.probs = None

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        MoE 前向传播

        标准流程:
        1. Router: 计算 routing scores
        2. Token Dispatch: 分发 tokens 到 expert GPUs
        3. Expert 计算: 每个 expert 处理分配的 tokens
        4. Token Combine: 收集 expert 输出
        5. 加权聚合: 根据 routing probs 聚合
        """

        # 保存输入 shape
        input_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, input_shape[-1])  # [num_tokens, hidden]

        # ===== 步骤 1: Router =====
        probs, routing_map = self.router(hidden_states)
        # probs: [num_tokens, num_experts] - probabilities for all experts
        # routing_map: [num_tokens, num_experts] - binary mask for selected top-k experts

        # ===== 步骤 2: Token Dispatch =====
        dispatched_input, tokens_per_expert, probs = self._dispatch(
            hidden_states,
            routing_map=routing_map,
            probs=probs,
        )

        # ===== 步骤 3: Expert 计算 (FluidGroupedMLP) =====
        expert_output, _ = self.experts(
            dispatched_input,
            tokens_per_expert,
            probs,
        )

        # ===== 步骤 4: Token Combine =====
        # Pass the flattened shape for unpermute
        output = self._combine(expert_output, restore_shape=hidden_states.shape)

        # 恢复原始 shape
        output = output.view(input_shape)

        return output, None  # 第二个返回值是 mlp_bias

    def _dispatch(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Token Dispatch: 将 tokens 分发到对应的 expert GPUs

        标准流程:
        1. Permute: 根据 routing_map 重排 tokens
        2. AllToAll (EP): 跨 GPU 分发 tokens
        3. AllGather (TP): 如果有 TP,gather tokens
        """

        # ===== 步骤 1: Local Permutation =====
        # permute() returns 3 values: (permuted_input, permuted_probs, sorted_indices)
        # Pass probs to permute() so it gets permuted correctly along with tokens
        permuted_tokens, permuted_probs_flat, self.permutation_map = permute(
            hidden_states,
            routing_map,
            probs=probs,  # Pass probs to be permuted
            num_out_tokens=routing_map.size(0) * self.config.moe_router_topk,
        )

        # Reshape permuted probs to [num_permuted_tokens, 1]
        permuted_probs = permuted_probs_flat.view(-1, 1) if permuted_probs_flat is not None else probs.view(-1, 1)

        # 计算每个 expert 接收的 token 数量
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()  # [num_experts]

        # ===== 步骤 2: AllToAll Dispatch (EP) =====
        if self.ep_size > 1:
            # 计算 split sizes
            input_splits = num_local_tokens_per_expert.reshape(
                self.ep_size, self.num_local_experts
            ).sum(axis=1)

            # Gather global token distribution
            num_global_tokens_per_expert = (
                gather_from_sequence_parallel_region(
                    num_local_tokens_per_expert, group=self.tp_ep_group
                )
                .reshape(self.ep_size, self.tp_size, -1)
                .transpose(0, 1)
            )

            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()

            num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
            output_splits = num_global_tokens_per_rank[
                self.pg_collection.expt_tp.rank()
            ]  # [ep_size]

            # Debug AllToAll sizes
            print(f"  permuted_tokens.shape = {permuted_tokens.shape}")
            print(f"  permuted_probs.shape = {permuted_probs.shape}")
            print(f"  input_splits = {input_splits}")
            print(f"  output_splits = {output_splits}")
            print(f"  input_splits.sum() = {input_splits.sum()}")
            print(f"  output_splits.sum() = {output_splits.sum()}")

            # 使用 Fluid MoE dispatch (不需要单独的 Dispatcher 类!)
            global_input_tokens = fluid_all_to_all_moe_dispatch(
                permuted_tokens,
                output_splits,
                input_splits,
                self.ep_group,
            )

            global_probs = fluid_all_to_all_moe_dispatch(
                permuted_probs,
                output_splits,
                input_splits,
                self.ep_group,
            )

            tokens_per_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))

            # Save splits for combine (combine reverses input/output)
            self.dispatch_input_splits = input_splits  # Will be output_splits in combine
            self.dispatch_output_splits = output_splits  # Will be input_splits in combine

        else:
            global_input_tokens = permuted_tokens
            global_probs = permuted_probs
            tokens_per_expert = num_local_tokens_per_expert

        # ===== 步骤 3: AllGather (TP) =====
        if self.tp_size > 1:
            global_input_tokens = gather_from_sequence_parallel_region(
                global_input_tokens, group=self.tp_group
            )
            global_probs = gather_from_sequence_parallel_region(
                global_probs, group=self.tp_group
            )

        self.probs = global_probs

        return global_input_tokens, tokens_per_expert, global_probs

    def _combine(
        self,
        expert_output: torch.Tensor,
        restore_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Token Combine: 将 expert 输出收集回原始 GPUs

        标准流程:
        1. ReduceScatter (TP): 如果有 TP
        2. AllToAll (EP): 跨 GPU 收集 tokens
        3. Unpermute: 恢复原始 token 顺序

        Args:
            expert_output: Expert computation output
            restore_shape: Original shape before permutation for unpermute
        """

        hidden_states = expert_output

        # ===== 步骤 1: ReduceScatter (TP) =====
        if self.tp_size > 1:
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states.to(self.probs.dtype),
                group=self.tp_group,
            ).to(hidden_states.dtype)

        # ===== 步骤 2: AllToAll Combine (EP) =====
        if self.ep_size > 1:
            # 使用 Fluid MoE combine (与 dispatch 对称)
            # 注意: combine 的 input/output splits 与 dispatch 相反
            # Combine reverses the splits from dispatch
            hidden_states = fluid_all_to_all_moe_combine(
                hidden_states,
                output_splits=self.dispatch_input_splits,  # Reverse of dispatch
                input_splits=self.dispatch_output_splits,  # Reverse of dispatch
                group=self.ep_group,
            )

        # ===== 步骤 3: Unpermute =====
        print(f"  hidden_states.shape = {hidden_states.shape}")
        print(f"  self.permutation_map.shape = {self.permutation_map.shape}")
        print(f"  restore_shape = {restore_shape}")

        output = unpermute(
            hidden_states,
            self.permutation_map,
            restore_shape,  # Pass the restore_shape parameter
        )

        return output

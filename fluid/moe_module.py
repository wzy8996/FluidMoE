# Copyright (c) 2024, FluidMoE Contributors.

"""
完全自定义的 Fluid MoE 模块
支持前向计算优化和通信重叠

Forward Overlap (v0.7):
- FC2 + AllToAll tile-level pipeline
- Enabled via FLUID_FORWARD_NUM_CHUNKS environment variable
- Timeline: FC2[i] || AllToAll[i-1]
"""

import os
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
    sort_chunks_by_idxs,
)
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

from fluid.moe_layers import FluidGroupedMLP, set_dispatch_alltoall_ctx, FusedForwardStandardBackward
from fluid.communication import (
    fluid_all_to_all_moe_dispatch,
    fluid_all_to_all_moe_combine,
    fluid_fused_combine_unpermute,
    get_dx_num_chunks,
)
# Forward mode: "baseline" or "fused"
# Can be overridden by environment variable FLUID_FORWARD_MODE
# NOTE: "fused" mode currently has overhead from dtype/layout conversion.
# Using "baseline" as default until C++ kernel is optimized.
import os
forward = os.environ.get('FLUID_FORWARD_MODE', 'baseline')


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

        # 初始化 sort_chunks_by_idxs 的索引 (用于 num_local_experts > 1 时的重排)
        # AllToAll 后数据布局是 source-rank-major: [R0_E0][R0_E1][R1_E0][R1_E1]...
        # GroupedGEMM 需要 expert-major: [E0_all][E1_all]...
        if self.num_local_experts > 1:
            num_input_chunks = config.num_moe_experts * self.tp_size
            input_chunk_idxs = torch.arange(num_input_chunks)
            # [num_local_experts, tp_size * ep_size] -> ravel to get expert-major order
            self.sort_input_by_local_experts = input_chunk_idxs.reshape(
                -1, self.num_local_experts
            ).T.ravel()
            # Reverse mapping: [num_local_experts, tp_size * ep_size] -> transpose -> ravel
            self.restore_output_by_local_experts = input_chunk_idxs.reshape(
                self.num_local_experts, -1
            ).T.ravel()
        else:
            self.sort_input_by_local_experts = None
            self.restore_output_by_local_experts = None

        # 保存 num_global_tokens_per_local_expert 用于 sort/unsort
        self.num_global_tokens_per_local_expert = None

        self._fused_nccl_initialized = False
        self._is_glu = config.gated_linear_unit

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 保存输入 shape
        input_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, input_shape[-1])

        # ===== 步骤 1: Router =====
        probs, routing_map = self.router(hidden_states)

        # ===== 选择执行路径 =====
        use_fused = forward == "fused" and self.ep_size > 1 and not self._is_glu

        if use_fused:
            # Fused path: AllToAll + FC1/FC2 深度重叠
            output = self._forward_fused(hidden_states, routing_map, probs)
        else:
            # Standard path: separate dispatch + expert + combine
            # 保存 probs 和 routing_map 用于 _combine 中的 unpermute
            self.original_probs = probs
            self.routing_map = routing_map

            dispatched_input, tokens_per_expert = self._dispatch(
                hidden_states,
                routing_map=routing_map,
                probs=probs,
            )

            # NOTE: probs 不再传给 experts，在 _combine 的 unpermute 里乘
            expert_output, _ = self.experts(
                dispatched_input,
                tokens_per_expert,
            )

            output = self._combine(expert_output, restore_shape=hidden_states.shape)

        # 恢复原始 shape
        output = output.view(input_shape)

        return output, None  # 第二个返回值是 mlp_bias

    def _forward_fused(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        非对称流水线前向传播：
        - 前向：AllToAll + FC1 + Activation + FC2 + AllToAll 深度重叠（融合算子）
        - 反向：标准 dX + dW 分离调度（FluidGroupedMLP autograd）

        这是 FluidMoE 的核心优化设计！

        Timeline:
          Forward (融合，计算-通信重叠):
            Self FC1:    [=========] <- 与 AllToAll 并行
            AllToAll:    [=========]
            Peer FC1:          [===][===] <- 数据到达即计算
            Peer FC2:    [===]
            Peer Send:        [=====] <- 与下一个 FC2 并行
            Self FC2:         [===]

          Backward (标准，dW 调度):
            dX:          [=========]
            AllToAll:    [=========] <- dX + AllToAll 分块重叠
            dW:                [===] <- 与下一层 AllToAll 重叠
        """
        from fluid.ops import fluid_kernels
        import torch.distributed as dist

        rank = self.pg_collection.ep.rank()
        world_size = self.ep_size

        # Initialize fused NCCL communicator on first call
        if not self._fused_nccl_initialized:
            nccl_id = fluid_kernels.get_moe_fused_nccl_unique_id()
            if rank == 0:
                nccl_id_tensor = torch.tensor(nccl_id, dtype=torch.int64, device='cuda')
            else:
                nccl_id_tensor = torch.zeros(len(nccl_id), dtype=torch.int64, device='cuda')
            dist.broadcast(nccl_id_tensor, src=0, group=self.ep_group)
            nccl_id_list = nccl_id_tensor.cpu().tolist()
            fluid_kernels.init_moe_fused_nccl(rank, world_size, nccl_id_list)
            self._fused_nccl_initialized = True
            if rank == 0:
                print(f"[FluidMoE] Fused NCCL communicator initialized")

        # ===== Step 1: Local Permutation =====
        permuted_tokens, permuted_probs_flat, permutation_map = permute(
            hidden_states,
            routing_map,
            probs=probs,
            num_out_tokens=routing_map.size(0) * self.config.moe_router_topk,
        )
        self.permutation_map = permutation_map  # 保存用于 combine

        # 计算每个 expert 接收的 token 数量
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

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
        ]

        # ===== 预计算 HOST 数组 =====
        # OPTIMIZED: 合并所有 D2H 传输为单次同步，避免多次 .cpu().tolist() 调用
        # 原来有 3 + (ep_size-1) 次同步，现在只有 3 次同步
        input_splits_list = input_splits.tolist()
        output_splits_list = output_splits.tolist()

        self_input_offset = sum(input_splits_list[:rank])
        self_input_count = input_splits_list[rank]

        # 一次性将整个 tensor 复制到 CPU，避免循环中多次同步
        tp_rank = self.pg_collection.expt_tp.rank()
        tokens_per_local_expert_cpu = num_global_tokens_per_local_expert[tp_rank].to(torch.int32).cpu()

        h_self_tokens_per_expert = tokens_per_local_expert_cpu[rank].tolist()

        h_peer_tokens_per_expert_all = []
        peer_token_counts = []
        for peer in range(self.ep_size):
            if peer == rank:
                continue
            # 直接从 CPU tensor 切片，不再触发 D2H 同步
            h_peer_tokens_per_expert_all.append(tokens_per_local_expert_cpu[peer].tolist())
            peer_token_counts.append(output_splits_list[peer])

        # 获取激活函数类型
        if self.config.activation_func == torch.nn.functional.silu:
            activation_type = 1
        elif self.config.activation_func == torch.nn.functional.relu:
            activation_type = 2
        else:
            activation_type = 0

        # ===== 融合前向 + 标准反向 =====
        # probs multiplication happens in unpermute (standard Megatron behavior)
        output = FusedForwardStandardBackward.apply(
            hidden_states,
            routing_map,
            probs,  # probs will be used in unpermute and backward
            permuted_tokens,
            permutation_map,
            self.experts.weight1,
            self.experts.weight2,
            self.num_local_experts,
            self.config.hidden_size,
            input_splits_list,
            output_splits_list,
            self_input_offset,
            self_input_count,
            peer_token_counts,
            h_self_tokens_per_expert,
            h_peer_tokens_per_expert_all,
            activation_type,
            self,
        )

        return output

    def _prepare_backward_ctx(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        dispatched_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为反向传播准备 context（不执行 AllToAll）

        用于融合前向时，我们已经有了 dispatched_input，
        只需要计算元信息和设置反向 AllToAll 的 context。

        Returns:
            (sorted_dispatched_input, tokens_per_expert, permuted_probs)
        """
        # Permute probs
        _, permuted_probs_flat, self.permutation_map = permute(
            hidden_states,
            routing_map,
            probs=probs,
            num_out_tokens=routing_map.size(0) * self.config.moe_router_topk,
        )
        permuted_probs = permuted_probs_flat.view(-1, 1) if permuted_probs_flat is not None else probs.view(-1, 1)

        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

        if self.ep_size > 1:
            input_splits = num_local_tokens_per_expert.reshape(
                self.ep_size, self.num_local_experts
            ).sum(axis=1)

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
            output_splits = num_global_tokens_per_rank[self.pg_collection.expt_tp.rank()]

            tokens_per_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))

            # 保存用于 combine
            self.dispatch_input_splits = input_splits
            self.dispatch_output_splits = output_splits
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert

            # 设置反向 AllToAll context
            set_dispatch_alltoall_ctx(
                ep_group=self.ep_group,
                input_splits=output_splits,
                output_splits=input_splits,
                enabled=True,
                num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
                sort_indices=self.sort_input_by_local_experts,
                restore_indices=self.restore_output_by_local_experts,
            )

            # AllToAll probs（这个很小，开销可忽略）
            global_probs = fluid_all_to_all_moe_dispatch(
                permuted_probs, output_splits, input_splits, self.ep_group,
            )
        else:
            global_probs = permuted_probs
            tokens_per_expert = num_local_tokens_per_expert
            self.num_global_tokens_per_local_expert = None

        # TP gather for probs
        if self.tp_size > 1:
            global_probs = gather_from_sequence_parallel_region(global_probs, group=self.tp_group)

        # Sort dispatched_input by local experts
        sorted_input = dispatched_input
        if self.num_local_experts > 1 and self.num_global_tokens_per_local_expert is not None:
            sorted_input, global_probs = sort_chunks_by_idxs(
                dispatched_input,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
                probs=global_probs,
            )

        self.probs = global_probs
        return sorted_input, tokens_per_expert, global_probs

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

            # 使用 Fluid MoE dispatch (不需要单独的 Dispatcher 类!)
            # NOTE: probs 不需要 AllToAll，在 combine 后的 unpermute 里乘（Megatron 标准行为）
            global_input_tokens = fluid_all_to_all_moe_dispatch(
                permuted_tokens,
                output_splits,
                input_splits,
                self.ep_group,
            )

            tokens_per_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))

            # Save for combine and sort/unsort
            self.dispatch_input_splits = input_splits  # Will be output_splits in combine
            self.dispatch_output_splits = output_splits  # Will be input_splits in combine
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert

            # Set dispatch context for chunked backward in Expert computation
            # The backward (combine direction) uses reversed splits
            set_dispatch_alltoall_ctx(
                ep_group=self.ep_group,
                input_splits=output_splits,  # Backward uses reversed splits
                output_splits=input_splits,  # Backward uses reversed splits
                enabled=True,
                num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
                sort_indices=self.sort_input_by_local_experts,
                restore_indices=self.restore_output_by_local_experts,
            )

        else:
            global_input_tokens = permuted_tokens
            tokens_per_expert = num_local_tokens_per_expert
            self.num_global_tokens_per_local_expert = None

        # ===== 步骤 3: AllGather (TP) =====
        if self.tp_size > 1:
            global_input_tokens = gather_from_sequence_parallel_region(
                global_input_tokens, group=self.tp_group
            )

        # ===== 步骤 4: Sort by local experts (当 num_local_experts > 1 时) =====
        # AllToAll 后数据布局是 source-rank-major: [R0_E0][R0_E1][R1_E0][R1_E1]
        # GroupedGEMM 需要 expert-major: [E0_all][E1_all]
        if self.num_local_experts > 1 and self.num_global_tokens_per_local_expert is not None:
            global_input_tokens, _ = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
            )

        return global_input_tokens, tokens_per_expert

    def _combine(
        self,
        expert_output: torch.Tensor,
        restore_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Token Combine: 将 expert 输出收集回原始 GPUs

        标准流程:
        1. Unsort by local experts (当 num_local_experts > 1 时)
        2. ReduceScatter (TP): 如果有 TP
        3. AllToAll (EP): 跨 GPU 收集 tokens
        4. Unpermute: 恢复原始 token 顺序

        Args:
            expert_output: Expert computation output
            restore_shape: Original shape before permutation for unpermute
        """

        hidden_states = expert_output

        # ===== 步骤 1: Unsort by local experts (当 num_local_experts > 1 时) =====
        # Expert 输出是 expert-major: [E0_all][E1_all]
        # AllToAll 需要 source-rank-major: [R0_E0][R0_E1][R1_E0][R1_E1]
        if self.num_local_experts > 1 and self.num_global_tokens_per_local_expert is not None:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),  # Note: transposed for reverse
                self.restore_output_by_local_experts,
            )

        # ===== 步骤 2: ReduceScatter (TP) =====
        if self.tp_size > 1:
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states,
                group=self.tp_group,
            )

        # ===== 步骤 3 & 4: AllToAll Combine + Unpermute =====
        # NOTE: probs 在 unpermute 里乘（Megatron 标准行为）
        num_chunks = get_dx_num_chunks()
        if self.ep_size > 1:
            if num_chunks > 1:
                # Use fused combine + unpermute for true dX + AllToAll pipeline
                output = fluid_fused_combine_unpermute(
                    hidden_states,
                    output_splits=self.dispatch_input_splits,  # Reverse of dispatch
                    input_splits=self.dispatch_output_splits,  # Reverse of dispatch
                    group=self.ep_group,
                    permutation_map=self.permutation_map,
                    restore_shape=restore_shape,
                    probs=self.original_probs,
                    routing_map=self.routing_map,
                )
            else:
                # Standard path: separate combine + unpermute
                hidden_states = fluid_all_to_all_moe_combine(
                    hidden_states,
                    output_splits=self.dispatch_input_splits,  # Reverse of dispatch
                    input_splits=self.dispatch_output_splits,  # Reverse of dispatch
                    group=self.ep_group,
                )
                output = unpermute(
                    hidden_states,
                    self.permutation_map,
                    restore_shape,
                    probs=self.original_probs,
                    routing_map=self.routing_map,
                )
        else:
            # No EP, just unpermute
            output = unpermute(
                hidden_states,
                self.permutation_map,
                restore_shape,
                probs=self.original_probs,
                routing_map=self.routing_map,
            )

        return output

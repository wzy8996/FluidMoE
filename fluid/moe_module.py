# Copyright (c) 2024, FluidMoE Contributors.

"""
完全自定义的 Fluid MoE 模块
支持反向计算-通信重叠（dW 与 AllToAll 并行）

前向：Baseline 模式（dispatch -> FC1 -> FC2 -> combine）
反向：dX 分块 + dW 调度（与 AllToAll 重叠）
"""

import torch
from typing import Optional, Tuple, List, Dict

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
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

from fluid.moe_layers import FluidGroupedMLP, set_dispatch_alltoall_ctx, get_dispatch_alltoall_ctx
from fluid.communication import (
    fluid_all_to_all_moe_dispatch,
    fluid_all_to_all_moe_combine,
    fluid_fused_combine_unpermute,
    get_dx_num_chunks,
)

import os

# 动态读取环境变量（允许运行时切换）
def get_forward_overlap_enabled():
    """Check if forward overlap mode is enabled (动态读取环境变量)"""
    return os.environ.get('FLUID_FORWARD_OVERLAP', '0') == '1'

def get_prealloc_buffers_enabled():
    """Check if pre-allocated buffers are enabled"""
    return os.environ.get('FLUID_PREALLOC_BUFFERS', '0') == '1'


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

        # ===== Pre-allocated P2P buffers for fine-grained mode =====
        # Lazy initialization: allocate on first use and reuse
        self._dispatch_recv_buffers = None  # {peer_rank: buffer}
        self._combine_recv_buffers = None   # {peer_rank: buffer}
        self._buffer_hidden_size = None
        self._buffer_max_tokens = None  # Maximum tokens per peer seen so far

    def _get_or_allocate_p2p_buffers(
        self,
        hidden_size: int,
        max_tokens_per_peer: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Get or allocate pre-allocated P2P buffers for dispatch and combine.

        Uses lazy allocation: allocate on first use, and re-allocate if larger
        buffers are needed.

        Args:
            hidden_size: Hidden dimension size
            max_tokens_per_peer: Maximum tokens expected from/to each peer
            dtype: Tensor dtype
            device: Tensor device

        Returns:
            (dispatch_recv_buffers, combine_recv_buffers)
        """
        my_rank = self.ep_group.rank()

        # Check if we need to (re)allocate
        need_realloc = (
            self._dispatch_recv_buffers is None
            or self._buffer_hidden_size != hidden_size
            or (self._buffer_max_tokens is not None and max_tokens_per_peer > self._buffer_max_tokens)
        )

        if need_realloc:
            # Allocate buffers for each peer
            self._dispatch_recv_buffers = {}
            self._combine_recv_buffers = {}

            for peer_rank in range(self.ep_size):
                if peer_rank == my_rank:
                    continue
                # Allocate with some extra margin (1.5x) to handle variance
                buffer_size = int(max_tokens_per_peer * 1.5) + 1
                self._dispatch_recv_buffers[peer_rank] = torch.empty(
                    buffer_size, hidden_size, dtype=dtype, device=device
                )
                self._combine_recv_buffers[peer_rank] = torch.empty(
                    buffer_size, hidden_size, dtype=dtype, device=device
                )

            self._buffer_hidden_size = hidden_size
            self._buffer_max_tokens = max_tokens_per_peer

        return self._dispatch_recv_buffers, self._combine_recv_buffers

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 保存输入 shape
        input_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, input_shape[-1])

        # ===== 步骤 1: Router =====
        probs, routing_map = self.router(hidden_states)

        # 保存 probs 和 routing_map 用于 _combine 中的 unpermute
        self.original_probs = probs
        self.routing_map = routing_map

        # ===== 选择前向路径 =====
        debug_path = os.environ.get('FLUID_DEBUG_OVERLAP', '0') == '1'
        if debug_path:
            print(f"[DEBUG] ep_size={self.ep_size}, overlap_enabled={get_forward_overlap_enabled()}", flush=True)
        if get_forward_overlap_enabled() and self.ep_size > 1:
            # 前向重叠模式（使用 overlap_forward.py 的新实现）
            if debug_path:
                print(f"[DEBUG] Taking OVERLAP path", flush=True)
            output = self._forward_with_overlap(hidden_states, routing_map, probs)
        else:
            # Baseline 模式：dispatch -> expert -> combine
            import time
            debug_baseline = os.environ.get('FLUID_DEBUG_FORWARD_TIMING', '0') == '1'
            if debug_baseline:
                torch.cuda.synchronize()
                t_start = time.perf_counter()

            dispatched_input, tokens_per_expert = self._dispatch(
                hidden_states,
                routing_map=routing_map,
                probs=probs,
            )

            if debug_baseline:
                torch.cuda.synchronize()
                t_dispatch = time.perf_counter()
                print(f"[Baseline] Dispatch (AllToAll): {(t_dispatch-t_start)*1000:.2f} ms", flush=True)

            # NOTE: probs 不再传给 experts，在 _combine 的 unpermute 里乘
            expert_output, _ = self.experts(
                dispatched_input,
                tokens_per_expert,
            )

            if debug_baseline:
                torch.cuda.synchronize()
                t_expert = time.perf_counter()
                print(f"[Baseline] Expert (FC1+Act+FC2): {(t_expert-t_dispatch)*1000:.2f} ms", flush=True)

            output = self._combine(expert_output, restore_shape=hidden_states.shape)

            if debug_baseline:
                torch.cuda.synchronize()
                t_combine = time.perf_counter()
                print(f"[Baseline] Combine (AllToAll): {(t_combine-t_expert)*1000:.2f} ms", flush=True)
                print(f"[Baseline] Total: {(t_combine-t_start)*1000:.2f} ms", flush=True)

        # 恢复原始 shape
        output = output.view(input_shape)

        return output, None  # 第二个返回值是 mlp_bias

    def _forward_with_overlap(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用多轮 P2P 重叠实现的 MoE 前向（统一实现）

        前向：本地 token 计算与远程 token 通信重叠
        反向：使用标准 AllToAll（保持反向调度不变）

        使用 Round-Robin Tournament 调度：
        - 将"每个rank和所有其他rank交换数据"拆成多轮
        - 若卡数P为偶数：总轮数 R = P-1（2卡时只有1轮）
        - 每轮每张卡只和一个peer通信，避免冲突
        - 通信流在跑第r轮的同时，计算流在吃第r-1轮的数据
        """
        from fluid.multicard_p2p import (
            MultiCardOverlapContext,
            moe_multicard_p2p_overlap_forward,
        )
        import time

        debug_timing = os.environ.get('FLUID_DEBUG_FORWARD_TIMING', '0') == '1'
        if debug_timing:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            print(f"[Forward Overlap MoE] Starting, tokens: {hidden_states.shape[0]}, ep_size: {self.ep_size}", flush=True)

        # ===== Step 1: Permute tokens by expert =====
        permuted_tokens, _, self.permutation_map = permute(
            hidden_states,
            routing_map,
            probs=probs,
            num_out_tokens=routing_map.size(0) * self.config.moe_router_topk,
        )

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

        # Save for unpermute
        self.dispatch_input_splits = input_splits
        self.dispatch_output_splits = output_splits
        self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert

        if debug_timing:
            torch.cuda.synchronize()
            t_permute = time.perf_counter()
            print(f"[Forward Overlap MoE] Permute: {(t_permute-t_start)*1000:.2f} ms", flush=True)

        # ===== Step 2: 计算 tokens_per_expert =====
        tokens_per_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))

        # ===== Step 3: 创建或获取 overlap context =====
        if not hasattr(self, '_multicard_overlap_ctx'):
            self._multicard_overlap_ctx = MultiCardOverlapContext(
                permuted_tokens.device, self.ep_size
            )

        if debug_timing:
            my_rank = self.ep_group.rank()
            print(f"[Forward Overlap MoE Rank {my_rank}] Using Round-Robin P2P with {self._multicard_overlap_ctx.num_rounds} rounds", flush=True)

        # ===== Step 4: 使用统一的多轮 P2P 重叠实现 =====
        combined_output = moe_multicard_p2p_overlap_forward(
            permuted_tokens,
            input_splits,
            output_splits,
            self.experts.weight1,
            self.experts.weight2,
            self.ep_group,
            self.experts.activation_func,
            self._multicard_overlap_ctx,
            layer_id=self.layer_number if self.layer_number is not None else 0,
            num_local_experts=self.num_local_experts,
            tokens_per_expert=tokens_per_expert,
            num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
        )

        if debug_timing:
            torch.cuda.synchronize()
            t_expert = time.perf_counter()
            print(f"[Forward Overlap MoE] P2P overlap expert: {(t_expert-t_permute)*1000:.2f} ms", flush=True)

        # ===== Step 5: Unpermute to restore original token order =====
        output = unpermute(
            combined_output,
            self.permutation_map,
            restore_shape=hidden_states.shape,
            probs=self.original_probs,
            routing_map=self.routing_map,
        )

        if debug_timing:
            torch.cuda.synchronize()
            t_unpermute = time.perf_counter()
            print(f"[Forward Overlap MoE] Unpermute: {(t_unpermute-t_expert)*1000:.2f} ms", flush=True)
            print(f"[Forward Overlap MoE] Total: {(t_unpermute-t_start)*1000:.2f} ms", flush=True)

        return output

    def _compute_fc1_per_expert(
        self,
        input_tokens: torch.Tensor,
        weight1: torch.Tensor,
        source_rank: int,
        output_splits_list: List[int],
    ) -> torch.Tensor:
        """
        对于 num_local_experts > 1，按 expert 分别计算 FC1

        数据按rank排列时，每个rank的token可能发往不同的local experts
        这里简化处理：假设token均匀分配给所有local experts
        """
        # 简化实现：直接用一个大的 matmul
        # 完整实现需要根据 routing 信息分配到各 expert
        return torch.mm(input_tokens, weight1)

    def _compute_fc2_per_expert(
        self,
        activated: torch.Tensor,
        weight2: torch.Tensor,
        source_rank: int,
        output_splits_list: List[int],
    ) -> torch.Tensor:
        """
        对于 num_local_experts > 1，按 expert 分别计算 FC2
        """
        # 简化实现：直接用一个大的 matmul
        return torch.mm(activated, weight2)

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

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn.functional as F

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from torch.nn.parameter import Parameter
from typing import Optional, List

# Import custom Fluid GroupGEMM kernels
import os
_USE_LOOP_FALLBACK = os.environ.get('USE_LOOP_FALLBACK', '0') == '1'

if _USE_LOOP_FALLBACK:
    FLUID_KERNELS_AVAILABLE = False
    print("[FluidMoE] Using loop-based matmul fallback (USE_LOOP_FALLBACK=1)")
else:
    try:
        from fluid.ops import fluid_kernels
        FLUID_KERNELS_AVAILABLE = True
        print("[FluidMoE] Using custom Fluid GroupGEMM kernels")
    except ImportError:
        FLUID_KERNELS_AVAILABLE = False
        print("[FluidMoE] Fluid kernels not available, using loop-based matmul fallback")

# Import chunking configuration
from fluid.communication import get_dx_num_chunks

# Import chunk reordering utilities
from fluid.chunk_reorder import RankChunkReorderContext, FullChunkReorderContext

# Check if chunked kernels are available
try:
    from fluid.ops import fluid_kernels
    HAS_CHUNKED_GEMM = hasattr(fluid_kernels, 'grouped_gemm_single_chunk')
except ImportError:
    HAS_CHUNKED_GEMM = False

# Global context for passing AllToAll parameters from dispatch to expert backward
# This is a thread-local storage to support multi-GPU training
import threading
_dispatch_ctx = threading.local()

def set_dispatch_alltoall_ctx(
    ep_group,
    input_splits: List[int],
    output_splits: List[int],
    enabled: bool = True,
    num_global_tokens_per_local_expert=None,
    sort_indices=None,
    restore_indices=None,
):
    """Set dispatch AllToAll context for chunked backward."""
    _dispatch_ctx.ep_group = ep_group
    _dispatch_ctx.input_splits = input_splits
    _dispatch_ctx.output_splits = output_splits
    _dispatch_ctx.enabled = enabled
    _dispatch_ctx.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert
    _dispatch_ctx.sort_indices = sort_indices
    _dispatch_ctx.restore_indices = restore_indices

def get_dispatch_alltoall_ctx():
    """Get dispatch AllToAll context."""
    return (
        getattr(_dispatch_ctx, 'ep_group', None),
        getattr(_dispatch_ctx, 'input_splits', None),
        getattr(_dispatch_ctx, 'output_splits', None),
        getattr(_dispatch_ctx, 'enabled', False),
        getattr(_dispatch_ctx, 'num_global_tokens_per_local_expert', None),
        getattr(_dispatch_ctx, 'sort_indices', None),
        getattr(_dispatch_ctx, 'restore_indices', None),
    )

def clear_dispatch_alltoall_ctx():
    """Clear dispatch AllToAll context."""
    _dispatch_ctx.ep_group = None
    _dispatch_ctx.input_splits = None
    _dispatch_ctx.output_splits = None
    _dispatch_ctx.enabled = False
    _dispatch_ctx.num_global_tokens_per_local_expert = None
    _dispatch_ctx.sort_indices = None
    _dispatch_ctx.restore_indices = None


# Activation gradient computation helpers
import math
_SQRT_2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)  # 1/√(2π) ≈ 0.3989422804

def _gelu_grad_exact(x):
    """
    Exact GELU gradient using erf (error function).

    GELU(x) = x * Φ(x) = 0.5 * x * (1 + erf(x/√2))

    GELU'(x) = Φ(x) + x * φ(x)
             = 0.5 * (1 + erf(x/√2)) + x * exp(-x²/2) / √(2π)

    where:
    - Φ(x) = CDF of standard normal = 0.5 * (1 + erf(x/√2))
    - φ(x) = PDF of standard normal = exp(-x²/2) / √(2π)
    """
    # Φ(x) = 0.5 * (1 + erf(x/√2))
    cdf = 0.5 * (1.0 + torch.erf(x / _SQRT_2))
    # φ(x) = exp(-x²/2) / √(2π)
    pdf = torch.exp(-0.5 * x * x) * _INV_SQRT_2PI
    return cdf + x * pdf

# 默认使用精确公式（与 PyTorch approximate='none' 一致）
def _gelu_grad_analytical(x):
    """GELU gradient - uses exact erf-based formula by default."""
    return _gelu_grad_exact(x)


# This is the new autograd function that implements the parallel backward pass.
class _FluidExpertComputation(torch.autograd.Function):
    """
    Custom autograd function for MoE expert computation that parallelizes
    the computation of dW and dX in the backward pass.
    """

    @staticmethod
    def forward(
        ctx,
        permuted_local_hidden_states,
        tokens_per_expert,
        permuted_probs,
        weight1,
        weight2,
        num_local_experts,
        hidden_size,
        ffn_hidden_size,
        gated_linear_unit,
        activation_func,
        moe_apply_probs_on_input,
        moe_router_topk,
        activation_func_type,  # 'gelu' or 'silu'
    ):
        # Save tensors and configs for backward
        ctx.save_for_backward(
            permuted_local_hidden_states, tokens_per_expert, permuted_probs, weight1, weight2
        )
        ctx.num_local_experts = num_local_experts
        ctx.hidden_size = hidden_size
        ctx.ffn_hidden_size = ffn_hidden_size
        ctx.gated_linear_unit = gated_linear_unit
        ctx.activation_func = activation_func
        ctx.moe_apply_probs_on_input = moe_apply_probs_on_input
        ctx.activation_func_type = activation_func_type

        # Capture dispatch AllToAll context for chunked backward
        # (thread-local doesn't persist across forward/backward, so save in ctx)
        (ep_group, input_splits, output_splits, ctx_enabled,
         num_global_tokens_per_local_expert, sort_indices, restore_indices) = get_dispatch_alltoall_ctx()
        ctx.dispatch_ep_group = ep_group
        ctx.dispatch_input_splits = input_splits
        ctx.dispatch_output_splits = output_splits
        ctx.dispatch_ctx_enabled = ctx_enabled
        ctx.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert
        ctx.sort_indices = sort_indices
        ctx.restore_indices = restore_indices
        # Clear thread-local context after capturing
        if ctx_enabled:
            clear_dispatch_alltoall_ctx()

        # --- Re-implement the forward pass from GroupedMLP ---
        if moe_apply_probs_on_input:
            assert moe_router_topk == 1, "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() == 0:
            # Make sure params of experts still have gradients even given zero tokens.
            w1 = weight1.view(hidden_size, -1)
            w2 = weight2.view(-1, hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = activation_func(h, permuted_probs.unsqueeze(-1))
            fc2_output = torch.matmul(h, w2)
            ctx.fc1_output = h
            ctx.intermediate_parallel = h  # Simplified for zero-element case
            return fc2_output

        # Reshape the weights for the grouped GEMMs.
        w1 = weight1.view(num_local_experts, hidden_size, -1)
        w2 = weight2.view(num_local_experts, -1, hidden_size)

        if FLUID_KERNELS_AVAILABLE:
            # Use custom Fluid GroupGEMM kernels
            tokens_per_expert_int = tokens_per_expert.to(torch.int32)
            fc1_output = fluid_kernels.grouped_gemm(
                permuted_local_hidden_states.half(), w1.half(),
                tokens_per_expert_int, trans_a=False, trans_b=False
            ).to(permuted_local_hidden_states.dtype)
            intermediate_parallel = activation_func(fc1_output, permuted_probs)
            fc2_output = fluid_kernels.grouped_gemm(
                intermediate_parallel.half(), w2.half(),
                tokens_per_expert_int, trans_a=False, trans_b=False
            ).to(permuted_local_hidden_states.dtype)
        else:
            # Loop fallback
            total_tokens = permuted_local_hidden_states.shape[0]
            ffn_size = w1.shape[2]
            fc1_output = torch.zeros(total_tokens, ffn_size, dtype=permuted_local_hidden_states.dtype, device=permuted_local_hidden_states.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                if n_tok > 0:
                    fc1_output[start:start+n_tok] = torch.matmul(permuted_local_hidden_states[start:start+n_tok], w1[exp_idx])
                    start += n_tok
            intermediate_parallel = activation_func(fc1_output, permuted_probs)
            fc2_output = torch.zeros(total_tokens, hidden_size, dtype=permuted_local_hidden_states.dtype, device=permuted_local_hidden_states.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                if n_tok > 0:
                    fc2_output[start:start+n_tok] = torch.matmul(intermediate_parallel[start:start+n_tok], w2[exp_idx])
                    start += n_tok

        # Pre-compute activation derivative in forward to speed up backward critical path
        # For non-GLU: save gelu'(fc1_output)
        # For GLU: save (gelu'(x_1), gelu(x_1), x_2) for grad_fc1 computation
        if gated_linear_unit:
            x_1, x_2 = torch.chunk(fc1_output, 2, dim=-1)
            if activation_func_type == 'silu':
                sig = torch.sigmoid(x_1)
                act_deriv = sig * (1 + x_1 * (1 - sig))
                act_val = x_1 * sig
            else:
                act_deriv = _gelu_grad_analytical(x_1)
                act_val = F.gelu(x_1)
            ctx.act_deriv = act_deriv  # activation'(x_1)
            ctx.act_val = act_val      # activation(x_1)
            ctx.x_2 = x_2              # x_2 for grad_x_1 = grad * act_deriv * x_2
        else:
            if activation_func_type == 'silu':
                sig = torch.sigmoid(fc1_output)
                ctx.act_deriv = sig * (1 + fc1_output * (1 - sig))
            else:
                ctx.act_deriv = _gelu_grad_analytical(fc1_output)

        # Save intermediate tensors for backward
        ctx.fc1_output = fc1_output
        ctx.intermediate_parallel = intermediate_parallel

        return fc2_output

    @staticmethod
    def backward(ctx, grad_fc2_output):
        # Retrieve saved tensors
        (
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
            weight1,
            weight2,
        ) = ctx.saved_tensors
        fc1_output = ctx.fc1_output
        intermediate_parallel = ctx.intermediate_parallel

        # Retrieve configs
        num_local_experts = ctx.num_local_experts
        hidden_size = ctx.hidden_size
        activation_func = ctx.activation_func
        moe_apply_probs_on_input = ctx.moe_apply_probs_on_input
        gated_linear_unit = ctx.gated_linear_unit
        activation_func_type = ctx.activation_func_type

        # Get Fluid scheduler
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        # Reshape weights
        w1 = weight1.view(num_local_experts, hidden_size, -1)
        w2 = weight2.view(num_local_experts, -1, hidden_size)
        ffn_hidden_size = ctx.ffn_hidden_size

        # Retrieve pre-computed activation derivatives from forward
        act_deriv = ctx.act_deriv

        # Ensure probs has correct shape
        probs = permuted_probs.view(-1, 1) if permuted_probs.dim() == 1 else permuted_probs
        if probs.dim() == 1:
            probs = probs.unsqueeze(-1)

        # Check if chunked dX + AllToAll pipeline should be used
        num_chunks = get_dx_num_chunks()
        use_chunked = (
            num_chunks > 1 and
            HAS_CHUNKED_GEMM and
            ctx.dispatch_ctx_enabled and
            grad_fc2_output.shape[0] > 0
        )

        if use_chunked:
            # ============================================================
            # TRUE Chunked dX + AllToAll Pipeline
            # ============================================================
            # 使用 grouped_gemm_single_chunk 实现真正的流水线：
            # 1. 计算 dX_chunk_i
            # 2. 立即启动 AllToAll_chunk_i (异步)
            # 3. 同时开始计算 dX_chunk_{i+1}
            #
            # Timeline:
            # default_stream: |-- dX_0 --|-- dX_1 --|-- dX_2 --|-- ...
            # comm_stream:               |--- A2A_0 ---|
            #                                         |--- A2A_1 ---|
            #                                                      |--- A2A_2 ---| <- dW here
            #
            # 注意：当 num_local_experts > 1 时，数据需要 unsort
            # 这里实现的是 num_local_experts == 1 的优化情况
            # num_local_experts > 1 时回退到标准路径（先完整计算再分块通信）
            # ============================================================
            from fluid.scheduler import get_backward_scheduler
            from megatron.core.tensor_parallel.mappings import _AllToAll

            scheduler = get_backward_scheduler()
            ep_group = ctx.dispatch_ep_group

            # Naming clarification:
            # ctx.dispatch_input_splits = what was received in dispatch = data we have now
            # ctx.dispatch_output_splits = what was sent in dispatch = what we'll receive after AllToAll
            data_per_rank = ctx.dispatch_input_splits
            recv_per_rank = ctx.dispatch_output_splits

            data_per_rank_list = data_per_rank.tolist() if hasattr(data_per_rank, 'tolist') else list(data_per_rank)
            recv_per_rank_list = recv_per_rank.tolist() if hasattr(recv_per_rank, 'tolist') else list(recv_per_rank)
            ep_size = len(data_per_rank_list)

            total_tokens = grad_fc2_output.shape[0]
            intermediate_dim = w2.shape[1]

            # 当 num_local_experts == 1 时，可以实现真正的 dX + AllToAll 流水线
            # 数据布局：[from_rank0][from_rank1]...（source-rank-major）
            # 每个 chunk 按 rank 分，计算该 chunk 的 dX，然后立即 AllToAll
            import os
            _DEBUG_TIMING = os.environ.get('FLUID_DEBUG_TIMING', '0') == '1'

            # CUDA事件计时 - 用于精确测量overlap
            if _DEBUG_TIMING:
                _cuda_events = {
                    'dx_start': [], 'dx_end': [],
                    'a2a_start': [], 'a2a_queued': [], 'a2a_end': [],
                    'a2a_wait_time': [],
                }

            if num_local_experts == 1:
                # ============================================================
                # TRUE PIPELINE for num_local_experts == 1
                # ============================================================
                # 优化：使用 grouped_gemm_single_chunk 直接按 chunk 计算
                # 避免 Python 循环中对每个 rank 调用 matmul 的开销
                #
                # 数据布局：[from_rank0][from_rank1]...（source-rank-major）
                # 每个 chunk 取每个 rank 的 1/N，拼接后仍然是连续的
                # ============================================================

                # 计算 rank offsets
                rank_offsets = [0]
                for split in data_per_rank_list:
                    rank_offsets.append(rank_offsets[-1] + split)

                # 存储异步 AllToAll 的信息：(work, output_buffer, chunk_output_splits)
                async_alltoall_infos = []
                # 存储每个 chunk 的 grad_fc1 和对应的索引
                chunk_grad_fc1_list = []
                chunk_indices_list = []

                # 预计算每个 chunk 的边界（避免重复计算）
                chunk_bounds = []  # [(chunk_start_per_rank, chunk_end_per_rank), ...]
                # 预计算所有chunk的信息（包括索引tensor）
                all_chunk_idx_tensors = []
                all_chunk_input_splits = []
                all_chunk_output_splits = []

                for chunk_idx in range(num_chunks):
                    bounds_this_chunk = []
                    chunk_indices = []
                    chunk_input_splits = []

                    for rank in range(ep_size):
                        rank_tokens = data_per_rank_list[rank]
                        if rank_tokens == 0:
                            bounds_this_chunk.append((0, 0))
                            chunk_input_splits.append(0)
                        else:
                            chunk_size = (rank_tokens + num_chunks - 1) // num_chunks
                            chunk_start = chunk_idx * chunk_size
                            chunk_end = min(chunk_start + chunk_size, rank_tokens)
                            bounds_this_chunk.append((chunk_start, chunk_end))
                            this_chunk_size = max(0, chunk_end - chunk_start)
                            chunk_input_splits.append(this_chunk_size)
                            if this_chunk_size > 0:
                                global_start = rank_offsets[rank] + chunk_start
                                global_end = rank_offsets[rank] + chunk_end
                                chunk_indices.extend(range(global_start, global_end))

                    chunk_bounds.append(bounds_this_chunk)
                    all_chunk_input_splits.append(chunk_input_splits)

                    # 预先创建索引tensor
                    if chunk_indices:
                        all_chunk_idx_tensors.append(
                            torch.tensor(chunk_indices, dtype=torch.int32, device=grad_fc2_output.device)
                        )
                    else:
                        all_chunk_idx_tensors.append(None)

                    # 预计算output splits
                    chunk_output_splits = []
                    for rank in range(ep_size):
                        rank_tokens = recv_per_rank_list[rank]
                        if rank_tokens == 0:
                            chunk_output_splits.append(0)
                            continue
                        chunk_size = (rank_tokens + num_chunks - 1) // num_chunks
                        chunk_start = chunk_idx * chunk_size
                        chunk_end = min(chunk_start + chunk_size, rank_tokens)
                        chunk_output_splits.append(max(0, chunk_end - chunk_start))
                    all_chunk_output_splits.append(chunk_output_splits)

                if _DEBUG_TIMING:
                    _loop_start_event = torch.cuda.Event(enable_timing=True)
                    _loop_start_event.record()

                # ============================================================
                # TRUE PIPELINED dX + AllToAll using C++ events
                # ============================================================
                # 1. Launch ALL dX kernels in C++ (non-blocking, returns immediately)
                # 2. C++ records a CUDA event after each chunk completes
                # 3. Python loops through chunks, waits on event, launches AllToAll
                #
                # Timeline:
                #   C++ default_stream: |-- dX_0 --|-- dX_1 --|-- dX_2 --|
                #                              ^event0    ^event1    ^event2
                #   Python comm_stream:        wait(e0) |--- A2A_0 ---|
                #                                       wait(e1) |--- A2A_1 ---|
                # ============================================================

                # Prepare data
                all_indices_list = [t for t in all_chunk_idx_tensors if t is not None]
                if all_indices_list:
                    all_chunk_indices = torch.cat(all_indices_list, dim=0)
                    # Compute offsets for each chunk
                    chunk_offsets = [0]
                    for t in all_chunk_idx_tensors:
                        if t is not None:
                            chunk_offsets.append(chunk_offsets[-1] + len(t))
                        else:
                            chunk_offsets.append(chunk_offsets[-1])
                    chunk_offsets_tensor = torch.tensor(chunk_offsets, dtype=torch.int32, device=grad_fc2_output.device)

                    if _DEBUG_TIMING:
                        _e_dx_start = torch.cuda.Event(enable_timing=True)
                        _e_dx_start.record()
                        _cuda_events['dx_start'].append(_e_dx_start)

                    # Launch ALL dX computations and get events for each chunk
                    if gated_linear_unit:
                        full_dx, full_grad_fc1, event_ptrs = fluid_kernels.grouped_gemm_dx_pipelined(
                            grad_fc2_output.half(),
                            probs.half(),
                            act_deriv.half(),
                            w1.half(),
                            w2.half(),
                            all_chunk_indices,
                            chunk_offsets_tensor,
                            num_chunks,
                            ctx.act_val.half(),
                            ctx.x_2.half()
                        )
                    else:
                        full_dx, full_grad_fc1, event_ptrs = fluid_kernels.grouped_gemm_dx_pipelined(
                            grad_fc2_output.half(),
                            probs.half(),
                            act_deriv.half(),
                            w1.half(),
                            w2.half(),
                            all_chunk_indices,
                            chunk_offsets_tensor,
                            num_chunks
                        )

                    full_dx = full_dx.to(grad_fc2_output.dtype)
                    full_grad_fc1 = full_grad_fc1.to(grad_fc2_output.dtype)

                    # Store grad_fc1 for each chunk
                    for chunk_idx in range(num_chunks):
                        chunk_start = chunk_offsets[chunk_idx]
                        chunk_end = chunk_offsets[chunk_idx + 1]
                        if chunk_end > chunk_start:
                            chunk_grad_fc1_list.append(full_grad_fc1[chunk_start:chunk_end])
                            chunk_indices_list.append(all_chunk_idx_tensors[chunk_idx])

                    # Get comm stream pointer for C++ wait
                    comm_stream_ptr = scheduler.comm_stream.cuda_stream

                    # Now launch AllToAll for each chunk, waiting on the corresponding event
                    for chunk_idx in range(num_chunks):
                        chunk_input_splits = all_chunk_input_splits[chunk_idx]
                        chunk_output_splits = all_chunk_output_splits[chunk_idx]
                        chunk_start = chunk_offsets[chunk_idx]
                        chunk_end = chunk_offsets[chunk_idx + 1]

                        if chunk_end > chunk_start:
                            chunk_dx = full_dx[chunk_start:chunk_end]
                        else:
                            chunk_dx = torch.zeros(0, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

                        chunk_dx_contiguous = chunk_dx.contiguous()
                        chunk_output_size = sum(chunk_output_splits)

                        # Allocate output buffer
                        chunk_output_buffer = chunk_dx_contiguous.new_empty(
                            size=[chunk_output_size] + list(chunk_dx_contiguous.size()[1:]),
                            dtype=chunk_dx_contiguous.dtype,
                            device=torch.cuda.current_device(),
                        )

                        if _DEBUG_TIMING:
                            _e_a2a_start = torch.cuda.Event(enable_timing=True)

                        with torch.cuda.stream(scheduler.comm_stream):
                            # Wait on the C++ event for this chunk (inside comm_stream context!)
                            event_ptr = event_ptrs[chunk_idx].item()
                            fluid_kernels.wait_cuda_event(event_ptr, comm_stream_ptr)

                            if _DEBUG_TIMING:
                                _e_a2a_start.record(scheduler.comm_stream)
                                _cuda_events['a2a_start'].append(_e_a2a_start)

                            work = torch.distributed.all_to_all_single(
                                chunk_output_buffer,
                                chunk_dx_contiguous,
                                output_split_sizes=chunk_output_splits,
                                input_split_sizes=chunk_input_splits,
                                group=ep_group,
                                async_op=True,
                            )

                        async_alltoall_infos.append((work, chunk_output_buffer, chunk_output_splits))

                        if chunk_idx == num_chunks - 1:
                            scheduler.on_alltoall_start(comm_type="moe_combine")

                    # Clean up CUDA events
                    fluid_kernels.destroy_cuda_events(event_ptrs)

                    if _DEBUG_TIMING:
                        _e_dx_end = torch.cuda.Event(enable_timing=True)
                        _e_dx_end.record()
                        _cuda_events['dx_end'].append(_e_dx_end)
                else:
                    # No indices - empty case
                    chunk_offsets = [0] * (num_chunks + 1)

                # ============================================================
                # 等待所有异步 AllToAll 完成并重组结果
                # ============================================================
                if _DEBUG_TIMING:
                    import time
                    _e_before_wait_stream = torch.cuda.Event(enable_timing=True)
                    _e_before_wait_stream.record()

                scheduler.default_stream.wait_stream(scheduler.comm_stream)

                if _DEBUG_TIMING:
                    _e_after_wait_stream = torch.cuda.Event(enable_timing=True)
                    _e_after_wait_stream.record()

                # 等待每个异步 AllToAll 的 work handle，并测量时间
                if _DEBUG_TIMING:
                    _wait_start = time.time()

                for i, (work, _, _) in enumerate(async_alltoall_infos):
                    if work is not None:
                        if _DEBUG_TIMING:
                            _t_wait_start = time.time()
                        work.wait()
                        if _DEBUG_TIMING:
                            _t_wait_end = time.time()
                            _cuda_events['a2a_wait_time'].append((_t_wait_end - _t_wait_start) * 1000)
                            _e_a2a_end = torch.cuda.Event(enable_timing=True)
                            _e_a2a_end.record()
                            _cuda_events['a2a_end'].append(_e_a2a_end)

                # 打印 CUDA 时间统计
                if _DEBUG_TIMING and len(_cuda_events['dx_start']) > 0:
                    torch.cuda.synchronize()
                    _wait_end = time.time()
                    total_dx_time = 0.0
                    total_a2a_time = 0.0

                    # 打印每个 chunk 的详细时间线
                    _VERBOSE_TIMING = os.environ.get('FLUID_VERBOSE_TIMING', '0') == '1'
                    if _VERBOSE_TIMING:
                        print(f"  [Timeline] ref_time = dx_start[0]")

                    for i in range(len(_cuda_events['dx_start'])):
                        dx_time = _cuda_events['dx_start'][i].elapsed_time(_cuda_events['dx_end'][i])
                        a2a_time = 0.0
                        if i < len(_cuda_events['a2a_start']) and i < len(_cuda_events['a2a_end']):
                            a2a_time = _cuda_events['a2a_start'][i].elapsed_time(_cuda_events['a2a_end'][i])

                        if _VERBOSE_TIMING:
                            # 计算相对于第一个 dX 开始的时间
                            dx_start_rel = _cuda_events['dx_start'][0].elapsed_time(_cuda_events['dx_start'][i])
                            dx_end_rel = _cuda_events['dx_start'][0].elapsed_time(_cuda_events['dx_end'][i])
                            if i < len(_cuda_events['a2a_start']):
                                a2a_start_rel = _cuda_events['dx_start'][0].elapsed_time(_cuda_events['a2a_start'][i])
                            else:
                                a2a_start_rel = -1
                            if i < len(_cuda_events['a2a_end']):
                                a2a_end_rel = _cuda_events['dx_start'][0].elapsed_time(_cuda_events['a2a_end'][i])
                            else:
                                a2a_end_rel = -1
                            print(f"  [Chunk {i}] dX: {dx_start_rel:.2f}-{dx_end_rel:.2f}ms ({dx_time:.2f}ms), "
                                  f"A2A: {a2a_start_rel:.2f}-{a2a_end_rel:.2f}ms ({a2a_time:.2f}ms)")

                        total_dx_time += dx_time
                        total_a2a_time += a2a_time
                    if _cuda_events['a2a_end']:
                        e2e_time = _cuda_events['dx_start'][0].elapsed_time(_cuda_events['a2a_end'][-1])
                    else:
                        e2e_time = total_dx_time
                    theoretical = total_dx_time + total_a2a_time
                    overlap_pct = 100 * (theoretical - e2e_time) / theoretical if theoretical > 0 else 0
                    print(f"[dX+A2A Pipeline] chunks={num_chunks}, dX={total_dx_time:.1f}ms, A2A={total_a2a_time:.1f}ms, "
                          f"E2E={e2e_time:.1f}ms, overlap={overlap_pct:.0f}%")

                # Reassemble results
                dest_rank_chunks = [[] for _ in range(ep_size)]
                for chunk_idx, (_, chunk_output_buffer, chunk_splits) in enumerate(async_alltoall_infos):
                    offset = 0
                    for dest_rank in range(ep_size):
                        split_size = chunk_splits[dest_rank]
                        if split_size > 0:
                            dest_rank_chunks[dest_rank].append(chunk_output_buffer[offset:offset+split_size])
                            offset += split_size

                final_parts = []
                for dest_rank in range(ep_size):
                    if dest_rank_chunks[dest_rank]:
                        final_parts.append(torch.cat(dest_rank_chunks[dest_rank], dim=0))

                if final_parts:
                    ctx.chunked_alltoall_result = torch.cat(final_parts, dim=0)
                else:
                    ctx.chunked_alltoall_result = torch.zeros(sum(recv_per_rank_list), hidden_size,
                                                              dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

                # Reassemble grad_fc1 from chunks (no recomputation needed!)
                # The kernel already computed grad_fc1 for each chunk
                fc1_dim = w1.shape[2]  # ffn_size for non-GLU, 2*intermediate_size for GLU
                if chunk_grad_fc1_list:
                    # Scatter chunk results back to original positions
                    grad_fc1 = torch.zeros(total_tokens, fc1_dim, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
                    for chunk_grad_fc1, chunk_indices in zip(chunk_grad_fc1_list, chunk_indices_list):
                        # chunk_indices contains the global indices for this chunk
                        grad_fc1[chunk_indices.long()] = chunk_grad_fc1
                else:
                    grad_fc1 = torch.zeros(total_tokens, fc1_dim, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

                # grad_permuted_local_hidden_states 已通过分块计算
                grad_permuted_local_hidden_states = torch.zeros(total_tokens, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

            else:
                # ============================================================
                # TRUE PIPELINE for num_local_experts > 1
                # ============================================================
                # 策略：对每个 (rank, expert) 对各取 1/N token
                #
                # 对每个 chunk:
                # 1. 提取 chunk 对应的输入数据（每个 (rank, expert) 的 1/N）
                # 2. 用普通 grouped_gemm 计算这个子集的 dX
                # 3. 启动 AllToAll (异步)
                # 4. 同时计算下一个 chunk
                #
                # 关键洞察：
                # - 数据布局是 expert-major: [E0_all][E1_all]...
                # - 每个 expert 内部是 rank-major: [R0_Ei][R1_Ei]...
                # - chunk_i 从每个 (rank, expert) 取第 i 个 1/N 段
                # - 这些段拼接后，用普通 grouped_gemm 计算 dX
                # - 结果直接就是要 AllToAll 的数据（source-rank-major）
                # ============================================================
                tokens_per_expert_list = tokens_per_expert.tolist()

                # Get token distribution per (rank, expert)
                num_global_per_expert = ctx.num_global_tokens_per_local_expert  # [tp, ep, num_local_experts]
                if num_global_per_expert is not None and num_global_per_expert.dim() == 3:
                    tokens_per_rank_expert = num_global_per_expert.sum(dim=0)  # [ep, num_local_experts]
                    tokens_per_rank_expert_list = tokens_per_rank_expert.tolist()
                else:
                    # Fallback to standard path if no per-rank info
                    tokens_per_rank_expert = None
                    tokens_per_rank_expert_list = None

                # Expert offsets in expert-major layout
                expert_offsets = [0]
                for exp_idx in range(num_local_experts):
                    expert_offsets.append(expert_offsets[-1] + int(tokens_per_expert_list[exp_idx]))

                # Precompute rank offsets within each expert
                # rank_offsets_in_expert[expert][rank] = start offset of rank's tokens in expert
                rank_offsets_in_expert = []
                if tokens_per_rank_expert_list is not None:
                    for exp_idx in range(num_local_experts):
                        offsets = [0]
                        for rank in range(ep_size):
                            offsets.append(offsets[-1] + int(tokens_per_rank_expert_list[rank][exp_idx]))
                        rank_offsets_in_expert.append(offsets)

                alltoall_results = []

                # 用于存储完整的 grad_permuted（用于 dW 计算）
                grad_permuted_full = torch.zeros(total_tokens, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)


                # 存储异步 AllToAll 的信息：(work, output_buffer, chunk_output_splits)
                async_alltoall_infos = []

                for chunk_idx in range(num_chunks):
                    # ============================================================
                    # Step 1: 提取 chunk 对应的输入数据
                    # ============================================================
                    # 策略：按每个 rank 的总数据量的 1/N 分块（而不是按 (rank, expert) 对分块）
                    # 这样可以保证 chunk_input_splits 和 chunk_output_splits 用相同的分块策略
                    #
                    # 数据布局（expert-major）: [E0_all][E1_all]...
                    #   其中 E_i_all = [R0_Ei][R1_Ei]...
                    #
                    # 我们需要按 rank 分块提取数据：
                    # 对 rank R，总数据量 = sum(tokens_per_rank_expert_list[R])
                    # chunk_i 取这个总量的第 i 个 1/N

                    chunk_grad_fc2_parts = []  # 按 rank 组织的输入
                    chunk_act_deriv_parts = []
                    chunk_probs_parts = []
                    chunk_act_val_parts = [] if gated_linear_unit else None
                    chunk_x_2_parts = [] if gated_linear_unit else None
                    chunk_input_splits = []

                    # 记录每个提取段的位置，用于写回 grad_permuted_full
                    chunk_positions = []  # [(global_src_start, global_src_end, chunk_dst_start, chunk_dst_end), ...]
                    chunk_offset = 0

                    for rank in range(ep_size):
                        # 计算这个 rank 的总数据量
                        rank_total = sum(int(tokens_per_rank_expert_list[rank][e]) for e in range(num_local_experts))

                        if rank_total == 0:
                            chunk_input_splits.append(0)
                            continue

                        # 按总量分块
                        per_chunk = (rank_total + num_chunks - 1) // num_chunks
                        chunk_start_in_rank = chunk_idx * per_chunk
                        chunk_end_in_rank = min(chunk_start_in_rank + per_chunk, rank_total)
                        chunk_size_for_rank = max(0, chunk_end_in_rank - chunk_start_in_rank)

                        if chunk_size_for_rank == 0:
                            chunk_input_splits.append(0)
                            continue

                        chunk_input_splits.append(chunk_size_for_rank)

                        # 从各 expert 中提取数据
                        # rank 的数据在各 expert 中分布：[R_E0][R_E1]...（expert-major 顺序）
                        # 我们需要从位置 chunk_start_in_rank 到 chunk_end_in_rank 提取
                        remaining_start = chunk_start_in_rank
                        remaining_size = chunk_size_for_rank

                        for exp_idx in range(num_local_experts):
                            if remaining_size <= 0:
                                break

                            rank_tokens_in_exp = int(tokens_per_rank_expert_list[rank][exp_idx])
                            if rank_tokens_in_exp == 0:
                                continue

                            # 检查是否跳过这个 expert
                            if remaining_start >= rank_tokens_in_exp:
                                remaining_start -= rank_tokens_in_exp
                                continue

                            # 从这个 expert 提取数据
                            local_start = remaining_start
                            local_end = min(local_start + remaining_size, rank_tokens_in_exp)
                            extract_size = local_end - local_start

                            # 在 expert-major 布局中的全局索引
                            rank_start_in_exp = rank_offsets_in_expert[exp_idx][rank]
                            global_start = expert_offsets[exp_idx] + rank_start_in_exp + local_start
                            global_end = expert_offsets[exp_idx] + rank_start_in_exp + local_end

                            # 提取数据
                            chunk_grad_fc2_parts.append(grad_fc2_output[global_start:global_end])
                            chunk_act_deriv_parts.append(act_deriv[global_start:global_end])
                            chunk_probs_parts.append(probs[global_start:global_end])
                            if gated_linear_unit:
                                chunk_act_val_parts.append(ctx.act_val[global_start:global_end])
                                chunk_x_2_parts.append(ctx.x_2[global_start:global_end])

                            # 记录位置用于写回
                            chunk_positions.append((global_start, global_end, chunk_offset, chunk_offset + extract_size, exp_idx))
                            chunk_offset += extract_size

                            remaining_start = 0
                            remaining_size -= extract_size

                    # chunk_output_splits 使用相同的分块策略
                    # recv_per_rank_list[rank] = 我从 rank 收到的总数据量（combine 时）
                    chunk_output_splits = []
                    for rank in range(ep_size):
                        rank_total_to_me = recv_per_rank_list[rank]
                        if rank_total_to_me == 0:
                            chunk_output_splits.append(0)
                            continue
                        per_chunk = (rank_total_to_me + num_chunks - 1) // num_chunks
                        c_start = chunk_idx * per_chunk
                        c_end = min(c_start + per_chunk, rank_total_to_me)
                        chunk_output_splits.append(max(0, c_end - c_start))

                    if not chunk_grad_fc2_parts:
                        # 空 chunk
                        chunk_dx = torch.zeros(0, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
                    else:
                        # ============================================================
                        # Step 2: 计算这个 chunk 的 dX
                        # ============================================================
                        # 策略：将 source-rank-major 数据重组为 expert-major，
                        #       使用 grouped_gemm 高效计算，再重组回 source-rank-major
                        #
                        # 当前数据布局 (source-rank-major): [R0_data][R1_data]...
                        #   其中 R_i_data 可能包含多个 expert 的数据
                        # 目标布局 (expert-major): [E0_all][E1_all]...
                        #
                        # chunk_positions 格式: [(global_src, global_end, chunk_start, chunk_end, exp_idx), ...]

                        chunk_total = sum(p[3] - p[2] for p in chunk_positions)

                        # Step 2a: 按 expert 分组，构建 expert-major 数据
                        expert_segments = [[] for _ in range(num_local_experts)]  # expert_segments[exp] = [(chunk_start, chunk_end, global_src_start), ...]
                        for global_src_start, global_src_end, chunk_dst_start, chunk_dst_end, exp_idx in chunk_positions:
                            if chunk_dst_end > chunk_dst_start:
                                expert_segments[exp_idx].append((chunk_dst_start, chunk_dst_end, global_src_start))

                        # 计算每个 expert 在 chunk 中的 token 数
                        chunk_tokens_per_expert = [sum(end - start for start, end, _ in segs) for segs in expert_segments]

                        # 构建 expert-major 的索引映射
                        # src_to_exp_major[i] = chunk 数据中位置 i 在 expert-major 布局中的位置
                        # exp_major_to_src[i] = expert-major 布局中位置 i 在 chunk 数据中的位置
                        exp_major_to_src = []
                        src_to_exp_major = [0] * chunk_total
                        exp_major_offset = 0
                        for exp_idx in range(num_local_experts):
                            for chunk_start, chunk_end, _ in expert_segments[exp_idx]:
                                for i in range(chunk_start, chunk_end):
                                    src_to_exp_major[i] = exp_major_offset
                                    exp_major_to_src.append(i)
                                    exp_major_offset += 1

                        # Step 2b: 重组数据为 expert-major
                        chunk_grad_fc2_src = torch.cat(chunk_grad_fc2_parts, dim=0)
                        chunk_act_deriv_src = torch.cat(chunk_act_deriv_parts, dim=0)
                        chunk_probs_src = torch.cat(chunk_probs_parts, dim=0)

                        # 使用索引重排
                        exp_major_indices = torch.tensor(exp_major_to_src, device=chunk_grad_fc2_src.device, dtype=torch.long)
                        chunk_grad_fc2_exp = chunk_grad_fc2_src[exp_major_indices]
                        chunk_act_deriv_exp = chunk_act_deriv_src[exp_major_indices]
                        chunk_probs_exp = chunk_probs_src[exp_major_indices]
                        if gated_linear_unit:
                            chunk_act_val_src = torch.cat(chunk_act_val_parts, dim=0)
                            chunk_x_2_src = torch.cat(chunk_x_2_parts, dim=0)
                            chunk_act_val_exp = chunk_act_val_src[exp_major_indices]
                            chunk_x_2_exp = chunk_x_2_src[exp_major_indices]

                        # Step 2c: 使用 grouped_gemm 计算 dX
                        chunk_tokens_per_expert_t = torch.tensor(chunk_tokens_per_expert, dtype=torch.int32, device=chunk_grad_fc2_exp.device)

                        if FLUID_KERNELS_AVAILABLE and chunk_total > 0:
                            # grad_intermediate = grad_fc2 @ W2.T
                            chunk_grad_inter_exp = fluid_kernels.grouped_gemm(
                                chunk_grad_fc2_exp.half(), w2.half(),
                                chunk_tokens_per_expert_t, trans_a=False, trans_b=True
                            ).to(chunk_grad_fc2_exp.dtype)

                            # grad_fc1 = grad_intermediate * activation_deriv * probs
                            if gated_linear_unit:
                                grad_x_1 = chunk_grad_inter_exp * chunk_act_deriv_exp * chunk_x_2_exp * chunk_probs_exp
                                grad_x_2 = chunk_grad_inter_exp * chunk_act_val_exp * chunk_probs_exp
                                chunk_grad_fc1_exp = torch.cat([grad_x_1, grad_x_2], dim=-1)
                            else:
                                chunk_grad_fc1_exp = chunk_grad_inter_exp * chunk_act_deriv_exp * chunk_probs_exp

                            # grad_input = grad_fc1 @ W1.T
                            chunk_dx_exp = fluid_kernels.grouped_gemm(
                                chunk_grad_fc1_exp.half(), w1.half(),
                                chunk_tokens_per_expert_t, trans_a=False, trans_b=True
                            ).to(chunk_grad_fc1_exp.dtype)
                        else:
                            # Fallback: 逐 expert 计算
                            chunk_dx_exp = torch.zeros(chunk_total, hidden_size, dtype=chunk_grad_fc2_exp.dtype, device=chunk_grad_fc2_exp.device)
                            offset = 0
                            for exp_idx in range(num_local_experts):
                                n_tok = chunk_tokens_per_expert[exp_idx]
                                if n_tok == 0:
                                    continue
                                exp_grad_fc2 = chunk_grad_fc2_exp[offset:offset+n_tok]
                                exp_act_deriv = chunk_act_deriv_exp[offset:offset+n_tok]
                                exp_probs = chunk_probs_exp[offset:offset+n_tok]

                                exp_grad_inter = torch.matmul(exp_grad_fc2, w2[exp_idx].t())
                                if gated_linear_unit:
                                    exp_act_val = chunk_act_val_exp[offset:offset+n_tok]
                                    exp_x_2 = chunk_x_2_exp[offset:offset+n_tok]
                                    grad_x_1 = exp_grad_inter * exp_act_deriv * exp_x_2 * exp_probs
                                    grad_x_2 = exp_grad_inter * exp_act_val * exp_probs
                                    exp_grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
                                else:
                                    exp_grad_fc1 = exp_grad_inter * exp_act_deriv * exp_probs

                                chunk_dx_exp[offset:offset+n_tok] = torch.matmul(exp_grad_fc1, w1[exp_idx].t())
                                offset += n_tok

                        # Step 2d: 重组回 source-rank-major
                        src_indices = torch.tensor(src_to_exp_major, device=chunk_dx_exp.device, dtype=torch.long)
                        chunk_dx = chunk_dx_exp[src_indices]

                        # 写回 grad_permuted_full（用于 dW 计算）
                        for global_src_start, global_src_end, chunk_dst_start, chunk_dst_end, exp_idx in chunk_positions:
                            if chunk_dst_end > chunk_dst_start:
                                grad_permuted_full[global_src_start:global_src_end] = chunk_dx[chunk_dst_start:chunk_dst_end]

                    # ============================================================
                    # Step 3: Launch AllToAll for this chunk ASYNCHRONOUSLY
                    # ============================================================
                    # 关键优化：使用 async_op=True 使 all_to_all_single 非阻塞
                    # 这样 Python 线程可以立即返回，继续计算下一个 chunk 的 dX
                    # 同时 NCCL 在后台执行 AllToAll

                    chunk_dx_contiguous = chunk_dx.contiguous()
                    chunk_output_size = sum(chunk_output_splits)

                    # 预分配输出 buffer
                    chunk_output_buffer = chunk_dx_contiguous.new_empty(
                        size=[chunk_output_size] + list(chunk_dx_contiguous.size()[1:]),
                        dtype=chunk_dx_contiguous.dtype,
                        device=torch.cuda.current_device(),
                    )

                    # 在 comm_stream 上启动异步 AllToAll
                    # 使用 CUDA event 确保 dX 计算完成后才开始通信
                    dx_complete_event = torch.cuda.Event()
                    dx_complete_event.record(scheduler.default_stream)

                    with torch.cuda.stream(scheduler.comm_stream):
                        scheduler.comm_stream.wait_event(dx_complete_event)

                        # 使用 async_op=True！这是关键
                        work = torch.distributed.all_to_all_single(
                            chunk_output_buffer,
                            chunk_dx_contiguous,
                            output_split_sizes=chunk_output_splits,
                            input_split_sizes=chunk_input_splits,
                            group=ep_group,
                            async_op=True,  # 关键：非阻塞
                        )

                        # 记录 AllToAll 完成的 event（在 work.wait() 之后才有效）
                        # 但我们现在不 wait，而是继续下一个 chunk

                    # 存储异步操作信息，稍后统一等待
                    async_alltoall_infos.append((work, chunk_output_buffer, chunk_output_splits))

                    # 最后一个 chunk：设置 dW overlap 触发
                    if chunk_idx == num_chunks - 1:
                        # 在最后一个 chunk 的 AllToAll 期间调度 dW
                        scheduler.on_alltoall_start(comm_type="moe_combine")

                # ============================================================
                # 等待所有异步 AllToAll 完成并重组结果
                # ============================================================
                # 等待 comm_stream 上的所有操作完成
                scheduler.default_stream.wait_stream(scheduler.comm_stream)

                # 等待每个异步 AllToAll 的 work handle
                for work, _, _ in async_alltoall_infos:
                    if work is not None:
                        work.wait()

                # 重组结果
                dest_rank_chunks = [[] for _ in range(ep_size)]
                for chunk_idx, (_, chunk_output_buffer, chunk_splits) in enumerate(async_alltoall_infos):
                    offset = 0
                    for dest_rank in range(ep_size):
                        split_size = chunk_splits[dest_rank]
                        if split_size > 0:
                            dest_rank_chunks[dest_rank].append(chunk_output_buffer[offset:offset+split_size])
                            offset += split_size

                final_parts = []
                for dest_rank in range(ep_size):
                    if dest_rank_chunks[dest_rank]:
                        final_parts.append(torch.cat(dest_rank_chunks[dest_rank], dim=0))

                if final_parts:
                    ctx.chunked_alltoall_result = torch.cat(final_parts, dim=0)
                else:
                    ctx.chunked_alltoall_result = torch.zeros(sum(recv_per_rank_list), hidden_size,
                                                              dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

                grad_permuted_local_hidden_states = grad_permuted_full

                # 需要为 dW 计算准备 grad_fc1
                # 由于我们在 chunk 循环中没有保存完整的 grad_fc1，这里重新计算
                tokens_per_expert_int = tokens_per_expert.to(torch.int32)
                grad_intermediate = fluid_kernels.grouped_gemm(
                    grad_fc2_output.half(), w2.half(),
                    tokens_per_expert_int, trans_a=False, trans_b=True
                ).to(grad_fc2_output.dtype)
                if gated_linear_unit:
                    grad_x_1 = grad_intermediate * act_deriv * ctx.x_2 * probs
                    grad_x_2 = grad_intermediate * ctx.act_val * probs
                    grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
                else:
                    grad_fc1 = grad_intermediate * act_deriv * probs

            # Mark that we did chunked AllToAll
            ctx.did_chunked_alltoall = True

        else:  # Standard backward path
            # === STANDARD dX COMPUTATION ===
            if FLUID_KERNELS_AVAILABLE:
                tokens_per_expert_int = tokens_per_expert.to(torch.int32)
                # grad_intermediate = grad_fc2_output @ w2.T
                grad_intermediate = fluid_kernels.grouped_gemm(
                    grad_fc2_output.half(), w2.half(),
                    tokens_per_expert_int, trans_a=False, trans_b=True
                ).to(grad_fc2_output.dtype)

                # Compute grad_fc1 using pre-computed activation derivatives (fast!)
                if gated_linear_unit:
                    act_val = ctx.act_val
                    x_2 = ctx.x_2
                    grad_x_1 = grad_intermediate * act_deriv * x_2 * probs
                    grad_x_2 = grad_intermediate * act_val * probs
                    grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
                else:
                    grad_fc1 = grad_intermediate * act_deriv * probs

                # grad_input = grad_fc1 @ w1.T
                grad_permuted_local_hidden_states = fluid_kernels.grouped_gemm(
                    grad_fc1.half(), w1.half(),
                    tokens_per_expert_int, trans_a=False, trans_b=True
                ).to(grad_fc2_output.dtype)
            else:
                # Loop fallback
                total_tokens = grad_fc2_output.shape[0]
                intermediate_dim = intermediate_parallel.shape[-1]
                grad_intermediate = torch.zeros(total_tokens, intermediate_dim, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_per_expert[exp_idx].item()
                    if n_tok > 0:
                        grad_intermediate[start:start+n_tok] = torch.matmul(grad_fc2_output[start:start+n_tok], w2[exp_idx].t())
                        start += n_tok

                # Compute grad_fc1 using pre-computed activation derivatives (fast!)
                if gated_linear_unit:
                    act_val = ctx.act_val
                    x_2 = ctx.x_2
                    grad_x_1 = grad_intermediate * act_deriv * x_2 * probs
                    grad_x_2 = grad_intermediate * act_val * probs
                    grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
                else:
                    grad_fc1 = grad_intermediate * act_deriv * probs

                grad_permuted_local_hidden_states = torch.zeros(total_tokens, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_per_expert[exp_idx].item()
                    if n_tok > 0:
                        grad_permuted_local_hidden_states[start:start+n_tok] = torch.matmul(grad_fc1[start:start+n_tok], w1[exp_idx].t())
                        start += n_tok

        # === LAZY REGISTRATION: Register dW computation ===
        # Detach tensors to avoid holding computation graph
        grad_fc2_output_saved = grad_fc2_output.detach()
        intermediate_parallel_saved = intermediate_parallel.detach()
        permuted_local_hidden_states_saved = permuted_local_hidden_states.detach()
        tokens_per_expert_saved = tokens_per_expert.detach()
        fc1_output_saved = fc1_output.detach()
        permuted_probs_saved = permuted_probs.detach()
        # Cache grad_fc1 directly - no need to recompute in compute_dw_weight1
        grad_fc1_saved = grad_fc1.detach()

        # Define dW computation function for all experts
        def compute_dw_weight2():
            """Compute grad_weight2 for all experts"""
            if FLUID_KERNELS_AVAILABLE:
                # Use custom Fluid GroupGEMM dW kernel
                # grad_w2 = intermediate.T @ grad_output
                # A = intermediate [total_tokens, ffn_hidden_size]
                # B = grad_fc2_output [total_tokens, hidden_size]
                tokens_per_expert_int = tokens_per_expert_saved.to(torch.int32)
                grad_w2_all = fluid_kernels.grouped_gemm_dw(
                    intermediate_parallel_saved.half(),
                    grad_fc2_output_saved.half(),
                    tokens_per_expert_int,
                    ffn_hidden_size,  # M: rows of dW (input dimension)
                    hidden_size       # N: cols of dW (output dimension)
                ).to(weight2.dtype)
                return grad_w2_all.view_as(weight2)
            else:
                # Loop fallback
                grad_w2_all = torch.zeros_like(weight2)
                w2_view = grad_w2_all.view(num_local_experts, -1, hidden_size)

                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_per_expert_saved[exp_idx].item()
                    if n_tok == 0:
                        continue
                    end = start + n_tok

                    # grad_w2 = intermediate.T @ grad_output
                    grad_w2 = torch.matmul(
                        intermediate_parallel_saved[start:end].t(),
                        grad_fc2_output_saved[start:end]
                    )
                    w2_view[exp_idx] = grad_w2
                    start = end

                return grad_w2_all

        def compute_dw_weight1():
            """Compute grad_weight1 for all experts using cached grad_fc1"""
            # Get actual ffn dimension from saved grad_fc1
            actual_ffn_dim = grad_fc1_saved.shape[-1]

            if FLUID_KERNELS_AVAILABLE:
                # Use custom Fluid GroupGEMM for dW
                # grad_w1 = input.T @ grad_fc1 (grad_fc1 already computed in dX path)
                tokens_per_expert_int = tokens_per_expert_saved.to(torch.int32)
                grad_w1_all = fluid_kernels.grouped_gemm_dw(
                    permuted_local_hidden_states_saved.half(),
                    grad_fc1_saved.half(),
                    tokens_per_expert_int,
                    hidden_size,      # M: rows of dW (input dimension)
                    actual_ffn_dim    # N: cols of dW (output dimension)
                ).to(weight1.dtype)
                return grad_w1_all.view_as(weight1)
            else:
                # Loop fallback
                grad_w1_all = torch.zeros_like(weight1)
                w1_view = grad_w1_all.view(num_local_experts, hidden_size, -1)

                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_per_expert_saved[exp_idx].item()
                    if n_tok == 0:
                        continue
                    end = start + n_tok

                    # grad_w1 = input.T @ grad_fc1 (use cached grad_fc1)
                    grad_w1 = torch.matmul(
                        permuted_local_hidden_states_saved[start:end].t(),
                        grad_fc1_saved[start:end]
                    )
                    w1_view[exp_idx] = grad_w1
                    start = end

                return grad_w1_all

        # Register dW tasks to scheduler
        # Higher priority for weight2 (closer to output)
        scheduler.register_dw_task(
            layer_name="moe_expert_weight2",
            layer_id=0,  # Could be parameterized if needed
            compute_fn=compute_dw_weight2,
            priority=100,
            weight_param=weight2,
        )

        scheduler.register_dw_task(
            layer_name="moe_expert_weight1",
            layer_id=0,
            compute_fn=compute_dw_weight1,
            priority=99,
            weight_param=weight1,
        )

        # Return None for weight gradients (scheduler will compute them)
        grad_weight1 = None
        grad_weight2 = None

        return (
            grad_permuted_local_hidden_states,  # 1
            None,  # 2: tokens_per_expert
            None,  # 3: permuted_probs
            grad_weight1,  # 4: None (scheduler computes)
            grad_weight2,  # 5: None (scheduler computes)
            None,  # 6: num_local_experts
            None,  # 7: hidden_size
            None,  # 8: ffn_hidden_size
            None,  # 9: gated_linear_unit
            None,  # 10: activation_func
            None,  # 11: moe_apply_probs_on_input
            None,  # 12: moe_router_topk
            None,  # 13: activation_func_type
        )



class FluidGroupedMLP(MegatronModule):
    """
    A version of GroupedMLP that uses a custom autograd function
    to enable a parallel backward pass.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert not config.add_bias_linear, f"bias not supported in FluidGroupedMLP (got {config.add_bias_linear})"

        self.expert_parallel = config.expert_model_parallel_size > 1

        # Activation function logic from GroupedMLP
        # Determine activation function type for gradient computation
        if self.config.activation_func == F.silu:
            self.activation_func_type = 'silu'
        else:
            self.activation_func_type = 'gelu'  # default to gelu

        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        def activation_func_with_probs(x, probs):
            dtype = x.dtype
            res = self.activation_func(x) * probs
            return res.to(dtype)

        self.activation_func_with_probs = activation_func_with_probs

        # Weight initialization from GroupedMLP
        tp_size = pg_collection.expt_tp.size()

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        init_device = 'cpu' if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight1 = Parameter(
            torch.empty(
                self.config.hidden_size,
                fc1_output_size_per_partition,
                device=init_device,
                dtype=config.params_dtype,
            )
        )
        self.weight2 = Parameter(
            torch.empty(
                fc2_input_size_per_partition,
                self.config.hidden_size,
                device=init_device,
                dtype=config.params_dtype,
            )
        )

        if config.perform_initialization:
            if config.use_cpu_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    1,
                    config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    0,
                    config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                )
            else:
                _initialize_affine_weight_gpu(
                    self.weight1, config.init_method, partition_dim=1, is_expert=True
                )
                _initialize_affine_weight_gpu(
                    self.weight2, config.output_layer_init_method, partition_dim=0, is_expert=True
                )

        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        output = _FluidExpertComputation.apply(
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
            self.weight1,
            self.weight2,
            self.num_local_experts,
            self.config.hidden_size,
            self.config.moe_ffn_hidden_size,
            self.config.gated_linear_unit,
            self.activation_func_with_probs,
            self.config.moe_apply_probs_on_input,
            self.config.moe_router_topk,
            self.activation_func_type,  # Pass activation type for gradient computation
        )
        # FluidGroupedMLP doesn't use bias, so return None for mlp_bias
        return output, None


# ============================================================
# FluidRouter - Router with dW overlap support
# ============================================================

class _FluidRouterFunc(torch.autograd.Function):
    """
    Fluid Router with lazy dW registration for overlap

    Router: Linear(hidden_size, num_experts, bias=False)
    """

    @staticmethod
    def forward(ctx, input, weight, layer_name, layer_id):
        """
        Forward: input @ weight.T

        Args:
            input: [num_tokens, hidden_size]
            weight: [num_experts, hidden_size]
        Returns:
            logits: [num_tokens, num_experts]
        """
        ctx.save_for_backward(input, weight)
        ctx.layer_name = layer_name
        ctx.layer_id = layer_id

        # Router forward: input @ weight.T
        # [num_tokens, hidden_size] @ [hidden_size, num_experts] = [num_tokens, num_experts]
        logits = torch.matmul(input, weight.t())

        return logits

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: Compute dX immediately, register dW for overlap

        Args:
            grad_output: [num_tokens, num_experts]
        Returns:
            grad_input: [num_tokens, hidden_size]
        """
        input, weight = ctx.saved_tensors

        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        # === CRITICAL PATH: Compute dX immediately ===
        # grad_input = grad_output @ weight
        # [num_tokens, num_experts] @ [num_experts, hidden_size] = [num_tokens, hidden_size]
        grad_input = torch.matmul(grad_output, weight)

        # === LAZY REGISTRATION: Register dW ===
        grad_output_saved = grad_output.detach()
        input_saved = input.detach()

        def compute_dw():
            # grad_weight = grad_output.T @ input
            # [num_experts, num_tokens] @ [num_tokens, hidden_size] = [num_experts, hidden_size]
            grad_weight = torch.matmul(grad_output_saved.t(), input_saved)
            return grad_weight

        scheduler.register_dw_task(
            layer_name=f"{ctx.layer_name}_router_weight",
            layer_id=ctx.layer_id,
            compute_fn=compute_dw,
            priority=0,
            weight_param=weight,
        )

        return grad_input, None, None, None


class FluidRouter(MegatronModule):
    """
    Fluid Router with dW overlap support

    Replaces Megatron's TopKRouter for computation-communication overlap.
    """

    def __init__(self, config: TransformerConfig, pg_collection=None, layer_number=None):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number

        # Router weight: [num_experts, hidden_size]
        self.weight = Parameter(
            torch.empty(
                config.num_moe_experts,
                config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )

        # Initialize weight
        if config.perform_initialization:
            _initialize_affine_weight_cpu(
                self.weight,
                config.num_moe_experts,
                config.hidden_size,
                config.num_moe_experts,
                partition_dim=0,
                init_method=config.init_method,
                params_dtype=config.params_dtype,
            )

    def forward(self, hidden_states: torch.Tensor):
        """
        Router forward with TopK selection

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            probs: [num_tokens, num_experts] - routing probabilities for all experts
            routing_map: [num_tokens, num_experts] - binary mask indicating selected top-k experts
        """
        # Use Fluid autograd function
        logits = _FluidRouterFunc.apply(
            hidden_states,
            self.weight,
            f"layer_{self.layer_number}" if self.layer_number is not None else "router",
            self.layer_number if self.layer_number is not None else 0,
        )

        # Apply softmax to get probabilities for all experts
        # probs: [num_tokens, num_experts]
        probs = F.softmax(logits, dim=-1)

        # TopK selection
        topk = self.config.moe_router_topk
        _, indices = torch.topk(probs, k=topk, dim=-1)

        # Create routing_map: [num_tokens, num_experts] binary mask (bool, same as Megatron)
        num_tokens = hidden_states.shape[0]
        num_experts = self.config.num_moe_experts
        routing_map = torch.zeros(
            num_tokens, num_experts,
            dtype=torch.int32,
            device=hidden_states.device
        )

        # Set selected experts to 1, then convert to bool (consistent with Megatron)
        routing_map.scatter_(1, indices, 1)
        routing_map = routing_map.bool()

        return probs, routing_map


# ============================================================
# Fused Forward + Standard Backward
# ============================================================
# 非对称流水线：前向用融合算子，反向走标准路径保留 dW 调度
# ============================================================
from megatron.core.transformer.moe.moe_utils import unpermute


class FusedForwardStandardBackward(torch.autograd.Function):
    """
    非对称流水线：融合前向 + 标准反向

    前向：C++ 融合算子（AllToAll + FC 深度重叠）
    反向：直接使用保存的中间值计算梯度，无需重新执行 expert forward

    优化设计：
    - 保存 dispatched_input (FC1 输入), fc1_pre_act (激活前), fc1_output (激活后)
    - 反向时直接计算 dX 和 dW，不重新执行前向
    - dW 通过 BackwardScheduler 调度，与下一层 AllToAll 重叠
    """

    @staticmethod
    def forward(ctx, hidden_states, routing_map, probs, permuted_tokens,
                permutation_map, weight1, weight2, num_local_experts, hidden_size,
                input_splits_list, output_splits_list, self_input_offset, self_input_count,
                peer_token_counts, h_self_tokens_per_expert, h_peer_tokens_per_expert_all,
                activation_type, moe_layer):
        from fluid.ops import fluid_kernels
        import os

        # Check if we should use the new expert-major function
        use_expert_major = os.environ.get('FLUID_USE_EXPERT_MAJOR', '1') == '1'

        debug_fwd_timing = os.environ.get('FLUID_DEBUG_FORWARD_TIMING', '0') == '1'
        if debug_fwd_timing:
            torch.cuda.synchronize()
            import time
            t_fwd_0 = time.perf_counter()

        fc1_weight = weight1.view(num_local_experts, hidden_size, -1)

        if use_expert_major:
            # ============================================================
            # NEW: Use moe_alltoall_fc1_fused_expert_major
            # ============================================================
            # Returns:
            #   - fc1_output: RANK-MAJOR (for FC2)
            #   - dispatched_input: EXPERT-MAJOR (for backward dW1)
            #   - fc1_pre_act: EXPERT-MAJOR (for backward activation)
            #   - reorder_indices, inverse_indices: computed on GPU
            fc1_output, segment_sizes, dispatched_input, fc1_pre_act, reorder_indices, inverse_indices = \
                fluid_kernels.moe_alltoall_fc1_fused_expert_major(
                    permuted_tokens.half(), fc1_weight.half(),
                    self_input_offset, self_input_count,
                    input_splits_list, output_splits_list, peer_token_counts,
                    h_self_tokens_per_expert, h_peer_tokens_per_expert_all, activation_type
                )
            # Convert dtypes
            fc1_output = fc1_output.to(permuted_tokens.dtype)
            dispatched_input = dispatched_input.to(permuted_tokens.dtype)
            fc1_pre_act = fc1_pre_act.to(permuted_tokens.dtype)

            if debug_fwd_timing:
                torch.cuda.synchronize()
                t_fwd_1 = time.perf_counter()
                print(f"[Forward Timing] moe_alltoall_fc1_fused_expert_major: {(t_fwd_1-t_fwd_0)*1000:.2f} ms")
                t_fwd_0 = t_fwd_1
        else:
            # ============================================================
            # Original path: moe_alltoall_fc1_fused + Python reorder_indices
            # ============================================================
            fc1_output, segment_sizes, dispatched_input, fc1_pre_act = fluid_kernels.moe_alltoall_fc1_fused(
                permuted_tokens.half(), fc1_weight.half(),
                self_input_offset, self_input_count,
                input_splits_list, output_splits_list, peer_token_counts,
                h_self_tokens_per_expert, h_peer_tokens_per_expert_all, activation_type
            )
            fc1_output = fc1_output.to(permuted_tokens.dtype)
            dispatched_input = dispatched_input.to(permuted_tokens.dtype)
            fc1_pre_act = fc1_pre_act.to(permuted_tokens.dtype)

            # Build reorder_indices in Python (old path)
            all_segments = [list(h_self_tokens_per_expert)] + [list(p) for p in h_peer_tokens_per_expert_all]
            segments_tensor = torch.tensor(all_segments, dtype=torch.long, device=fc1_output.device)
            total_tokens = int(segments_tensor.sum().item())
            num_segments = segments_tensor.shape[0]
            counts_flat = segments_tensor.flatten()
            expert_pattern = torch.arange(num_local_experts, device=fc1_output.device).repeat(num_segments)
            expert_ids = expert_pattern.repeat_interleave(counts_flat)
            sorted_indices = torch.argsort(expert_ids, stable=True)
            reorder_indices = sorted_indices
            inverse_indices = torch.empty_like(reorder_indices)
            inverse_indices.scatter_(0, reorder_indices, torch.arange(total_tokens, device=fc1_output.device))

            if debug_fwd_timing:
                torch.cuda.synchronize()
                t_fwd_1 = time.perf_counter()
                print(f"[Forward Timing] moe_alltoall_fc1_fused + reorder build: {(t_fwd_1-t_fwd_0)*1000:.2f} ms")
                t_fwd_0 = t_fwd_1

        # Stage 2: Fused FC2 + AllToAll
        fc2_weight = weight2.view(num_local_experts, -1, hidden_size)
        combined_output = fluid_kernels.moe_fc2_alltoall_fused(
            fc1_output.half(), fc2_weight.half(), segment_sizes,
            input_splits_list, h_self_tokens_per_expert, h_peer_tokens_per_expert_all
        )
        combined_output = combined_output.to(permuted_tokens.dtype)

        if debug_fwd_timing:
            torch.cuda.synchronize()
            t_fwd_2 = time.perf_counter()
            print(f"[Forward Timing] moe_fc2_alltoall_fused: {(t_fwd_2-t_fwd_0)*1000:.2f} ms")
            t_fwd_0 = t_fwd_2

        # Unpermute with probs multiplication (standard Megatron behavior)
        output = unpermute(combined_output, permutation_map, hidden_states.shape,
                          probs=probs, routing_map=routing_map)

        # 计算真正的 tokens_per_expert（每个 expert 处理的总 token 数）
        tokens_per_expert_list = list(h_self_tokens_per_expert)
        for peer_tokens in h_peer_tokens_per_expert_all:
            for i, count in enumerate(peer_tokens):
                tokens_per_expert_list[i] += count
        tokens_per_expert_tensor = torch.tensor(
            tokens_per_expert_list, dtype=torch.int64, device=fc1_output.device
        )

        # 保存反向需要的中间值（rank-major 布局）
        # 反向会使用 inverse_indices 将数据重排为 expert-major 布局
        ctx.save_for_backward(
            hidden_states, routing_map, probs, permuted_tokens, permutation_map,
            dispatched_input, fc1_pre_act, fc1_output,
            weight1, weight2, tokens_per_expert_tensor, reorder_indices, inverse_indices
        )
        ctx.moe_layer = moe_layer
        ctx.num_local_experts = num_local_experts
        ctx.hidden_size = hidden_size
        ctx.activation_type = activation_type
        ctx.input_splits_list = input_splits_list
        ctx.output_splits_list = output_splits_list

        return output

    @staticmethod
    def backward(ctx, grad_output):
        import os
        from fluid.scheduler import get_backward_scheduler
        from fluid.communication import fluid_all_to_all_moe_dispatch, fluid_all_to_all_moe_combine
        from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs
        from megatron.core.tensor_parallel import (
            gather_from_sequence_parallel_region,
            reduce_scatter_to_sequence_parallel_region,
        )
        if FLUID_KERNELS_AVAILABLE:
            from fluid.ops import fluid_kernels

        (hidden_states, routing_map, probs, permuted_tokens, permutation_map,
         dispatched_input, fc1_pre_act, fc1_output,
         weight1, weight2, tokens_per_expert_tensor, reorder_indices, inverse_indices) = ctx.saved_tensors
        # 数据是 rank-major 布局，需要在反向中转换为 expert-major
        # reorder_indices: expert-major -> rank-major
        # inverse_indices: rank-major -> expert-major
        moe_layer = ctx.moe_layer
        num_local_experts = ctx.num_local_experts
        hidden_size = ctx.hidden_size
        activation_type = ctx.activation_type
        input_splits_list = ctx.input_splits_list
        output_splits_list = ctx.output_splits_list

        scheduler = get_backward_scheduler()

        # 获取维度信息
        ffn_hidden_size = fc1_output.shape[-1]
        total_expert_tokens = fc1_output.shape[0]

        # Reshape weights
        w1 = weight1.view(num_local_experts, hidden_size, ffn_hidden_size)
        w2 = weight2.view(num_local_experts, ffn_hidden_size, hidden_size)

        # 添加timing分析
        import os
        debug_timing = os.environ.get('FLUID_DEBUG_BACKWARD_TIMING', '0') == '1'

        if debug_timing:
            torch.cuda.synchronize()
            import time
            t_step1_start = time.perf_counter()

        # ============================================================
        # Step 1: Unpermute backward (grad_output -> grad_combined_output)
        # ============================================================
        # unpermute forward with probs (standard Megatron):
        #   1. permuted_probs = probs.T.masked_select(routing_map.T)  [num_permuted_tokens]
        #   2. permuted_tokens_weighted = permuted_tokens * permuted_probs.unsqueeze(-1)
        #   3. output.scatter_add_(0, sorted_indices, permuted_tokens_weighted)
        # unpermute backward:
        #   grad_permuted[i] = grad_output[sorted_indices[i]] * permuted_probs[i]
        grad_output_flat = grad_output.view(-1, hidden_size)

        # Compute permuted_probs (same as in unpermute forward)
        # probs: [num_tokens, num_experts], routing_map: [num_tokens, num_experts]
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())

        # permutation_map (sorted_indices) tells us: permuted token i came from original position sorted_indices[i]
        # In unpermute with probs, we scatter permuted_input * permuted_probs back using sorted_indices
        # So backward: grad_permuted[i] = grad_output[sorted_indices[i]] * permuted_probs[i]
        grad_combined_output = grad_output_flat.index_select(0, permutation_map) * permuted_probs.unsqueeze(-1)

        if debug_timing:
            torch.cuda.synchronize()
            t_step1_end = time.perf_counter()
            print(f"[Backward Timing] Step 1 (unpermute bwd): {(t_step1_end-t_step1_start)*1000:.2f} ms")

        # ============================================================
        # Step 2: Combine backward (AllToAll: grad_combined -> grad_expert_output)
        # ============================================================
        # This AllToAll can overlap with PREVIOUS layer's dW tasks (cross-layer overlap)
        ep_size = moe_layer.ep_size

        if debug_timing:
            t_step2_start = time.perf_counter()

        if ep_size > 1:
            # Combine backward = Dispatch forward with reversed splits
            # grad_expert_output 是 expert-major 布局
            grad_after_combine = fluid_all_to_all_moe_dispatch(
                grad_combined_output,
                output_splits=torch.tensor(output_splits_list, device=grad_combined_output.device),
                input_splits=torch.tensor(input_splits_list, device=grad_combined_output.device),
                group=moe_layer.ep_group,
            )
            # Note: fluid_all_to_all_moe_dispatch internally triggers scheduler.on_alltoall_start
            # through _FluidAllToAll.backward when used in autograd context.
            # In manual backward, we trigger it explicitly below if there are pending dW tasks.
            if scheduler.dw_queue:
                scheduler.on_alltoall_start(comm_type="moe_combine_bwd")
        else:
            grad_after_combine = grad_combined_output

        # TP gather for grad
        if moe_layer.tp_size > 1:
            grad_after_combine = gather_from_sequence_parallel_region(
                grad_after_combine, group=moe_layer.tp_group
            )

        if debug_timing:
            torch.cuda.synchronize()
            t_step2_end = time.perf_counter()
            print(f"[Backward Timing] Step 2 (combine AllToAll): {(t_step2_end-t_step2_start)*1000:.2f} ms")

        # 使用保存的 tokens_per_expert_tensor（每个 expert 的真实 token 数）
        tokens_per_expert = tokens_per_expert_tensor

        # ============================================================
        # Step 3: Expert backward（使用grouped_gemm，需要expert-major布局）
        # ============================================================
        if debug_timing:
            t0 = time.perf_counter()

        # Check if we used expert-major function in forward
        use_expert_major = os.environ.get('FLUID_USE_EXPERT_MAJOR', '1') == '1'

        # 3.1 转换为 expert-major 布局
        # With FLUID_USE_EXPERT_MAJOR=1:
        #   - dispatched_input and fc1_pre_act are ALREADY expert-major (from C++ kernel)
        #   - Only need to reorder grad_after_combine and fc1_output
        # Without:
        #   - All tensors need to be reordered
        grad_expert_output_exp = grad_after_combine.index_select(0, inverse_indices)
        fc1_output_exp = fc1_output.index_select(0, inverse_indices)

        if use_expert_major:
            # dispatched_input and fc1_pre_act are already expert-major from forward!
            fc1_pre_act_exp = fc1_pre_act
            dispatched_input_exp = dispatched_input
        else:
            # Old path: need to reorder all tensors
            fc1_pre_act_exp = fc1_pre_act.index_select(0, inverse_indices)
            dispatched_input_exp = dispatched_input.index_select(0, inverse_indices)

        if debug_timing:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            reorder_desc = "2 tensors" if use_expert_major else "4 tensors"
            print(f"[Backward Timing] index_select reorder ({reorder_desc}): {(t1-t0)*1000:.2f} ms")

        # 3.2 FC2 backward: grad_intermediate = grad @ W2.T (使用 grouped_gemm)
        w2_t = w2.transpose(1, 2).contiguous()
        if FLUID_KERNELS_AVAILABLE:
            grad_intermediate_exp = fluid_kernels.grouped_gemm(
                grad_expert_output_exp.half(), w2_t.half(), tokens_per_expert.to(torch.int32)
            ).to(grad_expert_output_exp.dtype)
        else:
            # Fallback: loop-based grouped gemm
            grad_intermediate_exp = torch.empty(total_expert_tokens, ffn_hidden_size,
                                                dtype=grad_expert_output_exp.dtype,
                                                device=grad_expert_output_exp.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                if n_tok > 0:
                    grad_intermediate_exp[start:start+n_tok] = torch.matmul(
                        grad_expert_output_exp[start:start+n_tok], w2_t[exp_idx]
                    )
                    start += n_tok

        if debug_timing:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f"[Backward Timing] FC2 backward grouped_gemm: {(t2-t1)*1000:.2f} ms")

        # 3.3 Activation backward
        if activation_type == 0:  # GELU
            grad_act = _gelu_grad_analytical(fc1_pre_act_exp)
        elif activation_type == 1:  # SiLU
            sig = torch.sigmoid(fc1_pre_act_exp)
            grad_act = sig * (1 + fc1_pre_act_exp * (1 - sig))
        elif activation_type == 2:  # ReLU
            grad_act = (fc1_pre_act_exp > 0).to(fc1_pre_act_exp.dtype)
        else:
            grad_act = torch.ones_like(fc1_pre_act_exp)
        grad_fc1_exp = grad_intermediate_exp * grad_act

        if debug_timing:
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            print(f"[Backward Timing] Activation backward: {(t3-t2)*1000:.2f} ms")

        # 3.4 FC1 backward: grad_dispatched = grad_fc1 @ W1.T (使用 grouped_gemm)
        w1_t = w1.transpose(1, 2).contiguous()
        if FLUID_KERNELS_AVAILABLE:
            grad_dispatched_exp = fluid_kernels.grouped_gemm(
                grad_fc1_exp.half(), w1_t.half(), tokens_per_expert.to(torch.int32)
            ).to(grad_fc1_exp.dtype)
        else:
            # Fallback: loop-based grouped gemm
            grad_dispatched_exp = torch.empty(total_expert_tokens, hidden_size,
                                              dtype=grad_fc1_exp.dtype,
                                              device=grad_fc1_exp.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                if n_tok > 0:
                    grad_dispatched_exp[start:start+n_tok] = torch.matmul(
                        grad_fc1_exp[start:start+n_tok], w1_t[exp_idx]
                    )
                    start += n_tok

        if debug_timing:
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            print(f"[Backward Timing] FC1 backward grouped_gemm: {(t4-t3)*1000:.2f} ms")

        # 3.5 将 grad_dispatched 重排回 rank-major 布局
        grad_dispatched = grad_dispatched_exp.index_select(0, reorder_indices)

        if debug_timing:
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            print(f"[Backward Timing] index_select back: {(t5-t4)*1000:.2f} ms")
            print(f"[Backward Timing] Step 3 total: {(t5-t0)*1000:.2f} ms")

        # ============================================================
        # Step 4: Register dW tasks for overlap
        # ============================================================
        # 数据是 expert-major 布局，直接按 expert 切片计算 dW

        grad_expert_output_saved = grad_expert_output_exp.detach()
        fc1_output_saved = fc1_output_exp.detach()
        dispatched_input_saved = dispatched_input_exp.detach()
        grad_fc1_saved = grad_fc1_exp.detach()
        tokens_per_expert_saved = tokens_per_expert.clone()

        def compute_dw_weight2():
            """Compute grad_weight2: fc1_output.T @ grad_expert_output (expert-major)"""
            grad_w2_all = torch.zeros_like(weight2)
            w2_view = grad_w2_all.view(num_local_experts, ffn_hidden_size, hidden_size)

            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert_saved[exp_idx].item()
                if n_tok > 0:
                    grad_w2 = torch.matmul(
                        fc1_output_saved[start:start+n_tok].t(),
                        grad_expert_output_saved[start:start+n_tok]
                    )
                    w2_view[exp_idx] = grad_w2
                    start += n_tok

            return grad_w2_all

        def compute_dw_weight1():
            """Compute grad_weight1: dispatched_input.T @ grad_fc1 (expert-major)"""
            grad_w1_all = torch.zeros_like(weight1)
            w1_view = grad_w1_all.view(num_local_experts, hidden_size, ffn_hidden_size)

            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert_saved[exp_idx].item()
                if n_tok > 0:
                    grad_w1 = torch.matmul(
                        dispatched_input_saved[start:start+n_tok].t(),
                        grad_fc1_saved[start:start+n_tok]
                    )
                    w1_view[exp_idx] = grad_w1
                    start += n_tok

            return grad_w1_all

        # Register dW tasks
        scheduler.register_dw_task(
            layer_name="fused_moe_weight2",
            layer_id=0,
            compute_fn=compute_dw_weight2,
            priority=100,
            weight_param=weight2,
        )
        scheduler.register_dw_task(
            layer_name="fused_moe_weight1",
            layer_id=0,
            compute_fn=compute_dw_weight1,
            priority=99,
            weight_param=weight1,
        )

        # ============================================================
        # Step 5: Dispatch backward (AllToAll: grad_dispatched -> grad_permuted)
        # ============================================================
        if debug_timing:
            t_step5_start = time.perf_counter()

        # 融合前向的数据布局与反向 AllToAll 预期的布局一致，不需要重排
        grad_dispatched_unsorted = grad_dispatched

        # TP reduce scatter
        if moe_layer.tp_size > 1:
            grad_dispatched_unsorted = reduce_scatter_to_sequence_parallel_region(
                grad_dispatched_unsorted.to(probs.dtype),
                group=moe_layer.tp_group,
            ).to(grad_dispatched_unsorted.dtype)

        if ep_size > 1:
            # Dispatch backward = Combine forward with reversed splits
            # Use standard AllToAll, then trigger dW execution
            grad_permuted = fluid_all_to_all_moe_combine(
                grad_dispatched_unsorted,
                output_splits=torch.tensor(input_splits_list, device=grad_dispatched_unsorted.device),
                input_splits=torch.tensor(output_splits_list, device=grad_dispatched_unsorted.device),
                group=moe_layer.ep_group,
            )
            # Trigger dW execution after AllToAll completes
            # In a multi-layer model, this executes current layer's dW tasks
            # which can then overlap with next layer's AllToAll
            scheduler.on_alltoall_start(comm_type="moe_dispatch_bwd")
        else:
            grad_permuted = grad_dispatched_unsorted

        if debug_timing:
            torch.cuda.synchronize()
            t_step5_end = time.perf_counter()
            print(f"[Backward Timing] Step 5 (dispatch AllToAll): {(t_step5_end-t_step5_start)*1000:.2f} ms")

        # ============================================================
        # Step 6: Permute backward (grad_permuted -> grad_hidden_states)
        # ============================================================
        if debug_timing:
            t_step6_start = time.perf_counter()

        # permute forward: permuted[i] = hidden[routing_indices[i]]
        # permute backward: grad_hidden[routing_indices[i]] += grad_permuted[i]
        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_hidden_states.scatter_add_(
            0,
            permutation_map.unsqueeze(-1).expand(-1, hidden_size),
            grad_permuted
        )

        if debug_timing:
            torch.cuda.synchronize()
            t_step6_end = time.perf_counter()
            print(f"[Backward Timing] Step 6 (permute bwd): {(t_step6_end-t_step6_start)*1000:.2f} ms")
            total_bwd = t_step6_end - t_step1_start
            print(f"[Backward Timing] Total backward: {total_bwd*1000:.2f} ms")

        return (grad_hidden_states,) + (None,) * 17


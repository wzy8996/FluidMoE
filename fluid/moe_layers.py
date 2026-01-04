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

# Import chunking configuration and direct AllToAll
from fluid.communication import get_dx_num_chunks, _all_to_all

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
            # Use custom Fluid GroupGEMM kernels (native bf16 support)
            tokens_per_expert_int = tokens_per_expert.to(torch.int32)
            fc1_output = fluid_kernels.grouped_gemm(
                permuted_local_hidden_states, w1,
                tokens_per_expert_int, trans_a=False, trans_b=False
            )
            intermediate_parallel = activation_func(fc1_output, permuted_probs)
            fc2_output = fluid_kernels.grouped_gemm(
                intermediate_parallel, w2,
                tokens_per_expert_int, trans_a=False, trans_b=False
            )
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
        import os
        import time

        debug_timing = os.environ.get('FLUID_DEBUG_BACKWARD_TIMING', '0') == '1'
        if debug_timing:
            print(f"[_FluidExpertComputation] backward called, grad shape: {grad_fc2_output.shape}", flush=True)

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

        # ============================================================
        # 两层优化策略 (TRUE ASYNC):
        #
        # 第一层 (num_chunks=1): dW 与 AllToAll 重叠
        #   default: |====== dX ======| release |====== dW ======|
        #                                  ↓
        #   comm:                     acquire |====== A2A ======| release
        #                                                           ↓
        #   default:                                           acquire → continue
        #
        # 第二层 (num_chunks>1): dX 分块使 AllToAll 提前开始 + dW 重叠
        #   default: |=dX_0=| release |=dX_1=| release |====== dW ======|
        #                       ↓           ↓
        #   comm:          acquire |=A2A_0=|   |=A2A_1=| release
        #                                       ↓           ↓
        #   default:                       acquire     acquire → continue
        #
        # 关键点：
        # - dX 和 dW 分离计算
        # - dW 始终与 (最后一个) AllToAll 重叠
        # - 分块时，dX_chunk[i+1] 与 A2A_chunk[i] 重叠
        # ============================================================
        from fluid.communication import get_dx_num_chunks, set_dispatch_alltoall_done
        num_chunks = get_dx_num_chunks()

        # Check conditions for async overlap
        has_dispatch_ctx = (ctx.dispatch_ctx_enabled and ctx.dispatch_ep_group is not None)

        if debug_timing:
            print(f"[DEBUG] dispatch_ctx_enabled={ctx.dispatch_ctx_enabled}, "
                  f"ep_group={ctx.dispatch_ep_group is not None}, "
                  f"has_dispatch_ctx={has_dispatch_ctx}", flush=True)

        total_tokens = grad_fc2_output.shape[0]

        if has_dispatch_ctx and num_chunks > 1:
            # ============================================================
            # 第二层优化: dX 分块 + AllToAll 分块 + dW 重叠
            # ============================================================
            # Timeline (2 chunks example):
            #   default: |=dX_0=| event |=dX_1=| event |==== dW ====|
            #                       ↓           ↓
            #   comm:           wait |=A2A_0=|   |=A2A_1=| event
            #                                       ↓           ↓
            #   default:                        wait     wait → continue
            #
            # 关键点：
            # - dX_chunk[i+1] 与 A2A_chunk[i] 重叠
            # - dW 与最后一个 A2A 重叠
            # - 使用PyTorch streams和events实现异步（不使用C++ NCCL通信器）
            # ============================================================
            if debug_timing:
                print(f"[TRUE ASYNC L2] dX分块({num_chunks}) + dW重叠", flush=True)
                torch.cuda.synchronize()
                t_start = time.perf_counter()

            # Use scheduler's existing comm stream for consistency with ASYNC=0
            comm_stream = scheduler.comm_stream

            # Compute expert offsets for expert-major layout
            expert_offsets = [0]
            for exp_idx in range(num_local_experts - 1):
                expert_offsets.append(expert_offsets[-1] + tokens_per_expert[exp_idx].item())

            # Compute chunk ranges for each expert
            expert_chunk_ranges = []
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                chunk_size_exp = n_tok // num_chunks
                remainder_exp = n_tok % num_chunks
                ranges = []
                local_start = 0
                for chunk_idx in range(num_chunks):
                    if chunk_idx < remainder_exp:
                        this_size = chunk_size_exp + 1
                    else:
                        this_size = chunk_size_exp
                    ranges.append((local_start, local_start + this_size))
                    local_start += this_size
                expert_chunk_ranges.append(ranges)

            # Get dispatch AllToAll context
            ep_group = ctx.dispatch_ep_group
            dispatch_input_splits = ctx.dispatch_input_splits
            dispatch_output_splits = ctx.dispatch_output_splits
            ep_size = len(dispatch_input_splits) if dispatch_input_splits is not None else 1

            alltoall_results = []
            dx_chunk_times = [] if debug_timing else None
            grad_fc1_chunks = []

            # ========== Step 1: dX 分块计算 + AllToAll 分块启动 ==========
            for chunk_idx in range(num_chunks):
                if debug_timing:
                    torch.cuda.synchronize()
                    t_chunk_start = time.perf_counter()

                # Step A: FC2 backward for this chunk
                grad_intermediate_parts = []
                for exp_idx in range(num_local_experts):
                    local_start, local_end = expert_chunk_ranges[exp_idx][chunk_idx]
                    if local_end > local_start:
                        global_start = expert_offsets[exp_idx] + local_start
                        global_end = expert_offsets[exp_idx] + local_end
                        grad_inter = torch.matmul(
                            grad_fc2_output[global_start:global_end], w2[exp_idx].t()
                        )
                        grad_intermediate_parts.append((exp_idx, local_start, local_end, grad_inter))

                # Step B: Activation backward for this chunk
                grad_fc1_parts = []
                for exp_idx, local_start, local_end, grad_inter in grad_intermediate_parts:
                    global_start = expert_offsets[exp_idx] + local_start
                    global_end = expert_offsets[exp_idx] + local_end
                    # NOTE: probs multiplication now happens in unpermute (Megatron standard)
                    # So we don't multiply by probs here
                    if gated_linear_unit:
                        act_val = ctx.act_val
                        x_2 = ctx.x_2
                        grad_x_1 = grad_inter * act_deriv[global_start:global_end] * x_2[global_start:global_end]
                        grad_x_2 = grad_inter * act_val[global_start:global_end]
                        grad_fc1_part = torch.cat([grad_x_1, grad_x_2], dim=-1)
                    else:
                        grad_fc1_part = grad_inter * act_deriv[global_start:global_end]
                    grad_fc1_parts.append((exp_idx, local_start, local_end, grad_fc1_part))

                grad_fc1_chunks.append(grad_fc1_parts)

                # Step C: FC1 backward for this chunk
                grad_dx_parts = []
                for exp_idx, local_start, local_end, grad_fc1_part in grad_fc1_parts:
                    grad_dx = torch.matmul(grad_fc1_part, w1[exp_idx].t())
                    grad_dx_parts.append(grad_dx)

                grad_chunk = torch.cat(grad_dx_parts, dim=0) if grad_dx_parts else \
                    torch.empty(0, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

                if debug_timing:
                    torch.cuda.synchronize()
                    dx_chunk_times.append((time.perf_counter() - t_chunk_start) * 1000)

                # Step D: Record event and launch async AllToAll on comm stream
                # Compute chunk split sizes
                chunk_output_splits = []
                chunk_input_splits = []
                for rank_idx in range(ep_size):
                    if dispatch_output_splits is not None:
                        rank_tokens = int(dispatch_output_splits[rank_idx].item() if torch.is_tensor(dispatch_output_splits[rank_idx]) else dispatch_output_splits[rank_idx])
                    else:
                        rank_tokens = total_tokens // ep_size
                    chunk_size = rank_tokens // num_chunks
                    remainder = rank_tokens % num_chunks
                    this_size = chunk_size + 1 if chunk_idx < remainder else chunk_size
                    chunk_output_splits.append(this_size)

                for rank_idx in range(ep_size):
                    if dispatch_input_splits is not None:
                        rank_tokens = int(dispatch_input_splits[rank_idx].item() if torch.is_tensor(dispatch_input_splits[rank_idx]) else dispatch_input_splits[rank_idx])
                    else:
                        rank_tokens = total_tokens // ep_size
                    chunk_size = rank_tokens // num_chunks
                    remainder = rank_tokens % num_chunks
                    this_size = chunk_size + 1 if chunk_idx < remainder else chunk_size
                    chunk_input_splits.append(this_size)

                # Record event when dX chunk is ready
                dx_chunk_ready = torch.cuda.Event()
                dx_chunk_ready.record(torch.cuda.current_stream())

                # Launch AllToAll on comm stream (non-blocking from default stream)
                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_event(dx_chunk_ready)
                    chunk_result = _all_to_all(
                        grad_chunk.contiguous(),
                        chunk_input_splits,   # backward: recv from where we sent
                        chunk_output_splits,  # backward: send to where we received
                        ep_group,
                    )
                    a2a_chunk_done = torch.cuda.Event()
                    a2a_chunk_done.record(comm_stream)

                alltoall_results.append((chunk_result, a2a_chunk_done))

            # ========== Step 2: Reconstruct grad_fc1 for dW ==========
            intermediate_dim = intermediate_parallel.shape[-1]
            if gated_linear_unit:
                grad_fc1 = torch.empty(total_tokens, intermediate_dim * 2, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
            else:
                grad_fc1 = torch.empty(total_tokens, intermediate_dim, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

            for chunk_idx, grad_fc1_parts in enumerate(grad_fc1_chunks):
                for exp_idx, local_start, local_end, grad_fc1_part in grad_fc1_parts:
                    global_start = expert_offsets[exp_idx] + local_start
                    global_end = expert_offsets[exp_idx] + local_end
                    grad_fc1[global_start:global_end] = grad_fc1_part

            # ========== Step 3: dW 计算 (与最后的 AllToAll 并行) ==========
            # 注意：不要 acquire 所有结果，先执行 dW，让 dW 与 AllToAll 重叠
            if debug_timing:
                torch.cuda.synchronize()
                t_dw_start = time.perf_counter()

            # 直接计算 dW（不注册到 scheduler）
            tokens_per_expert_int = tokens_per_expert.to(torch.int32)

            # dW2 = intermediate.T @ grad_fc2_output
            if FLUID_KERNELS_AVAILABLE:
                grad_w2 = fluid_kernels.grouped_gemm_dw(
                    intermediate_parallel,
                    grad_fc2_output,
                    tokens_per_expert_int,
                    ffn_hidden_size,
                    hidden_size
                ).view_as(weight2)
            else:
                grad_w2 = torch.zeros_like(weight2)
                w2_view = grad_w2.view(num_local_experts, -1, hidden_size)
                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_per_expert[exp_idx].item()
                    if n_tok > 0:
                        w2_view[exp_idx] = torch.matmul(
                            intermediate_parallel[start:start+n_tok].t(),
                            grad_fc2_output[start:start+n_tok]
                        )
                        start += n_tok

            # dW1 = input.T @ grad_fc1
            actual_ffn_dim = grad_fc1.shape[-1]
            if FLUID_KERNELS_AVAILABLE:
                grad_w1 = fluid_kernels.grouped_gemm_dw(
                    permuted_local_hidden_states,
                    grad_fc1,
                    tokens_per_expert_int,
                    hidden_size,
                    actual_ffn_dim
                ).view_as(weight1)
            else:
                grad_w1 = torch.zeros_like(weight1)
                w1_view = grad_w1.view(num_local_experts, hidden_size, -1)
                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_per_expert[exp_idx].item()
                    if n_tok > 0:
                        w1_view[exp_idx] = torch.matmul(
                            permuted_local_hidden_states[start:start+n_tok].t(),
                            grad_fc1[start:start+n_tok]
                        )
                        start += n_tok

            if debug_timing:
                torch.cuda.synchronize()
                print(f"[TRUE ASYNC L2] dW计算: {(time.perf_counter()-t_dw_start)*1000:.2f} ms", flush=True)

            # ========== Step 4: Wait for all AllToAll results ==========
            final_results = []
            for chunk_result, a2a_done_event in alltoall_results:
                torch.cuda.current_stream().wait_event(a2a_done_event)
                final_results.append(chunk_result)

            grad_permuted_local_hidden_states = torch.cat(final_results, dim=0) if final_results else \
                torch.empty(0, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

            # Signal that dispatch AllToAll was handled here
            set_dispatch_alltoall_done(True)

            if debug_timing:
                torch.cuda.synchronize()
                t_end = time.perf_counter()
                print(f"[TRUE ASYNC L2] 总耗时: {(t_end-t_start)*1000:.2f} ms", flush=True)
                if dx_chunk_times:
                    print(f"  dX chunks: {dx_chunk_times} (total: {sum(dx_chunk_times):.2f} ms)", flush=True)

            # 直接返回 dW 梯度（不通过 scheduler）
            return (
                grad_permuted_local_hidden_states,
                None, None,
                grad_w1,  # 直接返回
                grad_w2,  # 直接返回
                None, None, None, None, None, None, None, None,
            )

        # NOTE: TRUE ASYNC L1 (use_true_async with num_chunks=1) is now handled by falling
        # through to the standard scheduler path below. The dW-AllToAll overlap happens
        # in _FluidAllToAll.backward() which has its own TRUE ASYNC path.
        #
        # This change was necessary because there are TWO dispatch AllToAll operations
        # (tokens and probs), and the original TRUE ASYNC L1 approach of doing AllToAll
        # here and setting a skip flag caused the wrong AllToAll to be skipped.

        if num_chunks > 1 and has_dispatch_ctx:
            # ============================================================
            # CHUNKED dX (FC2+Act+FC1) + DISPATCH ALLTOALL PIPELINE
            # ============================================================
            if debug_timing:
                print(f"[_FluidExpertComputation] Using chunked dX+A2A: {num_chunks} chunks", flush=True)
                torch.cuda.synchronize()
                t_start = time.perf_counter()

            # Compute expert offsets for expert-major layout
            expert_offsets = [0]
            for exp_idx in range(num_local_experts - 1):
                expert_offsets.append(expert_offsets[-1] + tokens_per_expert[exp_idx].item())

            # Compute chunk ranges for each expert
            expert_chunk_ranges = []
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                chunk_size_exp = n_tok // num_chunks
                remainder_exp = n_tok % num_chunks
                ranges = []
                local_start = 0
                for chunk_idx in range(num_chunks):
                    if chunk_idx < remainder_exp:
                        this_size = chunk_size_exp + 1
                    else:
                        this_size = chunk_size_exp
                    ranges.append((local_start, local_start + this_size))
                    local_start += this_size
                expert_chunk_ranges.append(ranges)

            # Get dispatch AllToAll context
            ep_group = ctx.dispatch_ep_group
            dispatch_input_splits = ctx.dispatch_input_splits
            dispatch_output_splits = ctx.dispatch_output_splits
            ep_size = len(dispatch_input_splits) if dispatch_input_splits is not None else 1

            alltoall_results = []
            dx_chunk_times = [] if debug_timing else None

            # Also collect grad_fc1 chunks for dW computation later
            grad_fc1_chunks = []

            for chunk_idx in range(num_chunks):
                if debug_timing:
                    torch.cuda.synchronize()
                    t_chunk_start = time.perf_counter()

                # ============================================================
                # Step A: FC2 backward for this chunk
                # ============================================================
                grad_intermediate_parts = []
                for exp_idx in range(num_local_experts):
                    local_start, local_end = expert_chunk_ranges[exp_idx][chunk_idx]
                    if local_end > local_start:
                        global_start = expert_offsets[exp_idx] + local_start
                        global_end = expert_offsets[exp_idx] + local_end
                        # FC2 backward: grad_fc2_output @ W2.T
                        grad_inter = torch.matmul(
                            grad_fc2_output[global_start:global_end], w2[exp_idx].t()
                        )
                        grad_intermediate_parts.append((exp_idx, local_start, local_end, grad_inter))

                # ============================================================
                # Step B: Activation backward for this chunk
                # NOTE: probs multiplication now happens in unpermute (Megatron standard)
                # ============================================================
                grad_fc1_parts = []
                for exp_idx, local_start, local_end, grad_inter in grad_intermediate_parts:
                    global_start = expert_offsets[exp_idx] + local_start
                    global_end = expert_offsets[exp_idx] + local_end
                    if gated_linear_unit:
                        act_val = ctx.act_val
                        x_2 = ctx.x_2
                        grad_x_1 = grad_inter * act_deriv[global_start:global_end] * x_2[global_start:global_end]
                        grad_x_2 = grad_inter * act_val[global_start:global_end]
                        grad_fc1_part = torch.cat([grad_x_1, grad_x_2], dim=-1)
                    else:
                        grad_fc1_part = grad_inter * act_deriv[global_start:global_end]
                    grad_fc1_parts.append((exp_idx, local_start, local_end, grad_fc1_part))

                # Save grad_fc1 chunks for dW computation
                grad_fc1_chunks.append(grad_fc1_parts)

                # ============================================================
                # Step C: FC1 backward for this chunk
                # ============================================================
                grad_dx_parts = []
                for exp_idx, local_start, local_end, grad_fc1_part in grad_fc1_parts:
                    # FC1 backward: grad_fc1 @ W1.T
                    grad_dx = torch.matmul(grad_fc1_part, w1[exp_idx].t())
                    grad_dx_parts.append(grad_dx)

                # Concatenate chunk results (expert-major order)
                grad_chunk = torch.cat(grad_dx_parts, dim=0) if grad_dx_parts else \
                    torch.empty(0, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

                # Record event after dX chunk computation
                dx_done_event = torch.cuda.Event()
                dx_done_event.record()

                if debug_timing:
                    torch.cuda.synchronize()
                    dx_chunk_times.append((time.perf_counter() - t_chunk_start) * 1000)

                # ============================================================
                # Step D: Launch AllToAll for this chunk on comm_stream
                # ============================================================
                # Compute chunk split sizes
                chunk_output_splits = []
                chunk_input_splits = []
                for rank_idx in range(ep_size):
                    if dispatch_output_splits is not None:
                        rank_tokens = int(dispatch_output_splits[rank_idx].item() if torch.is_tensor(dispatch_output_splits[rank_idx]) else dispatch_output_splits[rank_idx])
                    else:
                        rank_tokens = total_tokens // ep_size
                    chunk_size = rank_tokens // num_chunks
                    remainder = rank_tokens % num_chunks
                    if chunk_idx < remainder:
                        this_size = chunk_size + 1
                    else:
                        this_size = chunk_size
                    chunk_output_splits.append(this_size)
                for rank_idx in range(ep_size):
                    if dispatch_input_splits is not None:
                        rank_tokens = int(dispatch_input_splits[rank_idx].item() if torch.is_tensor(dispatch_input_splits[rank_idx]) else dispatch_input_splits[rank_idx])
                    else:
                        rank_tokens = total_tokens // ep_size
                    chunk_size = rank_tokens // num_chunks
                    remainder = rank_tokens % num_chunks
                    if chunk_idx < remainder:
                        this_size = chunk_size + 1
                    else:
                        this_size = chunk_size
                    chunk_input_splits.append(this_size)

                # Launch dispatch AllToAll (async on comm_stream)
                with torch.cuda.stream(scheduler.comm_stream):
                    scheduler.comm_stream.wait_event(dx_done_event)
                    chunk_result = _all_to_all(
                        grad_chunk.contiguous(),
                        chunk_output_splits,
                        chunk_input_splits,
                        ep_group,
                    )
                    a2a_done_event = torch.cuda.Event()
                    a2a_done_event.record(scheduler.comm_stream)

                alltoall_results.append(chunk_result)

                # For the last chunk, set up dW overlap
                if chunk_idx == num_chunks - 1:
                    scheduler.set_alltoall_end_event(a2a_done_event)
                    scheduler.on_alltoall_start(comm_type="moe_dispatch_bwd")

            # Wait for all AllToAll chunks to complete
            scheduler.default_stream.wait_stream(scheduler.comm_stream)

            # Concatenate results
            grad_permuted_local_hidden_states = torch.cat(alltoall_results, dim=0) if alltoall_results else \
                torch.empty(0, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

            # Reconstruct full grad_fc1 for dW computation
            # We need to reassemble the chunks in the correct order
            intermediate_dim = intermediate_parallel.shape[-1]
            if gated_linear_unit:
                grad_fc1 = torch.empty(total_tokens, intermediate_dim * 2, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
            else:
                grad_fc1 = torch.empty(total_tokens, intermediate_dim, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

            for chunk_idx, grad_fc1_parts in enumerate(grad_fc1_chunks):
                for exp_idx, local_start, local_end, grad_fc1_part in grad_fc1_parts:
                    global_start = expert_offsets[exp_idx] + local_start
                    global_end = expert_offsets[exp_idx] + local_end
                    grad_fc1[global_start:global_end] = grad_fc1_part

            # Signal that dispatch AllToAll was handled here
            set_dispatch_alltoall_done(True)

            if debug_timing:
                torch.cuda.synchronize()
                t_end = time.perf_counter()
                print(f"[_FluidExpertComputation] Chunked dX+A2A: {(t_end-t_start)*1000:.2f} ms", flush=True)
                if dx_chunk_times:
                    print(f"  dX chunks: {dx_chunk_times} (total: {sum(dx_chunk_times):.2f} ms)", flush=True)

        else:
            # ============================================================
            # STANDARD (non-chunked) dX computation
            # ============================================================
            if debug_timing:
                torch.cuda.synchronize()
                t_dx_start = time.perf_counter()

            # === Step 1: FC2 backward (full, grouped_gemm) ===
            if FLUID_KERNELS_AVAILABLE:
                tokens_per_expert_int = tokens_per_expert.to(torch.int32)
                grad_intermediate = fluid_kernels.grouped_gemm(
                    grad_fc2_output, w2,
                    tokens_per_expert_int, trans_a=False, trans_b=True
                )
            else:
                intermediate_dim = intermediate_parallel.shape[-1]
                grad_intermediate = torch.zeros(total_tokens, intermediate_dim, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_per_expert[exp_idx].item()
                    if n_tok > 0:
                        grad_intermediate[start:start+n_tok] = torch.matmul(grad_fc2_output[start:start+n_tok], w2[exp_idx].t())
                        start += n_tok

            # === Step 2: Activation backward (full, element-wise) ===
            if gated_linear_unit:
                act_val = ctx.act_val
                x_2 = ctx.x_2
                grad_x_1 = grad_intermediate * act_deriv * x_2 * probs
                grad_x_2 = grad_intermediate * act_val * probs
                grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
            else:
                grad_fc1 = grad_intermediate * act_deriv * probs

            # === Step 3: FC1 backward (full, grouped_gemm) ===
            if FLUID_KERNELS_AVAILABLE:
                grad_permuted_local_hidden_states = fluid_kernels.grouped_gemm(
                    grad_fc1, w1,
                    tokens_per_expert_int, trans_a=False, trans_b=True
                )
            else:
                grad_permuted_local_hidden_states = torch.zeros(total_tokens, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_per_expert[exp_idx].item()
                    if n_tok > 0:
                        grad_permuted_local_hidden_states[start:start+n_tok] = torch.matmul(grad_fc1[start:start+n_tok], w1[exp_idx].t())
                        start += n_tok

            if debug_timing:
                torch.cuda.synchronize()
                print(f"[_FluidExpertComputation] Standard dX (FC2+Act+FC1): {(time.perf_counter()-t_dx_start)*1000:.2f} ms", flush=True)

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
                # Use custom Fluid GroupGEMM dW kernel (native bf16 support)
                # grad_w2 = intermediate.T @ grad_output
                # A = intermediate [total_tokens, ffn_hidden_size]
                # B = grad_fc2_output [total_tokens, hidden_size]
                tokens_per_expert_int = tokens_per_expert_saved.to(torch.int32)
                grad_w2_all = fluid_kernels.grouped_gemm_dw(
                    intermediate_parallel_saved,
                    grad_fc2_output_saved,
                    tokens_per_expert_int,
                    ffn_hidden_size,  # M: rows of dW (input dimension)
                    hidden_size       # N: cols of dW (output dimension)
                )
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
                # Use custom Fluid GroupGEMM for dW (native bf16 support)
                # grad_w1 = input.T @ grad_fc1 (grad_fc1 already computed in dX path)
                tokens_per_expert_int = tokens_per_expert_saved.to(torch.int32)
                grad_w1_all = fluid_kernels.grouped_gemm_dw(
                    permuted_local_hidden_states_saved,
                    grad_fc1_saved,
                    tokens_per_expert_int,
                    hidden_size,      # M: rows of dW (input dimension)
                    actual_ffn_dim    # N: cols of dW (output dimension)
                )
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

        # NOTE: probs multiplication now happens in unpermute (Megatron standard)
        # So activation_func_with_probs just ignores probs
        def activation_func_with_probs(x, probs):
            return self.activation_func(x)

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
        permuted_probs: torch.Tensor = None,
    ):
        """
        Expert computation.

        Args:
            permuted_local_hidden_states: Input tokens after dispatch
            tokens_per_expert: Number of tokens per expert
            permuted_probs: Routing probabilities (optional, for backward compatibility)
                           NOTE: probs multiplication now happens in unpermute (Megatron standard)
        """
        # 如果没有传入 probs，使用全 1（probs 在 unpermute 里乘）
        if permuted_probs is None:
            permuted_probs = torch.ones(
                permuted_local_hidden_states.shape[0],
                dtype=permuted_local_hidden_states.dtype,
                device=permuted_local_hidden_states.device,
            )

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
                    permuted_tokens, fc1_weight,
                    self_input_offset, self_input_count,
                    input_splits_list, output_splits_list, peer_token_counts,
                    h_self_tokens_per_expert, h_peer_tokens_per_expert_all, activation_type
                )

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
                permuted_tokens, fc1_weight,
                self_input_offset, self_input_count,
                input_splits_list, output_splits_list, peer_token_counts,
                h_self_tokens_per_expert, h_peer_tokens_per_expert_all, activation_type
            )

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

        # Stage 2: Fused FC2 + AllToAll (native bf16 support)
        fc2_weight = weight2.view(num_local_experts, -1, hidden_size)
        combined_output = fluid_kernels.moe_fc2_alltoall_fused(
            fc1_output, fc2_weight, segment_sizes,
            input_splits_list, h_self_tokens_per_expert, h_peer_tokens_per_expert_all
        )

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

        # ============================================================
        # OPTIMIZATION: Pre-compute activation derivative in forward
        # ============================================================
        # This avoids expensive _gelu_grad_analytical call in backward
        # (saves ~10ms per backward pass)
        if activation_type == 0:  # GELU
            act_deriv = _gelu_grad_analytical(fc1_pre_act)
        elif activation_type == 1:  # SiLU
            sig = torch.sigmoid(fc1_pre_act)
            act_deriv = sig * (1 + fc1_pre_act * (1 - sig))
        elif activation_type == 2:  # ReLU
            act_deriv = (fc1_pre_act > 0).to(fc1_pre_act.dtype)
        else:
            act_deriv = torch.ones_like(fc1_pre_act)

        # 保存反向需要的中间值（rank-major 布局）
        # 反向会使用 inverse_indices 将数据重排为 expert-major 布局
        # NOTE: 保存 act_deriv 而不是 fc1_pre_act，因为 backward 不需要 fc1_pre_act
        ctx.save_for_backward(
            hidden_states, routing_map, probs, permuted_tokens, permutation_map,
            dispatched_input, act_deriv, fc1_output,
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
         dispatched_input, act_deriv, fc1_output,
         weight1, weight2, tokens_per_expert_tensor, reorder_indices, inverse_indices) = ctx.saved_tensors
        # 数据是 rank-major 布局，需要在反向中转换为 expert-major
        # reorder_indices: expert-major -> rank-major
        # inverse_indices: rank-major -> expert-major
        # NOTE: act_deriv 是预计算的 activation derivative（在 forward 中计算）
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
            print(f"[Backward] ep_size={ep_size}, num_chunks={num_chunks}, num_local_experts={num_local_experts}")
            t_step2_start = time.perf_counter()

        if ep_size > 1:
            # Combine backward = Dispatch forward with reversed splits
            # grad_expert_output 是 expert-major 布局
            #
            # OPTIMIZATION: Use async AllToAll on comm_stream for overlap with dW
            # The synchronous call was preventing overlap!
            output_splits = output_splits_list
            input_splits = input_splits_list

            # Launch AllToAll on comm_stream (async)
            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                grad_after_combine = _all_to_all(
                    grad_combined_output, output_splits, input_splits, moe_layer.ep_group
                )
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)
                scheduler.set_alltoall_end_event(event)

            # Trigger dW execution while AllToAll is running
            if scheduler.dw_queue:
                scheduler.on_alltoall_start(comm_type="moe_combine_bwd")

            # Wait for AllToAll to complete before continuing
            scheduler.default_stream.wait_stream(scheduler.comm_stream)
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
        # Step 3-5: Fused dX + AllToAll Pipeline
        # ============================================================
        # When num_chunks > 1 and ep_size > 1, use chunked pipeline:
        #   default: |= dX_1 =|= dX_2 =|
        #   comm:              |= A2A_1 =|= A2A_2 =|
        # This allows dX_2 to overlap with A2A_1, saving AllToAll/2 time
        # ============================================================

        if debug_timing:
            t0 = time.perf_counter()

        # Check if we used expert-major function in forward
        use_expert_major = os.environ.get('FLUID_USE_EXPERT_MAJOR', '1') == '1'
        num_chunks = get_dx_num_chunks()

        # 3.1 转换为 expert-major 布局
        if num_local_experts == 1:
            grad_expert_output_exp = grad_after_combine
            fc1_output_exp = fc1_output
            act_deriv_exp = act_deriv
            dispatched_input_exp = dispatched_input
        else:
            grad_expert_output_exp = grad_after_combine.index_select(0, inverse_indices)
            fc1_output_exp = fc1_output.index_select(0, inverse_indices)
            if use_expert_major:
                act_deriv_exp = act_deriv
                dispatched_input_exp = dispatched_input
            else:
                act_deriv_exp = act_deriv.index_select(0, inverse_indices)
                dispatched_input_exp = dispatched_input.index_select(0, inverse_indices)

        if debug_timing:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if num_local_experts == 1:
                reorder_desc = "skipped (1 expert)"
            elif use_expert_major:
                reorder_desc = "2 tensors"
            else:
                reorder_desc = "4 tensors"
            print(f"[Backward Timing] index_select reorder ({reorder_desc}): {(t1-t0)*1000:.2f} ms")

        # Prepare weights for matmul
        w2_t = w2.transpose(1, 2).contiguous()
        w1_t = w1.transpose(1, 2).contiguous()

        # ============================================================
        # Chunked dX + AllToAll pipeline is DISABLED
        # Reason: In 2-GPU NVLink environment, chunking overhead > overlap benefit
        # May re-enable for cross-node scenarios with high communication latency
        # ============================================================
        use_chunked_pipeline = False  # Disabled: chunking doesn't help in current setup

        if use_chunked_pipeline:
            # ============================================================
            # Chunked FC1 + AllToAll Pipeline (只对 FC1 分块)
            # ============================================================
            # FC2 + Activation 使用 grouped_gemm，不分块
            # 只对 FC1 backward 分块，因为它紧跟 AllToAll
            # 分块策略：每个 expert 取 1/N 的 tokens
            # ============================================================
            if debug_timing:
                print(f"[Backward Timing] Using chunked FC1+A2A pipeline: {num_chunks} chunks")

            # Step 1: FC2 backward (不分块，使用 grouped_gemm)
            if FLUID_KERNELS_AVAILABLE:
                grad_intermediate_exp = fluid_kernels.grouped_gemm(
                    grad_expert_output_exp, w2_t, tokens_per_expert.to(torch.int32)
                )
            else:
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

            # Step 2: Activation backward (不分块)
            grad_fc1_exp = grad_intermediate_exp * act_deriv_exp

            if debug_timing:
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                print(f"[Backward Timing] Activation mul: {(t3-t2)*1000:.2f} ms")

            # Step 3: FC1 backward + AllToAll (分块 pipeline)
            # 分块策略：每个 expert 的 tokens 分成 num_chunks 份
            # chunk_i 包含每个 expert 的第 i 份 tokens

            if debug_timing:
                torch.cuda.synchronize()
                t_fc1_start = time.perf_counter()

            # 计算每个 expert 在每个 chunk 中的 token 范围
            # expert_chunk_ranges[exp_idx][chunk_idx] = (start, end) in expert's local coords
            expert_chunk_ranges = []
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                chunk_size_exp = n_tok // num_chunks
                remainder_exp = n_tok % num_chunks
                ranges = []
                local_start = 0
                for chunk_idx in range(num_chunks):
                    if chunk_idx < remainder_exp:
                        this_size = chunk_size_exp + 1
                    else:
                        this_size = chunk_size_exp
                    ranges.append((local_start, local_start + this_size))
                    local_start += this_size
                expert_chunk_ranges.append(ranges)

            # 计算每个 chunk 的总 token 数
            chunk_total_tokens = []
            for chunk_idx in range(num_chunks):
                total = sum(expert_chunk_ranges[exp_idx][chunk_idx][1] - expert_chunk_ranges[exp_idx][chunk_idx][0]
                           for exp_idx in range(num_local_experts))
                chunk_total_tokens.append(total)

            alltoall_results = []

            # Expert 起始偏移（在 expert-major 布局中）
            expert_offsets = [0]
            for exp_idx in range(num_local_experts - 1):
                expert_offsets.append(expert_offsets[-1] + tokens_per_expert[exp_idx].item())

            # ============================================================
            # 两阶段 Pipeline：
            # 阶段 1：批量启动所有 GEMM（在 default_stream 上，连续执行）
            # 阶段 2：批量启动所有 A2A（在 comm_stream 上，每个等待对应的 GEMM）
            # 这样 A2A_0 可以和 GEMM_1, GEMM_2, ... 真正并行
            # ============================================================

            # 阶段 1：启动所有 GEMM，记录 events
            gemm_results = []
            gemm_events = []
            chunk_splits = []

            for chunk_idx in range(num_chunks):
                this_chunk_total = chunk_total_tokens[chunk_idx]
                if this_chunk_total == 0:
                    gemm_results.append(None)
                    gemm_events.append(None)
                    chunk_splits.append(None)
                    continue

                # FC1 backward GEMM
                if num_local_experts == 1:
                    local_start, local_end = expert_chunk_ranges[0][chunk_idx]
                    if local_end > local_start:
                        grad_dispatched_chunk = torch.matmul(
                            grad_fc1_exp[local_start:local_end], w1_t[0]
                        ).contiguous()
                    else:
                        grad_dispatched_chunk = torch.empty(
                            0, hidden_size, dtype=grad_fc1_exp.dtype, device=grad_fc1_exp.device
                        )
                else:
                    grad_fc1_chunk_parts = []
                    for exp_idx in range(num_local_experts):
                        local_start, local_end = expert_chunk_ranges[exp_idx][chunk_idx]
                        if local_end > local_start:
                            global_start = expert_offsets[exp_idx] + local_start
                            global_end = expert_offsets[exp_idx] + local_end
                            grad_dispatched_part = torch.matmul(
                                grad_fc1_exp[global_start:global_end], w1_t[exp_idx]
                            )
                            grad_fc1_chunk_parts.append(grad_dispatched_part)
                    grad_dispatched_chunk = torch.cat(grad_fc1_chunk_parts, dim=0) if grad_fc1_chunk_parts else \
                        torch.empty(0, hidden_size, dtype=grad_fc1_exp.dtype, device=grad_fc1_exp.device)
                    grad_dispatched_chunk = grad_dispatched_chunk.contiguous()

                gemm_results.append(grad_dispatched_chunk)

                # 记录 event（这个 GEMM 完成的标记）
                event = torch.cuda.Event()
                event.record(scheduler.default_stream)
                gemm_events.append(event)

                # 预计算 splits
                chunk_output_splits = [this_chunk_total // ep_size] * ep_size
                chunk_input_splits = [this_chunk_total // ep_size] * ep_size
                for i in range(this_chunk_total % ep_size):
                    chunk_output_splits[i] += 1
                    chunk_input_splits[i] += 1
                chunk_splits.append((chunk_output_splits, chunk_input_splits))

            # 阶段 2：启动所有 A2A（在 comm_stream 上，每个等待对应的 GEMM event）
            # 这样 A2A_0 执行时，GEMM_1, GEMM_2, ... 可能还在执行
            for chunk_idx in range(num_chunks):
                if gemm_results[chunk_idx] is None:
                    continue

                with torch.cuda.stream(scheduler.comm_stream):
                    # 只等待这个 chunk 的 GEMM 完成
                    scheduler.comm_stream.wait_event(gemm_events[chunk_idx])
                    chunk_output_splits, chunk_input_splits = chunk_splits[chunk_idx]
                    chunk_result = _all_to_all(
                        gemm_results[chunk_idx],
                        chunk_output_splits,
                        chunk_input_splits,
                        moe_layer.ep_group,
                    )
                    event = torch.cuda.Event()
                    event.record(scheduler.comm_stream)

                alltoall_results.append(chunk_result)

                # For the last chunk, set up dW overlap
                if chunk_idx == num_chunks - 1:
                    scheduler.set_alltoall_end_event(event)
                    scheduler.on_alltoall_start(comm_type="moe_dispatch_bwd")

            # ============================================================
            # Step D: Wait and reassemble results
            # ============================================================
            scheduler.default_stream.wait_stream(scheduler.comm_stream)
            grad_permuted = torch.cat(alltoall_results, dim=0) if alltoall_results else \
                torch.empty(0, hidden_size, dtype=grad_fc1_exp.dtype, device=grad_fc1_exp.device)

            if debug_timing:
                torch.cuda.synchronize()
                t_step5_end = time.perf_counter()
                print(f"[Backward Timing] Step 3-5 (chunked FC1+A2A pipeline): {(t_step5_end-t0)*1000:.2f} ms")
                if fc1_chunk_times:
                    print(f"  FC1 per chunk: {fc1_chunk_times} (total: {sum(fc1_chunk_times):.2f} ms)")
                if a2a_launch_times:
                    print(f"  A2A launch per chunk: {a2a_launch_times} (total: {sum(a2a_launch_times):.2f} ms)")

        else:
            # ============================================================
            # Original non-chunked implementation
            # ============================================================
            # Check if we should use cuBLAS optimized version
            use_cublas_backward = os.environ.get('FLUID_USE_CUBLAS_BACKWARD', '0') == '1'

            if use_cublas_backward and FLUID_KERNELS_AVAILABLE:
                # ============================================================
                # cuBLAS Optimized: FC2 + Activation + FC1 in 3 kernel launches
                # ============================================================
                # Prepare probs for cuBLAS (needs to reorder to expert-major)
                probs_exp = probs.T.contiguous().masked_select(routing_map.T.contiguous())
                if num_local_experts > 1:
                    probs_exp = probs_exp.index_select(0, inverse_indices)

                # cuBLAS expects: w1=[E, ffn, hidden], w2=[E, hidden, ffn]
                # Current: w1=[E, hidden, ffn], w2=[E, ffn, hidden]
                # So we pass w1_t and w2_t
                tokens_per_expert_list = [tokens_per_expert[i].item() for i in range(num_local_experts)]

                grad_dispatched_exp, grad_fc1_exp = fluid_kernels.moe_backward_dx_cublas(
                    grad_expert_output_exp,  # grad_fc2: [total_tokens, hidden_size]
                    act_deriv_exp,           # act_deriv: [total_tokens, ffn_hidden_size]
                    probs_exp,               # probs: [total_tokens]
                    w1_t,                    # [E, ffn, hidden]
                    w2_t,                    # [E, hidden, ffn]
                    tokens_per_expert_list
                )

                if debug_timing:
                    torch.cuda.synchronize()
                    t4 = time.perf_counter()
                    print(f"[Backward Timing] cuBLAS fused FC2+Act+FC1: {(t4-t1)*1000:.2f} ms")
            else:
                # ============================================================
                # Original: Separate grouped_gemm calls
                # ============================================================
                # 3.2 FC2 backward: grad_intermediate = grad @ W2.T
                if FLUID_KERNELS_AVAILABLE:
                    grad_intermediate_exp = fluid_kernels.grouped_gemm(
                        grad_expert_output_exp, w2_t, tokens_per_expert.to(torch.int32)
                    )
                else:
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
                grad_fc1_exp = grad_intermediate_exp * act_deriv_exp

                if debug_timing:
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    print(f"[Backward Timing] Activation mul (pre-computed): {(t3-t2)*1000:.2f} ms")

                # 3.4 FC1 backward: grad_dispatched = grad_fc1 @ W1.T
                if FLUID_KERNELS_AVAILABLE:
                    grad_dispatched_exp = fluid_kernels.grouped_gemm(
                        grad_fc1_exp, w1_t, tokens_per_expert.to(torch.int32)
                    )
                else:
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

            # 3.5 Reorder to rank-major
            if num_local_experts == 1:
                grad_dispatched = grad_dispatched_exp
            else:
                grad_dispatched = grad_dispatched_exp.index_select(0, reorder_indices)

            if debug_timing:
                torch.cuda.synchronize()
                t5 = time.perf_counter()
                print(f"[Backward Timing] index_select back: {(t5-t4)*1000:.2f} ms")
                print(f"[Backward Timing] Step 3 total: {(t5-t0)*1000:.2f} ms")

            # Step 5: Dispatch backward AllToAll
            if debug_timing:
                t_step5_start = time.perf_counter()

            grad_dispatched_unsorted = grad_dispatched

            # TP reduce scatter
            if moe_layer.tp_size > 1:
                grad_dispatched_unsorted = reduce_scatter_to_sequence_parallel_region(
                    grad_dispatched_unsorted.to(probs.dtype),
                    group=moe_layer.tp_group,
                ).to(grad_dispatched_unsorted.dtype)

            if ep_size > 1:
                output_splits = input_splits_list
                input_splits = output_splits_list

                with torch.cuda.stream(scheduler.comm_stream):
                    scheduler.comm_stream.wait_stream(scheduler.default_stream)
                    grad_permuted = _all_to_all(
                        grad_dispatched_unsorted, output_splits, input_splits, moe_layer.ep_group
                    )
                    event = torch.cuda.Event()
                    event.record(scheduler.comm_stream)
                    scheduler.set_alltoall_end_event(event)

                scheduler.on_alltoall_start(comm_type="moe_dispatch_bwd")
                scheduler.default_stream.wait_stream(scheduler.comm_stream)
            else:
                grad_permuted = grad_dispatched_unsorted

            if debug_timing:
                torch.cuda.synchronize()
                t_step5_end = time.perf_counter()
                print(f"[Backward Timing] Step 5 (dispatch AllToAll): {(t_step5_end-t_step5_start)*1000:.2f} ms")

        # ============================================================
        # Step 4: Register dW tasks for overlap
        # ============================================================
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


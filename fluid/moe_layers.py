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

# 使用 PyTorch 原生的循环矩阵乘法（不使用自定义 CUDA 内核）
import os

# Import chunking configuration and direct AllToAll
from fluid.communication import get_dx_num_chunks, _all_to_all

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
        # Save tensors for backward (与 OVERLAP 模式保持一致，只保存必要的 tensor)
        # 只保存 input, weight1, weight2，其他信息通过 ctx 属性传递
        ctx.save_for_backward(permuted_local_hidden_states, weight1, weight2)

        # 非 tensor 数据通过 ctx 属性传递（与 OVERLAP 一致）
        ctx.tokens_per_expert_list = tokens_per_expert.tolist() if torch.is_tensor(tokens_per_expert) else list(tokens_per_expert)
        ctx.permuted_probs = permuted_probs.detach()  # 保存 probs 引用

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
            # 优化版：不再保存中间结果，backward 时重计算
            return fc2_output

        # Reshape the weights for the grouped GEMMs.
        w1 = weight1.view(num_local_experts, hidden_size, -1)
        w2 = weight2.view(num_local_experts, -1, hidden_size)

        # 使用 PyTorch 循环实现 GroupedGEMM
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

        # 优化版：不再预计算激活导数，backward 时重计算（与 OVERLAP 模式一致）
        # 这减少了 forward 的内存保存开销，用 backward 时的计算换取内存

        return fc2_output

    @staticmethod
    def backward(ctx, grad_fc2_output):
        import os
        import time

        debug_timing = os.environ.get('FLUID_DEBUG_BACKWARD_TIMING', '0') == '1'
        if debug_timing:
            print(f"[_FluidExpertComputation] backward called, grad shape: {grad_fc2_output.shape}", flush=True)

        # Retrieve saved tensors (与 OVERLAP 模式保持一致)
        permuted_local_hidden_states, weight1, weight2 = ctx.saved_tensors

        # 从 ctx 属性获取其他信息
        tokens_per_expert_list = ctx.tokens_per_expert_list
        permuted_probs = ctx.permuted_probs

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

        # ===== 优化版：重计算中间激活（用计算换内存，与 OVERLAP 模式一致）=====
        total_tokens = permuted_local_hidden_states.shape[0]
        ffn_size = w1.shape[2]

        # 重计算 fc1_output
        fc1_output = torch.zeros(total_tokens, ffn_size, dtype=permuted_local_hidden_states.dtype, device=permuted_local_hidden_states.device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = tokens_per_expert_list[exp_idx]
            if n_tok > 0:
                fc1_output[start:start+n_tok] = torch.matmul(permuted_local_hidden_states[start:start+n_tok], w1[exp_idx])
                start += n_tok

        # 重计算 intermediate_parallel (activation output)
        intermediate_parallel = activation_func(fc1_output, permuted_probs)

        # 重计算 act_deriv
        if gated_linear_unit:
            x_1, x_2 = torch.chunk(fc1_output, 2, dim=-1)
            if activation_func_type == 'silu':
                sig = torch.sigmoid(x_1)
                act_deriv = sig * (1 + x_1 * (1 - sig))
                act_val = x_1 * sig
            else:
                act_deriv = _gelu_grad_analytical(x_1)
                act_val = F.gelu(x_1)
        else:
            if activation_func_type == 'silu':
                sig = torch.sigmoid(fc1_output)
                act_deriv = sig * (1 + fc1_output * (1 - sig))
            else:
                act_deriv = _gelu_grad_analytical(fc1_output)
            act_val = None
            x_2 = None

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
            # FC1-only Chunked Backward with CommQueue
            # ============================================================
            # New design: FC2 and Activation use efficient grouped_gemm (no chunking)
            # Only FC1 backward is chunked, with each chunk submitted to CommQueue
            #
            # Timeline:
            #   default: |== FC2 ==|== Act ==|=FC1_c0=|=FC1_c1=|...| wait |
            #                                    ↓        ↓
            #   comm:                         |A2A_c0| |A2A_c1|...
            #
            # Benefits:
            # - FC2 uses grouped_gemm (efficient)
            # - Activation backward is fast (element-wise)
            # - Only FC1 is chunked for AllToAll overlap
            # - CommQueue simplifies async logic
            # ============================================================
            if debug_timing:
                print(f"[FC1 Chunked] Using CommQueue with {num_chunks} chunks", flush=True)
                torch.cuda.synchronize()
                t_start = time.perf_counter()

            # Get dispatch AllToAll context
            ep_group = ctx.dispatch_ep_group
            dispatch_input_splits = ctx.dispatch_input_splits
            dispatch_output_splits = ctx.dispatch_output_splits
            ep_size = len(dispatch_input_splits) if dispatch_input_splits is not None else 1

            # ========== Step 1: FC2 backward (full, grouped_gemm) ==========
            intermediate_dim = intermediate_parallel.shape[-1]
            grad_intermediate = torch.zeros(total_tokens, intermediate_dim, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert_list[exp_idx]
                if n_tok > 0:
                    grad_intermediate[start:start+n_tok] = torch.matmul(
                        grad_fc2_output[start:start+n_tok], w2[exp_idx].t()
                    )
                    start += n_tok

            # ========== Step 2: Activation backward (full, element-wise) ==========
            # 使用前面重计算的 act_val 和 x_2
            if gated_linear_unit:
                grad_x_1 = grad_intermediate * act_deriv * x_2
                grad_x_2 = grad_intermediate * act_val
                grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
            else:
                grad_fc1 = grad_intermediate * act_deriv

            if debug_timing:
                torch.cuda.synchronize()
                t_fc2_act = time.perf_counter()
                print(f"[FC1 Chunked] FC2+Act: {(t_fc2_act-t_start)*1000:.2f} ms", flush=True)

            # ========== Step 3: FC1 backward (chunked) + CommQueue ==========
            # Create CommQueue for async AllToAll
            comm_queue = CommQueue(scheduler.comm_stream)

            # Compute expert offsets for expert-major layout
            expert_offsets = [0]
            for exp_idx in range(num_local_experts - 1):
                expert_offsets.append(expert_offsets[-1] + tokens_per_expert_list[exp_idx])

            # Compute chunk ranges for each expert
            expert_chunk_ranges = []
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert_list[exp_idx]
                chunk_size_exp = n_tok // num_chunks
                remainder_exp = n_tok % num_chunks
                ranges = []
                local_start = 0
                for chunk_idx in range(num_chunks):
                    this_size = chunk_size_exp + (1 if chunk_idx < remainder_exp else 0)
                    ranges.append((local_start, local_start + this_size))
                    local_start += this_size
                expert_chunk_ranges.append(ranges)

            dx_chunk_times = [] if debug_timing else None

            for chunk_idx in range(num_chunks):
                if debug_timing:
                    torch.cuda.synchronize()
                    t_chunk_start = time.perf_counter()

                # FC1 backward for this chunk: grad_fc1 @ W1.T
                grad_dx_parts = []
                for exp_idx in range(num_local_experts):
                    local_start, local_end = expert_chunk_ranges[exp_idx][chunk_idx]
                    if local_end > local_start:
                        global_start = expert_offsets[exp_idx] + local_start
                        global_end = expert_offsets[exp_idx] + local_end
                        grad_dx = torch.matmul(grad_fc1[global_start:global_end], w1[exp_idx].t())
                        grad_dx_parts.append(grad_dx)

                grad_chunk = torch.cat(grad_dx_parts, dim=0) if grad_dx_parts else \
                    torch.empty(0, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

                if debug_timing:
                    torch.cuda.synchronize()
                    dx_chunk_times.append((time.perf_counter() - t_chunk_start) * 1000)

                # Compute chunk split sizes for AllToAll
                chunk_output_splits = []
                chunk_input_splits = []
                for rank_idx in range(ep_size):
                    if dispatch_output_splits is not None:
                        rank_tokens = int(dispatch_output_splits[rank_idx].item() if torch.is_tensor(dispatch_output_splits[rank_idx]) else dispatch_output_splits[rank_idx])
                    else:
                        rank_tokens = total_tokens // ep_size
                    chunk_size = rank_tokens // num_chunks
                    remainder = rank_tokens % num_chunks
                    this_size = chunk_size + (1 if chunk_idx < remainder else 0)
                    chunk_output_splits.append(this_size)

                for rank_idx in range(ep_size):
                    if dispatch_input_splits is not None:
                        rank_tokens = int(dispatch_input_splits[rank_idx].item() if torch.is_tensor(dispatch_input_splits[rank_idx]) else dispatch_input_splits[rank_idx])
                    else:
                        rank_tokens = total_tokens // ep_size
                    chunk_size = rank_tokens // num_chunks
                    remainder = rank_tokens % num_chunks
                    this_size = chunk_size + (1 if chunk_idx < remainder else 0)
                    chunk_input_splits.append(this_size)

                # Submit to CommQueue (async)
                comm_queue.submit_alltoall(
                    grad_chunk,
                    chunk_input_splits,   # backward: recv from where we sent
                    chunk_output_splits,  # backward: send to where we received
                    ep_group,
                )

            # ========== Step 4: Wait for all AllToAll and gather results ==========
            alltoall_results = comm_queue.wait_all()

            grad_permuted_local_hidden_states = torch.cat(alltoall_results, dim=0) if alltoall_results else \
                torch.empty(0, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)

            # Signal that dispatch AllToAll was handled here
            set_dispatch_alltoall_done(True)

            if debug_timing:
                torch.cuda.synchronize()
                t_end = time.perf_counter()
                print(f"[FC1 Chunked] Total: {(t_end-t_start)*1000:.2f} ms", flush=True)
                if dx_chunk_times:
                    print(f"  FC1 chunks: {dx_chunk_times} (total: {sum(dx_chunk_times):.2f} ms)", flush=True)

        else:
            # ============================================================
            # STANDARD (non-chunked) dX computation
            # ============================================================
            if debug_timing:
                torch.cuda.synchronize()
                t_dx_start = time.perf_counter()

            # === Step 1: FC2 backward (full, grouped_gemm) ===
            intermediate_dim = intermediate_parallel.shape[-1]
            grad_intermediate = torch.zeros(total_tokens, intermediate_dim, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert_list[exp_idx]
                if n_tok > 0:
                    grad_intermediate[start:start+n_tok] = torch.matmul(grad_fc2_output[start:start+n_tok], w2[exp_idx].t())
                    start += n_tok

            # === Step 2: Activation backward (full, element-wise) ===
            # 使用前面重计算的 act_val 和 x_2
            if gated_linear_unit:
                grad_x_1 = grad_intermediate * act_deriv * x_2 * probs
                grad_x_2 = grad_intermediate * act_val * probs
                grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
            else:
                grad_fc1 = grad_intermediate * act_deriv * probs

            # === Step 3: FC1 backward (full, grouped_gemm) ===
            grad_permuted_local_hidden_states = torch.zeros(total_tokens, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert_list[exp_idx]
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
        # tokens_per_expert_list 已经是 Python list，不需要 detach
        tokens_per_expert_saved = tokens_per_expert_list
        fc1_output_saved = fc1_output.detach()
        permuted_probs_saved = permuted_probs.detach()
        # Cache grad_fc1 directly - no need to recompute in compute_dw_weight1
        grad_fc1_saved = grad_fc1.detach()

        # Define dW computation function for all experts
        def compute_dw_weight2():
            """Compute grad_weight2 for all experts"""
            grad_w2_all = torch.zeros_like(weight2)
            w2_view = grad_w2_all.view(num_local_experts, -1, hidden_size)

            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert_saved[exp_idx]
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
            grad_w1_all = torch.zeros_like(weight1)
            w1_view = grad_w1_all.view(num_local_experts, hidden_size, -1)

            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert_saved[exp_idx]
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


# 旧的细粒度 MoE 重叠类已移除，新实现在 overlap_forward.py 中


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



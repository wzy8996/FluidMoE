"""
MoE Backward Operations with AllToAll + Chunked Overlap + dW Scheduling

This module provides all backward operations for MoE layers with:
- Combine AllToAll backward with dW overlap
- Expert computation backward (grad_fc1, grad_tokens)
- Dispatch AllToAll backward with optional chunked dX overlap
- dW task registration for weight1 and weight2

Key functions:
- register_moe_dw_tasks: Register dW tasks for weight1 and weight2
- combine_backward: Combine AllToAll backward with dW overlap
- expert_backward: Expert computation backward
- dispatch_backward: Dispatch AllToAll with optional chunked dX + AllToAll overlap

Design:
- dW tasks are registered BEFORE AllToAll to execute during communication
- Chunked dX computation overlaps with AllToAll (hidden dimension chunking)
- Two AllToAll operations: Combine backward and Dispatch backward

Timeline:
  Combine AllToAll: Launch async -> Execute queued dW tasks -> Wait
  Expert backward:  Compute grad_fc1, grad_tokens
  Register dW:      Register weight1/weight2 dW tasks
  Dispatch AllToAll: Chunked dX + AllToAll overlap (if num_chunks > 1)
"""

import os
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Dict

from fluid.core import _all_to_all, _sort_chunks_by_idxs
from fluid.core.scheduler import get_backward_scheduler
from fluid.core.nvtx import nvtx_range_push, nvtx_range_pop, nvtx_mark


# =============================================================================
# dW Task Registration
# =============================================================================

def register_moe_dw_tasks(
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    all_expert_tokens: torch.Tensor,
    act_output: torch.Tensor,
    grad_all_fc2: torch.Tensor,
    grad_all_fc1: torch.Tensor,
    num_local_experts: int,
    all_tokens_per_expert: List[int],
    layer_id: int,
    orig_weight1: torch.Tensor,
    orig_weight2: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Register dW tasks for weight1 and weight2 to execute during AllToAll communication.

    This should be called BEFORE the Dispatch AllToAll to allow dW computation
    to overlap with communication.

    Args:
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight (3D view)
        weight2: [num_local_experts, ffn_hidden, hidden] FC2 weight (3D view)
        all_expert_tokens: [total_recv, hidden] all tokens (expert-major order)
        act_output: [total_recv, ffn_hidden] activation output (from expert_backward)
        grad_all_fc2: [total_recv, hidden] gradient w.r.t. FC2 output
        grad_all_fc1: [total_recv, ffn_hidden] gradient w.r.t. FC1 output
        num_local_experts: Number of local experts
        all_tokens_per_expert: Token count per expert
        layer_id: Layer ID for task naming
        orig_weight1: Original weight1 tensor (2D) for gradient assignment
        orig_weight2: Original weight2 tensor (2D) for gradient assignment

    Returns:
        (grad_weight1, grad_weight2) if scheduler disabled, else (None, None)
        Note: gradients are returned in 2D format matching orig_weight shapes
    """
    scheduler = get_backward_scheduler()

    # Debug output
    if os.environ.get('FLUID_DEBUG_SCHEDULER_REGISTER', '0') == '1':
        print(f"[DEBUG] register_moe_dw_tasks(L{layer_id}): scheduler.is_enabled()={scheduler.is_enabled()}", flush=True)

    # Get dimensions
    ffn_hidden = weight1.shape[-1]
    hidden_size = weight2.shape[-1]

    if scheduler.is_enabled():
        # Scheduler enabled: register dW tasks for execution during later AllToAll
        num_local_experts_saved = num_local_experts
        all_tokens_per_expert_saved = all_tokens_per_expert
        grad_all_fc2_saved = grad_all_fc2.detach()
        grad_all_fc1_saved = grad_all_fc1.detach()
        act_output_saved = act_output.detach()
        all_expert_tokens_saved = all_expert_tokens.detach()
        ffn_hidden_saved = ffn_hidden
        hidden_size_saved = hidden_size

        def compute_dw_weight2():
            # Compute gradients directly in 3D [E, ffn, hidden] - no permute needed
            # Weight shape: [E, ffn, hidden], gradient same shape
            device = grad_all_fc2_saved.device
            dtype = grad_all_fc2_saved.dtype
            grad_w2_3d = torch.zeros(num_local_experts_saved, ffn_hidden_saved, hidden_size_saved,
                                     dtype=dtype, device=device)
            start = 0
            for exp_idx in range(num_local_experts_saved):
                n_tok = all_tokens_per_expert_saved[exp_idx]
                if n_tok > 0:
                    grad_w2_3d[exp_idx] = torch.matmul(
                        act_output_saved[start:start+n_tok].t(),
                        grad_all_fc2_saved[start:start+n_tok]
                    )
                    start += n_tok
            return grad_w2_3d  # Return 3D directly

        def compute_dw_weight1():
            # Compute gradients directly in 3D [E, hidden, ffn] - no permute needed
            # Weight shape: [E, hidden, ffn], gradient same shape
            device = grad_all_fc1_saved.device
            dtype = grad_all_fc1_saved.dtype
            grad_w1_3d = torch.zeros(num_local_experts_saved, hidden_size_saved, ffn_hidden_saved,
                                     dtype=dtype, device=device)
            start = 0
            for exp_idx in range(num_local_experts_saved):
                n_tok = all_tokens_per_expert_saved[exp_idx]
                if n_tok > 0:
                    grad_w1_3d[exp_idx] = torch.matmul(
                        all_expert_tokens_saved[start:start+n_tok].t(),
                        grad_all_fc1_saved[start:start+n_tok]
                    )
                    start += n_tok
            return grad_w1_3d  # Return 3D directly

        scheduler.register_dw_task(
            layer_name=f"moe_weight2_L{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight2,
            priority=100,
            weight_param=orig_weight2,
            needs_ar=False,  # Expert params use EP, not DP
        )
        scheduler.register_dw_task(
            layer_name=f"moe_weight1_L{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight1,
            priority=99,
            weight_param=orig_weight1,
            needs_ar=False,  # Expert params use EP, not DP
        )
        return None, None
    else:
        # Scheduler disabled: compute dW directly in 3D (no permute needed)
        device = grad_all_fc2.device
        dtype = grad_all_fc2.dtype

        # Compute in 3D [E, ffn, hidden] for weight2
        grad_w2_3d = torch.zeros(num_local_experts, ffn_hidden, hidden_size, dtype=dtype, device=device)
        # Compute in 3D [E, hidden, ffn] for weight1
        grad_w1_3d = torch.zeros(num_local_experts, hidden_size, ffn_hidden, dtype=dtype, device=device)

        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                grad_w2_3d[exp_idx] = torch.matmul(
                    act_output[start:start+n_tok].t(),
                    grad_all_fc2[start:start+n_tok]
                )
                grad_w1_3d[exp_idx] = torch.matmul(
                    all_expert_tokens[start:start+n_tok].t(),
                    grad_all_fc1[start:start+n_tok]
                )
                start += n_tok

        # Return 3D gradients directly (weights are now 3D)
        return grad_w1_3d, grad_w2_3d


# =============================================================================
# FC1 Recomputation (for Activation Recomputation optimization)
# =============================================================================

def recompute_fc1(
    all_expert_tokens: torch.Tensor,
    weight1: torch.Tensor,
    num_local_experts: int,
    all_tokens_per_expert: List[int],
) -> torch.Tensor:
    """
    Recompute FC1 outputs from tokens and weight1.

    This is called during backward to avoid saving FC1 in forward,
    reducing memory copy overhead (~2.5ms savings in forward).

    The recomputation takes ~5ms but can be hidden by Combine AllToAll (~28ms).

    Args:
        all_expert_tokens: [total_recv, hidden] all tokens (expert-major order)
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight (3D view)
        num_local_experts: Number of local experts
        all_tokens_per_expert: Token count per expert

    Returns:
        all_fc1: [total_recv, ffn_hidden] FC1 outputs (expert-major order)
    """
    device = all_expert_tokens.device
    dtype = all_expert_tokens.dtype
    total_recv = all_expert_tokens.shape[0]
    ffn_hidden = weight1.shape[-1]

    if total_recv == 0:
        return torch.empty(0, ffn_hidden, dtype=dtype, device=device)

    all_fc1 = torch.zeros(total_recv, ffn_hidden, dtype=dtype, device=device)

    start = 0
    for exp_idx in range(num_local_experts):
        n_tok = all_tokens_per_expert[exp_idx]
        if n_tok > 0:
            # FC1 = tokens @ weight1
            all_fc1[start:start + n_tok] = torch.matmul(
                all_expert_tokens[start:start + n_tok],
                weight1[exp_idx]
            )
            start += n_tok

    return all_fc1


# =============================================================================
# Combine Backward AllToAll with FC1 Recomputation
# =============================================================================

def combine_backward(
    grad_output: torch.Tensor,
    input_splits_list: List[int],
    output_splits_list: List[int],
    ep_group,
    layer_id: int,
    all_expert_tokens: Optional[torch.Tensor] = None,
    weight1: Optional[torch.Tensor] = None,
    num_local_experts: int = 0,
    all_tokens_per_expert: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Combine backward AllToAll with dW overlap and optional FC1 recomputation.

    Uses comm thread for AllToAll execution:
    1. Submit AllToAll to comm thread
    2. Recompute FC1 on main thread (overlaps with AllToAll)
    3. Execute queued dW tasks (overlaps with AllToAll)
    4. Wait for AllToAll completion

    FC1 recomputation (~5ms) is hidden by Combine AllToAll (~28ms).

    Args:
        grad_output: [total_output, hidden] gradient w.r.t. output
        input_splits_list: AllToAll input split sizes
        output_splits_list: AllToAll output split sizes
        ep_group: Expert parallel process group
        layer_id: Layer ID for debugging
        all_expert_tokens: [total_recv, hidden] tokens for FC1 recomputation (optional)
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight for recomputation (optional)
        num_local_experts: Number of local experts (required if recomputing FC1)
        all_tokens_per_expert: Token count per expert (required if recomputing FC1)

    Returns:
        grad_combined: [total_recv, hidden] gradient after AllToAll
        all_fc1: [total_recv, ffn_hidden] recomputed FC1 or None
    """
    nvtx_range_push("combine_backward")
    scheduler = get_backward_scheduler()
    all_fc1 = None

    if scheduler.is_enabled():
        # Prepare input tensor
        grad_output_contig = grad_output.contiguous()

        # Create closure for AllToAll execution
        result_holder = [None]
        def do_alltoall():
            result_holder[0] = _all_to_all(
                grad_output_contig,
                output_split_sizes=output_splits_list,
                input_split_sizes=input_splits_list,
                group=ep_group
            )
            return result_holder[0]

        # Submit AllToAll to comm thread (async)
        nvtx_range_push("combine_alltoall_submit")
        task_id = scheduler.submit_alltoall(do_alltoall)
        nvtx_range_pop()

        # Recompute FC1 on main thread while AllToAll is running
        # This overlaps with communication (~5ms compute vs ~28ms AllToAll)
        if all_expert_tokens is not None and weight1 is not None:
            nvtx_range_push("fc1_recompute")
            all_fc1 = recompute_fc1(
                all_expert_tokens, weight1,
                num_local_experts, all_tokens_per_expert
            )
            nvtx_range_pop()

        # Execute dW tasks while AllToAll is running
        if not os.environ.get('FLUID_SKIP_ALL_DW', '0') == '1':
            nvtx_range_push("dw_tasks_execute")
            scheduler.execute_dw_tasks()
            nvtx_range_pop()

        # Wait for AllToAll to complete
        nvtx_range_push("combine_alltoall_wait")
        scheduler.wait_alltoall(task_id)
        nvtx_range_pop()

        grad_combined = result_holder[0]
    else:
        nvtx_range_push("combine_alltoall_sync")
        grad_combined = _all_to_all(
            grad_output.contiguous(),
            output_split_sizes=output_splits_list,
            input_split_sizes=input_splits_list,
            group=ep_group
        )
        nvtx_range_pop()
        # Recompute FC1 (no overlap in non-scheduler mode)
        if all_expert_tokens is not None and weight1 is not None:
            nvtx_range_push("fc1_recompute")
            all_fc1 = recompute_fc1(
                all_expert_tokens, weight1,
                num_local_experts, all_tokens_per_expert
            )
            nvtx_range_pop()

    nvtx_range_pop()  # combine_backward
    return grad_combined, all_fc1


# =============================================================================
# Combined Expert + Dispatch Backward with Chunked dX + AllToAll Overlap
# =============================================================================

def expert_dispatch_backward(
    grad_all_fc2: torch.Tensor,
    all_fc1: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    activation_func,
    num_local_experts: int,
    all_tokens_per_expert: List[int],
    split_sizes_exp_major: List[int],
    sorted_idxs_exp_to_rank: List[int],
    input_splits_list: List[int],
    output_splits_list: List[int],
    ep_group,
    layer_id: int = 0,
    num_chunks: int = 1,
    comm_stream: Optional[torch.cuda.Stream] = None,
    # For dW registration (optional - if provided, will register dW tasks)
    all_expert_tokens: Optional[torch.Tensor] = None,
    orig_weight1: Optional[torch.Tensor] = None,
    orig_weight2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Combined expert backward + dispatch AllToAll with optional chunked dX overlap.

    This function combines expert_backward, register_moe_dw_tasks, and dispatch_backward:
    1. Compute grad_fc1 from grad_fc2 through FC2 and activation
    2. Register dW tasks for weight1/weight2 (if params provided)
    3. Compute dX (grad_tokens = grad_fc1 @ weight1.T) with optional chunking
    4. Do AllToAll (overlapped with dX computation when chunked, dW tasks during A2A)

    When num_chunks > 1: dX is computed in chunks, each immediately sent via AllToAll
    while the next chunk is being computed. This overlaps compute with communication.

    When num_chunks == 1: dX is computed in one pass, then AllToAll is performed.

    Args:
        grad_all_fc2: [total_recv, hidden] gradient w.r.t. FC2 output
        all_fc1: [total_recv, ffn_hidden] all FC1 outputs (expert-major order)
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight
        weight2: [num_local_experts, ffn_hidden, hidden] FC2 weight
        activation_func: Activation function
        num_local_experts: Number of local experts
        all_tokens_per_expert: Token count per expert
        split_sizes_exp_major: Chunk sizes in expert-major order
        sorted_idxs_exp_to_rank: Indices for expert-major -> rank-major reorder
        input_splits_list: AllToAll input split sizes
        output_splits_list: AllToAll output split sizes
        ep_group: Expert parallel process group
        layer_id: Layer ID for debugging
        num_chunks: Number of chunks for hidden dimension (1 = no chunking)
        comm_stream: CUDA stream for communication
        all_expert_tokens: [total_recv, hidden] input tokens (for dW registration)
        orig_weight1: Original weight1 tensor for gradient assignment
        orig_weight2: Original weight2 tensor for gradient assignment

    Returns:
        grad_all_fc1: [total_recv, ffn_hidden] gradient w.r.t. FC1 output
        act_output: [total_recv, ffn_hidden] activation output
        grad_tokens: [total_send, hidden] gradient w.r.t. input tokens (after AllToAll)
        grad_weight1: Gradient for weight1 (if scheduler disabled, else None)
        grad_weight2: Gradient for weight2 (if scheduler disabled, else None)

    Timeline (when num_chunks > 1):
        1. Compute grad_fc1 (FC2 backward + activation backward)
        2. Register dW tasks for weight1/weight2
        3. Chunk 0: compute dX[:, 0:H/4] → submit A2A
        4. Chunk 1: compute dX[:, H/4:H/2] → submit A2A (while A2A_0 runs)
        5. Chunk 2: compute dX[:, H/2:3H/4] → submit A2A (while A2A_1 runs)
        6. Chunk 3: compute dX[:, 3H/4:H] → submit A2A (while A2A_2 runs)
        7. Execute dW tasks during final A2A
        8. Wait for all A2A to complete
    """
    nvtx_range_push("expert_dispatch_backward")
    scheduler = get_backward_scheduler()
    device = grad_all_fc2.device
    dtype = grad_all_fc2.dtype
    total_recv = grad_all_fc2.shape[0]
    hidden_size = weight2.shape[-1]
    ffn_hidden = weight1.shape[-1]

    # Convert to list if tensor
    if torch.is_tensor(split_sizes_exp_major):
        split_sizes_exp_major = split_sizes_exp_major.tolist()
    if torch.is_tensor(sorted_idxs_exp_to_rank):
        sorted_idxs_exp_to_rank = sorted_idxs_exp_to_rank.tolist()

    # Validate chunk size
    if num_chunks > 1 and hidden_size % num_chunks != 0:
        num_chunks = 1

    # Get streams
    if comm_stream is None and scheduler.is_enabled():
        comm_stream = scheduler.comm_stream
    default_stream = torch.cuda.current_stream()

    # =========================================================================
    # Step 1: Expert backward - compute grad_fc1 from grad_fc2
    # =========================================================================
    nvtx_range_push("expert_backward_fc2_to_fc1")

    # Compute grad_exp_act = grad_fc2 @ weight2.T for all experts
    grad_all_exp_act = torch.zeros(total_recv, ffn_hidden, dtype=dtype, device=device)
    start = 0
    for exp_idx in range(num_local_experts):
        n_tok = all_tokens_per_expert[exp_idx]
        if n_tok > 0:
            grad_all_exp_act[start:start+n_tok] = torch.matmul(
                grad_all_fc2[start:start+n_tok], weight2[exp_idx].t()
            )
            start += n_tok

    # Activation backward using autograd (much faster than explicit formula)
    with torch.enable_grad():
        fc1_with_grad = all_fc1.detach().requires_grad_(True)
        act_output = activation_func(fc1_with_grad)
        grad_all_fc1, = torch.autograd.grad(
            act_output, fc1_with_grad, grad_all_exp_act, retain_graph=False
        )
    nvtx_range_pop()

    # =========================================================================
    # Step 2: Register dW tasks (if params provided)
    # =========================================================================
    grad_weight1 = None
    grad_weight2 = None
    if all_expert_tokens is not None and orig_weight1 is not None and orig_weight2 is not None:
        grad_weight1, grad_weight2 = register_moe_dw_tasks(
            weight1, weight2, all_expert_tokens, act_output.detach(),
            grad_all_fc2, grad_all_fc1,
            num_local_experts, all_tokens_per_expert, layer_id,
            orig_weight1, orig_weight2
        )

    # =========================================================================
    # Step 3: Dispatch backward - compute dX and AllToAll
    # =========================================================================
    if num_chunks == 1:
        # -----------------------------------------------------------------
        # Non-chunked path: compute full dX, then AllToAll via comm thread
        # -----------------------------------------------------------------
        nvtx_range_push("dispatch_non_chunked")

        # Compute dX = grad_fc1 @ weight1.T
        nvtx_range_push("dx_compute")
        grad_all_tokens = torch.zeros(total_recv, hidden_size, dtype=dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                grad_all_tokens[start:start+n_tok] = torch.matmul(
                    grad_all_fc1[start:start+n_tok], weight1[exp_idx].t()
                )
                start += n_tok
        nvtx_range_pop()

        # Reorder expert-major -> rank-major
        nvtx_range_push("reorder")
        grad_dispatched = _sort_chunks_by_idxs(
            grad_all_tokens, split_sizes_exp_major, sorted_idxs_exp_to_rank
        )
        nvtx_range_pop()

        # AllToAll via comm thread with dW overlap
        if scheduler.is_enabled():
            grad_dispatched_contig = grad_dispatched.contiguous()
            result_holder = [None]
            def do_alltoall():
                result_holder[0] = _all_to_all(
                    grad_dispatched_contig,
                    output_split_sizes=input_splits_list,
                    input_split_sizes=output_splits_list,
                    group=ep_group
                )
                return result_holder[0]

            nvtx_range_push("alltoall_submit")
            task_id = scheduler.submit_alltoall(do_alltoall)
            nvtx_range_pop()

            # Execute dW tasks while AllToAll is running
            if not (os.environ.get('FLUID_SKIP_DISPATCH_DW', '0') == '1' or
                    os.environ.get('FLUID_SKIP_ALL_DW', '0') == '1'):
                nvtx_range_push("dw_tasks")
                scheduler.execute_dw_tasks()
                nvtx_range_pop()

            nvtx_range_push("alltoall_wait")
            scheduler.wait_alltoall(task_id)
            nvtx_range_pop()

            grad_tokens = result_holder[0]
        else:
            nvtx_range_push("alltoall_sync")
            grad_tokens = _all_to_all(
                grad_dispatched.contiguous(),
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group
            )
            nvtx_range_pop()

        nvtx_range_pop()  # dispatch_non_chunked

    else:
        # -----------------------------------------------------------------
        # Chunked path: compute dX in chunks overlapped with AllToAll
        # Uses comm thread for AllToAll execution
        # Pipeline: dX_0 -> submit A2A_0 -> dX_1 (parallel) -> submit A2A_1 -> ...
        # -----------------------------------------------------------------
        nvtx_range_push("dispatch_chunked")
        chunk_size = hidden_size // num_chunks

        # Pre-allocate output buffer
        total_send = sum(input_splits_list)
        grad_tokens = torch.empty(total_send, hidden_size, dtype=dtype, device=device)

        # Pre-allocate chunk buffers
        chunk_dx_buffers = [torch.empty(total_recv, chunk_size, dtype=dtype, device=device)
                           for _ in range(num_chunks)]
        chunk_reorder_buffers = [torch.empty(total_recv, chunk_size, dtype=dtype, device=device)
                                 for _ in range(num_chunks)]

        # Track AllToAll results and task IDs
        chunk_results = [None] * num_chunks
        task_ids = []

        scheduler_enabled = scheduler.is_enabled()

        # Pipeline: compute dX_i, submit AllToAll_i (overlaps with dX_{i+1})
        for chunk_idx in range(num_chunks):
            h_start = chunk_idx * chunk_size
            h_end = h_start + chunk_size

            # ----- Step 1: Compute dX chunk -----
            nvtx_range_push(f"dx_chunk_{chunk_idx}")
            chunk_dx_buffers[chunk_idx].zero_()
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = all_tokens_per_expert[exp_idx]
                if n_tok > 0:
                    chunk_dx_buffers[chunk_idx][start:start + n_tok] = torch.matmul(
                        grad_all_fc1[start:start + n_tok],
                        weight1[exp_idx, h_start:h_end, :].t()
                    )
                    start += n_tok

            # Reorder for AllToAll
            reordered = _sort_chunks_by_idxs(
                chunk_dx_buffers[chunk_idx],
                split_sizes_exp_major,
                sorted_idxs_exp_to_rank
            )
            chunk_reorder_buffers[chunk_idx].copy_(reordered)
            nvtx_range_pop()

            # ----- Step 2: Submit AllToAll to comm thread (overlaps with next dX) -----
            nvtx_range_push(f"alltoall_submit_{chunk_idx}")
            if scheduler_enabled:
                # Capture chunk_idx for closure
                _chunk_idx = chunk_idx
                _input_buf = chunk_reorder_buffers[chunk_idx].contiguous()
                _total_send = total_send
                _chunk_size = chunk_size

                def make_alltoall_fn(idx, input_buf, out_size, c_size):
                    def do_alltoall():
                        output_buf = torch.empty(out_size, c_size, dtype=dtype, device=device)
                        dist.all_to_all_single(
                            output_buf,
                            input_buf,
                            output_split_sizes=input_splits_list,
                            input_split_sizes=output_splits_list,
                            group=ep_group,
                        )
                        chunk_results[idx] = output_buf
                        return output_buf
                    return do_alltoall

                task_id = scheduler.submit_alltoall(make_alltoall_fn(_chunk_idx, _input_buf, _total_send, _chunk_size))
                task_ids.append(task_id)
            else:
                # Synchronous execution
                output_buf = torch.empty(total_send, chunk_size, dtype=dtype, device=device)
                dist.all_to_all_single(
                    output_buf,
                    chunk_reorder_buffers[chunk_idx],
                    output_split_sizes=input_splits_list,
                    input_split_sizes=output_splits_list,
                    group=ep_group,
                )
                chunk_results[chunk_idx] = output_buf
            nvtx_range_pop()

        # Execute dW tasks while AllToAll is running
        if scheduler_enabled:
            if not (os.environ.get('FLUID_SKIP_DISPATCH_DW', '0') == '1' or
                    os.environ.get('FLUID_SKIP_ALL_DW', '0') == '1'):
                nvtx_range_push("dw_tasks")
                scheduler.execute_dw_tasks()
                nvtx_range_pop()

            # Wait for the last AllToAll only (FIFO guarantees earlier ones are done)
            # Pass num_tasks to correctly decrement the in_progress counter
            nvtx_range_push("alltoall_wait")
            if task_ids:
                scheduler.wait_alltoall(task_ids[-1], num_tasks=len(task_ids))
            nvtx_range_pop()

        # Gather results
        nvtx_range_push("gather_alltoall_results")
        for chunk_idx in range(num_chunks):
            h_start = chunk_idx * chunk_size
            h_end = h_start + chunk_size
            grad_tokens[:, h_start:h_end].copy_(chunk_results[chunk_idx])
        nvtx_range_pop()

        nvtx_range_pop()  # dispatch_chunked

    nvtx_range_pop()  # expert_dispatch_backward
    return grad_all_fc1, act_output.detach(), grad_tokens, grad_weight1, grad_weight2


# =============================================================================
# Router Backward
# =============================================================================

def router_backward(
    grad_permuted_probs: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_indices: torch.Tensor,
    permuted_probs: torch.Tensor,
    router_probs: torch.Tensor,
    top_indices: torch.Tensor,
    router_weight: torch.Tensor,
    num_tokens: int,
    top_k: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute router backward: gradients through softmax, top-k, and linear projection.

    This computes:
    1. Gradient through prob permutation/sorting
    2. Gradient through prob normalization
    3. Gradient through top-k selection
    4. Gradient through softmax
    5. Gradient through router linear (input @ weight -> logits)

    Args:
        grad_permuted_probs: [num_tokens * top_k] gradient w.r.t. permuted probabilities
        sorted_indices: [num_tokens * top_k] indices used to sort tokens
        restore_indices: [num_tokens * top_k] indices to restore original order
        permuted_probs: [num_tokens * top_k] permuted routing probabilities
        router_probs: [num_tokens, num_experts] full router probabilities (after softmax)
        top_indices: [num_tokens, top_k] indices of top-k experts
        router_weight: [hidden_size, num_experts] router weight matrix
        num_tokens: Number of input tokens
        top_k: Number of experts per token
        dtype: Data type for output

    Returns:
        grad_hidden_from_router: [num_tokens, hidden_size] gradient w.r.t. hidden_states
        grad_router_logits: [num_tokens, num_experts] gradient w.r.t. router logits (for dW)
    """
    # Step 1: Backward through prob permutation
    # permuted_probs = expanded_probs[sorted_indices]
    # expanded_probs = top_probs.reshape(-1)
    grad_expanded_probs = torch.zeros_like(grad_permuted_probs)
    grad_expanded_probs[sorted_indices] = grad_permuted_probs
    grad_top_probs = grad_expanded_probs.view(num_tokens, top_k)

    # Step 2: Backward through normalization
    # top_probs = raw_top_probs / sum(raw_top_probs)
    # grad_raw[i] = grad_top[i] / s - (grad_top · top_probs) * top_probs[i] / s
    top_probs_saved = permuted_probs[restore_indices].view(num_tokens, top_k)
    grad_dot = (grad_top_probs * top_probs_saved).sum(dim=-1, keepdim=True)
    grad_raw_top_probs = (grad_top_probs - grad_dot * top_probs_saved) / top_probs_saved.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    # Step 3: Backward through top-k selection
    # top_probs, top_indices = torch.topk(router_probs, k=top_k, dim=-1)
    # grad_router_probs is zero except at top_indices positions
    grad_router_probs = torch.zeros_like(router_probs)
    grad_router_probs.scatter_(1, top_indices, grad_raw_top_probs)

    # Step 4: Backward through softmax
    # router_probs = softmax(router_logits)
    # grad_logits = router_probs * (grad_probs - sum(grad_probs * router_probs))
    sum_grad_probs = (grad_router_probs * router_probs).sum(dim=-1, keepdim=True)
    grad_router_logits = router_probs * (grad_router_probs - sum_grad_probs)

    # Step 5: Backward through router linear: logits = hidden @ weight
    # grad_hidden = grad_logits @ weight.T
    grad_hidden_from_router = torch.matmul(grad_router_logits.float(), router_weight.t().float()).to(dtype)

    return grad_hidden_from_router, grad_router_logits


def register_router_dw_task(
    hidden_states: torch.Tensor,
    grad_router_logits: torch.Tensor,
    router_weight: torch.Tensor,
    layer_id: int,
) -> Optional[torch.Tensor]:
    """
    Register router weight gradient task or compute directly if scheduler disabled.

    The router weight gradient is computed as: grad_weight = hidden.T @ grad_logits

    Args:
        hidden_states: [num_tokens, hidden_size] input hidden states
        grad_router_logits: [num_tokens, num_experts] gradient w.r.t. router logits
        router_weight: [hidden_size, num_experts] original router weight (for gradient assignment)
        layer_id: Layer ID for task naming

    Returns:
        grad_router_weight: Gradient if scheduler disabled, else None (registered as task)
    """
    scheduler = get_backward_scheduler()

    if scheduler.is_enabled():
        hidden_saved = hidden_states.detach()
        grad_logits_saved = grad_router_logits.detach()

        def compute_router_dw():
            grad_weight = torch.matmul(hidden_saved.t().float(), grad_logits_saved.float())
            return grad_weight

        scheduler.register_dw_task(
            layer_name=f"router_weight_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_router_dw,
            priority=0,  # Low priority, execute after other dW tasks
            weight_param=router_weight,
        )
        return None
    else:
        grad_router_weight = torch.matmul(hidden_states.t().float(), grad_router_logits.float())
        return grad_router_weight


__all__ = [
    # FC1 recomputation
    'recompute_fc1',
    # dW registration
    'register_moe_dw_tasks',
    # Combine backward
    'combine_backward',
    # Expert backward
    'expert_backward',
    # Dispatch backward (unified)
    'dispatch_backward',
    # Router backward
    'router_backward',
    'register_router_dw_task',
]

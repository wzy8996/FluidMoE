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
from typing import List, Tuple, Optional, Dict

from fluid.core import _all_to_all, _sort_chunks_by_idxs
from fluid.core.scheduler import get_backward_scheduler


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
        )
        scheduler.register_dw_task(
            layer_name=f"moe_weight1_L{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight1,
            priority=99,
            weight_param=orig_weight1,
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

    Launches AllToAll on comm_stream (async), then:
    1. Recomputes FC1 on default_stream (if tokens/weight1 provided)
    2. Executes queued dW tasks
    3. Waits for AllToAll completion

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
    scheduler = get_backward_scheduler()
    all_fc1 = None

    if scheduler.is_enabled():
        comm_stream = scheduler.comm_stream
        default_stream = scheduler.default_stream
        # Launch AllToAll on comm_stream (async)
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_stream(default_stream)
            grad_combined = _all_to_all(
                grad_output.contiguous(),
                output_split_sizes=output_splits_list,
                input_split_sizes=input_splits_list,
                group=ep_group
            )
            scheduler.record_alltoall_end(comm_stream)

        # Recompute FC1 on default_stream while AllToAll is running
        # This overlaps with communication (~5ms compute vs ~28ms AllToAll)
        if all_expert_tokens is not None and weight1 is not None:
            all_fc1 = recompute_fc1(
                all_expert_tokens, weight1,
                num_local_experts, all_tokens_per_expert
            )

        # Execute dW tasks from queue while AllToAll is running
        scheduler.on_alltoall_start(comm_type=f"moe_combine_L{layer_id}")
        # Wait for AllToAll to complete
        default_stream.wait_stream(comm_stream)
    else:
        grad_combined = _all_to_all(
            grad_output.contiguous(),
            output_split_sizes=output_splits_list,
            input_split_sizes=input_splits_list,
            group=ep_group
        )
        # Recompute FC1 (no overlap in non-scheduler mode)
        if all_expert_tokens is not None and weight1 is not None:
            all_fc1 = recompute_fc1(
                all_expert_tokens, weight1,
                num_local_experts, all_tokens_per_expert
            )

    return grad_combined, all_fc1


# =============================================================================
# Expert Backward Computation
# =============================================================================

def expert_backward(
    grad_all_fc2: torch.Tensor,
    all_fc1: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    activation_func,
    num_local_experts: int,
    all_tokens_per_expert: List[int],
    compute_dx: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Expert computation backward: compute grad_fc1 and optionally grad_tokens (dX).

    Uses torch.autograd.grad for activation backward which is significantly faster
    than explicitly computing GELU gradient formula (~10ms savings).

    Args:
        grad_all_fc2: [total_recv, hidden] gradient w.r.t. FC2 output
        all_fc1: [total_recv, ffn_hidden] all FC1 outputs (expert-major order)
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight
        weight2: [num_local_experts, ffn_hidden, hidden] FC2 weight
        activation_func: Activation function
        num_local_experts: Number of local experts
        all_tokens_per_expert: Token count per expert
        compute_dx: Whether to compute grad_tokens (False when using chunked backward)

    Returns:
        grad_all_fc1: [total_recv, ffn_hidden] gradient w.r.t. FC1 output
        act_output: [total_recv, ffn_hidden] activation output (for dW computation)
        grad_all_tokens: [total_recv, hidden] or None - gradient w.r.t. input tokens
    """
    device = grad_all_fc2.device
    dtype = grad_all_fc2.dtype
    total_recv = grad_all_fc2.shape[0]
    hidden_size = weight2.shape[-1]
    ffn_hidden = weight1.shape[-1]

    # First pass: compute grad_exp_act for all experts (need this for autograd)
    grad_all_exp_act = torch.zeros(total_recv, ffn_hidden, dtype=dtype, device=device)
    start = 0
    for exp_idx in range(num_local_experts):
        n_tok = all_tokens_per_expert[exp_idx]
        if n_tok > 0:
            # grad_exp_act = grad_fc2 @ weight2.T
            grad_all_exp_act[start:start+n_tok] = torch.matmul(
                grad_all_fc2[start:start+n_tok], weight2[exp_idx].t()
            )
            start += n_tok

    # Compute activation backward using autograd (fused, much faster than explicit formula)
    # This computes grad_fc1 from grad_exp_act in one operation
    # Need to enable_grad() since we're inside a backward function where grad mode is disabled
    with torch.enable_grad():
        fc1_with_grad = all_fc1.detach().requires_grad_(True)
        act_output = activation_func(fc1_with_grad)

        # Use autograd.grad to compute activation backward efficiently
        grad_all_fc1, = torch.autograd.grad(
            act_output,
            fc1_with_grad,
            grad_all_exp_act,
            retain_graph=False,
        )

    # Second pass: compute grad_tokens if needed
    if compute_dx:
        grad_all_tokens = torch.zeros(total_recv, hidden_size, dtype=dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                # grad_tokens = grad_fc1 @ weight1.T
                grad_all_tokens[start:start+n_tok] = torch.matmul(
                    grad_all_fc1[start:start+n_tok], weight1[exp_idx].t()
                )
                start += n_tok
    else:
        grad_all_tokens = None

    # Return activation output for dW computation (already computed above)
    return grad_all_fc1, act_output.detach(), grad_all_tokens


# =============================================================================
# Dispatch Backward with Optional Chunked dX + AllToAll Overlap
# =============================================================================

def dispatch_backward(
    grad_all_fc1: torch.Tensor,
    weight1: torch.Tensor,
    split_sizes_exp_major: List[int],
    sorted_idxs_exp_to_rank: List[int],
    all_tokens_per_expert: List[int],
    input_splits_list: List[int],
    output_splits_list: List[int],
    ep_group,
    layer_id: int = 0,
    num_chunks: int = 1,
    grad_all_tokens: Optional[torch.Tensor] = None,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    MoE Dispatch backward with optional hidden-dimension chunking for dX + AllToAll overlap.

    When num_chunks > 1: computes grad_tokens = grad_fc1 @ weight1.t() in chunks,
    where each chunk is immediately sent via AllToAll while the next chunk is being computed.

    When num_chunks == 1: uses the pre-computed grad_all_tokens directly (non-chunked path).

    Args:
        grad_all_fc1: [total_recv, ffn_hidden] gradient w.r.t. FC1 output
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight
        split_sizes_exp_major: Chunk sizes in expert-major order
        sorted_idxs_exp_to_rank: Indices for expert-major -> rank-major reorder
        all_tokens_per_expert: Token count per expert
        input_splits_list: AllToAll input split sizes
        output_splits_list: AllToAll output split sizes
        ep_group: Expert parallel process group
        layer_id: Layer ID for debugging
        num_chunks: Number of chunks for hidden dimension (1 = no chunking)
        grad_all_tokens: Pre-computed gradient (required when num_chunks=1)
        comm_stream: CUDA stream for communication

    Returns:
        grad_tokens: [total_send, hidden] gradient w.r.t. input tokens

    Timeline (when num_chunks > 1):
        Chunk 0: compute dX[:, 0:H/4] → submit A2A
        Chunk 1: compute dX[:, H/4:H/2] → submit A2A (while A2A_0 runs)
        Chunk 2: compute dX[:, H/2:3H/4] → submit A2A (while A2A_1 runs)
        Chunk 3: compute dX[:, 3H/4:H] → submit A2A (while A2A_2 runs)
        Wait for all A2A to complete, concatenate results
    """
    scheduler = get_backward_scheduler()
    device = grad_all_fc1.device
    dtype = grad_all_fc1.dtype
    total_recv = grad_all_fc1.shape[0]
    num_local_experts = weight1.shape[0]
    hidden_size = weight1.shape[1]

    # Convert to list if tensor
    if torch.is_tensor(split_sizes_exp_major):
        split_sizes_exp_major = split_sizes_exp_major.tolist()
    if torch.is_tensor(sorted_idxs_exp_to_rank):
        sorted_idxs_exp_to_rank = sorted_idxs_exp_to_rank.tolist()

    # Validate chunk size
    if num_chunks > 1 and hidden_size % num_chunks != 0:
        # Fall back to non-chunked if not divisible
        num_chunks = 1

    # Debug: print chunk info
    if os.environ.get('FLUID_DEBUG_CHUNKS'):
        print(f"[dispatch_backward] num_chunks={num_chunks}, hidden_size={hidden_size}, total_recv={total_recv}")

    # Get streams
    if comm_stream is None and scheduler.is_enabled():
        comm_stream = scheduler.comm_stream
    default_stream = torch.cuda.current_stream()

    # =========================================================================
    # Non-chunked path (num_chunks == 1)
    # =========================================================================
    if num_chunks == 1:
        # Use pre-computed grad_all_tokens if provided, otherwise compute
        if grad_all_tokens is None:
            grad_all_tokens = torch.zeros(total_recv, hidden_size, dtype=dtype, device=device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = all_tokens_per_expert[exp_idx]
                if n_tok > 0:
                    grad_all_tokens[start:start + n_tok] = torch.matmul(
                        grad_all_fc1[start:start + n_tok],
                        weight1[exp_idx].t()
                    )
                    start += n_tok

        # Reorder expert-major -> rank-major
        grad_dispatched = _sort_chunks_by_idxs(
            grad_all_tokens, split_sizes_exp_major, sorted_idxs_exp_to_rank
        )

        # AllToAll with dW overlap
        if scheduler.is_enabled():
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_tokens = _all_to_all(
                    grad_dispatched.contiguous(),
                    output_split_sizes=input_splits_list,
                    input_split_sizes=output_splits_list,
                    group=ep_group
                )
                scheduler.record_alltoall_end(comm_stream)
            scheduler.on_alltoall_start(comm_type=f"moe_dispatch_L{layer_id}")
            default_stream.wait_stream(comm_stream)
        else:
            grad_tokens = _all_to_all(
                grad_dispatched.contiguous(),
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group
            )

        return grad_tokens

    # =========================================================================
    # Chunked path (num_chunks > 1)
    # Using alternating pattern (like Forward P2P) to ensure proper overlap
    # =========================================================================
    chunk_size = hidden_size // num_chunks

    # Reuse pre-created Event from scheduler to reduce overhead
    compute_event = scheduler.get_compute_sync_event()

    # Storage for AllToAll results
    output_chunks = []

    # Helper function to compute one chunk
    def compute_chunk(h_start, h_end):
        grad_chunk = torch.zeros(total_recv, chunk_size, dtype=dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                grad_chunk[start:start + n_tok] = torch.matmul(
                    grad_all_fc1[start:start + n_tok],
                    weight1[exp_idx, h_start:h_end, :].t()
                )
                start += n_tok
        return _sort_chunks_by_idxs(grad_chunk, split_sizes_exp_major, sorted_idxs_exp_to_rank)

    # Pre-compute first chunk (like Forward pre-computes first round)
    grad_chunk_reordered = compute_chunk(0, chunk_size)
    compute_event.record(default_stream)

    # Pipeline loop: submit AllToAll, then compute next chunk (alternating)
    for chunk_idx in range(num_chunks):
        # ============================================
        # Step 1: Submit current chunk's AllToAll (wait for current data)
        # ============================================
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(compute_event)  # Wait for current chunk

            output_chunk = _all_to_all(
                grad_chunk_reordered,
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group
            )
            output_chunks.append(output_chunk)

            # Record event after the last chunk's AllToAll
            if chunk_idx == num_chunks - 1:
                scheduler.record_alltoall_end(comm_stream)

        # ============================================
        # Step 2: Compute next chunk (parallel with current AllToAll)
        # ============================================
        if chunk_idx + 1 < num_chunks:
            next_h_start = (chunk_idx + 1) * chunk_size
            next_h_end = (chunk_idx + 2) * chunk_size
            grad_chunk_reordered = compute_chunk(next_h_start, next_h_end)
            compute_event.record(default_stream)  # Record AFTER wait, so no overwrite issue

    # ============================================
    # Step 3: Execute dW tasks while AllToAll is in progress
    # ============================================
    if scheduler.is_enabled():
        scheduler.on_alltoall_start(comm_type=f"moe_dispatch_chunked_L{layer_id}")

    # ============================================
    # Step 4: Wait for all AllToAll and concatenate
    # ============================================
    default_stream.wait_stream(comm_stream)

    # Concatenate chunks along hidden dimension
    grad_tokens = torch.cat(output_chunks, dim=-1)

    return grad_tokens


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

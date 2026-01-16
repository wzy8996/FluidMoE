"""
MoE Chunked Backward Pass with dX + AllToAll Overlap

This module implements chunked backward computation for MoE Dispatch layer,
enabling overlap between dX computation and AllToAll communication.

Key idea:
- Split dX computation along hidden dimension
- As each chunk completes, immediately submit AllToAll to comm_stream
- Compute next chunk while previous AllToAll is in progress

Timeline:
  default_stream: |dX_c0|--dX_c1--|--dX_c2--|--dX_c3--|wait|
                      ↓       ↓        ↓        ↓
  comm_stream:        |A2A_c0-|A2A_c1--|A2A_c2--|A2A_c3--|
                           overlap!
"""

import os
import torch
from typing import List, Optional

from fluid.core import _all_to_all, _sort_chunks_by_idxs
from fluid.core.scheduler import get_backward_scheduler


def backward_dispatch_chunked(
    grad_all_fc1: torch.Tensor,
    weight1: torch.Tensor,
    split_sizes_exp_major: List[int],
    sorted_idxs_exp_to_rank: List[int],
    tokens_per_expert_list: List[int],
    input_splits_list: List[int],
    output_splits_list: List[int],
    ep_group,
    num_chunks: int = 4,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    MoE Dispatch backward with hidden-dimension chunking for dX + AllToAll overlap.

    This function computes grad_tokens = grad_fc1 @ weight1.t() in chunks,
    where each chunk is immediately sent via AllToAll while the next chunk
    is being computed.

    Args:
        grad_all_fc1: Gradient w.r.t. FC1 output, shape [total_recv, ffn_hidden]
        weight1: FC1 weight, shape [num_experts, hidden_size, ffn_hidden]
        split_sizes_exp_major: Chunk sizes in expert-major order (for sort_chunks_by_idxs)
        sorted_idxs_exp_to_rank: Indices to reorder from expert-major to rank-major
        tokens_per_expert_list: Number of tokens per expert
        input_splits_list: AllToAll input split sizes
        output_splits_list: AllToAll output split sizes
        ep_group: Expert parallel process group
        num_chunks: Number of chunks to split hidden dimension
        comm_stream: CUDA stream for communication (uses scheduler's if None)

    Returns:
        grad_tokens: Gradient w.r.t. input tokens, shape [total_send, hidden_size]

    Timeline:
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

    # Validate chunk size
    if hidden_size % num_chunks != 0:
        # Fall back to non-chunked if not divisible
        num_chunks = 1

    # Debug: print chunk info (only once per run)
    if os.environ.get('FLUID_DEBUG_CHUNKS'):
        print(f"[backward_dispatch_chunked] num_chunks={num_chunks}, hidden_size={hidden_size}, total_recv={total_recv}")

    if num_chunks == 1:
        # Non-chunked path: compute full dX then AllToAll with dW overlap
        grad_all_tokens = torch.zeros(total_recv, hidden_size, dtype=dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = tokens_per_expert_list[exp_idx]
            if n_tok > 0:
                grad_all_tokens[start:start + n_tok] = torch.matmul(
                    grad_all_fc1[start:start + n_tok],
                    weight1[exp_idx].t()
                )
                start += n_tok

        # Reorder using sort_chunks_by_idxs
        grad_dispatched = _sort_chunks_by_idxs(
            grad_all_tokens, split_sizes_exp_major, sorted_idxs_exp_to_rank
        )

        # AllToAll with dW overlap
        if comm_stream is None:
            comm_stream = scheduler.comm_stream
        default_stream = torch.cuda.current_stream()

        if scheduler.is_enabled():
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_tokens = _all_to_all(
                    grad_dispatched,
                    output_split_sizes=input_splits_list,
                    input_split_sizes=output_splits_list,
                    group=ep_group
                )
                # Record AllToAll end event for incremental dW execution
                event = torch.cuda.Event()
                event.record(comm_stream)
                scheduler.set_alltoall_end_event(event)
            # Execute dW tasks incrementally (check if comm done after each dW)
            scheduler.on_alltoall_start(comm_type="moe_dispatch")
            default_stream.wait_stream(comm_stream)
            return grad_tokens
        else:
            return _all_to_all(
                grad_dispatched,
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group
            )

    # Chunked path
    chunk_size = hidden_size // num_chunks

    # Get streams
    if comm_stream is None:
        comm_stream = scheduler.comm_stream
    default_stream = torch.cuda.current_stream()

    # Convert to list (more efficient for _sort_chunks_by_idxs)
    if torch.is_tensor(split_sizes_exp_major):
        split_sizes_exp_major = split_sizes_exp_major.tolist()
    if torch.is_tensor(sorted_idxs_exp_to_rank):
        sorted_idxs_exp_to_rank = sorted_idxs_exp_to_rank.tolist()

    # Pre-create Events (avoid repeated creation in loop)
    compute_events = [torch.cuda.Event() for _ in range(num_chunks)]

    # Storage for AllToAll results
    output_chunks = []

    for chunk_idx in range(num_chunks):
        h_start = chunk_idx * chunk_size
        h_end = (chunk_idx + 1) * chunk_size

        # ============================================
        # Step 1: Compute dX for current chunk (on default_stream)
        # ============================================
        # grad_chunk: [total_recv, chunk_size]
        grad_chunk = torch.zeros(total_recv, chunk_size, dtype=dtype, device=device)

        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = tokens_per_expert_list[exp_idx]
            if n_tok > 0:
                # weight1[exp_idx, h_start:h_end, :].t(): [ffn_hidden, chunk_size]
                # matmul: [n_tok, ffn_hidden] @ [ffn_hidden, chunk_size] = [n_tok, chunk_size]
                grad_chunk[start:start + n_tok] = torch.matmul(
                    grad_all_fc1[start:start + n_tok],
                    weight1[exp_idx, h_start:h_end, :].t()
                )
                start += n_tok

        # ============================================
        # Step 2: Reorder expert-major → rank-major using sort_chunks_by_idxs
        # ============================================
        grad_chunk_reordered = _sort_chunks_by_idxs(
            grad_chunk, split_sizes_exp_major, sorted_idxs_exp_to_rank
        )

        # ============================================
        # Step 3: Submit AllToAll to comm_stream
        # ============================================
        # Record event to mark dX computation done
        compute_events[chunk_idx].record(default_stream)

        # Execute AllToAll on comm_stream
        with torch.cuda.stream(comm_stream):
            # Wait for current chunk computation to complete
            comm_stream.wait_event(compute_events[chunk_idx])

            # AllToAll for this chunk
            output_chunk = _all_to_all(
                grad_chunk_reordered,
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group
            )
            output_chunks.append(output_chunk)

            # Record event after the last chunk's AllToAll
            if chunk_idx == num_chunks - 1:
                event = torch.cuda.Event()
                event.record(comm_stream)
                scheduler.set_alltoall_end_event(event)

    # ============================================
    # Step 4: Execute dW tasks while AllToAll is in progress
    # ============================================
    # Execute dW tasks incrementally (check if comm done after each dW)
    if scheduler.is_enabled():
        scheduler.on_alltoall_start(comm_type="moe_dispatch_chunked")

    # ============================================
    # Step 5: Wait for all AllToAll and concatenate
    # ============================================
    default_stream.wait_stream(comm_stream)

    # Concatenate chunks along hidden dimension
    # Each chunk: [total_send, chunk_size] → [total_send, hidden_size]
    grad_tokens = torch.cat(output_chunks, dim=-1)

    return grad_tokens

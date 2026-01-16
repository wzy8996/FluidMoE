"""
Attention Chunked Backward Pass with dX + AllToAll Overlap

This module implements chunked backward computation for Attention output projection,
enabling overlap between dX computation and sp2hp AllToAll communication.

Key idea:
- Split dX computation along sequence dimension
- As each chunk completes, immediately submit sp2hp AllToAll to comm_stream
- Compute next chunk while previous AllToAll is in progress

Timeline:
  default_stream: |dX_c0|--dX_c1--|--dX_c2--|--dX_c3--|wait|
                      ↓       ↓        ↓        ↓
  comm_stream:        |A2A_c0-|A2A_c1--|A2A_c2--|A2A_c3--|
                           overlap!
"""

import os
import torch
from typing import Optional

from fluid.core import _all_to_all_sp2hp_forward
from fluid.core.scheduler import get_backward_scheduler


def backward_output_proj_chunked(
    grad_output: torch.Tensor,
    weight_proj: torch.Tensor,
    total_heads: int,
    head_dim: int,
    cp_group,
    num_chunks: int = 4,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    Attention output projection backward with seq-dimension chunking for dX + AllToAll overlap.

    This function computes grad_attn = grad_output @ weight_proj in chunks along seq dimension,
    where each chunk is immediately sent via sp2hp AllToAll while the next chunk is being computed.

    Args:
        grad_output: Gradient w.r.t. output, shape [seq_local, B, hidden]
        weight_proj: Projection weight, shape [hidden, total_heads * head_dim]
        total_heads: Total number of attention heads
        head_dim: Dimension per head
        cp_group: Context parallel process group
        num_chunks: Number of chunks to split seq dimension
        comm_stream: CUDA stream for communication

    Returns:
        grad_attn_output: Gradient w.r.t. attention output, shape [seq_full, B, heads_local, head_dim]

    Timeline:
        Chunk 0: compute dX[0:S/4] → submit sp2hp A2A
        Chunk 1: compute dX[S/4:S/2] → submit sp2hp A2A (while A2A_0 runs)
        ...
        Wait for all A2A to complete, concatenate results along seq dimension
    """
    scheduler = get_backward_scheduler()
    device = grad_output.device
    dtype = grad_output.dtype

    seq_local, batch_size, hidden_size = grad_output.shape
    cp_size = cp_group.size()
    heads_local = total_heads // cp_size

    # Validate chunk size
    if seq_local % num_chunks != 0:
        # Fall back to non-chunked if not divisible
        num_chunks = 1

    # Debug: print chunk info
    if os.environ.get('FLUID_DEBUG_CHUNKS'):
        print(f"[backward_output_proj_chunked] num_chunks={num_chunks}, seq_local={seq_local}")

    if num_chunks == 1:
        # Non-chunked path: compute full dX then AllToAll with dW overlap
        # dX: [seq_local, B, hidden] @ [hidden, total_heads * head_dim] → [seq_local, B, total_heads * head_dim]
        grad_attn_flat = torch.matmul(grad_output, weight_proj)

        # reshape: [seq_local, B, total_heads, head_dim]
        grad_attn_sp = grad_attn_flat.view(seq_local, batch_size, total_heads, head_dim)

        # sp2hp AllToAll with dW overlap
        if comm_stream is None:
            comm_stream = scheduler.comm_stream
        default_stream = torch.cuda.current_stream()

        if scheduler.is_enabled():
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_attn_output = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
                # Record AllToAll end event
                event = torch.cuda.Event()
                event.record(comm_stream)
                scheduler.set_alltoall_end_event(event)
            # Execute dW tasks incrementally
            scheduler.on_alltoall_start(comm_type="attn_sp2hp")
            default_stream.wait_stream(comm_stream)
            return grad_attn_output
        else:
            return _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)

    # Chunked path
    seq_chunk = seq_local // num_chunks

    # Get streams
    if comm_stream is None:
        comm_stream = scheduler.comm_stream
    default_stream = torch.cuda.current_stream()

    # Pre-create Events
    compute_events = [torch.cuda.Event() for _ in range(num_chunks)]

    # Storage for AllToAll results
    output_chunks = []

    for chunk_idx in range(num_chunks):
        s_start = chunk_idx * seq_chunk
        s_end = (chunk_idx + 1) * seq_chunk

        # ============================================
        # Step 1: Compute dX for current seq chunk (on default_stream)
        # ============================================
        # grad_chunk: [seq_chunk, B, hidden] @ [hidden, total_heads * head_dim]
        #           → [seq_chunk, B, total_heads * head_dim]
        grad_chunk = torch.matmul(grad_output[s_start:s_end], weight_proj)

        # reshape: [seq_chunk, B, total_heads, head_dim]
        grad_chunk = grad_chunk.view(seq_chunk, batch_size, total_heads, head_dim)

        # ============================================
        # Step 2: Submit sp2hp AllToAll to comm_stream
        # ============================================
        # Record event to mark dX computation done
        compute_events[chunk_idx].record(default_stream)

        # Execute AllToAll on comm_stream
        with torch.cuda.stream(comm_stream):
            # Wait for current chunk computation to complete
            comm_stream.wait_event(compute_events[chunk_idx])

            # sp2hp AllToAll for this chunk
            # [seq_chunk, B, total_heads, D] → [seq_chunk * cp_size, B, heads_local, D]
            output_chunk = _all_to_all_sp2hp_forward(grad_chunk, cp_group)
            output_chunks.append(output_chunk)

            # Record event after the last chunk's AllToAll
            if chunk_idx == num_chunks - 1:
                event = torch.cuda.Event()
                event.record(comm_stream)
                scheduler.set_alltoall_end_event(event)

    # ============================================
    # Step 3: Execute dW tasks while AllToAll is in progress
    # ============================================
    # Execute dW tasks incrementally
    if scheduler.is_enabled():
        scheduler.on_alltoall_start(comm_type="attn_sp2hp_chunked")

    # ============================================
    # Step 4: Wait for all AllToAll and concatenate
    # ============================================
    default_stream.wait_stream(comm_stream)

    # Concatenate chunks along seq dimension
    # Each chunk: [seq_chunk * cp_size, B, heads_local, D] → [seq_full, B, heads_local, D]
    grad_attn_output = torch.cat(output_chunks, dim=0)

    return grad_attn_output

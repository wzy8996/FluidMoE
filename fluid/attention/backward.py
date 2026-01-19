"""
Attention Backward Operations with AllToAll + dX/dW Scheduling

This module implements all backward operations for context-parallel attention:
1. Output projection backward: chunked dX + sp2hp AllToAll + dW overlap
2. Attention backward: recompute + chunked grad_Q/K/V + hp2sp AllToAll + dW overlap
3. QKV projection backward: dX + dW

Key design principle:
- Backward uses AllToAll (not P2P) for communication
- dX computation is chunked and overlapped with AllToAll communication
- dW tasks are registered and executed during AllToAll communication

Timeline (Output Projection Backward - chunked):
    default_stream: |dX_c0|--dX_c1--|--dX_c2--|--dX_c3--|wait|
                        ↓       ↓        ↓        ↓
    comm_stream:        |A2A_c0-|A2A_c1--|A2A_c2--|A2A_c3--|
                             overlap!
    After last chunk: register dW task -> execute during remaining A2A

Timeline (Attention Backward - chunked):
    default_stream: |recompute|grad_QKV_c0|--c1--|--c2--|--c3--|wait|
                                   ↓        ↓      ↓      ↓
    comm_stream:                   |A2A_c0--|A2A_c1|A2A_c2|A2A_c3|
                                        overlap!
    After last chunk: register dW tasks -> execute during remaining A2A
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple

from fluid.core import _all_to_all_sp2hp_forward, _all_to_all_hp2sp_forward
from fluid.core.scheduler import get_backward_scheduler


def output_projection_backward_chunked(
    grad_output: torch.Tensor,
    weight_proj: torch.Tensor,
    total_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    num_chunks: int = 4,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    Output projection backward with chunked dX + sp2hp AllToAll overlap.

    IMPORTANT: dX computation is chunked together with sp2hp AllToAll,
    NOT computed all at once before AllToAll.

    Args:
        grad_output: [seq_local, batch, hidden] - gradient w.r.t. output
        weight_proj: [hidden, total_heads * head_dim] - projection weight
        total_heads: total number of attention heads
        head_dim: dimension per head
        cp_group: context parallel process group
        num_chunks: number of chunks to split seq dimension
        comm_stream: CUDA stream for communication

    Returns:
        grad_attn_output: [seq_full, batch, heads_local, head_dim] - gradient in HP format

    Timeline:
        Chunk 0: compute dX[0:S/4] -> submit sp2hp A2A
        Chunk 1: compute dX[S/4:S/2] -> submit sp2hp A2A (while A2A_0 runs)
        ...
        After last chunk: register dW task -> execute during remaining A2A
        Wait for all A2A to complete, concatenate results
    """
    scheduler = get_backward_scheduler()
    device = grad_output.device
    dtype = grad_output.dtype

    seq_local, batch_size, hidden_size = grad_output.shape
    cp_size = cp_group.size()
    heads_local = total_heads // cp_size

    # Validate chunk size
    if seq_local % num_chunks != 0:
        num_chunks = 1

    # Debug output
    if os.environ.get('FLUID_DEBUG_CHUNKS'):
        print(f"[output_projection_backward_chunked] num_chunks={num_chunks}, seq_local={seq_local}")

    if num_chunks == 1:
        # Non-chunked path: compute full dX then AllToAll with dW overlap
        # dX: [seq_local, B, hidden] @ [hidden, total_heads * head_dim]
        #   -> [seq_local, B, total_heads * head_dim]
        grad_attn_flat = torch.matmul(grad_output, weight_proj)
        grad_attn_sp = grad_attn_flat.view(seq_local, batch_size, total_heads, head_dim)

        # sp2hp AllToAll with dW overlap
        if comm_stream is None:
            comm_stream = scheduler.comm_stream
        default_stream = torch.cuda.current_stream()

        if scheduler.is_enabled():
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_attn_output = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
                scheduler.record_alltoall_end(comm_stream)
            scheduler.on_alltoall_start(comm_type="attn_proj_sp2hp")
            default_stream.wait_stream(comm_stream)
            return grad_attn_output
        else:
            return _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)

    # ========================================================================
    # Chunked path: dX computation overlapped with sp2hp AllToAll
    # Using alternating pattern (like Forward P2P) to ensure proper overlap
    # ========================================================================
    seq_chunk = seq_local // num_chunks

    if comm_stream is None:
        comm_stream = scheduler.comm_stream
    default_stream = torch.cuda.current_stream()

    # Reuse pre-created Event from scheduler to reduce overhead
    compute_event = scheduler.get_compute_sync_event()

    # Storage for AllToAll results
    output_chunks = []

    # Pre-compute first chunk (like Forward pre-computes first round)
    s_start = 0
    s_end = seq_chunk
    grad_chunk = torch.matmul(grad_output[s_start:s_end], weight_proj)
    grad_chunk = grad_chunk.view(seq_chunk, batch_size, total_heads, head_dim)
    compute_event.record(default_stream)

    # Pipeline loop: submit AllToAll, then compute next chunk (alternating)
    for chunk_idx in range(num_chunks):
        # ============================================
        # Step 1: Submit current chunk's AllToAll (wait for current data)
        # ============================================
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(compute_event)  # Wait for current chunk's dX

            # sp2hp AllToAll: [seq_chunk, B, total_heads, D] -> [seq_chunk * cp_size, B, heads_local, D]
            output_chunk = _all_to_all_sp2hp_forward(grad_chunk, cp_group)
            output_chunks.append(output_chunk)

            # Record event after the last chunk's AllToAll
            if chunk_idx == num_chunks - 1:
                scheduler.record_alltoall_end(comm_stream)

        # ============================================
        # Step 2: Compute next chunk's dX (parallel with current AllToAll)
        # ============================================
        if chunk_idx + 1 < num_chunks:
            next_s_start = (chunk_idx + 1) * seq_chunk
            next_s_end = (chunk_idx + 2) * seq_chunk
            grad_chunk = torch.matmul(grad_output[next_s_start:next_s_end], weight_proj)
            grad_chunk = grad_chunk.view(seq_chunk, batch_size, total_heads, head_dim)
            compute_event.record(default_stream)  # Record AFTER wait, so no overwrite issue

    # ============================================
    # Step 3: Execute dW tasks while AllToAll is in progress
    # ============================================
    # dW task can now be executed since all dX chunks have been computed
    if scheduler.is_enabled():
        scheduler.on_alltoall_start(comm_type="attn_proj_sp2hp_chunked")

    # ============================================
    # Step 4: Wait for all AllToAll and concatenate
    # ============================================
    default_stream.wait_stream(comm_stream)

    # Concatenate chunks along seq dimension
    # Each chunk: [seq_chunk * cp_size, B, heads_local, D] -> [seq_full, B, heads_local, D]
    grad_attn_output = torch.cat(output_chunks, dim=0)

    return grad_attn_output


def attention_backward_chunked(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    cp_group: dist.ProcessGroup,
    num_chunks: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Attention backward with recomputation and chunked grad_Q/K/V + hp2sp AllToAll overlap.

    Memory-efficient: saves Q, K, V instead of attention matrix, recomputes attn_probs.

    Args:
        grad_output: [batch, heads_local, seq_full, head_dim] - gradient w.r.t. attention output
        query: [batch, heads_local, seq_full, head_dim] - saved Q
        key: [batch, heads_local, seq_full, head_dim] - saved K
        value: [batch, heads_local, seq_full, head_dim] - saved V
        scale: attention scale factor (1/sqrt(head_dim))
        cp_group: context parallel process group
        num_chunks: number of chunks for head_dim splitting

    Returns:
        grad_q: [seq_local, batch, heads_full, head_dim] - gradient in SP format
        grad_k: [seq_local, batch, heads_full, head_dim] - gradient in SP format
        grad_v: [seq_local, batch, heads_full, head_dim] - gradient in SP format

    Timeline:
        Step 1: Recompute attn_probs (full)
        Step 2: Compute grad_attn_scores (full)
        Step 3: For each head_dim chunk:
                - Compute grad_Q/K/V chunk
                - Submit hp2sp AllToAll
        Step 4: Execute dW tasks during AllToAll
        Step 5: Wait and concatenate
    """
    batch, heads_local, seq_full, head_dim = query.shape
    device = query.device
    dtype = query.dtype
    cp_size = cp_group.size()

    scheduler = get_backward_scheduler()

    # Validate chunk size
    if head_dim % num_chunks != 0:
        num_chunks = 1

    if os.environ.get('FLUID_DEBUG_CHUNKS'):
        print(f"[attention_backward_chunked] num_chunks={num_chunks}, head_dim={head_dim}")

    # ============================================
    # Step 1: Recompute attention (full)
    # ============================================
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Causal mask
    causal_mask = torch.triu(
        torch.ones(seq_full, seq_full, device=device, dtype=torch.bool),
        diagonal=1
    )
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    attn_probs = F.softmax(attn_scores, dim=-1)

    # ============================================
    # Step 2: Compute grad_attn_scores (full)
    # ============================================
    # grad_attn_probs = grad_output @ V.T
    grad_attn_probs = torch.matmul(grad_output, value.transpose(-2, -1))

    # Softmax backward: grad_attn_scores = attn_probs * (grad_attn_probs - sum)
    grad_attn_scores = attn_probs * (
        grad_attn_probs - (grad_attn_probs * attn_probs).sum(dim=-1, keepdim=True)
    )

    # ============================================
    # Step 3: Non-chunked path (single GPU or num_chunks=1)
    # ============================================
    if num_chunks == 1 or cp_size == 1:
        # Compute full gradients
        grad_q = torch.matmul(grad_attn_scores, key) * scale
        grad_k = torch.matmul(grad_attn_scores.transpose(-2, -1), query) * scale
        grad_v = torch.matmul(attn_probs.transpose(-2, -1), grad_output)

        # Reshape: [batch, heads_local, seq_full, head_dim] -> [seq_full, batch, heads_local, head_dim]
        grad_q = grad_q.permute(2, 0, 1, 3)
        grad_k = grad_k.permute(2, 0, 1, 3)
        grad_v = grad_v.permute(2, 0, 1, 3)

        if cp_size > 1:
            # hp2sp AllToAll with dW overlap
            grad_qkv_merged = torch.cat([grad_q, grad_k, grad_v], dim=-1)
            if scheduler.is_enabled():
                comm_stream = scheduler.comm_stream
                default_stream = torch.cuda.current_stream()
                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_stream(default_stream)
                    grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_merged, cp_group)
                    scheduler.record_alltoall_end(comm_stream)
                scheduler.on_alltoall_start(comm_type="attn_qkv_hp2sp")
                default_stream.wait_stream(comm_stream)
            else:
                grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_merged, cp_group)
            grad_q, grad_k, grad_v = torch.split(grad_qkv_sp, head_dim, dim=-1)

        return grad_q, grad_k, grad_v

    # ============================================
    # Step 4: Chunked path with AllToAll overlap
    # Using alternating pattern (like Forward P2P) to ensure proper overlap
    # ============================================
    chunk_size = head_dim // num_chunks

    comm_stream = scheduler.comm_stream if scheduler.is_enabled() else torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()

    # Reuse pre-created Event from scheduler to reduce overhead
    compute_event = scheduler.get_compute_sync_event()
    output_q_chunks = []
    output_k_chunks = []
    output_v_chunks = []

    # Helper function to compute one chunk
    def compute_chunk(d_start, d_end):
        grad_q_chunk = torch.matmul(grad_attn_scores, key[:, :, :, d_start:d_end]) * scale
        grad_k_chunk = torch.matmul(
            grad_attn_scores.transpose(-2, -1), query[:, :, :, d_start:d_end]
        ) * scale
        grad_v_chunk = torch.matmul(
            attn_probs.transpose(-2, -1), grad_output[:, :, :, d_start:d_end]
        )
        grad_q_chunk = grad_q_chunk.permute(2, 0, 1, 3).contiguous()
        grad_k_chunk = grad_k_chunk.permute(2, 0, 1, 3).contiguous()
        grad_v_chunk = grad_v_chunk.permute(2, 0, 1, 3).contiguous()
        return torch.cat([grad_q_chunk, grad_k_chunk, grad_v_chunk], dim=-1)

    # Pre-compute first chunk (like Forward pre-computes first round)
    grad_qkv_chunk = compute_chunk(0, chunk_size)
    compute_event.record(default_stream)

    # Pipeline loop: submit AllToAll, then compute next chunk (alternating)
    for chunk_idx in range(num_chunks):
        # ============================================
        # Step 4a: Submit current chunk's AllToAll (wait for current data)
        # ============================================
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(compute_event)  # Wait for current chunk

            # hp2sp AllToAll: [seq_full, batch, heads_local, 3*chunk_size]
            #              -> [seq_local, batch, heads_full, 3*chunk_size]
            grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_chunk, cp_group)

            # Split back to Q, K, V
            grad_q_sp, grad_k_sp, grad_v_sp = torch.split(grad_qkv_sp, chunk_size, dim=-1)
            output_q_chunks.append(grad_q_sp)
            output_k_chunks.append(grad_k_sp)
            output_v_chunks.append(grad_v_sp)

            # Record event after last chunk's AllToAll
            if chunk_idx == num_chunks - 1 and scheduler.is_enabled():
                scheduler.record_alltoall_end(comm_stream)

        # ============================================
        # Step 4b: Compute next chunk (parallel with current AllToAll)
        # ============================================
        if chunk_idx + 1 < num_chunks:
            next_d_start = (chunk_idx + 1) * chunk_size
            next_d_end = (chunk_idx + 2) * chunk_size
            grad_qkv_chunk = compute_chunk(next_d_start, next_d_end)
            compute_event.record(default_stream)  # Record AFTER wait, so no overwrite issue

    # ============================================
    # Step 5: Execute dW tasks while AllToAll in progress
    # ============================================
    if scheduler.is_enabled():
        scheduler.on_alltoall_start(comm_type="attn_qkv_hp2sp_chunked")

    # ============================================
    # Step 6: Wait for all AllToAll and concatenate
    # ============================================
    default_stream.wait_stream(comm_stream)

    # Concatenate chunks along head_dim dimension
    grad_q = torch.cat(output_q_chunks, dim=-1)
    grad_k = torch.cat(output_k_chunks, dim=-1)
    grad_v = torch.cat(output_v_chunks, dim=-1)

    return grad_q, grad_k, grad_v


def qkv_projection_backward(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    tokens: torch.Tensor,
    weight_qkv: torch.Tensor,
    cp_group: dist.ProcessGroup,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_id: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    QKV projection backward: compute grad_tokens (dX) and register dW task.

    Args:
        grad_q: [seq_local, batch, num_heads, head_dim] - gradient w.r.t. Q
        grad_k: [seq_local, batch, num_kv_heads, head_dim] - gradient w.r.t. K
        grad_v: [seq_local, batch, num_kv_heads, head_dim] - gradient w.r.t. V
        tokens: [seq_local, batch, hidden] - input tokens (saved from forward)
        weight_qkv: [total_proj, hidden] - QKV weight
        cp_group: context parallel process group
        num_heads: total Q heads
        num_kv_heads: total K/V heads
        head_dim: dimension per head
        layer_id: layer ID for dW task naming

    Returns:
        grad_tokens: [seq_local, batch, hidden] - gradient w.r.t. input tokens
        grad_weight: [total_proj, hidden] or None - gradient w.r.t. weight (None if scheduler enabled)
    """
    cp_size = cp_group.size()
    seq_local, batch, hidden_size = tokens.shape
    device = tokens.device
    dtype = tokens.dtype

    q_per_group = num_heads // num_kv_heads
    groups_per_rank = num_kv_heads // cp_size
    group_size = (q_per_group + 2) * head_dim

    scheduler = get_backward_scheduler()

    # ============================================
    # Step 1: Handle GQA - sum K/V gradients back to kv_heads
    # ============================================
    if q_per_group > 1:
        # grad_k/v come in expanded form [seq_local, batch, num_heads, head_dim]
        # Need to sum back to [seq_local, batch, num_kv_heads, head_dim]
        grad_k = grad_k.view(seq_local, batch, num_kv_heads, q_per_group, head_dim).sum(dim=3)
        grad_v = grad_v.view(seq_local, batch, num_kv_heads, q_per_group, head_dim).sum(dim=3)

    # ============================================
    # Step 2: Reassemble to interleaved grad_qkv format
    # ============================================
    # grad_q: [seq_local, batch, num_heads, head_dim]
    # -> [seq_local, batch, num_kv_heads, q_per_group * head_dim]
    grad_q_grouped = grad_q.view(seq_local, batch, num_kv_heads, q_per_group * head_dim)

    # Concatenate [Q_group, K, V] for each group
    grad_qkv = torch.cat([grad_q_grouped, grad_k, grad_v], dim=-1)
    # Shape: [seq_local, batch, num_kv_heads, group_size]
    grad_qkv = grad_qkv.view(seq_local, batch, -1)
    # Shape: [seq_local, batch, total_proj]

    # ============================================
    # Step 3: Compute dX for QKV projection
    # ============================================
    # Direct matrix multiplication (same as Baseline)
    # grad_qkv: [seq_local, batch, total_proj] @ weight_qkv: [total_proj, hidden]
    # -> grad_tokens: [seq_local, batch, hidden]
    grad_tokens = torch.matmul(grad_qkv, weight_qkv)

    # ============================================
    # Step 4: Register dW task (or compute directly)
    # ============================================
    if scheduler.is_enabled():
        tokens_saved = tokens.detach()
        grad_qkv_saved = grad_qkv.view(-1, grad_qkv.shape[-1]).detach()
        weight_qkv_saved = weight_qkv

        def compute_dw_qkv():
            # Direct matrix multiplication (same as Baseline)
            # grad_qkv.T @ tokens -> [total_proj, hidden]
            tokens_flat = tokens_saved.view(-1, hidden_size)
            return torch.matmul(grad_qkv_saved.t(), tokens_flat)

        scheduler.register_dw_task(
            layer_name=f"qkv_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_qkv,
            priority=100,
            weight_param=weight_qkv_saved,
        )
        grad_weight = None
    else:
        # Direct matrix multiplication (same as Baseline)
        # grad_qkv.T @ tokens -> [total_proj, hidden]
        tokens_flat = tokens.view(-1, hidden_size)
        grad_qkv_flat = grad_qkv.view(-1, grad_qkv.shape[-1])
        grad_weight = torch.matmul(grad_qkv_flat.t(), tokens_flat)

    return grad_tokens, grad_weight


def output_projection_register_dw(
    grad_output: torch.Tensor,
    attn_input_full: torch.Tensor,
    weight_proj: torch.Tensor,
    layer_id: int = 0,
) -> Optional[torch.Tensor]:
    """
    Register dW task for output projection.

    This should be called after output_projection_backward_chunked has computed dX.
    The dW task will be executed during AllToAll communication.

    Args:
        grad_output: [seq_local, batch, hidden] - gradient w.r.t. output
        attn_input_full: [seq_local, batch, all_heads * head_dim] - saved from forward
        weight_proj: [hidden, total_heads * head_dim] - projection weight
        layer_id: layer ID for dW task naming

    Returns:
        grad_weight: [hidden, total_heads * head_dim] or None if scheduler enabled
    """
    scheduler = get_backward_scheduler()
    seq_local, batch_size, hidden_size = grad_output.shape

    if scheduler.is_enabled():
        attn_full_flat_saved = attn_input_full.reshape(seq_local * batch_size, -1).detach()
        grad_output_flat_saved = grad_output.reshape(seq_local * batch_size, hidden_size).detach()
        weight_proj_saved = weight_proj

        def compute_dw_proj():
            # dW = grad_output.T @ attn_input_full
            grad_weight = torch.matmul(grad_output_flat_saved.t(), attn_full_flat_saved)
            return grad_weight

        scheduler.register_dw_task(
            layer_name=f"output_proj_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_proj,
            priority=99,
            weight_param=weight_proj_saved,
        )
        return None
    else:
        # Compute dW directly
        attn_full_flat = attn_input_full.view(seq_local * batch_size, -1)
        grad_output_flat = grad_output.reshape(seq_local * batch_size, hidden_size)
        grad_weight = torch.matmul(grad_output_flat.t(), attn_full_flat)
        return grad_weight


__all__ = [
    'output_projection_backward_chunked',
    'attention_backward_chunked',
    'qkv_projection_backward',
    'output_projection_register_dw',
]

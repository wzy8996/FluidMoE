"""
Attention Backward Operations with AllToAll + dX/dW Scheduling

This module implements all backward operations for context-parallel attention:
1. Output projection backward: chunked dX + sp2hp AllToAll + attention recompute overlap
2. Attention backward: chunked grad_Q/K/V + hp2sp AllToAll + dW overlap
3. QKV projection backward: dX + dW

Key design principle:
- Backward uses AllToAll (not P2P) for communication
- dX computation is chunked and overlapped with AllToAll communication
- Attention recomputation is overlapped with output projection AllToAll
- dW tasks are registered and executed during AllToAll communication

Timeline (Output Projection Backward with Attention Recompute):
    default_stream: |dX_c0|--dX_c1--|--dX_c2--|--dX_c3--|attn_recompute|dW|wait|
                        ↓       ↓        ↓        ↓
    comm_stream:        |A2A_c0-|A2A_c1--|A2A_c2--|A2A_c3-----------------|
                                        overlap!

Timeline (Attention Backward - uses precomputed attn_probs/grad_attn_scores):
    default_stream: |grad_QKV_c0|--c1--|--c2--|--c3--|dW|wait|
                          ↓        ↓      ↓      ↓
    comm_stream:          |A2A_c0--|A2A_c1|A2A_c2|A2A_c3----|
                                        overlap!
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
    # Attention recompute params (optional - for overlap with AllToAll)
    query: Optional[torch.Tensor] = None,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Output projection backward with chunked dX + sp2hp AllToAll + attention recompute overlap.

    IMPORTANT: dX computation is chunked together with sp2hp AllToAll,
    and attention recomputation is done during AllToAll communication.

    Args:
        grad_output: [seq_local, batch, hidden] - gradient w.r.t. output
        weight_proj: [hidden, total_heads * head_dim] - projection weight
        total_heads: total number of attention heads
        head_dim: dimension per head
        cp_group: context parallel process group
        num_chunks: number of chunks to split seq dimension
        comm_stream: CUDA stream for communication
        query: [batch, heads_local, seq_full, head_dim] - Q for attention recompute (optional)
        key: [batch, heads_local, seq_full, head_dim] - K for attention recompute (optional)
        value: [batch, heads_local, seq_full, head_dim] - V for attention recompute (optional)
        scale: attention scale factor (optional)

    Returns:
        grad_attn_output: [seq_full, batch, heads_local, head_dim] - gradient in HP format
        attn_probs: [batch, heads_local, seq_full, seq_full] - recomputed attention probs (or None)
        grad_attn_scores: [batch, heads_local, seq_full, seq_full] - gradient of attn scores (or None)

    Timeline:
        Chunk 0-N: compute dX chunks -> submit sp2hp A2A
        During A2A: attention recompute + dW tasks
        Wait for all A2A to complete, concatenate results
    """
    scheduler = get_backward_scheduler()
    device = grad_output.device
    dtype = grad_output.dtype

    seq_local, batch_size, hidden_size = grad_output.shape
    cp_size = cp_group.size()
    heads_local = total_heads // cp_size

    # Check if attention recompute is requested
    do_attn_recompute = query is not None and key is not None and value is not None and scale is not None

    # Validate chunk size
    if seq_local % num_chunks != 0:
        num_chunks = 1

    # Debug output
    if os.environ.get('FLUID_DEBUG_CHUNKS'):
        print(f"[output_projection_backward_chunked] num_chunks={num_chunks}, seq_local={seq_local}, attn_recompute={do_attn_recompute}")

    # Helper function for attention recompute (overlaps with AllToAll)
    def recompute_attention():
        """Recompute attn_probs and grad_attn_scores during AllToAll."""
        if not do_attn_recompute:
            return None, None

        seq_full = query.shape[2]

        # Recompute attention scores and probs
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_full, seq_full, device=device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)

        return attn_probs, None  # grad_attn_scores computed later with grad_output

    if num_chunks == 1:
        # Non-chunked path: compute full dX then AllToAll via comm thread
        grad_attn_flat = torch.matmul(grad_output, weight_proj)
        grad_attn_sp = grad_attn_flat.view(seq_local, batch_size, total_heads, head_dim)

        attn_probs = None
        if scheduler.is_enabled():
            result_holder = [None]
            def do_alltoall():
                result_holder[0] = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
                return result_holder[0]

            task_id = scheduler.submit_alltoall(do_alltoall)

            # Attention recompute during AllToAll (overlaps with communication)
            attn_probs, _ = recompute_attention()

            # Execute dW tasks while AllToAll is running
            scheduler.execute_dw_tasks()

            # Wait for AllToAll
            scheduler.wait_alltoall(task_id)
            return result_holder[0], attn_probs, None
        else:
            grad_attn_output = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
            attn_probs, _ = recompute_attention()
            return grad_attn_output, attn_probs, None

    # ========================================================================
    # Chunked path: dX computation overlapped with sp2hp AllToAll via comm thread
    # ========================================================================
    seq_chunk = seq_local // num_chunks
    output_chunks = [None] * num_chunks
    task_ids = []

    scheduler_enabled = scheduler.is_enabled()

    # Pipeline loop: compute dX, submit AllToAll (overlaps with next dX)
    for chunk_idx in range(num_chunks):
        s_start = chunk_idx * seq_chunk
        s_end = s_start + seq_chunk

        grad_chunk = torch.matmul(grad_output[s_start:s_end], weight_proj)
        grad_chunk = grad_chunk.view(seq_chunk, batch_size, total_heads, head_dim)

        if scheduler_enabled:
            _chunk_idx = chunk_idx
            _grad_chunk = grad_chunk.contiguous()

            def make_alltoall_fn(idx, data):
                def do_alltoall():
                    result = _all_to_all_sp2hp_forward(data, cp_group)
                    output_chunks[idx] = result
                    return result
                return do_alltoall

            task_id = scheduler.submit_alltoall(make_alltoall_fn(_chunk_idx, _grad_chunk))
            task_ids.append(task_id)
        else:
            output_chunks[chunk_idx] = _all_to_all_sp2hp_forward(grad_chunk, cp_group)

    # ============================================
    # During AllToAll: attention recompute + dW tasks
    # ============================================
    attn_probs = None
    if scheduler_enabled:
        # Attention recompute during AllToAll (overlaps with communication)
        attn_probs, _ = recompute_attention()

        # Execute dW tasks
        scheduler.execute_dw_tasks()

        # Wait for the last AllToAll only
        if task_ids:
            scheduler.wait_alltoall(task_ids[-1], num_tasks=len(task_ids))
    else:
        attn_probs, _ = recompute_attention()

    # Concatenate chunks
    grad_attn_output = torch.cat(output_chunks, dim=0)

    return grad_attn_output, attn_probs, None


def attention_backward_chunked(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    cp_group: dist.ProcessGroup,
    num_chunks: int = 4,
    attn_probs_precomputed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Attention backward with chunked grad_Q/K/V + hp2sp AllToAll overlap.

    If attn_probs_precomputed is provided (from output_projection_backward),
    skips the recomputation step for better overlap.

    Args:
        grad_output: [batch, heads_local, seq_full, head_dim] - gradient w.r.t. attention output
        query: [batch, heads_local, seq_full, head_dim] - saved Q
        key: [batch, heads_local, seq_full, head_dim] - saved K
        value: [batch, heads_local, seq_full, head_dim] - saved V
        scale: attention scale factor (1/sqrt(head_dim))
        cp_group: context parallel process group
        num_chunks: number of chunks for head_dim splitting
        attn_probs_precomputed: [batch, heads_local, seq_full, seq_full] - precomputed attn probs (optional)

    Returns:
        grad_q: [seq_local, batch, heads_full, head_dim] - gradient in SP format
        grad_k: [seq_local, batch, heads_full, head_dim] - gradient in SP format
        grad_v: [seq_local, batch, heads_full, head_dim] - gradient in SP format

    Timeline (with precomputed attn_probs):
        Step 1: Compute grad_attn_scores (using precomputed attn_probs)
        Step 2: For each seq chunk: compute grad_Q/K/V -> submit hp2sp AllToAll
        Step 3: Execute dW tasks during AllToAll
        Step 4: Wait and concatenate
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
        print(f"[attention_backward_chunked] num_chunks={num_chunks}, head_dim={head_dim}, precomputed={attn_probs_precomputed is not None}")

    # ============================================
    # Step 1: Use precomputed or recompute attention
    # ============================================
    if attn_probs_precomputed is not None:
        # Use precomputed attn_probs (already computed during output_projection AllToAll)
        attn_probs = attn_probs_precomputed
    else:
        # Recompute attention (fallback if not precomputed)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
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
            # hp2sp AllToAll via comm thread with dW overlap
            grad_qkv_merged = torch.cat([grad_q, grad_k, grad_v], dim=-1)
            if scheduler.is_enabled():
                result_holder = [None]
                def do_alltoall():
                    result_holder[0] = _all_to_all_hp2sp_forward(grad_qkv_merged, cp_group)
                    return result_holder[0]

                task_id = scheduler.submit_alltoall(do_alltoall)
                scheduler.execute_dw_tasks()
                scheduler.wait_alltoall(task_id)
                grad_qkv_sp = result_holder[0]
            else:
                grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_merged, cp_group)
            grad_q, grad_k, grad_v = torch.split(grad_qkv_sp, head_dim, dim=-1)

        return grad_q, grad_k, grad_v

    # ============================================
    # Step 4: Chunked path with AllToAll overlap
    # Chunk by SEQ dimension (not head_dim) for efficient matmul
    # ============================================
    # Validate: seq_full must be divisible by num_chunks
    if seq_full % num_chunks != 0:
        num_chunks = 1
        # Fall back to non-chunked path
        grad_q = torch.matmul(grad_attn_scores, key) * scale
        grad_k = torch.matmul(grad_attn_scores.transpose(-2, -1), query) * scale
        grad_v = torch.matmul(attn_probs.transpose(-2, -1), grad_output)
        grad_q = grad_q.permute(2, 0, 1, 3)
        grad_k = grad_k.permute(2, 0, 1, 3)
        grad_v = grad_v.permute(2, 0, 1, 3)
        grad_qkv_merged = torch.cat([grad_q, grad_k, grad_v], dim=-1)
        grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_merged, cp_group)
        grad_q, grad_k, grad_v = torch.split(grad_qkv_sp, head_dim, dim=-1)
        return grad_q, grad_k, grad_v

    seq_chunk_size = seq_full // num_chunks

    scheduler_enabled = scheduler.is_enabled()

    # Storage for AllToAll results and task IDs
    output_chunks = [None] * num_chunks
    task_ids = []

    # Helper function to compute one seq chunk (efficient: full key/query/grad_output used)
    def compute_seq_chunk(s_start, s_end):
        # grad_q: [B, H, chunk, head_dim] - uses full key
        grad_q_chunk = torch.matmul(grad_attn_scores[:, :, s_start:s_end, :], key) * scale
        # grad_k: [B, H, chunk, head_dim] - uses grad_attn_scores columns and full query
        grad_k_chunk = torch.matmul(
            grad_attn_scores[:, :, :, s_start:s_end].transpose(-2, -1), query
        ) * scale
        # grad_v: [B, H, chunk, head_dim] - uses attn_probs columns and full grad_output
        grad_v_chunk = torch.matmul(
            attn_probs[:, :, :, s_start:s_end].transpose(-2, -1), grad_output
        )
        # Permute: [B, H, chunk, D] -> [chunk, B, H, D]
        grad_q_chunk = grad_q_chunk.permute(2, 0, 1, 3).contiguous()
        grad_k_chunk = grad_k_chunk.permute(2, 0, 1, 3).contiguous()
        grad_v_chunk = grad_v_chunk.permute(2, 0, 1, 3).contiguous()
        # Merge Q, K, V for AllToAll: [chunk, B, H, 3*D]
        return torch.cat([grad_q_chunk, grad_k_chunk, grad_v_chunk], dim=-1)

    # Pipeline loop: compute grad_qkv chunk, submit AllToAll (overlaps with next chunk)
    for chunk_idx in range(num_chunks):
        s_start = chunk_idx * seq_chunk_size
        s_end = s_start + seq_chunk_size

        # ============================================
        # Step 4a: Compute grad_qkv chunk
        # ============================================
        grad_qkv_chunk = compute_seq_chunk(s_start, s_end)

        # ============================================
        # Step 4b: Submit AllToAll to comm thread (overlaps with next chunk)
        # ============================================
        if scheduler_enabled:
            _chunk_idx = chunk_idx
            _grad_qkv_chunk = grad_qkv_chunk.contiguous()

            def make_alltoall_fn(idx, data):
                def do_alltoall():
                    result = _all_to_all_hp2sp_forward(data, cp_group)
                    output_chunks[idx] = result
                    return result
                return do_alltoall

            task_id = scheduler.submit_alltoall(make_alltoall_fn(_chunk_idx, _grad_qkv_chunk))
            task_ids.append(task_id)
        else:
            output_chunks[chunk_idx] = _all_to_all_hp2sp_forward(grad_qkv_chunk, cp_group)

    # ============================================
    # Step 5: Execute dW tasks while AllToAll in progress
    # ============================================
    if scheduler_enabled:
        scheduler.execute_dw_tasks()

        # Wait for the last AllToAll only (FIFO guarantees earlier ones are done)
        # Pass num_tasks to correctly decrement the in_progress counter
        if task_ids:
            scheduler.wait_alltoall(task_ids[-1], num_tasks=len(task_ids))

    # Concatenate chunks along seq dimension (dim=0)
    grad_qkv_full = torch.cat(output_chunks, dim=0)
    # Split back to Q, K, V along head_dim
    grad_q, grad_k, grad_v = torch.split(grad_qkv_full, head_dim, dim=-1)

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

    # Debug output
    if os.environ.get('FLUID_DEBUG_SCHEDULER_REGISTER', '0') == '1':
        print(f"[DEBUG] qkv_projection_backward(L{layer_id}): scheduler.is_enabled()={scheduler.is_enabled()}", flush=True)

    # ============================================
    # Step 1: Handle GQA - sum K/V gradients back to kv_heads
    # ============================================
    if q_per_group > 1:
        # grad_k/v come in expanded form [seq_local, batch, num_heads, head_dim]
        # Need to sum back to [seq_local, batch, num_kv_heads, head_dim]
        grad_k = grad_k.reshape(seq_local, batch, num_kv_heads, q_per_group, head_dim).sum(dim=3)
        grad_v = grad_v.reshape(seq_local, batch, num_kv_heads, q_per_group, head_dim).sum(dim=3)

    # ============================================
    # Step 2: Reassemble to interleaved grad_qkv format
    # ============================================
    # grad_q: [seq_local, batch, num_heads, head_dim]
    # -> [seq_local, batch, num_kv_heads, q_per_group * head_dim]
    grad_q_grouped = grad_q.reshape(seq_local, batch, num_kv_heads, q_per_group * head_dim)

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

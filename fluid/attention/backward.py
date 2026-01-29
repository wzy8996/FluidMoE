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


def outproj_sp2hp_backward(
    grad_output: torch.Tensor,
    weight_proj: torch.Tensor,
    total_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    num_chunks: int = 4,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    Output projection backward with chunked dX + sp2hp AllToAll.

    SDPA backward handles attention recomputation internally, so this function
    only computes output projection dX and sp2hp AllToAll.

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
        Chunk 0-N: compute dX chunks -> submit sp2hp A2A
        During A2A: dW tasks
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
        print(f"[outproj_sp2hp_backward] num_chunks={num_chunks}, seq_local={seq_local}")

    if num_chunks == 1:
        # Non-chunked path: compute full dX then AllToAll via comm thread
        grad_attn_flat = torch.matmul(grad_output, weight_proj)
        grad_attn_sp = grad_attn_flat.view(seq_local, batch_size, total_heads, head_dim)

        if scheduler.is_enabled():
            result_holder = [None]
            def do_alltoall():
                result_holder[0] = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
                return result_holder[0]

            task_id = scheduler.submit_alltoall(do_alltoall)

            # Execute dW tasks while AllToAll is running
            scheduler.execute_dw_tasks()

            # Wait for AllToAll
            scheduler.wait_alltoall(task_id)
            return result_holder[0]
        else:
            grad_attn_output = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
            return grad_attn_output

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
    # During AllToAll: dW tasks
    # ============================================
    if scheduler_enabled:
        # Execute dW tasks
        scheduler.execute_dw_tasks()

        # Wait for the last AllToAll only
        if task_ids:
            scheduler.wait_alltoall(task_ids[-1], num_tasks=len(task_ids))

    # Concatenate chunks
    grad_attn_output = torch.cat(output_chunks, dim=0)

    return grad_attn_output



# =============================================================================
# Attention Score Backward (replaceable with FlashAttention)
# =============================================================================

def attention_score_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute grad_Q, grad_K, grad_V via PyTorch native SDPA backward.

    Uses torch.autograd.grad through F.scaled_dot_product_attention,
    which automatically selects the best kernel (FlashAttention, etc.).
    No manual recomputation needed - SDPA handles internally.

    Args:
        grad_output: [batch, heads_local, seq_full, head_dim]
        query: [batch, heads_local, seq_full, head_dim]
        key: [batch, heads_local, seq_full, head_dim]
        value: [batch, heads_local, seq_full, head_dim]
        scale: attention scale factor (1/sqrt(head_dim))

    Returns:
        grad_q: [batch, heads_local, seq_full, head_dim]
        grad_k: [batch, heads_local, seq_full, head_dim]
        grad_v: [batch, heads_local, seq_full, head_dim]
    """
    with torch.enable_grad():
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )

        grad_q, grad_k, grad_v = torch.autograd.grad(
            attn_out, (q, k, v), grad_output, retain_graph=False
        )

    return grad_q, grad_k, grad_v


# =============================================================================
# Region 4: hp2sp AllToAll → QKV dX (Communication-First Pipeline)
# =============================================================================

def hp2sp_qkv_backward(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    tokens: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_id: int = 0,
    num_chunks: int = 4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Region 4: hp2sp AllToAll + QKV dX (communication-first pipeline).

    Takes grad_Q/K/V from attention_score_backward and performs:
      1. GQA contraction + QKV reassembly in HP format
      2. Submit hp2sp AllToAll chunks (chunked along seq dimension)
      3. As each chunk arrives (SP format): compute QKV dX for that chunk
      4. dW tasks overlap with AllToAll

    comm_thread: |A2A_0|A2A_1|A2A_2|A2A_3|
    default:     |gqa+merge|dW|wait0+dx0|wait1+dx1|...|

    Args:
        grad_q: [batch, heads_local, seq_full, head_dim] gradient w.r.t. Q
        grad_k: [batch, heads_local, seq_full, head_dim] gradient w.r.t. K
        grad_v: [batch, heads_local, seq_full, head_dim] gradient w.r.t. V
        cp_group: context parallel process group
        tokens: [seq_local, batch, hidden] input tokens (for dW registration)
        weight_qkv: [total_proj, hidden] QKV weight
        num_heads: total Q heads
        num_kv_heads: total K/V heads
        head_dim: dimension per head
        layer_id: layer ID for dW task naming
        num_chunks: number of chunks for seq dimension

    Returns:
        grad_tokens: [seq_local, batch, hidden] gradient w.r.t. input tokens
        grad_weight: [total_proj, hidden] or None if scheduler enabled
    """
    batch, heads_local, seq_full, _ = grad_q.shape
    device = grad_q.device
    dtype = grad_q.dtype
    cp_size = cp_group.size()
    seq_local = seq_full // cp_size

    scheduler = get_backward_scheduler()
    q_per_group = num_heads // num_kv_heads
    groups_per_rank = num_kv_heads // cp_size
    _, batch_t, hidden_size = tokens.shape

    if cp_size == 1:
        # No AllToAll needed, just do QKV projection backward directly
        grad_q_sp = grad_q.permute(2, 0, 1, 3)  # [seq, batch, heads, head_dim]
        grad_k_sp = grad_k.permute(2, 0, 1, 3)
        grad_v_sp = grad_v.permute(2, 0, 1, 3)
        return _qkv_dx_and_dw(
            grad_q_sp, grad_k_sp, grad_v_sp, tokens, weight_qkv,
            num_heads, num_kv_heads, head_dim, layer_id, cp_size
        )

    # ============================================
    # Step 1: GQA contraction + QKV reassembly in HP format
    # ============================================
    # Permute to [seq_full, batch, heads_local, head_dim]
    grad_q_hp = grad_q.permute(2, 0, 1, 3)  # [seq_full, batch, heads_local, head_dim]
    grad_k_hp = grad_k.permute(2, 0, 1, 3)
    grad_v_hp = grad_v.permute(2, 0, 1, 3)

    # GQA contraction for K/V: sum expanded heads back to kv_heads
    if q_per_group > 1:
        # grad_k_hp: [seq_full, batch, heads_local, head_dim]
        # heads_local = num_heads // cp_size, need to sum to num_kv_heads // cp_size
        kv_heads_local = num_kv_heads // cp_size
        grad_k_hp = grad_k_hp.reshape(seq_full, batch, kv_heads_local, q_per_group, head_dim).sum(dim=3)
        grad_v_hp = grad_v_hp.reshape(seq_full, batch, kv_heads_local, q_per_group, head_dim).sum(dim=3)
        # Reshape Q to grouped form
        grad_q_hp_grouped = grad_q_hp.reshape(seq_full, batch, kv_heads_local, q_per_group * head_dim)
    else:
        kv_heads_local = heads_local
        grad_q_hp_grouped = grad_q_hp.reshape(seq_full, batch, kv_heads_local, head_dim)

    # Merge Q/K/V per group: [seq_full, batch, kv_heads_local, group_size]
    grad_qkv_hp = torch.cat([grad_q_hp_grouped, grad_k_hp, grad_v_hp], dim=-1)
    # shape: [seq_full, batch, kv_heads_local, (q_per_group + 2) * head_dim]

    if not scheduler.is_enabled() or num_chunks <= 1:
        # ---- Non-chunked fallback ----
        # hp2sp AllToAll
        if scheduler.is_enabled():
            result_holder = [None]
            _grad_qkv_hp = grad_qkv_hp.contiguous()
            def do_alltoall():
                result_holder[0] = _all_to_all_hp2sp_forward(_grad_qkv_hp, cp_group)
                return result_holder[0]
            task_id = scheduler.submit_alltoall(do_alltoall)
            scheduler.execute_dw_tasks()
            scheduler.wait_alltoall(task_id)
            grad_qkv_sp = result_holder[0]
        else:
            grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_hp.contiguous(), cp_group)

        # grad_qkv_sp: [seq_local, batch, num_kv_heads, (q_per_group+2)*head_dim]
        # Flatten to [seq_local, batch, total_proj]
        grad_qkv_flat = grad_qkv_sp.view(seq_local, batch, -1)
        grad_tokens = torch.matmul(grad_qkv_flat, weight_qkv)

        # Register dW
        grad_weight = _register_qkv_dw(
            grad_qkv_flat, tokens, weight_qkv, hidden_size, layer_id
        )

        return grad_tokens, grad_weight

    # ============================================
    # Step 3: Communication-first pipeline
    # Submit all hp2sp AllToAll chunks, then compute QKV dX as each arrives
    # ============================================
    # Validate chunk size (chunk along seq_full)
    if seq_full % num_chunks != 0:
        num_chunks = 1
        # Fallback
        grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_hp.contiguous(), cp_group)
        grad_qkv_flat = grad_qkv_sp.view(seq_local, batch, -1)
        grad_tokens = torch.matmul(grad_qkv_flat, weight_qkv)
        grad_weight = _register_qkv_dw(grad_qkv_flat, tokens, weight_qkv, hidden_size, layer_id)
        return grad_tokens, grad_weight

    seq_chunk_full = seq_full // num_chunks
    seq_chunk_local = seq_local // num_chunks

    task_ids = []
    chunk_results = [None] * num_chunks
    grad_qkv_hp_contig = grad_qkv_hp.contiguous()

    # Submit all AllToAll chunks
    for chunk_idx in range(num_chunks):
        s_start = chunk_idx * seq_chunk_full
        s_end = s_start + seq_chunk_full

        _chunk_idx = chunk_idx
        _input_chunk = grad_qkv_hp_contig[s_start:s_end].contiguous()

        def make_alltoall_fn(idx, data):
            def do_alltoall():
                result = _all_to_all_hp2sp_forward(data, cp_group)
                chunk_results[idx] = result
                return result
            return do_alltoall

        task_id = scheduler.submit_alltoall(make_alltoall_fn(_chunk_idx, _input_chunk))
        task_ids.append(task_id)

    # Execute dW tasks during AllToAll
    scheduler.execute_dw_tasks()

    # Process each chunk as it arrives: QKV dX
    # After hp2sp AllToAll: [seq_chunk_local, batch, num_kv_heads, group_size]
    # total_proj = num_kv_heads * group_size (NOT kv_heads_local * group_size)
    group_size = grad_qkv_hp.shape[3]
    total_proj = num_kv_heads * group_size
    grad_tokens = torch.empty(seq_local, batch, hidden_size, dtype=tokens.dtype, device=device)

    # Accumulate grad_qkv_flat for dW
    grad_qkv_flat_all = torch.empty(seq_local, batch, total_proj, dtype=dtype, device=device)

    for chunk_idx in range(num_chunks):
        s_start_local = chunk_idx * seq_chunk_local
        s_end_local = s_start_local + seq_chunk_local

        # Wait for this chunk
        scheduler.wait_alltoall(task_ids[chunk_idx])
        grad_qkv_sp_chunk = chunk_results[chunk_idx]
        # [seq_chunk_local, batch, num_kv_heads, group_size]

        # Flatten and compute dX
        grad_qkv_chunk_flat = grad_qkv_sp_chunk.reshape(seq_chunk_local, batch, -1)
        grad_tokens[s_start_local:s_end_local] = torch.matmul(grad_qkv_chunk_flat, weight_qkv)
        grad_qkv_flat_all[s_start_local:s_end_local] = grad_qkv_chunk_flat

    # Register dW
    grad_weight = _register_qkv_dw(
        grad_qkv_flat_all, tokens, weight_qkv, hidden_size, layer_id
    )

    return grad_tokens, grad_weight


def _qkv_dx_and_dw(
    grad_q, grad_k, grad_v, tokens, weight_qkv,
    num_heads, num_kv_heads, head_dim, layer_id, cp_size,
):
    """Helper: GQA contraction + QKV reassembly + dX + dW for single-GPU case."""
    seq_local, batch, hidden_size = tokens.shape
    q_per_group = num_heads // num_kv_heads

    if q_per_group > 1:
        grad_k = grad_k.reshape(seq_local, batch, num_kv_heads, q_per_group, head_dim).sum(dim=3)
        grad_v = grad_v.reshape(seq_local, batch, num_kv_heads, q_per_group, head_dim).sum(dim=3)

    grad_q_grouped = grad_q.reshape(seq_local, batch, num_kv_heads, q_per_group * head_dim)
    grad_qkv = torch.cat([grad_q_grouped, grad_k, grad_v], dim=-1)
    grad_qkv = grad_qkv.view(seq_local, batch, -1)

    grad_tokens = torch.matmul(grad_qkv, weight_qkv)
    grad_weight = _register_qkv_dw(grad_qkv, tokens, weight_qkv, hidden_size, layer_id)
    return grad_tokens, grad_weight


def _register_qkv_dw(grad_qkv_flat, tokens, weight_qkv, hidden_size, layer_id):
    """Helper: register QKV dW task or compute directly."""
    scheduler = get_backward_scheduler()

    if scheduler.is_enabled():
        tokens_saved = tokens.detach()
        grad_qkv_saved = grad_qkv_flat.reshape(-1, grad_qkv_flat.shape[-1]).detach()

        def compute_dw_qkv():
            tokens_flat = tokens_saved.view(-1, hidden_size)
            return torch.matmul(grad_qkv_saved.t(), tokens_flat)

        scheduler.register_dw_task(
            layer_name=f"qkv_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_qkv,
            priority=100,
            weight_param=weight_qkv,
        )
        return None
    else:
        tokens_flat = tokens.view(-1, hidden_size)
        grad_qkv_2d = grad_qkv_flat.reshape(-1, grad_qkv_flat.shape[-1])
        return torch.matmul(grad_qkv_2d.t(), tokens_flat)



def output_projection_register_dw(
    grad_output: torch.Tensor,
    attn_input_full: torch.Tensor,
    weight_proj: torch.Tensor,
    layer_id: int = 0,
) -> Optional[torch.Tensor]:
    """
    Register dW task for output projection.

    This should be called after outproj_sp2hp_backward has computed dX.
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
    'outproj_sp2hp_backward',
    'attention_score_backward',
    'hp2sp_qkv_backward',
    'output_projection_register_dw',
]

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

import torch
import torch.distributed as dist
from typing import Optional, Tuple

from fluid.core import _all_to_all_sp2hp_forward, _all_to_all_hp2sp_forward
from fluid.core.scheduler import get_backward_scheduler
from fluid.core.python_profile import profile_section
from fluid.core.nvtx import nvtx_range_push, nvtx_range_pop


def _alltoall_sp2hp_into(data: torch.Tensor, group: dist.ProcessGroup, out: torch.Tensor):
    return _all_to_all_sp2hp_forward(data, group, output=out)


def _alltoall_hp2sp_into(data: torch.Tensor, group: dist.ProcessGroup, out: torch.Tensor):
    return _all_to_all_hp2sp_forward(data, group, output=out)


def outproj_sp2hp_backward(
    grad_output: torch.Tensor,
    weight_proj: torch.Tensor,
    total_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    num_chunks: int = 4,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    with profile_section("attn_bwd.outproj_sp2hp"):
        return _outproj_sp2hp_backward_impl(
            grad_output, weight_proj, total_heads, head_dim, cp_group,
            num_chunks=num_chunks, comm_stream=comm_stream,
        )


def _outproj_sp2hp_backward_impl(
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

    # Ablation: --no-stage2 forces num_chunks=1.
    if not get_backward_scheduler().stage2_enabled:
        num_chunks = 1
    # Validate chunk size
    if seq_local % num_chunks != 0:
        num_chunks = 1


    if num_chunks == 1:
        # Non-chunked path: compute full dX then AllToAll via comm thread
        grad_attn_flat = torch.matmul(grad_output, weight_proj)
        grad_attn_sp = grad_attn_flat.view(seq_local, batch_size, total_heads, head_dim)

        if scheduler.is_enabled():
            task_id = scheduler.submit_alltoall_call(
                _all_to_all_sp2hp_forward, grad_attn_sp, cp_group)

            # Execute dW tasks bounded by per-region cap; leftover defers.
            scheduler.execute_dw_tasks(max_defer_tasks=2)

            # Wait for AllToAll
            return scheduler.wait_alltoall(task_id)
        else:
            grad_attn_output = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
            return grad_attn_output

    # ========================================================================
    # Chunked path: dX computation overlapped with sp2hp AllToAll via comm thread
    # ========================================================================
    seq_chunk = seq_local // num_chunks
    # Each sp2hp AllToAll chunk produces [seq_chunk * cp_size, batch, heads_local, head_dim]
    seq_chunk_out = seq_chunk * cp_size
    # Pre-allocate output to avoid torch.cat at the end
    grad_attn_output = torch.empty(
        seq_chunk_out * num_chunks, batch_size, heads_local, head_dim,
        dtype=dtype, device=device,
    )
    task_ids = []

    scheduler_enabled = scheduler.is_enabled()

    if scheduler_enabled:
        for chunk_idx in range(num_chunks):
            s_start = chunk_idx * seq_chunk
            s_end = s_start + seq_chunk
            o_start = chunk_idx * seq_chunk_out
            o_end = o_start + seq_chunk_out

            nvtx_range_push(f"attn_proj_dx_chunk_{chunk_idx}")
            grad_chunk = torch.matmul(grad_output[s_start:s_end], weight_proj)
            grad_chunk = grad_chunk.view(seq_chunk, batch_size, total_heads, head_dim)
            nvtx_range_pop()

            nvtx_range_push(f"attn_proj_a2a_submit_{chunk_idx}")
            task_ids.append(scheduler.submit_alltoall_call(
                _alltoall_sp2hp_into,
                grad_chunk,
                cp_group,
                grad_attn_output[o_start:o_end],
            ))
            nvtx_range_pop()
    else:
        for chunk_idx in range(num_chunks):
            s_start = chunk_idx * seq_chunk
            s_end = s_start + seq_chunk
            o_start = chunk_idx * seq_chunk_out
            o_end = o_start + seq_chunk_out

            grad_chunk = torch.matmul(grad_output[s_start:s_end], weight_proj)
            grad_chunk = grad_chunk.view(seq_chunk, batch_size, total_heads, head_dim)
            _all_to_all_sp2hp_forward(grad_chunk, cp_group, output=grad_attn_output[o_start:o_end])

    # ============================================
    # During AllToAll: dW tasks (bounded by per-region cap)
    # ============================================
    if scheduler_enabled:
        scheduler.execute_dw_tasks(max_defer_tasks=2)

        # Wait for the last AllToAll only
        if task_ids:
            scheduler.wait_alltoall(task_ids[-1], num_tasks=len(task_ids))

    return grad_attn_output



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
    with profile_section("attn_bwd.hp2sp_qkv"):
        return _hp2sp_qkv_backward_impl(
            grad_q, grad_k, grad_v, cp_group, tokens, weight_qkv,
            num_heads, num_kv_heads, head_dim, layer_id=layer_id,
            num_chunks=num_chunks,
        )


def _hp2sp_qkv_backward_impl(
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

    Takes grad_Q/K/V from SDPA autograd.grad and performs:
      1. QKV reassembly in HP format (native GQA: K/V already have kv_heads shape)
      2. Submit hp2sp AllToAll chunks (chunked along seq dimension)
      3. As each chunk arrives (SP format): compute QKV dX for that chunk
      4. dW tasks overlap with AllToAll

    comm_stream:    |A2A_0|A2A_1|A2A_2|A2A_3|
    default_stream: |merge|dW|wait0+dx0|wait1+dx1|...|

    Args:
        grad_q: [batch, q_heads_local, seq_full, head_dim] gradient w.r.t. Q
        grad_k: [batch, kv_heads_local, seq_full, head_dim] gradient w.r.t. K (native GQA)
        grad_v: [batch, kv_heads_local, seq_full, head_dim] gradient w.r.t. V (native GQA)
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
    # Step 1: QKV reassembly in HP format (native GQA)
    # ============================================
    # Permute to [seq_full, batch, heads_local, head_dim]
    # Native GQA: grad_q has q_heads_local, grad_k/grad_v have kv_heads_local
    grad_q_hp = grad_q.permute(2, 0, 1, 3)  # [seq_full, batch, q_heads_local, head_dim]
    grad_k_hp = grad_k.permute(2, 0, 1, 3)  # [seq_full, batch, kv_heads_local, head_dim]
    grad_v_hp = grad_v.permute(2, 0, 1, 3)  # [seq_full, batch, kv_heads_local, head_dim]

    # QKV reassembly (native GQA: K/V already have kv_heads shape, no contraction needed)
    kv_heads_local = num_kv_heads // cp_size
    # Reshape Q to grouped form: [seq_full, batch, kv_heads_local, q_per_group * head_dim]
    grad_q_hp_grouped = grad_q_hp.reshape(seq_full, batch, kv_heads_local, q_per_group * head_dim)

    # Merge Q/K/V per group: [seq_full, batch, kv_heads_local, group_size]
    grad_qkv_hp = torch.cat([grad_q_hp_grouped, grad_k_hp, grad_v_hp], dim=-1)
    # shape: [seq_full, batch, kv_heads_local, (q_per_group + 2) * head_dim]

    if not scheduler.is_enabled() or num_chunks <= 1:
        # ---- Non-chunked fallback ----
        # hp2sp AllToAll
        if scheduler.is_enabled():
            task_id = scheduler.submit_alltoall_call(
                _all_to_all_hp2sp_forward, grad_qkv_hp, cp_group)
            scheduler.execute_dw_tasks(max_defer_tasks=2)
            grad_qkv_sp = scheduler.wait_alltoall(task_id)
        else:
            grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_hp, cp_group)

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
    # Ablation: --no-stage2 forces num_chunks=1.
    if not get_backward_scheduler().stage2_enabled:
        num_chunks = 1
    # Validate chunk size (chunk along seq_full)
    if seq_full % num_chunks != 0:
        num_chunks = 1
        # Fallback
        grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_hp, cp_group)
        grad_qkv_flat = grad_qkv_sp.view(seq_local, batch, -1)
        grad_tokens = torch.matmul(grad_qkv_flat, weight_qkv)
        grad_weight = _register_qkv_dw(grad_qkv_flat, tokens, weight_qkv, hidden_size, layer_id)
        return grad_tokens, grad_weight

    seq_chunk_full = seq_full // num_chunks
    seq_chunk_local = seq_local // num_chunks

    # Pre-allocate chunk results buffer to avoid per-chunk allocation inside AllToAll
    # hp2sp output shape: [seq_chunk_local, batch, num_kv_heads, group_size]
    group_size = grad_qkv_hp.shape[3]
    total_proj = num_kv_heads * group_size
    chunk_results_buf = torch.empty(
        num_chunks, seq_chunk_local, batch, num_kv_heads, group_size,
        dtype=dtype, device=device,
    )

    comm_args = []
    for chunk_idx in range(num_chunks):
        s_start = chunk_idx * seq_chunk_full
        s_end = s_start + seq_chunk_full
        comm_args.append((grad_qkv_hp[s_start:s_end], cp_group, chunk_results_buf[chunk_idx]))

    # Batch submit: single stream switch for all chunks
    task_ids = scheduler.submit_alltoall_batch_call(_alltoall_hp2sp_into, comm_args)

    # Execute dW tasks during AllToAll (bounded by per-region cap)
    scheduler.execute_dw_tasks(max_defer_tasks=2)

    # Process each chunk as it arrives: QKV dX only
    grad_tokens = torch.empty(seq_local, batch, hidden_size, dtype=tokens.dtype, device=device)
    grad_qkv_full = torch.empty(seq_local * batch, total_proj, dtype=dtype, device=device)

    for chunk_idx in range(num_chunks):
        s_start_local = chunk_idx * seq_chunk_local
        s_end_local = s_start_local + seq_chunk_local

        is_last = (chunk_idx == num_chunks - 1)
        scheduler.wait_alltoall(task_ids[chunk_idx], try_trickle=is_last)
        grad_qkv_sp_chunk = chunk_results_buf[chunk_idx]

        # Flatten and compute dX
        grad_qkv_chunk_flat = grad_qkv_sp_chunk.reshape(seq_chunk_local, batch, -1)
        grad_tokens[s_start_local:s_end_local] = torch.matmul(grad_qkv_chunk_flat, weight_qkv)
        flat_start = s_start_local * batch
        flat_end = s_end_local * batch
        grad_qkv_full[flat_start:flat_end].copy_(grad_qkv_chunk_flat.reshape(-1, total_proj))

    # Register single QKV dW task (complete gradient, single atomic write)
    grad_weight = _register_qkv_dw(grad_qkv_full, tokens.detach(), weight_qkv, hidden_size, layer_id)

    return grad_tokens, grad_weight


def _qkv_dx_and_dw(
    grad_q, grad_k, grad_v, tokens, weight_qkv,
    num_heads, num_kv_heads, head_dim, layer_id, cp_size,
):
    """Helper: QKV reassembly + dX + dW for single-GPU case (native GQA)."""
    seq_local, batch, hidden_size = tokens.shape
    q_per_group = num_heads // num_kv_heads

    # Native GQA: grad_k/grad_v already have kv_heads shape, no contraction needed
    # Reshape Q to grouped form: [seq_local, batch, num_kv_heads, q_per_group * head_dim]
    grad_q_grouped = grad_q.reshape(seq_local, batch, num_kv_heads, q_per_group * head_dim)
    # grad_k/grad_v: [seq_local, batch, num_kv_heads, head_dim]
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

        def compute_dw_proj():
            return torch.matmul(grad_output_flat_saved.t(), attn_full_flat_saved)

        scheduler.register_dw_task(
            layer_name=f"output_proj_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_proj,
            weight_param=weight_proj,
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
    'hp2sp_qkv_backward',
    'output_projection_register_dw',
]

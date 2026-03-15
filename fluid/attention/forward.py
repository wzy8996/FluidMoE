"""
Attention Forward Operations with P2P Overlap

This module implements all forward operations for context-parallel attention:
1. QKV projection + sp2hp with P2P overlap
2. Scaled dot-product attention computation
3. hp2sp + output projection with P2P overlap

Key design principle:
- Forward uses P2P overlap (Round-Robin Tournament scheduling)
- Each round exchanges data with one peer while computing for the next peer
- This maximizes compute-communication overlap

Timeline (QKV + sp2hp):
    Round 0: Compute QKV for partner_0 -> Start P2P_0
    Round i: P2P_{i-1} running + Compute QKV for partner_i -> Start P2P_i
    Final: Last P2P running + Compute local QKV -> Wait all P2P

Timeline (hp2sp + output projection):
    Each rank collects all heads data at my_seq position via P2P
    Then computes output projection while receiving next round's data
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Optional

from fluid.core.comm import MultiCardOverlapContext
from fluid.moe.forward import _get_group_ranks


def qkv_projection_p2p_forward(
    tokens: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    te_qkv_linear=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    QKV projection with P2P overlap for sp2hp transformation.

    This function computes QKV projections while overlapping with P2P communication,
    implementing the sp2hp (sequence-parallel to head-parallel) transformation.

    Args:
        tokens: [seq_local, batch, hidden] - input tokens in SP format
        weight_qkv: [total_proj, hidden] - QKV projection weight (interleaved layout)
        num_heads: total number of Q heads
        num_kv_heads: total number of K/V heads (groups for GQA)
        head_dim: dimension per head
        cp_group: context parallel process group
        overlap_ctx: P2P overlap context for round-robin scheduling

    Returns:
        q: [seq_full, batch, heads_local, head_dim] - Q in HP format
        k: [seq_full, batch, heads_local, head_dim] - K in HP format
        v: [seq_full, batch, heads_local, head_dim] - V in HP format

    Timeline:
        default_stream: |QKV_p0|--QKV_p1--|--QKV_p2--|--local--|wait|
                            ↓        ↓          ↓
        comm_stream:        |P2P_0---|P2P_1-----|P2P_2----|
                               overlap!
    """
    seq_local, batch, hidden_size = tokens.shape
    cp_size = cp_group.size()
    my_rank = cp_group.rank()
    # Build local-to-global rank mapping for P2P ops
    global_ranks = _get_group_ranks(cp_group)
    device = tokens.device
    dtype = tokens.dtype

    q_per_group = num_heads // num_kv_heads
    group_size = (q_per_group + 2) * head_dim
    groups_per_rank = num_kv_heads // cp_size
    heads_local = groups_per_rank * q_per_group
    proj_per_rank = groups_per_rank * group_size

    if cp_size == 1:
        # Single GPU: full QKV projection, no P2P needed
        if te_qkv_linear is not None:
            with torch.no_grad():
                qkv_all = te_qkv_linear(tokens)
        else:
            qkv_all = torch.matmul(tokens, weight_qkv.t())
        qkv = qkv_all.view(seq_local, batch, num_kv_heads, group_size)
        q_dim = q_per_group * head_dim
        q = qkv[:, :, :, :q_dim].reshape(seq_local, batch, num_heads, head_dim)
        k = qkv[:, :, :, q_dim:q_dim + head_dim]
        v = qkv[:, :, :, q_dim + head_dim:]
        return q, k, v

    # Multi-GPU: per-partner QKV compute overlapped with P2P (sp2hp)
    # Timeline:
    #   Round 0: Compute QKV for partner_0 heads -> Start P2P_0
    #   Round i: P2P_{i-1} running + Compute QKV for partner_i -> Start P2P_i
    #   Final:   Last P2P running + Compute local QKV -> Wait all P2P
    num_rounds = cp_size - 1
    seq_full = seq_local * cp_size

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    # Build partner list
    partners = []
    for round_idx in range(num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner != -1:
            partners.append(partner)

    # Result buffer: [seq_full, batch, groups_per_rank, group_size]
    qkv_full = torch.empty(seq_full, batch, groups_per_rank, group_size,
                           dtype=dtype, device=device)

    # Per-partner weight slices: weight_qkv is [num_kv_heads * group_size, hidden]
    # Partner p owns head groups [p*groups_per_rank : (p+1)*groups_per_rank]
    # corresponding to weight rows [p*proj_per_rank : (p+1)*proj_per_rank]
    weight_per_partner = {}
    for partner in partners:
        w_start = partner * proj_per_rank
        weight_per_partner[partner] = weight_qkv[w_start:w_start + proj_per_rank, :]

    all_reqs = []
    send_bufs = []  # keep references alive until P2P completes
    last_p2p_event = None

    for round_idx, partner in enumerate(partners):
        # Compute QKV for this partner's head groups
        # tokens: [seq_local, batch, hidden] @ weight_slice.t() -> [seq_local, batch, proj_per_rank]
        send_buf = torch.matmul(tokens, weight_per_partner[partner].t())
        send_buf = send_buf.view(seq_local, batch, groups_per_rank, group_size).contiguous()
        send_bufs.append(send_buf)

        # Start P2P: send my QKV for partner's heads, receive partner's QKV for my heads
        partner_seq_start = partner * seq_local
        recv_buffer = qkv_full[partner_seq_start:partner_seq_start + seq_local]

        with torch.cuda.stream(comm_stream):
            comm_stream.wait_stream(default_stream)
            global_partner = global_ranks[partner]
            p2p_ops = [
                dist.P2POp(dist.irecv, recv_buffer, global_partner, group=cp_group),
                dist.P2POp(dist.isend, send_buf, global_partner, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(p2p_ops)
            all_reqs.extend(reqs)
            last_p2p_event = torch.cuda.Event()
            last_p2p_event.record(comm_stream)

    # While last P2P is running, compute local QKV (overlaps with comm)
    local_w_start = my_rank * proj_per_rank
    weight_local = weight_qkv[local_w_start:local_w_start + proj_per_rank, :]
    local_qkv = torch.matmul(tokens, weight_local.t())
    local_qkv = local_qkv.view(seq_local, batch, groups_per_rank, group_size)

    local_seq_start = my_rank * seq_local
    qkv_full[local_seq_start:local_seq_start + seq_local] = local_qkv

    # GPU-level wait for all P2P to complete (no CPU blocking)
    if last_p2p_event is not None:
        default_stream.wait_event(last_p2p_event)

    # NCCL cleanup (P2P already done, returns immediately)
    for req in all_reqs:
        req.wait()

    # Separate Q, K, V
    q_size = q_per_group * head_dim
    q, k, v = torch.split(qkv_full, [q_size, head_dim, head_dim], dim=-1)
    q = q.reshape(seq_full, batch, heads_local, head_dim)

    return q, k, v


def scaled_dot_product_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = True,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Scaled dot-product attention using PyTorch native SDPA.

    Uses F.scaled_dot_product_attention which automatically selects the best
    backend (FlashAttention, Memory-Efficient, or Math) based on hardware.
    No manual recomputation needed - SDPA handles this internally during backward.

    Args:
        query: [batch, q_heads, seq, head_dim] - batch-first format
        key: [batch, kv_heads, seq, head_dim] - can have fewer heads if enable_gqa=True
        value: [batch, kv_heads, seq, head_dim] - can have fewer heads if enable_gqa=True
        scale: optional scale factor (default: 1/sqrt(head_dim))
        is_causal: whether to apply causal mask
        enable_gqa: if True, use native GQA (PyTorch 2.5+), no need to expand K/V heads

    Returns:
        output: [batch, q_heads, seq, head_dim] - same format as query
    """
    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    return output


def output_projection_p2p_forward(
    attn_output: torch.Tensor,
    weight_proj: torch.Tensor,
    bias_proj: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    te_proj_linear=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Output projection with P2P overlap for hp2sp transformation.

    This function computes output projection while overlapping with P2P communication,
    implementing the hp2sp (head-parallel to sequence-parallel) transformation.

    Args:
        attn_output: [seq_full, batch, heads_local, head_dim] - attention output in HP format
        weight_proj: [hidden, total_heads * head_dim] - output projection weight
        bias_proj: [hidden] or None - output projection bias
        cp_group: context parallel process group
        overlap_ctx: P2P overlap context for round-robin scheduling

    Returns:
        output: [seq_local, batch, hidden] - projected output in SP format
        attn_input_full: [seq_local, batch, all_heads * head_dim] - saved for backward

    Timeline:
        Each rank collects all heads data at my_seq position via P2P
        Compute partial output while receiving next round's data
    """
    cp_size = cp_group.size()
    my_rank = cp_group.rank()
    # Build local-to-global rank mapping for P2P ops
    global_ranks = _get_group_ranks(cp_group)
    device = attn_output.device
    dtype = attn_output.dtype

    if cp_size == 1:
        # Single GPU: compute directly
        attn_flat = attn_output.view(attn_output.shape[0], attn_output.shape[1], -1)
        if te_proj_linear is not None:
            with torch.no_grad():
                output = te_proj_linear(attn_flat)
        else:
            output = torch.matmul(attn_flat, weight_proj.t())
        if bias_proj is not None:
            output = output + bias_proj
        return output, attn_flat

    seq_full, batch_size, heads_local, head_dim = attn_output.shape
    seq_local = seq_full // cp_size
    hidden_size = weight_proj.shape[0]
    input_dim_per_rank = heads_local * head_dim
    num_rounds = cp_size - 1

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    # Local sequence position
    local_seq_start = my_rank * seq_local

    # Build partner list
    partners = []
    for round_idx in range(num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner != -1:
            partners.append(partner)

    # Prepare send data and receive buffers
    send_data_dict = {}
    recv_buffers = {}
    recv_buffers_flat = {}

    for partner in partners:
        partner_seq_start = partner * seq_local
        send_data_dict[partner] = attn_output[partner_seq_start:partner_seq_start + seq_local]

        recv_buf = torch.empty(
            seq_local, batch_size, heads_local, head_dim, dtype=dtype, device=device
        )
        recv_buffers[partner] = recv_buf
        recv_buffers_flat[partner] = recv_buf.view(seq_local, batch_size, -1)

    # --- Per-partner matmul with P2P overlap (event-based, no CPU blocking) ---
    weight_local_start = my_rank * input_dim_per_rank
    weight_local = weight_proj[:, weight_local_start:weight_local_start + input_dim_per_rank]
    weight_per_partner = {}
    for partner in partners:
        partner_weight_start = partner * input_dim_per_rank
        weight_per_partner[partner] = weight_proj[:, partner_weight_start:partner_weight_start + input_dim_per_rank]

    attn_local_seq = attn_output[local_seq_start:local_seq_start + seq_local]
    attn_local_flat = attn_local_seq.reshape(seq_local, batch_size, -1)

    # Pre-allocate attn_input_full, copy into it during compute to avoid torch.cat at end
    total_head_dim = cp_size * input_dim_per_rank
    attn_input_full = torch.empty(seq_local, batch_size, total_head_dim,
                                  dtype=dtype, device=device)
    local_dim_start = my_rank * input_dim_per_rank
    attn_input_full[:, :, local_dim_start:local_dim_start + input_dim_per_rank] = attn_local_flat

    prev_partner = None
    all_reqs = []
    p2p_events = []
    output = None

    for round_idx, partner in enumerate(partners):
        with torch.cuda.stream(comm_stream):
            if round_idx == 0:
                comm_stream.wait_stream(default_stream)
            global_partner = global_ranks[partner]
            p2p_ops = [
                dist.P2POp(dist.irecv, recv_buffers[partner], global_partner, group=cp_group),
                dist.P2POp(dist.isend, send_data_dict[partner], global_partner, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(p2p_ops)
            all_reqs.extend(reqs)
            evt = torch.cuda.Event()
            evt.record(comm_stream)
            p2p_events.append(evt)

        if round_idx == 0:
            output = torch.matmul(attn_local_flat, weight_local.t())
        else:
            # GPU-level wait for previous P2P (no CPU blocking)
            default_stream.wait_event(p2p_events[round_idx - 1])
            # Copy received data into pre-allocated buffer (overlaps with current P2P)
            prev_dim_start = prev_partner * input_dim_per_rank
            attn_input_full[:, :, prev_dim_start:prev_dim_start + input_dim_per_rank] = recv_buffers_flat[prev_partner]
            output = output + torch.matmul(recv_buffers_flat[prev_partner], weight_per_partner[prev_partner].t())

        prev_partner = partner

    if len(partners) > 0:
        default_stream.wait_event(p2p_events[-1])
        # Copy last partner's data into pre-allocated buffer
        prev_dim_start = prev_partner * input_dim_per_rank
        attn_input_full[:, :, prev_dim_start:prev_dim_start + input_dim_per_rank] = recv_buffers_flat[prev_partner]
        output = output + torch.matmul(recv_buffers_flat[prev_partner], weight_per_partner[prev_partner].t())

    # NCCL work cleanup (P2P already done at this point, returns immediately)
    for req in all_reqs:
        req.wait()

    if bias_proj is not None:
        output = output + bias_proj

    return output, attn_input_full


__all__ = [
    'qkv_projection_p2p_forward',
    'scaled_dot_product_attention_forward',
    'output_projection_p2p_forward',
]

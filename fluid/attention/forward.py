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


def qkv_projection_p2p_forward(
    tokens: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
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
    device = tokens.device
    dtype = tokens.dtype

    q_per_group = num_heads // num_kv_heads
    group_size = (q_per_group + 2) * head_dim
    groups_per_rank = num_kv_heads // cp_size
    heads_local = groups_per_rank * q_per_group
    proj_per_rank = groups_per_rank * group_size

    if cp_size == 1:
        # Single GPU: compute directly without P2P
        qkv = torch.matmul(tokens, weight_qkv.t())
        qkv = qkv.view(seq_local, batch, num_kv_heads, group_size)
        q_dim = q_per_group * head_dim
        q = qkv[:, :, :, :q_dim].reshape(seq_local, batch, num_heads, head_dim)
        k = qkv[:, :, :, q_dim:q_dim + head_dim]
        v = qkv[:, :, :, q_dim + head_dim:]
        # K, V: [seq_local, batch, num_kv_heads, head_dim] - NOT expanded for GQA
        # Caller should expand if needed
        return q, k, v

    # Multi-GPU: P2P overlap
    num_rounds = cp_size - 1
    seq_full = seq_local * cp_size

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    # Split weight by group: [num_groups, group_size, hidden]
    weight_grouped = weight_qkv.view(num_kv_heads, group_size, hidden_size)

    # Weight for local groups
    local_group_start = my_rank * groups_per_rank
    weight_local = weight_grouped[local_group_start:local_group_start + groups_per_rank]
    weight_local = weight_local.reshape(-1, hidden_size)

    # Pre-prepare weight for each peer
    weight_per_partner = {}
    partners = []
    for round_idx in range(num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner == -1:
            continue
        partners.append(partner)
        r_group_start = partner * groups_per_rank
        weight_r = weight_grouped[r_group_start:r_group_start + groups_per_rank]
        weight_per_partner[partner] = weight_r.reshape(-1, hidden_size)

    # Pipeline overlap implementation
    qkv_full = torch.empty(seq_full, batch, proj_per_rank, dtype=dtype, device=device)
    send_data_dict = {}

    # Reuse single Event from context to reduce overhead
    send_data_event = overlap_ctx.data_ready_event

    # Pre-compute first round's send_data (before pipeline starts)
    if len(partners) > 0:
        first_partner = partners[0]
        send_data_dict[first_partner] = torch.matmul(
            tokens, weight_per_partner[first_partner].t()
        )
        send_data_event.record(default_stream)

    # Pipeline loop: each round's computation overlaps with previous P2P
    # GPU sync version: comm_stream waits for computation, no CPU blocking
    all_reqs = []
    qkv_local = None

    for round_idx, partner in enumerate(partners):
        # Receive buffer at partner's sequence position
        partner_seq_start = partner * seq_local
        recv_buffer = qkv_full[partner_seq_start:partner_seq_start + seq_local]

        # Start P2P communication
        # GPU sync: comm_stream waits for send_data to be ready
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(send_data_event)  # GPU sync instead of CPU sync
            p2p_ops = [
                dist.P2POp(dist.irecv, recv_buffer, partner, group=cp_group),
                dist.P2POp(dist.isend, send_data_dict[partner], partner, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(p2p_ops)
            all_reqs.extend(reqs)

        # Parallel with current P2P: compute next round's data on default_stream
        if round_idx + 1 < len(partners):
            next_partner = partners[round_idx + 1]
            send_data_dict[next_partner] = torch.matmul(
                tokens, weight_per_partner[next_partner].t()
            )
            send_data_event.record(default_stream)
        else:
            # Last round: compute local QKV (parallel with last P2P)
            qkv_local = torch.matmul(tokens, weight_local.t())

    # Wait for all P2P to complete (release NCCL resources)
    for req in all_reqs:
        req.wait()

    # Handle no partners case (cp_size=1)
    if len(partners) == 0:
        qkv_local = torch.matmul(tokens, weight_local.t())

    # Assemble result: write local data to corresponding position
    local_seq_start = my_rank * seq_local
    qkv_full[local_seq_start:local_seq_start + seq_local] = qkv_local

    # Separate Q, K, V
    qkv_full = qkv_full.view(seq_full, batch, groups_per_rank, group_size)
    q_size = q_per_group * head_dim
    q, k, v = torch.split(qkv_full, [q_size, head_dim, head_dim], dim=-1)
    q = q.reshape(seq_full, batch, heads_local, head_dim)
    # K, V: [seq_full, batch, kv_heads_local, head_dim] - NOT expanded for GQA
    # Caller should expand if needed

    return q, k, v


def scaled_dot_product_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Scaled dot-product attention using PyTorch native SDPA.

    Uses F.scaled_dot_product_attention which automatically selects the best
    backend (FlashAttention, Memory-Efficient, or Math) based on hardware.
    No manual recomputation needed - SDPA handles this internally during backward.

    Args:
        query: [batch, heads, seq, head_dim] - batch-first format
        key: [batch, heads, seq, head_dim]
        value: [batch, heads, seq, head_dim]
        scale: optional scale factor (default: 1/sqrt(head_dim))
        is_causal: whether to apply causal mask

    Returns:
        output: [batch, heads, seq, head_dim] - same format as input
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
    )

    return output


def output_projection_p2p_forward(
    attn_output: torch.Tensor,
    weight_proj: torch.Tensor,
    bias_proj: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
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
    device = attn_output.device
    dtype = attn_output.dtype

    if cp_size == 1:
        # Single GPU: compute directly
        attn_flat = attn_output.view(attn_output.shape[0], attn_output.shape[1], -1)
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

    # Weight for local heads
    weight_local_start = my_rank * input_dim_per_rank
    weight_local = weight_proj[:, weight_local_start:weight_local_start + input_dim_per_rank]

    # Prepare send data and receive buffers
    send_data_dict = {}
    recv_buffers = {}
    recv_buffers_flat = {}  # Pre-computed flat views
    weight_per_partner = {}
    partners = []

    for round_idx in range(num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner == -1:
            continue
        partners.append(partner)

        # Send data: attn_output at peer's sequence position
        # Slice along dim 0 of contiguous tensor is already contiguous - no need for .contiguous()
        partner_seq_start = partner * seq_local
        send_data_dict[partner] = attn_output[partner_seq_start:partner_seq_start + seq_local]

        # Receive buffer + pre-computed flat view
        recv_buf = torch.empty(
            seq_local, batch_size, heads_local, head_dim, dtype=dtype, device=device
        )
        recv_buffers[partner] = recv_buf
        recv_buffers_flat[partner] = recv_buf.view(seq_local, batch_size, -1)

        # Cache partner's weight
        partner_weight_start = partner * input_dim_per_rank
        weight_per_partner[partner] = weight_proj[:, partner_weight_start:partner_weight_start + input_dim_per_rank]

    # Pipeline overlap with delayed req.wait()
    # Key insight: Start P2P_i first, then wait for P2P_{i-1} to complete
    # This ensures P2P_i runs in background while we process P2P_{i-1}'s data
    attn_local_seq = attn_output[local_seq_start:local_seq_start + seq_local]
    attn_local_flat = attn_local_seq.reshape(seq_local, batch_size, -1)

    prev_partner = None
    prev_reqs = []
    output = None

    for round_idx, partner in enumerate(partners):
        # 1. Start current round's P2P (async, returns immediately)
        # GPU sync: comm_stream waits for default_stream (send_data preparation)
        with torch.cuda.stream(comm_stream):
            if round_idx == 0:
                comm_stream.wait_stream(default_stream)  # Wait for send_data preparation
            p2p_ops = [
                dist.P2POp(dist.irecv, recv_buffers[partner], partner, group=cp_group),
                dist.P2POp(dist.isend, send_data_dict[partner], partner, group=cp_group),
            ]
            curr_reqs = dist.batch_isend_irecv(p2p_ops)

        # 2. Wait for PREVIOUS round's P2P to complete (current round runs in background)
        if round_idx > 0:
            for req in prev_reqs:
                req.wait()

        # 3. Compute (overlaps with current round's P2P)
        if round_idx == 0:
            # First round: compute local partial (parallel with P2P_0)
            output = torch.matmul(attn_local_flat, weight_local.t())
        else:
            # Use previous round's received data (now guaranteed complete)
            output = output + torch.matmul(recv_buffers_flat[prev_partner], weight_per_partner[prev_partner].t())

        prev_partner = partner
        prev_reqs = curr_reqs

    # Process last round: wait for last P2P and compute
    if len(partners) > 0:
        for req in prev_reqs:
            req.wait()
        output = output + torch.matmul(recv_buffers_flat[prev_partner], weight_per_partner[prev_partner].t())

    if bias_proj is not None:
        output = output + bias_proj

    # Collect all heads' data at my_seq position for backward (use pre-computed flat views)
    attn_parts = {my_rank: attn_local_flat}
    for partner in partners:
        attn_parts[partner] = recv_buffers_flat[partner]
    attn_input_full = torch.cat([attn_parts[r] for r in range(cp_size)], dim=-1)

    return output, attn_input_full


__all__ = [
    'qkv_projection_p2p_forward',
    'scaled_dot_product_attention_forward',
    'output_projection_p2p_forward',
]

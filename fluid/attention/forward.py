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


class CPPlan:
    """Lightweight pre-computed static plan for attention CP P2P communication.

    Caches only Python values and event references — zero GPU buffer allocation.
    """
    __slots__ = (
        'cp_size', 'my_rank', 'seq_local',
        'q_per_group', 'group_size', 'groups_per_rank',
        'heads_local', 'proj_per_rank',
        'partners', 'global_partners', 'n_partners',
        # QKV forward
        'qkv_weight_offsets', 'qkv_recv_seq_starts',
        'qkv_events', 'qkv_ready_events',
        'local_w_start',
        # Output proj forward
        'proj_weight_offsets', 'proj_dim_starts',
        'proj_events',
        'input_dim_per_rank', 'local_seq_start',
        'weight_local_start',
        # Shared slots (set externally, shared across layers)
        'qkv_send_slots',   # [n_partners] pre-allocated send bufs, or None
        'proj_recv_slots',   # [n_partners] pre-allocated recv bufs, or None
        'proj_recv_flat',    # [n_partners] flattened views, or None
    )

    def __init__(self, seq_local, batch, hidden_size,
                 num_heads, num_kv_heads, head_dim,
                 cp_group, overlap_ctx):
        self.cp_size = cp_group.size()
        self.my_rank = cp_group.rank()
        self.seq_local = seq_local

        self.q_per_group = num_heads // num_kv_heads
        self.group_size = (self.q_per_group + 2) * head_dim
        self.groups_per_rank = num_kv_heads // self.cp_size
        self.heads_local = self.groups_per_rank * self.q_per_group
        self.proj_per_rank = self.groups_per_rank * self.group_size

        # Round-robin partners
        num_rounds = self.cp_size - 1 if self.cp_size % 2 == 0 else self.cp_size
        self.partners = []
        for r in range(num_rounds):
            p = overlap_ctx.get_partner(self.my_rank, r)
            if p != -1:
                self.partners.append(p)
        global_ranks = _get_group_ranks(cp_group)
        self.global_partners = [global_ranks[p] for p in self.partners]
        self.n_partners = len(self.partners)

        # QKV: weight slice offsets per partner
        self.qkv_weight_offsets = [p * self.proj_per_rank for p in self.partners]
        self.qkv_recv_seq_starts = [p * seq_local for p in self.partners]
        self.local_w_start = self.my_rank * self.proj_per_rank

        # QKV: pre-allocated events
        self.qkv_events = [overlap_ctx.get_round_event("attn_qkv", i)
                           for i in range(self.n_partners)]
        self.qkv_ready_events = [overlap_ctx.get_round_event("attn_qkv_ready", i)
                                 for i in range(self.n_partners)]

        # Output proj: weight slice offsets and dim starts per partner
        self.input_dim_per_rank = self.heads_local * head_dim
        self.local_seq_start = self.my_rank * seq_local
        self.weight_local_start = self.my_rank * self.input_dim_per_rank
        self.proj_weight_offsets = [p * self.input_dim_per_rank for p in self.partners]
        self.proj_dim_starts = [p * self.input_dim_per_rank for p in self.partners]
        self.proj_events = [overlap_ctx.get_round_event("attn_output", i)
                            for i in range(self.n_partners)]

        # Shared slots: set via set_shared_slots() after creation.
        # Shared across layers — only 1 set allocated for the whole model.
        self.qkv_send_slots = None
        self.proj_recv_slots = None
        self.proj_recv_flat = None

    def set_shared_slots(self, qkv_send_slots, proj_recv_slots):
        """Attach shared pre-allocated buffers (called once by TransformerModel)."""
        self.qkv_send_slots = qkv_send_slots
        self.proj_recv_slots = proj_recv_slots
        if proj_recv_slots is not None:
            s0 = proj_recv_slots[0]
            batch = s0.shape[1]
            self.proj_recv_flat = [s.view(self.seq_local, batch, -1)
                                   for s in proj_recv_slots]
        else:
            self.proj_recv_flat = None


def qkv_projection_p2p_forward(
    tokens: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    te_qkv_linear=None,
    cp_plan: 'CPPlan' = None,
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

    seq_full = seq_local * cp_size
    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    # Use cp_plan for cached schedule/events, or build dynamically
    if cp_plan is not None:
        partners = cp_plan.partners
        global_partners = cp_plan.global_partners
        events = cp_plan.qkv_events
        ready_events = cp_plan.qkv_ready_events
        n_partners = cp_plan.n_partners
        local_w_start = cp_plan.local_w_start
    else:
        partners = []
        for r in range(cp_size - 1):
            p = overlap_ctx.get_partner(my_rank, r)
            if p != -1:
                partners.append(p)
        global_partners = [global_ranks[p] for p in partners]
        events = [overlap_ctx.get_round_event("attn_qkv", i) for i in range(len(partners))]
        ready_events = [None] * len(partners)  # will use default_stream.record_event()
        n_partners = len(partners)
        local_w_start = my_rank * proj_per_rank

    # Result buffer — symmetric heap when NVSHMEM active (recv target).
    # Views/slices of symmetric tensors are valid symmetric addresses.
    from fluid.core.p2p_backend import get_p2p_backend
    _p2p = get_p2p_backend()
    _qkv_numel = seq_full * batch * groups_per_rank * group_size
    qkv_full = _p2p.alloc_recv_buffer(
        "attn_qkv_full", _qkv_numel, dtype, device
    ).view(seq_full, batch, groups_per_rank, group_size)

    _use_slots = cp_plan is not None and cp_plan.qkv_send_slots is not None
    send_bufs = [] if not _use_slots else None
    last_p2p_event = None

    for i in range(n_partners):
        w_start = (cp_plan.qkv_weight_offsets[i] if cp_plan is not None
                   else partners[i] * proj_per_rank)

        if _use_slots:
            send_buf = cp_plan.qkv_send_slots[i]
            send_flat = send_buf.view(seq_local, batch, proj_per_rank)
            torch.matmul(tokens, weight_qkv[w_start:w_start + proj_per_rank, :].t(),
                         out=send_flat)
        else:
            send_buf = torch.matmul(tokens, weight_qkv[w_start:w_start + proj_per_rank, :].t())
            send_buf = send_buf.view(seq_local, batch, groups_per_rank, group_size).contiguous()
            send_bufs.append(send_buf)

        if cp_plan is not None:
            ready_events[i].record(default_stream)
            ready_evt = ready_events[i]
        else:
            ready_evt = default_stream.record_event()
        seq_start = (cp_plan.qkv_recv_seq_starts[i] if cp_plan is not None
                     else partners[i] * seq_local)
        recv_buffer = qkv_full[seq_start:seq_start + seq_local]
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(ready_evt)
            recv_buffer.record_stream(comm_stream)
            if not _use_slots:
                send_buf.record_stream(comm_stream)
            _p2p.exchange(
                send_buf=send_buf, recv_buf=recv_buffer,
                partner_global_rank=global_partners[i],
                partner_local_rank=partners[i],
                group=cp_group, comm_stream=comm_stream, event=events[i],
            )
            last_p2p_event = events[i]

    # Compute local QKV (overlaps with last P2P)
    weight_local = weight_qkv[local_w_start:local_w_start + proj_per_rank, :]
    local_qkv = torch.matmul(tokens, weight_local.t())
    local_qkv = local_qkv.view(seq_local, batch, groups_per_rank, group_size)

    _my_rank = cp_plan.my_rank if cp_plan is not None else my_rank
    local_seq_start = _my_rank * seq_local
    qkv_full[local_seq_start:local_seq_start + seq_local] = local_qkv

    # GPU-level wait for all P2P to complete (no CPU blocking)
    if last_p2p_event is not None:
        default_stream.wait_event(last_p2p_event)

    if _p2p.needs_final_wait():
        _p2p.final_wait()

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
    cp_plan: 'CPPlan' = None,
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

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    # Use cp_plan for cached schedule, or build dynamically
    if cp_plan is not None:
        input_dim_per_rank = cp_plan.input_dim_per_rank
        local_seq_start = cp_plan.local_seq_start
        partners = cp_plan.partners
        global_partners = cp_plan.global_partners
        p2p_events = cp_plan.proj_events
        n_partners = cp_plan.n_partners
        weight_local_start = cp_plan.weight_local_start
    else:
        input_dim_per_rank = heads_local * head_dim
        local_seq_start = my_rank * seq_local
        partners = []
        for r in range(cp_size - 1):
            p = overlap_ctx.get_partner(my_rank, r)
            if p != -1:
                partners.append(p)
        global_partners = [global_ranks[p] for p in partners]
        p2p_events = [overlap_ctx.get_round_event("attn_output", i) for i in range(len(partners))]
        n_partners = len(partners)
        weight_local_start = my_rank * input_dim_per_rank

    weight_local = weight_proj[:, weight_local_start:weight_local_start + input_dim_per_rank]

    from fluid.core.p2p_backend import get_p2p_backend
    _p2p_proj = get_p2p_backend()
    _use_slots = cp_plan is not None and cp_plan.proj_recv_slots is not None
    _has_plan = cp_plan is not None

    send_data_list = []
    if not _has_plan:
        weight_slices = []
        partner_dim_starts = []
    if not _use_slots:
        recv_bufs = []
        recv_bufs_flat = []

    for i, partner in enumerate(partners):
        partner_seq_start = partner * seq_local
        send_data_list.append(attn_output[partner_seq_start:partner_seq_start + seq_local])
        if not _use_slots:
            _proj_recv_numel = seq_local * batch_size * heads_local * head_dim
            recv_buf = _p2p_proj.alloc_recv_buffer(
                f"attn_proj_recv_{i}", _proj_recv_numel, dtype, device
            ).view(seq_local, batch_size, heads_local, head_dim)
            recv_bufs.append(recv_buf)
            recv_bufs_flat.append(recv_buf.view(seq_local, batch_size, -1))
        if not _has_plan:
            weight_slices.append(weight_proj[:, partner * input_dim_per_rank:partner * input_dim_per_rank + input_dim_per_rank])
            partner_dim_starts.append(partner * input_dim_per_rank)

    attn_local_seq = attn_output[local_seq_start:local_seq_start + seq_local]
    attn_local_flat = attn_local_seq.reshape(seq_local, batch_size, -1)

    total_head_dim = cp_size * input_dim_per_rank
    attn_input_full = torch.empty(seq_local, batch_size, total_head_dim, dtype=dtype, device=device)
    local_dim_start = weight_local_start
    attn_input_full[:, :, local_dim_start:local_dim_start + input_dim_per_rank] = attn_local_flat

    output = None

    for i in range(n_partners):
        recv_buf_i = cp_plan.proj_recv_slots[i] if _use_slots else recv_bufs[i]

        with torch.cuda.stream(comm_stream):
            if i == 0:
                comm_stream.wait_stream(default_stream)
            if not _use_slots:
                recv_buf_i.record_stream(comm_stream)
            send_data_list[i].record_stream(comm_stream)
            _p2p_proj.exchange(
                send_buf=send_data_list[i], recv_buf=recv_buf_i,
                partner_global_rank=global_partners[i],
                partner_local_rank=partners[i],
                group=cp_group, comm_stream=comm_stream, event=p2p_events[i],
            )

        if i == 0:
            output = torch.matmul(attn_local_flat, weight_local.t())
        else:
            default_stream.wait_event(p2p_events[i - 1])
            prev_flat = cp_plan.proj_recv_flat[i - 1] if _use_slots else recv_bufs_flat[i - 1]
            if _has_plan:
                d_start = cp_plan.proj_dim_starts[i - 1]
                w_off = cp_plan.proj_weight_offsets[i - 1]
                attn_input_full[:, :, d_start:d_start + input_dim_per_rank] = prev_flat
                output = output + torch.matmul(prev_flat, weight_proj[:, w_off:w_off + input_dim_per_rank].t())
            else:
                attn_input_full[:, :, partner_dim_starts[i-1]:partner_dim_starts[i-1] + input_dim_per_rank] = prev_flat
                output = output + torch.matmul(prev_flat, weight_slices[i-1].t())

    if n_partners > 0:
        last = n_partners - 1
        default_stream.wait_event(p2p_events[last])
        last_flat = cp_plan.proj_recv_flat[last] if _use_slots else recv_bufs_flat[last]
        if _has_plan:
            d_start = cp_plan.proj_dim_starts[last]
            w_off = cp_plan.proj_weight_offsets[last]
            attn_input_full[:, :, d_start:d_start + input_dim_per_rank] = last_flat
            output = output + torch.matmul(last_flat, weight_proj[:, w_off:w_off + input_dim_per_rank].t())
        else:
            attn_input_full[:, :, partner_dim_starts[last]:partner_dim_starts[last] + input_dim_per_rank] = last_flat
            output = output + torch.matmul(last_flat, weight_slices[last].t())

    if _p2p_proj.needs_final_wait():
        _p2p_proj.final_wait()

    if bias_proj is not None:
        output = output + bias_proj

    return output, attn_input_full


__all__ = [
    'qkv_projection_p2p_forward',
    'scaled_dot_product_attention_forward',
    'output_projection_p2p_forward',
]

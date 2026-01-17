"""
Attention P2P Forward Overlap Implementation

Uses Round-Robin Tournament scheduling for multi-card P2P communication overlap.

Key functions:
- qkv_sp2hp_multicard_overlap: QKV projection + sp2hp with P2P overlap (with gradient support)
- hp2sp_output_proj_multicard_overlap: hp2sp + output projection with P2P overlap (with gradient support)

The P2P overlap strategy for Attention:
1. QKV phase: Split sp2hp AllToAll into P2P rounds
   - Each round exchanges Q/K/V data with one peer
   - Overlap round r communication with computation on round r-1 data
2. Output phase: Split hp2sp AllToAll similarly
   - Each round exchanges attention output with one peer

Core Principle (Pipeline Overlap):
- Round 0: Compute QKV for partner_0 -> Start P2P_0
- Round i (i > 0): P2P_{i-1} running + Compute QKV for partner_i -> Start P2P_i
- Final: Last P2P running + Compute local QKV -> Wait for all P2P

This way each round's QKV computation overlaps with the previous round's P2P!
"""

import os
import torch
import torch.distributed as dist
from typing import Tuple, Optional

from fluid.core.forward_comm import AttentionMultiCardOverlapContext, MultiCardOverlapContext


# =============================================================================
# Autograd Functions (with integrated implementation)
# =============================================================================

class _QKVSp2HpMultiCardFunction(torch.autograd.Function):
    """QKV sp2hp multi-card P2P overlap autograd function

    Forward: QKV computation + sp2hp with P2P overlap
    Backward: standard AllToAll + dW scheduling
    """

    @staticmethod
    def forward(ctx, hidden_states, weight_qkv, num_heads, num_kv_heads, head_dim,
                cp_group, overlap_ctx, layer_id):
        # Check if backward is needed (skip saving for inference)
        needs_grad = hidden_states.requires_grad
        ctx.needs_grad = needs_grad

        ctx.cp_group = cp_group
        ctx.num_heads = num_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.head_dim = head_dim
        ctx.layer_id = layer_id

        if needs_grad:
            ctx.save_for_backward(hidden_states, weight_qkv)

        # =====================================================================
        # QKV sp2hp multi-card P2P overlap implementation
        # =====================================================================
        cp_size = cp_group.size()
        my_rank = cp_group.rank()
        device = hidden_states.device
        dtype = hidden_states.dtype
        seq_local, batch_size, hidden_size = hidden_states.shape

        if cp_size == 1:
            # Single card: compute directly
            qkv = torch.matmul(hidden_states, weight_qkv.t())
            qkv = qkv.view(seq_local, batch_size, num_kv_heads, -1)
            q_per_group = num_heads // num_kv_heads
            q_size = q_per_group * head_dim
            q, k, v = torch.split(qkv, [q_size, head_dim, head_dim], dim=-1)
            q = q.reshape(seq_local, batch_size, num_heads, head_dim)
            return q, k, v

        # Multi-card setup
        num_rounds = cp_size - 1
        seq_full = seq_local * cp_size
        q_per_group = num_heads // num_kv_heads
        groups_per_rank = num_kv_heads // cp_size
        heads_local = groups_per_rank * q_per_group
        group_size = (q_per_group + 2) * head_dim
        proj_per_rank = groups_per_rank * group_size

        default_stream = torch.cuda.current_stream(device)
        comm_stream = overlap_ctx.get_stream()

        # Split weight by group: [num_groups, group_size, hidden]
        weight_grouped = weight_qkv.view(num_kv_heads, group_size, hidden_size)

        # Weight for local groups
        local_group_start = my_rank * groups_per_rank
        weight_local = weight_grouped[local_group_start:local_group_start + groups_per_rank]
        weight_local = weight_local.reshape(-1, hidden_size)  # [proj_per_rank, hidden]

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

        # =====================================================================
        # Pipeline overlap: each remote matmul parallel with previous P2P
        # =====================================================================
        qkv_full = torch.empty(seq_full, batch_size, proj_per_rank, dtype=dtype, device=device)
        send_data_dict = {}
        send_data_events = {}

        # Round -1: Pre-compute first round's send_data
        if len(partners) > 0:
            first_partner = partners[0]
            send_data_dict[first_partner] = torch.matmul(
                hidden_states, weight_per_partner[first_partner].t()
            )
            send_data_events[first_partner] = torch.cuda.Event()
            send_data_events[first_partner].record(default_stream)

        # Pipeline loop
        all_reqs = []
        for round_idx, partner in enumerate(partners):
            # Receive buffer
            partner_seq_start = partner * seq_local
            recv_buffer = qkv_full[partner_seq_start:partner_seq_start + seq_local]

            # CPU wait for send_data to be ready
            send_data_events[partner].synchronize()

            # Start P2P (send_data is ready)
            with torch.cuda.stream(comm_stream):
                p2p_ops = [
                    dist.P2POp(dist.irecv, recv_buffer, partner, group=cp_group),
                    dist.P2POp(dist.isend, send_data_dict[partner], partner, group=cp_group),
                ]
                reqs = dist.batch_isend_irecv(p2p_ops)
                all_reqs.extend(reqs)

            # Parallel with current P2P: compute next round data on default_stream
            if round_idx + 1 < len(partners):
                next_partner = partners[round_idx + 1]
                send_data_dict[next_partner] = torch.matmul(
                    hidden_states, weight_per_partner[next_partner].t()
                )
                send_data_events[next_partner] = torch.cuda.Event()
                send_data_events[next_partner].record(default_stream)
            else:
                # Last round: compute local QKV (parallel with last P2P)
                qkv_local = torch.matmul(hidden_states, weight_local.t())

        # Wait for all P2P to complete
        for req in all_reqs:
            req.wait()

        # Handle no partners case (cp_size=1)
        if len(partners) == 0:
            qkv_local = torch.matmul(hidden_states, weight_local.t())

        # Assemble result: write local data to corresponding position
        local_seq_start = my_rank * seq_local
        qkv_full[local_seq_start:local_seq_start + seq_local] = qkv_local

        # Separate Q, K, V
        qkv_full = qkv_full.view(seq_full, batch_size, groups_per_rank, group_size)
        q_size = q_per_group * head_dim
        q, k, v = torch.split(qkv_full, [q_size, head_dim, head_dim], dim=-1)
        q = q.reshape(seq_full, batch_size, heads_local, head_dim)
        # k, v: [seq_full, B, kv_heads_local, head_dim]

        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        """Backward uses standard AllToAll"""
        from fluid.core.alltoall import _all_to_all_hp2sp_forward

        hidden_states, weight_qkv = ctx.saved_tensors
        cp_group = ctx.cp_group
        num_heads = ctx.num_heads
        num_kv_heads = ctx.num_kv_heads
        head_dim = ctx.head_dim

        cp_size = cp_group.size()
        my_rank = cp_group.rank()
        device = hidden_states.device
        seq_local, batch_size, hidden_size = hidden_states.shape

        q_per_group = num_heads // num_kv_heads
        groups_per_rank = num_kv_heads // cp_size
        group_size = (q_per_group + 2) * head_dim
        heads_local = groups_per_rank * q_per_group
        seq_full = grad_q.shape[0]

        # Merge grad_q, grad_k, grad_v into interleaved format
        grad_q_grouped = grad_q.view(seq_full, batch_size, groups_per_rank, q_per_group * head_dim)
        grad_qkv = torch.cat([grad_q_grouped, grad_k, grad_v], dim=-1)

        # hp2sp AllToAll: change seq dimension from full to local
        grad_qkv_flat = grad_qkv.view(seq_full, batch_size, -1)

        # AllToAll: seq_full -> seq_local, collect all ranks' groups
        grad_qkv_parts = []
        for r in range(cp_size):
            r_seq_start = r * seq_local
            grad_qkv_parts.append(grad_qkv_flat[r_seq_start:r_seq_start + seq_local])

        # Use AllToAll to exchange
        grad_qkv_send = torch.stack(grad_qkv_parts, dim=0)
        grad_qkv_recv = torch.empty_like(grad_qkv_send)

        dist.all_to_all_single(
            grad_qkv_recv.view(cp_size, -1),
            grad_qkv_send.view(cp_size, -1),
            group=cp_group
        )

        # Reassemble to [seq_local, B, total_groups * group_size]
        grad_qkv_recv = grad_qkv_recv.permute(1, 2, 0, 3).contiguous()
        grad_qkv_sp = grad_qkv_recv.view(seq_local, batch_size, -1)

        # Compute grad_hidden
        weight_grouped = weight_qkv.view(num_kv_heads, group_size, hidden_size)
        grad_hidden = torch.zeros(seq_local, batch_size, hidden_size, dtype=hidden_states.dtype, device=device)

        for rank in range(cp_size):
            rank_group_start = rank * groups_per_rank
            weight_rank = weight_grouped[rank_group_start:rank_group_start + groups_per_rank]
            weight_rank = weight_rank.reshape(-1, hidden_size)

            grad_start = rank * groups_per_rank * group_size
            grad_end = grad_start + groups_per_rank * group_size
            grad_rank = grad_qkv_sp[:, :, grad_start:grad_end]

            grad_hidden += torch.matmul(grad_rank, weight_rank)

        # Register dW task
        from fluid.core.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        if scheduler.is_enabled():
            hidden_flat_saved = hidden_states.view(-1, hidden_size).detach()
            grad_qkv_sp_saved = grad_qkv_sp.view(-1, grad_qkv_sp.shape[-1]).detach()
            weight_qkv_saved = weight_qkv
            num_kv_heads_saved = num_kv_heads
            groups_per_rank_saved = groups_per_rank
            group_size_saved = group_size
            cp_size_saved = cp_size
            layer_id_saved = ctx.layer_id

            def compute_dw_qkv():
                grad_weight = torch.zeros_like(weight_qkv_saved)
                for rank in range(cp_size_saved):
                    rank_group_start = rank * groups_per_rank_saved
                    grad_start = rank * groups_per_rank_saved * group_size_saved
                    grad_end = grad_start + groups_per_rank_saved * group_size_saved
                    grad_rank = grad_qkv_sp_saved[:, grad_start:grad_end]

                    grad_weight_rank = torch.matmul(grad_rank.t(), hidden_flat_saved)
                    weight_start = rank_group_start * group_size_saved
                    weight_end = weight_start + groups_per_rank_saved * group_size_saved
                    grad_weight[weight_start:weight_end] = grad_weight_rank

                return grad_weight

            scheduler.register_dw_task(
                layer_name=f"qkv_multicard_layer{layer_id_saved}",
                layer_id=layer_id_saved,
                compute_fn=compute_dw_qkv,
                priority=100,
                weight_param=weight_qkv_saved,
            )
            grad_weight = None
        else:
            hidden_flat = hidden_states.view(-1, hidden_size)
            grad_qkv_sp_flat = grad_qkv_sp.view(-1, grad_qkv_sp.shape[-1])

            grad_weight = torch.zeros_like(weight_qkv)
            for rank in range(cp_size):
                rank_group_start = rank * groups_per_rank
                grad_start = rank * groups_per_rank * group_size
                grad_end = grad_start + groups_per_rank * group_size
                grad_rank = grad_qkv_sp_flat[:, grad_start:grad_end]

                grad_weight_rank = torch.matmul(grad_rank.t(), hidden_flat)
                weight_start = rank_group_start * group_size
                weight_end = weight_start + groups_per_rank * group_size
                grad_weight[weight_start:weight_end] = grad_weight_rank

        return (grad_hidden, grad_weight, None, None, None, None, None, None)


class _HP2SpOutputProjMultiCardFunction(torch.autograd.Function):
    """hp2sp + output projection multi-card P2P overlap autograd function

    Forward: each rank collects all heads data at my_seq position via P2P, computes output
    Backward: uses forward-saved attn_input_full to compute grad_weight
    """

    @staticmethod
    def forward(ctx, attn_output, weight_proj, bias_proj, cp_group, overlap_ctx):
        # Check if backward is needed (skip saving for inference)
        needs_grad = attn_output.requires_grad
        ctx.needs_grad = needs_grad

        ctx.cp_group = cp_group
        ctx.has_bias = bias_proj is not None

        # =====================================================================
        # hp2sp + output projection multi-card P2P overlap implementation
        # =====================================================================
        cp_size = cp_group.size()
        my_rank = cp_group.rank()
        device = attn_output.device
        dtype = attn_output.dtype

        if cp_size == 1:
            attn_flat = attn_output.view(attn_output.shape[0], attn_output.shape[1], -1)
            output = torch.matmul(attn_flat, weight_proj.t())
            if bias_proj is not None:
                output = output + bias_proj
            if needs_grad:
                ctx.save_for_backward(attn_flat, weight_proj)
                ctx.seq_full = attn_output.shape[0]
                ctx.heads_local = attn_output.shape[2]
                ctx.head_dim = attn_output.shape[3]
            return output

        seq_full, batch_size, heads_local, head_dim = attn_output.shape
        seq_local = seq_full // cp_size
        hidden_size = weight_proj.shape[0]
        input_dim_per_rank = heads_local * head_dim
        num_rounds = cp_size - 1

        default_stream = torch.cuda.current_stream(device)
        comm_stream = overlap_ctx.get_stream()

        # Local sequence start position
        local_seq_start = my_rank * seq_local

        # Weight for local heads
        weight_local_start = my_rank * input_dim_per_rank
        weight_local = weight_proj[:, weight_local_start:weight_local_start + input_dim_per_rank]

        # Prepare all data to send and receive buffers
        send_data_dict = {}
        recv_buffers = {}
        weight_per_partner = {}
        partners = []

        for round_idx in range(num_rounds):
            partner = overlap_ctx.get_partner(my_rank, round_idx)
            if partner == -1:
                continue
            partners.append(partner)

            # Prepare send data: attn_output at peer's sequence position (my heads data)
            partner_seq_start = partner * seq_local
            send_data_dict[partner] = attn_output[partner_seq_start:partner_seq_start + seq_local].contiguous()

            # Prepare receive buffer
            recv_buffers[partner] = torch.empty(seq_local, batch_size, heads_local, head_dim, dtype=dtype, device=device)

            # Cache partner's weight
            partner_weight_start = partner * input_dim_per_rank
            weight_per_partner[partner] = weight_proj[:, partner_weight_start:partner_weight_start + input_dim_per_rank]

        # =====================================================================
        # Pipeline overlap: each round's partial computation overlaps with next P2P
        # =====================================================================
        attn_local_seq = attn_output[local_seq_start:local_seq_start + seq_local]
        attn_local_flat = attn_local_seq.view(seq_local, batch_size, -1)

        prev_reqs = None
        prev_partner = None

        for round_idx, partner in enumerate(partners):
            # Start current round's P2P communication
            with torch.cuda.stream(comm_stream):
                p2p_ops = [
                    dist.P2POp(dist.irecv, recv_buffers[partner], partner, group=cp_group),
                    dist.P2POp(dist.isend, send_data_dict[partner], partner, group=cp_group),
                ]
                curr_reqs = dist.batch_isend_irecv(p2p_ops)

            # Parallel with P2P: compute previous round's received data's partial
            if round_idx == 0:
                # First round: compute local partial (parallel with P2P_0)
                output = torch.matmul(attn_local_flat, weight_local.t())
            else:
                # Wait for previous round P2P to complete
                for req in prev_reqs:
                    req.wait()
                # Compute previous round's received data's partial (parallel with current P2P)
                recv_flat = recv_buffers[prev_partner].view(seq_local, batch_size, -1)
                output = output + torch.matmul(recv_flat, weight_per_partner[prev_partner].t())

            prev_reqs = curr_reqs
            prev_partner = partner

        # Wait for last P2P to complete, compute last partial
        if prev_reqs is not None:
            for req in prev_reqs:
                req.wait()
            recv_flat = recv_buffers[prev_partner].view(seq_local, batch_size, -1)
            output = output + torch.matmul(recv_flat, weight_per_partner[prev_partner].t())

        if bias_proj is not None:
            output = output + bias_proj

        # Save for backward if needed
        if needs_grad:
            # Collect all heads' data at my_seq position
            attn_parts = {my_rank: attn_local_flat}
            for partner in partners:
                attn_parts[partner] = recv_buffers[partner].view(seq_local, batch_size, -1)
            # Concatenate in rank order
            attn_input_full = torch.cat([attn_parts[r] for r in range(cp_size)], dim=-1)

            ctx.save_for_backward(attn_input_full, weight_proj)
            ctx.seq_full = seq_full
            ctx.heads_local = heads_local
            ctx.head_dim = head_dim

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: dX + sp2hp AllToAll, optional chunked implementation"""
        import os
        attn_input_full, weight_proj = ctx.saved_tensors
        cp_group = ctx.cp_group
        has_bias = ctx.has_bias
        seq_full = ctx.seq_full
        heads_local = ctx.heads_local
        head_dim = ctx.head_dim

        cp_size = cp_group.size()
        my_rank = cp_group.rank()
        device = grad_output.device
        dtype = grad_output.dtype

        # attn_input_full: [seq_local, batch, all_heads * head_dim]
        seq_local, batch_size, total_input_dim = attn_input_full.shape
        hidden_size = weight_proj.shape[0]
        total_heads = total_input_dim // head_dim

        from fluid.core.scheduler import get_backward_scheduler
        from fluid.core.alltoall import _all_to_all_sp2hp_forward
        scheduler = get_backward_scheduler()

        # Check if using chunked backward
        use_chunked = os.environ.get('FLUID_USE_CHUNKED_BACKWARD', '0') == '1'
        num_chunks = int(os.environ.get('FLUID_CHUNKED_NUM_CHUNKS', '4'))

        # =====================================================================
        # Step 1: Compute dX + sp2hp AllToAll (optional chunked implementation)
        # =====================================================================
        if use_chunked and scheduler.is_enabled() and cp_size > 1:
            # Use chunked backward: dX computation overlaps with sp2hp AllToAll
            from fluid.attention.chunked_backward import backward_output_proj_chunked
            grad_attn_output = backward_output_proj_chunked(
                grad_output,
                weight_proj,
                total_heads,
                head_dim,
                cp_group,
                num_chunks=num_chunks,
                comm_stream=scheduler.comm_stream,
            )
        else:
            # Standard path: compute full dX then AllToAll
            grad_attn_flat = torch.matmul(grad_output, weight_proj)
            grad_attn_sp = grad_attn_flat.view(seq_local, batch_size, total_heads, head_dim)

            if cp_size > 1:
                # sp2hp AllToAll with dW overlap
                if scheduler.is_enabled():
                    comm_stream = scheduler.comm_stream
                    default_stream = torch.cuda.current_stream()
                    with torch.cuda.stream(comm_stream):
                        comm_stream.wait_stream(default_stream)
                        grad_attn_output = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
                        event = torch.cuda.Event()
                        event.record(comm_stream)
                        scheduler.set_alltoall_end_event(event)
                    scheduler.on_alltoall_start(comm_type="attn_proj_sp2hp")
                    default_stream.wait_stream(comm_stream)
                else:
                    grad_attn_output = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
            else:
                grad_attn_output = grad_attn_sp

        # =====================================================================
        # Step 2: Register dW task (using forward-saved attn_input_full)
        # =====================================================================
        if scheduler.is_enabled():
            attn_full_flat_saved = attn_input_full.reshape(seq_local * batch_size, -1).detach()
            grad_output_flat_saved = grad_output.reshape(seq_local * batch_size, hidden_size).detach()
            weight_proj_saved = weight_proj

            def compute_dw_proj():
                # dW = grad_output.T @ attn_input_full
                grad_weight = torch.matmul(grad_output_flat_saved.t(), attn_full_flat_saved)
                return grad_weight

            scheduler.register_dw_task(
                layer_name=f"proj_multicard_layer{my_rank}",
                layer_id=my_rank,
                compute_fn=compute_dw_proj,
                priority=99,
                weight_param=weight_proj_saved,
            )
            grad_weight = None
        else:
            # Compute full grad_weight directly
            attn_full_flat = attn_input_full.view(seq_local * batch_size, -1)
            grad_output_flat = grad_output.reshape(seq_local * batch_size, hidden_size)

            # dW = grad_output.T @ attn_input_full
            grad_weight = torch.matmul(grad_output_flat.t(), attn_full_flat)

        # Bias gradient
        if has_bias:
            grad_bias = grad_output.sum(dim=(0, 1))
        else:
            grad_bias = None

        return (grad_attn_output, grad_weight, grad_bias, None, None)


# =============================================================================
# Public API
# =============================================================================

def qkv_sp2hp_multicard_overlap(
    hidden_states: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    layer_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV computation + sp2hp multi-card P2P overlap

    Multi-card version of qkv_sp2hp_heads_split, supports any number of GPUs.

    Args:
        hidden_states: [seq_local, B, hidden]
        weight_qkv: [total_proj, hidden] full QKV weight
        num_heads: total Q heads
        num_kv_heads: total K/V heads (groups)
        head_dim: dimension per head
        cp_group: Context Parallel process group
        overlap_ctx: multi-card overlap context
        layer_id: layer ID (for dW task registration)

    Returns:
        q, k, v: [seq_full, B, heads_local, head_dim]
    """
    return _QKVSp2HpMultiCardFunction.apply(
        hidden_states, weight_qkv, num_heads, num_kv_heads, head_dim,
        cp_group, overlap_ctx, layer_id
    )


def hp2sp_output_proj_multicard_overlap(
    attn_output: torch.Tensor,
    weight_proj: torch.Tensor,
    bias_proj: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
) -> torch.Tensor:
    """hp2sp + output projection multi-card P2P overlap

    Multi-card version of hp2sp_output_proj_overlap, supports any number of GPUs.

    Args:
        attn_output: [seq_full, B, heads_local, head_dim]
        weight_proj: [hidden, total_heads * head_dim]
        bias_proj: [hidden] or None
        cp_group: Context Parallel process group
        overlap_ctx: multi-card overlap context

    Returns:
        output: [seq_local, B, hidden]
    """
    return _HP2SpOutputProjMultiCardFunction.apply(
        attn_output, weight_proj, bias_proj, cp_group, overlap_ctx
    )


__all__ = [
    # Context
    'AttentionMultiCardOverlapContext',
    # Public API with gradient support
    'qkv_sp2hp_multicard_overlap',
    'hp2sp_output_proj_multicard_overlap',
    # Autograd functions
    '_QKVSp2HpMultiCardFunction',
    '_HP2SpOutputProjMultiCardFunction',
]

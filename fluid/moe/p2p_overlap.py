"""
MoE P2P Forward Overlap Implementation

Uses Round-Robin Tournament scheduling for multi-card P2P communication overlap.

Design Principles:
1. Split "each rank exchanges data with all other ranks" into multiple P2P rounds
2. Each round, each card only communicates with one peer, avoiding conflicts
3. Communication stream runs round r while compute stream processes round r-1 data

Timeline (Dispatch + FC1):
    Round 0: P2P_0 communication || local FC1+Act computation
    Round 1: P2P_1 communication || P2P_0 data FC1+Act computation
    ...
    Round N: no communication || P2P_{N-1} data FC1+Act computation

Timeline (FC2 + Combine):
    Round 0: FC2(peer_0) computation -> P2P_0 communication
    Round 1: P2P_0 done || FC2(peer_1) computation -> P2P_1 communication
    ...
    Round N: P2P_{N-1} done || local FC2 computation
"""

import os
import time
import torch
import torch.distributed as dist
from typing import List, Tuple, Dict, Optional

from fluid.core.forward_comm import (
    MultiCardOverlapContext,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_fc1_act_per_source(
    tokens: torch.Tensor,
    w1: torch.Tensor,
    activation_func,
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    source_rank: int,
) -> torch.Tensor:
    """
    Compute FC1+activation for tokens from a single source rank (without FC2)

    Args:
        tokens: [num_tokens, hidden] tokens from source_rank
        w1: [num_local_experts, hidden, ffn_hidden]
        activation_func: activation function
        num_local_experts: number of local experts
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]
        source_rank: source rank of data

    Returns:
        act_output: [num_tokens, ffn_hidden]
    """
    device = tokens.device
    ffn_hidden = w1.shape[-1]

    if tokens.numel() == 0:
        return torch.empty(0, ffn_hidden, dtype=tokens.dtype, device=device)

    if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
        act_output = torch.zeros(tokens.shape[0], ffn_hidden, dtype=tokens.dtype, device=device)

        offset = 0
        for exp_idx in range(num_local_experts):
            n_tok = num_global_tokens_per_local_expert[0, source_rank, exp_idx].item()
            if n_tok > 0:
                exp_tokens = tokens[offset:offset + n_tok]
                exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                act_output[offset:offset + n_tok] = activation_func(exp_fc1)
                offset += n_tok

        return act_output
    else:
        fc1 = torch.matmul(tokens, w1[0])
        return activation_func(fc1)


def _compute_fc2_per_source(
    act: torch.Tensor,
    w2: torch.Tensor,
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    source_rank: int,
) -> torch.Tensor:
    """
    Compute FC2 for activation values from a single source rank

    Args:
        act: [num_tokens, ffn_hidden] activation values
        w2: [num_local_experts, ffn_hidden, hidden]
        num_local_experts: number of local experts
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]
        source_rank: source rank of data

    Returns:
        fc2_output: [num_tokens, hidden]
    """
    device = act.device
    hidden_size = w2.shape[-1]

    if act.numel() == 0:
        return torch.empty(0, hidden_size, dtype=act.dtype, device=device)

    if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
        fc2_output = torch.zeros(act.shape[0], hidden_size, dtype=act.dtype, device=device)

        offset = 0
        for exp_idx in range(num_local_experts):
            n_tok = num_global_tokens_per_local_expert[0, source_rank, exp_idx].item()
            if n_tok > 0:
                exp_act = act[offset:offset + n_tok]
                fc2_output[offset:offset + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                offset += n_tok

        return fc2_output
    else:
        return torch.matmul(act, w2[0])


def _merge_tokens_and_fc1_expert_major(
    local_tokens: torch.Tensor,
    all_peer_tokens: torch.Tensor,
    local_fc1: torch.Tensor,
    all_peer_fc1: torch.Tensor,
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    my_rank: int,
    ep_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Merge local and peer tokens and fc1 into expert-major order simultaneously
    (Compute offsets once, rearrange both tensors)

    Args:
        local_tokens: [local_count, hidden_size] local tokens (in expert order)
        all_peer_tokens: [peer_count, hidden_size] all peer tokens (cat in rank order)
        local_fc1: [local_count, ffn_hidden] local fc1 results
        all_peer_fc1: [peer_count, ffn_hidden] all peer fc1 results (cat in rank order)

    Returns:
        all_expert_tokens: [total, hidden_size] expert-major order
        all_fc1: [total, ffn_hidden] expert-major order
        all_tokens_per_expert: [num_local_experts] token count per expert
    """
    hidden_size = local_tokens.shape[-1] if local_tokens.numel() > 0 else all_peer_tokens.shape[-1]
    ffn_hidden = local_fc1.shape[-1] if local_fc1 is not None and local_fc1.numel() > 0 else all_peer_fc1.shape[-1]
    dtype = local_tokens.dtype if local_tokens.numel() > 0 else all_peer_tokens.dtype

    # Compute total token count per expert
    all_tokens_per_expert = []
    for exp_idx in range(num_local_experts):
        total = 0
        for rank in range(ep_size):
            total += num_global_tokens_per_local_expert[0, rank, exp_idx].item()
        all_tokens_per_expert.append(total)

    total_tokens = sum(all_tokens_per_expert)

    if total_tokens == 0:
        return (torch.empty(0, hidden_size, dtype=dtype, device=device),
                torch.empty(0, ffn_hidden, dtype=dtype, device=device),
                all_tokens_per_expert)

    all_expert_tokens = torch.zeros(total_tokens, hidden_size, dtype=dtype, device=device)
    all_fc1 = torch.zeros(total_tokens, ffn_hidden, dtype=dtype, device=device)

    # Precompute start offset for each rank in all_peer_tokens/all_peer_fc1
    peer_rank_offsets = {}
    offset = 0
    for rank in range(ep_size):
        if rank == my_rank:
            continue
        peer_rank_offsets[rank] = offset
        for exp_idx in range(num_local_experts):
            offset += num_global_tokens_per_local_expert[0, rank, exp_idx].item()

    # Fill in expert-major order (process both tokens and fc1)
    write_offset = 0
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            n_tok = num_global_tokens_per_local_expert[0, rank, exp_idx].item()
            if n_tok == 0:
                continue

            if rank == my_rank:
                # Extract from local (local is in expert order)
                local_exp_offset = sum(
                    num_global_tokens_per_local_expert[0, my_rank, e].item()
                    for e in range(exp_idx)
                )
                all_expert_tokens[write_offset:write_offset + n_tok] = \
                    local_tokens[local_exp_offset:local_exp_offset + n_tok]
                if local_fc1 is not None and local_fc1.numel() > 0:
                    all_fc1[write_offset:write_offset + n_tok] = \
                        local_fc1[local_exp_offset:local_exp_offset + n_tok]
            else:
                # Extract from peer (peer is cat in rank order, each rank in expert order)
                peer_base = peer_rank_offsets[rank]
                peer_exp_offset = sum(
                    num_global_tokens_per_local_expert[0, rank, e].item()
                    for e in range(exp_idx)
                )
                src_offset = peer_base + peer_exp_offset
                all_expert_tokens[write_offset:write_offset + n_tok] = \
                    all_peer_tokens[src_offset:src_offset + n_tok]
                if all_peer_fc1.numel() > 0:
                    all_fc1[write_offset:write_offset + n_tok] = \
                        all_peer_fc1[src_offset:src_offset + n_tok]

            write_offset += n_tok

    return all_expert_tokens, all_fc1, all_tokens_per_expert


def _precompute_backward_sort_indices(ctx, num_local_experts, ep_size,
                                       num_global_tokens_per_local_expert, device):
    """Precompute sort indices needed for backward"""
    # rank-major chunk sizes: [R0_E0, R0_E1, R1_E0, R1_E1, ...]
    split_sizes_rank_major = []
    for rank in range(ep_size):
        for exp_idx in range(num_local_experts):
            split_sizes_rank_major.append(num_global_tokens_per_local_expert[0, rank, exp_idx].item())

    # rank-major -> expert-major indices
    sorted_idxs_rank_to_exp = []
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            sorted_idxs_rank_to_exp.append(rank * num_local_experts + exp_idx)

    # expert-major chunk sizes: [E0_R0, E0_R1, ..., E1_R0, E1_R1, ...]
    split_sizes_exp_major = []
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            split_sizes_exp_major.append(num_global_tokens_per_local_expert[0, rank, exp_idx].item())

    # expert-major -> rank-major indices
    sorted_idxs_exp_to_rank = []
    for rank in range(ep_size):
        for exp_idx in range(num_local_experts):
            sorted_idxs_exp_to_rank.append(exp_idx * ep_size + rank)

    ctx.split_sizes_rank_major = torch.tensor(split_sizes_rank_major, dtype=torch.int64, device=device)
    ctx.sorted_idxs_rank_to_exp = torch.tensor(sorted_idxs_rank_to_exp, dtype=torch.int64, device=device)
    ctx.split_sizes_exp_major = torch.tensor(split_sizes_exp_major, dtype=torch.int64, device=device)
    ctx.sorted_idxs_exp_to_rank = torch.tensor(sorted_idxs_exp_to_rank, dtype=torch.int64, device=device)


# =============================================================================
# Autograd Function
# =============================================================================

class _MoEMultiCardP2POverlapFunction(torch.autograd.Function):
    """
    MoE Multi-card P2P Overlap Autograd Wrapper

    Forward: Uses multi-round P2P overlap (dispatch and combine)
    Backward: Uses standard AllToAll (keeps backward scheduling unchanged)

    Key Design (Two Phases):

    Phase 1: Dispatch + FC1 Overlap
    -----------------------------------------
    - Local FC1+Act overlaps with first P2P round
    - Round r P2P overlaps with Round r-1 FC1+Act
    - Only compute FC1 and activation, not FC2

    Phase 2: FC2 + Combine Overlap (after all Dispatch and FC1 complete)
    -----------------------------------------
    - First compute remote data FC2 (overlap with Combine P2P)
    - Then compute local data FC2
    - Save rearranged activation values for backward after local FC2

    Data Concatenation:
    - Each device processes data according to Round-Robin partner order
    - combined_output arranged in original token order
    """

    @staticmethod
    def forward(ctx, tokens, input_splits, output_splits, weight1, weight2,
                ep_group, activation_func, overlap_ctx, layer_id,
                num_local_experts=1, tokens_per_expert=None,
                num_global_tokens_per_local_expert=None):
        """
        Multi-card pipeline overlap implementation:

        Dispatch Phase (communication -> computation) pipeline:
        -----------------------------------------
        Round 0: Start P2P_0, compute local FC1 + Act
        Round i: req.wait(P2P_{i-1}), start P2P_i, compute recv_{i-1} FC1 + Act
        Final:   req.wait(last round), compute last FC1 + Act

        Combine Phase (computation -> communication) pipeline:
        -----------------------------------------
        Round -1: Compute first peer's FC2
        Round i:  event.synchronize(), start P2P_i, compute next peer's FC2
        Final:    Compute local FC2 (parallel with last P2P)
        """
        ctx.ep_group = ep_group
        ctx.activation_func = activation_func
        ctx.layer_id = layer_id
        ctx.num_local_experts = num_local_experts

        # Save original weight references (for backward gradient computation)
        orig_weight1 = weight1
        orig_weight2 = weight2

        my_rank = ep_group.rank()
        ep_size = ep_group.size()
        device = tokens.device
        hidden_size = tokens.shape[-1]
        dtype = tokens.dtype

        # Weight dimensions
        total_ffn_hidden = weight1.shape[-1]
        ffn_hidden = total_ffn_hidden // num_local_experts

        w1 = weight1.view(num_local_experts, hidden_size, ffn_hidden)
        w2 = weight2.view(num_local_experts, ffn_hidden, hidden_size)

        input_splits_list = input_splits.tolist() if torch.is_tensor(input_splits) else list(input_splits)
        output_splits_list = output_splits.tolist() if torch.is_tensor(output_splits) else list(output_splits)

        ctx.input_splits_list = input_splits_list
        ctx.output_splits_list = output_splits_list
        ctx.my_rank = my_rank
        ctx.ffn_hidden = ffn_hidden

        # Compute offsets
        input_offsets = [0]
        for s in input_splits_list:
            input_offsets.append(input_offsets[-1] + s)

        output_offsets = [0]
        for s in output_splits_list:
            output_offsets.append(output_offsets[-1] + s)

        default_stream = torch.cuda.current_stream(device)
        comm_stream = overlap_ctx.get_stream()

        local_count = input_splits_list[my_rank]
        local_start = input_offsets[my_rank]
        ctx.local_count = local_count
        ctx.local_start = local_start

        # Compute local token count per expert (CPU operation, prepare ahead)
        local_tokens_per_expert = None
        if num_global_tokens_per_local_expert is not None:
            local_tokens_per_expert = [
                num_global_tokens_per_local_expert[0, my_rank, exp_idx].item()
                for exp_idx in range(num_local_experts)
            ]

        # Get Round-Robin scheduled partners
        partners = []
        for round_idx in range(overlap_ctx.num_rounds):
            partner = overlap_ctx.get_partner(my_rank, round_idx)
            if partner != -1:
                partners.append(partner)

        # Extract local tokens
        local_tokens = tokens[local_start:local_start + local_count].clone() if local_count > 0 else torch.empty(0, hidden_size, dtype=dtype, device=device)

        # Prepare send data (token slices, nearly instant)
        send_chunks = {}
        for partner in partners:
            if input_splits_list[partner] > 0:
                send_chunks[partner] = tokens[input_offsets[partner]:input_offsets[partner+1]].contiguous()

        # Prepare receive buffers (by partner order)
        recv_buffers = {}
        for partner in partners:
            recv_size = output_splits_list[partner]
            if recv_size > 0:
                recv_buffers[partner] = torch.empty(recv_size, hidden_size, dtype=dtype, device=device)

        # =========================================================================
        # Pre-allocate buffers and precompute indices (before Dispatch P2P loop)
        # =========================================================================

        # 优化1: 预分配 FC1/Act buffers 避免每轮 torch.zeros()
        total_recv = sum(output_splits_list)
        all_recv_fc1_buffer = torch.empty(total_recv, ffn_hidden, dtype=dtype, device=device) if total_recv > 0 else None
        all_recv_act_buffer = torch.empty(total_recv, ffn_hidden, dtype=dtype, device=device) if total_recv > 0 else None

        # 计算每个 partner 在 buffer 中的偏移
        recv_offsets = {}
        offset = 0
        for i in range(ep_size):
            if i != my_rank:
                recv_offsets[i] = offset
                offset += output_splits_list[i]

        # =========================================================================
        # Dispatch Phase Pipeline (communication -> computation)
        # =========================================================================
        # Round 0: Start P2P_0, compute local FC1 + Act
        # Round i: req.wait(P2P_{i-1}), start P2P_i, compute recv_{i-1} FC1 + Act
        # Note: act_deriv computed uniformly when saving activations, not here

        prev_reqs = None
        prev_partner = None
        recv_act_results = {}  # Store Act results for each partner (views into buffer)
        recv_fc1_results = {}  # Store FC1 results for each partner (views into buffer)
        local_fc1_saved = None  # Save local FC1 results

        # Detach weights before loop (very fast ~15us, cleaner code)
        weight1_detached = orig_weight1.detach()
        weight2_detached = orig_weight2.detach()

        for round_idx, partner in enumerate(partners):
            # Start current round P2P
            with torch.cuda.stream(comm_stream):
                p2p_ops = []
                if partner in recv_buffers:
                    p2p_ops.append(dist.P2POp(dist.irecv, recv_buffers[partner], partner, group=ep_group))
                if partner in send_chunks:
                    p2p_ops.append(dist.P2POp(dist.isend, send_chunks[partner], partner, group=ep_group))
                curr_reqs = dist.batch_isend_irecv(p2p_ops) if p2p_ops else []

            # Parallel with P2P: compute FC1 + Act, save FC1 for backward
            if round_idx == 0:
                # First round: compute local FC1 + Act (parallel with P2P_0)
                if local_count > 0 and local_tokens_per_expert is not None:
                    local_act = torch.zeros(local_count, ffn_hidden, dtype=dtype, device=device)
                    local_fc1_saved = torch.zeros(local_count, ffn_hidden, dtype=dtype, device=device)
                    start = 0
                    for exp_idx in range(num_local_experts):
                        n_tok = local_tokens_per_expert[exp_idx]
                        if n_tok > 0:
                            exp_tokens = local_tokens[start:start + n_tok]
                            exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                            local_fc1_saved[start:start + n_tok] = exp_fc1
                            local_act[start:start + n_tok] = activation_func(exp_fc1)
                            start += n_tok
                elif local_count > 0:
                    local_fc1_saved = torch.matmul(local_tokens, w1[0])
                    local_act = activation_func(local_fc1_saved)
                else:
                    local_act = None
                    local_fc1_saved = None
            else:
                # Wait for previous round P2P
                for req in prev_reqs:
                    req.wait()
                # Compute previous round received data FC1 + Act (parallel with current P2P)
                if prev_partner in recv_buffers:
                    recv_data = recv_buffers[prev_partner]
                    recv_count = recv_data.shape[0]
                    buf_offset = recv_offsets[prev_partner]
                    # 使用预分配 buffer 的 view，避免新分配
                    recv_fc1 = all_recv_fc1_buffer[buf_offset:buf_offset + recv_count]
                    recv_act = all_recv_act_buffer[buf_offset:buf_offset + recv_count]
                    if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
                        offset = 0
                        for exp_idx in range(num_local_experts):
                            n_tok = num_global_tokens_per_local_expert[0, prev_partner, exp_idx].item()
                            if n_tok > 0:
                                exp_tokens = recv_data[offset:offset + n_tok]
                                exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                                recv_fc1[offset:offset + n_tok] = exp_fc1
                                recv_act[offset:offset + n_tok] = activation_func(exp_fc1)
                                offset += n_tok
                    else:
                        recv_fc1.copy_(torch.matmul(recv_data, w1[0]))
                        recv_act.copy_(activation_func(recv_fc1))
                    recv_act_results[prev_partner] = recv_act
                    recv_fc1_results[prev_partner] = recv_fc1

            prev_reqs = curr_reqs
            prev_partner = partner

        # Wait for last round P2P, compute last FC1 + Act
        if prev_reqs is not None:
            for req in prev_reqs:
                req.wait()
            if prev_partner in recv_buffers:
                recv_data = recv_buffers[prev_partner]
                recv_count = recv_data.shape[0]
                buf_offset = recv_offsets[prev_partner]
                # 使用预分配 buffer 的 view，避免新分配
                recv_fc1 = all_recv_fc1_buffer[buf_offset:buf_offset + recv_count]
                recv_act = all_recv_act_buffer[buf_offset:buf_offset + recv_count]
                if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
                    offset = 0
                    for exp_idx in range(num_local_experts):
                        n_tok = num_global_tokens_per_local_expert[0, prev_partner, exp_idx].item()
                        if n_tok > 0:
                            exp_tokens = recv_data[offset:offset + n_tok]
                            exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                            recv_fc1[offset:offset + n_tok] = exp_fc1
                            recv_act[offset:offset + n_tok] = activation_func(exp_fc1)
                            offset += n_tok
                else:
                    recv_fc1.copy_(torch.matmul(recv_data, w1[0]))
                    recv_act.copy_(activation_func(recv_fc1))
                recv_act_results[prev_partner] = recv_act
                recv_fc1_results[prev_partner] = recv_fc1

        # =========================================================================
        # Combine Phase Pipeline (computation -> communication)
        # =========================================================================
        # Round -1: Compute first peer's FC2
        # Round i:  event.synchronize(), start P2P_i, compute next peer's FC2
        # Final:    Compute local FC2 (parallel with last P2P)

        total_output = sum(input_splits_list)
        combined_output = torch.empty(total_output, hidden_size, dtype=dtype, device=device)

        peer_fc2_results = {}
        fc2_events = {}

        # Round -1: Pre-compute first peer's FC2
        if len(partners) > 0:
            first_partner = partners[0]
            if first_partner in recv_act_results:
                recv_act = recv_act_results[first_partner]
                if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
                    peer_fc2 = torch.zeros(recv_act.shape[0], hidden_size, dtype=dtype, device=device)
                    offset = 0
                    for exp_idx in range(num_local_experts):
                        n_tok = num_global_tokens_per_local_expert[0, first_partner, exp_idx].item()
                        if n_tok > 0:
                            exp_act = recv_act[offset:offset + n_tok]
                            peer_fc2[offset:offset + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                            offset += n_tok
                else:
                    peer_fc2 = torch.matmul(recv_act, w2[0])
                peer_fc2_results[first_partner] = peer_fc2
                fc2_events[first_partner] = torch.cuda.Event()
                fc2_events[first_partner].record(default_stream)

        # Pipeline loop
        all_combine_reqs = []
        for round_idx, partner in enumerate(partners):
            # CPU wait for FC2 computation done
            if partner in fc2_events:
                fc2_events[partner].synchronize()

            # Start P2P (send FC2 result to partner, receive from partner)
            with torch.cuda.stream(comm_stream):
                p2p_ops = []
                # Receive: FC2 result from partner
                recv_size = input_splits_list[partner]
                if recv_size > 0:
                    recv_chunk = combined_output[input_offsets[partner]:input_offsets[partner+1]]
                    p2p_ops.append(dist.P2POp(dist.irecv, recv_chunk, partner, group=ep_group))
                # Send: my FC2 result to partner
                if partner in peer_fc2_results:
                    p2p_ops.append(dist.P2POp(dist.isend, peer_fc2_results[partner], partner, group=ep_group))
                reqs = dist.batch_isend_irecv(p2p_ops) if p2p_ops else []
                all_combine_reqs.extend(reqs)

            # Parallel with current P2P: compute next round FC2 or local FC2
            if round_idx + 1 < len(partners):
                next_partner = partners[round_idx + 1]
                if next_partner in recv_act_results:
                    recv_act = recv_act_results[next_partner]
                    if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
                        peer_fc2 = torch.zeros(recv_act.shape[0], hidden_size, dtype=dtype, device=device)
                        offset = 0
                        for exp_idx in range(num_local_experts):
                            n_tok = num_global_tokens_per_local_expert[0, next_partner, exp_idx].item()
                            if n_tok > 0:
                                exp_act = recv_act[offset:offset + n_tok]
                                peer_fc2[offset:offset + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                                offset += n_tok
                    else:
                        peer_fc2 = torch.matmul(recv_act, w2[0])
                    peer_fc2_results[next_partner] = peer_fc2
                    fc2_events[next_partner] = torch.cuda.Event()
                    fc2_events[next_partner].record(default_stream)
            else:
                # Last round: compute local FC2 (parallel with last P2P)
                if local_act is not None and local_tokens_per_expert is not None:
                    local_fc2 = torch.zeros(local_count, hidden_size, dtype=dtype, device=device)
                    start = 0
                    for exp_idx in range(num_local_experts):
                        n_tok = local_tokens_per_expert[exp_idx]
                        if n_tok > 0:
                            exp_act = local_act[start:start + n_tok]
                            local_fc2[start:start + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                            start += n_tok
                elif local_act is not None:
                    local_fc2 = torch.matmul(local_act, w2[0])
                else:
                    local_fc2 = None

        # =========================================================================
        # Save for backward: merge and rearrange activations (parallel with Combine P2P)
        # =========================================================================

        # 优化: 使用预分配的 buffer，避免 torch.cat 合并
        # all_recv_fc1_buffer 已经按 rank 顺序排列
        if total_recv > 0:
            # 合并 recv_buffers (按 rank 顺序)
            all_peer_tokens_list = []
            for i in range(ep_size):
                if i == my_rank:
                    continue
                if i in recv_buffers:
                    all_peer_tokens_list.append(recv_buffers[i])
            all_peer_tokens = torch.cat(all_peer_tokens_list, dim=0) if all_peer_tokens_list else torch.empty(0, hidden_size, dtype=dtype, device=device)
            # FC1 buffer 已经按 rank 顺序预分配，直接使用
            all_peer_fc1 = all_recv_fc1_buffer
        else:
            all_peer_tokens = torch.empty(0, hidden_size, dtype=dtype, device=device)
            all_peer_fc1 = torch.empty(0, ffn_hidden, dtype=dtype, device=device)

        # Merge local and peer tokens/fc1 for backward (expert-major order)
        if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
            # Use _merge_tokens_and_fc1_expert_major to rearrange both tokens and fc1
            all_expert_tokens, all_fc1, all_tokens_per_expert = _merge_tokens_and_fc1_expert_major(
                local_tokens, all_peer_tokens,
                local_fc1_saved, all_peer_fc1,
                num_local_experts, num_global_tokens_per_local_expert,
                my_rank, ep_size, device
            )
        else:
            all_expert_tokens = torch.cat([local_tokens, all_peer_tokens], dim=0) if all_peer_tokens.numel() > 0 else local_tokens
            all_tokens_per_expert = [all_expert_tokens.shape[0]]
            # Merge fc1 results (single expert doesn't need rearranging)
            if all_peer_fc1.numel() > 0:
                if local_fc1_saved is not None:
                    all_fc1 = torch.cat([local_fc1_saved, all_peer_fc1], dim=0)
                else:
                    all_fc1 = all_peer_fc1
            else:
                all_fc1 = local_fc1_saved if local_fc1_saved is not None else torch.empty(0, ffn_hidden, dtype=dtype, device=device)

        # 只在需要 backward 时才保存中间结果（inference 时跳过）
        needs_grad = tokens.requires_grad
        ctx.needs_grad = needs_grad

        if needs_grad:
            # 优化: 预计算 backward sort indices (与 Combine P2P req.wait() 重叠)
            if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
                _precompute_backward_sort_indices(ctx, num_local_experts, ep_size,
                                                  num_global_tokens_per_local_expert, device)

            # 优化: ctx 赋值移到 req.wait() 前 (与 Combine P2P 通信重叠)
            ctx._all_expert_tokens = all_expert_tokens
            ctx._weight1 = weight1_detached
            ctx._weight2 = weight2_detached
            ctx._orig_weight1 = orig_weight1
            ctx._orig_weight2 = orig_weight2
            ctx._all_fc1 = all_fc1
            ctx.all_tokens_per_expert = all_tokens_per_expert
            ctx.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert

        # Wait for all Combine P2P to complete
        for req in all_combine_reqs:
            req.wait()

        # Handle no partners case (ep_size=1): compute local FC2
        if len(partners) == 0:
            if local_act is not None and local_tokens_per_expert is not None:
                local_fc2 = torch.zeros(local_count, hidden_size, dtype=dtype, device=device)
                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = local_tokens_per_expert[exp_idx]
                    if n_tok > 0:
                        exp_act = local_act[start:start + n_tok]
                        local_fc2[start:start + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                        start += n_tok
            elif local_act is not None:
                local_fc2 = torch.matmul(local_act, w2[0])
            else:
                local_fc2 = None

        # Write local result to combined_output
        if local_fc2 is not None:
            combined_output[local_start:local_start + local_count] = local_fc2

        # Note: ctx 赋值已移到 req.wait() 前 (Lines 1020-1028) 与 Combine P2P 通信重叠
        # _precompute_backward_sort_indices 也已在前面提前执行

        return combined_output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward with scheduler-based dW-AllToAll overlap.

        Uses same pattern as baseline MoE:
        1. Launch AllToAll on comm_stream (async)
        2. Call on_alltoall_start() to execute dW tasks during AllToAll
        3. Register current layer's dW tasks for execution during later AllToAll
        """
        if not ctx.needs_grad:
            return (None,) * 12

        DEBUG_TIMING = os.environ.get('FLUID_DEBUG_MoE_BACKWARD', '0') == '1'
        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t_start = time.perf_counter()

        from fluid.core.utils import _compute_activation_grad, _compute_activation_derivative
        from fluid.core.alltoall import _all_to_all, _sort_chunks_by_idxs
        from fluid.core.scheduler import get_backward_scheduler

        all_expert_tokens = ctx._all_expert_tokens
        weight1 = ctx._weight1
        weight2 = ctx._weight2
        all_fc1 = ctx._all_fc1

        ep_group = ctx.ep_group
        activation_func = ctx.activation_func
        input_splits_list = ctx.input_splits_list
        output_splits_list = ctx.output_splits_list
        num_local_experts = ctx.num_local_experts
        ffn_hidden = ctx.ffn_hidden
        all_tokens_per_expert = ctx.all_tokens_per_expert
        layer_id = ctx.layer_id

        device = grad_output.device
        hidden_size = grad_output.shape[-1]

        w1 = weight1.view(num_local_experts, hidden_size, ffn_hidden)
        w2 = weight2.view(num_local_experts, ffn_hidden, hidden_size)

        total_all_tokens = all_expert_tokens.shape[0]

        # Compute act and act_deriv from saved fc1
        act_output = activation_func(all_fc1)
        act_deriv = _compute_activation_derivative(all_fc1, activation_func, gated_linear_unit=False)

        # Get scheduler for dW-AllToAll overlap
        scheduler = get_backward_scheduler()

        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t_setup = time.perf_counter()
            print(f"[MoE backward] Setup: {(t_setup - t_start)*1000:.2f}ms", flush=True)

        # =========================================================================
        # Combine Backward AllToAll with dW overlap
        # =========================================================================
        if scheduler.is_enabled():
            comm_stream = scheduler.comm_stream
            default_stream = scheduler.default_stream
            # Launch AllToAll on comm_stream (async)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_combined = _all_to_all(
                    grad_output.contiguous(),
                    output_split_sizes=output_splits_list,
                    input_split_sizes=input_splits_list,
                    group=ep_group
                )
                scheduler.record_alltoall_end(comm_stream)
            # Execute dW tasks from queue while AllToAll is running
            scheduler.on_alltoall_start(comm_type=f"moe_combine_L{layer_id}")
            # Wait for AllToAll to complete
            default_stream.wait_stream(comm_stream)
        else:
            grad_combined = _all_to_all(
                grad_output.contiguous(),
                output_split_sizes=output_splits_list,
                input_split_sizes=input_splits_list,
                group=ep_group
            )

        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t_alltoall1 = time.perf_counter()
            print(f"[MoE backward] AllToAll1: {(t_alltoall1 - t_setup)*1000:.2f}ms", flush=True)

        # Convert layout: rank-major -> expert-major using precomputed indices
        if hasattr(ctx, 'split_sizes_rank_major'):
            grad_all_fc2 = _sort_chunks_by_idxs(
                grad_combined,
                ctx.split_sizes_rank_major,
                ctx.sorted_idxs_rank_to_exp,
            )
        else:
            grad_all_fc2 = grad_combined

        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t_sort1 = time.perf_counter()
            print(f"[MoE backward] Sort1: {(t_sort1 - t_alltoall1)*1000:.2f}ms", flush=True)

        # Check if using chunked backward for dX + AllToAll overlap
        use_chunked = os.environ.get('FLUID_USE_CHUNKED_BACKWARD', '0') == '1'

        # Compute grad_tokens and grad_fc1 (dX - critical path)
        # When using chunked backward, we only compute grad_all_fc1 here
        # and let chunked backward compute dX + AllToAll with overlap
        grad_all_fc1 = torch.zeros(total_all_tokens, ffn_hidden, dtype=grad_output.dtype, device=device)

        if use_chunked:
            # Only compute grad_fc1, skip dX (will be done in chunked backward)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = all_tokens_per_expert[exp_idx]
                if n_tok > 0:
                    grad_exp_act = torch.matmul(grad_all_fc2[start:start+n_tok], w2[exp_idx].t())
                    grad_exp_fc1 = _compute_activation_grad(
                        grad_exp_act, act_deriv[start:start+n_tok], gated_linear_unit=False
                    )
                    grad_all_fc1[start:start+n_tok] = grad_exp_fc1
                    start += n_tok
            grad_all_tokens = None  # Will be computed by chunked backward
        else:
            # Compute both grad_fc1 and grad_tokens (dX)
            grad_all_tokens = torch.zeros(total_all_tokens, hidden_size, dtype=grad_output.dtype, device=device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = all_tokens_per_expert[exp_idx]
                if n_tok > 0:
                    grad_exp_act = torch.matmul(grad_all_fc2[start:start+n_tok], w2[exp_idx].t())
                    grad_exp_fc1 = _compute_activation_grad(
                        grad_exp_act, act_deriv[start:start+n_tok], gated_linear_unit=False
                    )
                    grad_all_tokens[start:start+n_tok] = torch.matmul(grad_exp_fc1, w1[exp_idx].t())
                    grad_all_fc1[start:start+n_tok] = grad_exp_fc1
                    start += n_tok

        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t_dx = time.perf_counter()
            print(f"[MoE backward] dX: {(t_dx - t_sort1)*1000:.2f}ms", flush=True)

        # =========================================================================
        # Compute or Register dW tasks
        # =========================================================================
        orig_weight1 = ctx._orig_weight1
        orig_weight2 = ctx._orig_weight2

        if scheduler.is_enabled():
            # Scheduler enabled: register dW tasks for execution during later AllToAll
            num_local_experts_saved = num_local_experts
            all_tokens_per_expert_saved = all_tokens_per_expert
            grad_all_fc2_saved = grad_all_fc2.detach()
            grad_all_fc1_saved = grad_all_fc1.detach()
            act_output_saved = act_output.detach()
            all_expert_tokens_saved = all_expert_tokens.detach()

            def compute_dw_weight2():
                grad_w2 = torch.zeros_like(weight2)
                grad_w2_view = grad_w2.view(num_local_experts_saved, ffn_hidden, hidden_size)
                start = 0
                for exp_idx in range(num_local_experts_saved):
                    n_tok = all_tokens_per_expert_saved[exp_idx]
                    if n_tok > 0:
                        grad_w2_view[exp_idx] = torch.matmul(
                            act_output_saved[start:start+n_tok].t(),
                            grad_all_fc2_saved[start:start+n_tok]
                        )
                        start += n_tok
                return grad_w2

            def compute_dw_weight1():
                grad_w1 = torch.zeros_like(weight1)
                grad_w1_view = grad_w1.view(num_local_experts_saved, hidden_size, ffn_hidden)
                start = 0
                for exp_idx in range(num_local_experts_saved):
                    n_tok = all_tokens_per_expert_saved[exp_idx]
                    if n_tok > 0:
                        grad_w1_view[exp_idx] = torch.matmul(
                            all_expert_tokens_saved[start:start+n_tok].t(),
                            grad_all_fc1_saved[start:start+n_tok]
                        )
                        start += n_tok
                return grad_w1

            scheduler.register_dw_task(
                layer_name=f"moe_weight2_L{layer_id}",
                layer_id=layer_id,
                compute_fn=compute_dw_weight2,
                priority=100,
                weight_param=orig_weight2,
            )
            scheduler.register_dw_task(
                layer_name=f"moe_weight1_L{layer_id}",
                layer_id=layer_id,
                compute_fn=compute_dw_weight1,
                priority=99,
                weight_param=orig_weight1,
            )
            grad_w1 = None
            grad_w2 = None
        else:
            # Scheduler disabled: compute dW directly
            grad_w2 = torch.zeros_like(weight2)
            grad_w2_view = grad_w2.view(num_local_experts, ffn_hidden, hidden_size)
            grad_w1 = torch.zeros_like(weight1)
            grad_w1_view = grad_w1.view(num_local_experts, hidden_size, ffn_hidden)

            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = all_tokens_per_expert[exp_idx]
                if n_tok > 0:
                    grad_w2_view[exp_idx] = torch.matmul(
                        act_output[start:start+n_tok].t(),
                        grad_all_fc2[start:start+n_tok]
                    )
                    grad_w1_view[exp_idx] = torch.matmul(
                        all_expert_tokens[start:start+n_tok].t(),
                        grad_all_fc1[start:start+n_tok]
                    )
                    start += n_tok

        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t_dw = time.perf_counter()
            print(f"[MoE backward] dW: {(t_dw - t_dx)*1000:.2f}ms", flush=True)

        # Convert layout: expert-major -> rank-major using precomputed indices
        # Skip this when using chunked backward (it does sort internally)
        if not use_chunked:
            if hasattr(ctx, 'split_sizes_exp_major'):
                grad_dispatched = _sort_chunks_by_idxs(
                    grad_all_tokens,
                    ctx.split_sizes_exp_major,
                    ctx.sorted_idxs_exp_to_rank,
                )
            else:
                grad_dispatched = grad_all_tokens
        else:
            grad_dispatched = None  # Not used when chunked backward is enabled

        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t_sort2 = time.perf_counter()
            print(f"[MoE backward] Sort2: {(t_sort2 - t_dw)*1000:.2f}ms", flush=True)

        # =========================================================================
        # Dispatch Backward AllToAll with dW overlap
        # Option: use chunked backward for dX + AllToAll overlap
        # =========================================================================
        num_chunks = int(os.environ.get('FLUID_CHUNKED_NUM_CHUNKS', '4'))

        if use_chunked and scheduler.is_enabled():
            # Use chunked backward: dX computation overlaps with AllToAll
            from fluid.moe.chunked_backward import backward_dispatch_chunked
            grad_tokens = backward_dispatch_chunked(
                grad_all_fc1,
                w1,  # Already in 3D format [E, hidden, ffn]
                ctx.split_sizes_exp_major if hasattr(ctx, 'split_sizes_exp_major') else all_tokens_per_expert,
                ctx.sorted_idxs_exp_to_rank if hasattr(ctx, 'sorted_idxs_exp_to_rank') else list(range(len(all_tokens_per_expert))),
                all_tokens_per_expert,
                input_splits_list,
                output_splits_list,
                ep_group,
                num_chunks=num_chunks,
                comm_stream=comm_stream,
            )
        elif scheduler.is_enabled():
            # Launch AllToAll on comm_stream (async)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_tokens = _all_to_all(
                    grad_dispatched.contiguous(),
                    output_split_sizes=input_splits_list,
                    input_split_sizes=output_splits_list,
                    group=ep_group
                )
                scheduler.record_alltoall_end(comm_stream)
            # Execute dW tasks from queue while AllToAll is running
            scheduler.on_alltoall_start(comm_type=f"moe_dispatch_L{layer_id}")
            # Wait for AllToAll to complete
            default_stream.wait_stream(comm_stream)
        else:
            grad_tokens = _all_to_all(
                grad_dispatched.contiguous(),
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group
            )

        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t_alltoall2 = time.perf_counter()
            print(f"[MoE backward] AllToAll2: {(t_alltoall2 - t_sort2)*1000:.2f}ms", flush=True)
            print(f"[MoE backward] Total: {(t_alltoall2 - t_start)*1000:.2f}ms", flush=True)

        # When scheduler is enabled, grad_w1/grad_w2 are None (set via weight_param.grad)
        # When scheduler is disabled, grad_w1/grad_w2 are computed directly
        return (grad_tokens, None, None, grad_w1, grad_w2, None, None, None, None,
                None, None, None)

# =============================================================================
# Public API
# =============================================================================

def moe_multicard_p2p_overlap_forward(
    tokens: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    ep_group: dist.ProcessGroup,
    activation_func,
    overlap_ctx: MultiCardOverlapContext,
    layer_id: int = 0,
    num_local_experts: int = 1,
    tokens_per_expert: torch.Tensor = None,
    num_global_tokens_per_local_expert: torch.Tensor = None,
) -> torch.Tensor:
    """
    MoE Multi-card P2P Overlap Forward: local token computation overlapped with remote token communication

    Uses multi-round P2P communication scheduling, each round each card only communicates with one peer.
    Communication stream runs round r while compute stream processes round r-1 data.

    Args:
        tokens: [num_tokens, hidden] input tokens (sorted by expert)
        input_splits: [ep_size] token count to send to each rank
        output_splits: [ep_size] token count to receive from each rank
        weight1: [hidden, ffn_hidden * num_local_experts] first layer weights
        weight2: [ffn_hidden * num_local_experts, hidden] second layer weights
        ep_group: Expert Parallel process group
        activation_func: activation function
        overlap_ctx: multi-card overlap context
        layer_id: layer ID
        num_local_experts: number of local experts
        tokens_per_expert: [num_local_experts] token count per local expert
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]

    Returns:
        output: [num_tokens, hidden]
    """
    return _MoEMultiCardP2POverlapFunction.apply(
        tokens, input_splits, output_splits, weight1, weight2,
        ep_group, activation_func, overlap_ctx, layer_id,
        num_local_experts, tokens_per_expert, num_global_tokens_per_local_expert
    )


__all__ = [
    # Main API
    'moe_multicard_p2p_overlap_forward',
    # Autograd function
    '_MoEMultiCardP2POverlapFunction',
    # Helper functions
    '_compute_fc1_act_per_source',
    '_compute_fc2_per_source',
    '_merge_tokens_and_fc1_expert_major',
    '_precompute_backward_sort_indices',
]

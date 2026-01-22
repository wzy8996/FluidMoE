"""
MoE Forward Operations with P2P Overlap

This module provides all forward operations for MoE layers with P2P communication overlap.

Key functions:
- dispatch_fc1_p2p_forward: Dispatch phase with P2P overlap (parallel FC1+Act)
- fc2_combine_p2p_forward: FC2 + Combine phase with P2P overlap

Design Principles (Two Phases):

Phase 1: Dispatch + FC1 Overlap
-----------------------------------------
- Local FC1+Act overlaps with first P2P round
- Round r P2P overlaps with Round r-1 FC1+Act
- Only compute FC1 and activation, not FC2

Timeline:
    Round 0: P2P_0 communication || local FC1+Act computation
    Round 1: P2P_1 communication || P2P_0 data FC1+Act computation
    ...
    Round N: no communication || P2P_{N-1} data FC1+Act computation

Phase 2: FC2 + Combine Overlap
-----------------------------------------
- First compute remote data FC2 (overlap with Combine P2P)
- Then compute local data FC2
- Save rearranged activation values for backward after local FC2

Timeline:
    Round -1: Compute first peer's FC2
    Round i:  event.synchronize(), start P2P_i, compute next peer's FC2
    ...
    Final:    Compute local FC2 (parallel with last P2P)
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Tuple, Optional, Dict

from fluid.core.comm import MultiCardOverlapContext


# =============================================================================
# Router Forward
# =============================================================================

def router_forward(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    num_experts: int,
    top_k: int,
    ep_group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Router forward computation: token-to-expert assignment with top-k selection.

    This function computes:
    1. Router logits via linear projection
    2. Softmax probabilities
    3. Top-k expert selection
    4. Token permutation by expert assignment
    5. AllToAll split sizes computation

    Args:
        hidden_states: [num_tokens, hidden_size] input tokens
        router_weight: [hidden_size, num_experts] router weight matrix
        num_experts: Total number of experts across all ranks
        top_k: Number of experts each token is sent to
        ep_group: Expert Parallel process group

    Returns:
        permuted_tokens: [num_tokens * top_k, hidden_size] tokens sorted by expert
        permuted_probs: [num_tokens * top_k] routing probabilities (sorted)
        restore_indices: [num_tokens * top_k] indices to restore original order
        sorted_indices: [num_tokens * top_k] indices used for sorting
        input_splits: [ep_size] token count to send to each rank
        output_splits: [ep_size] token count to receive from each rank
        tokens_per_expert: [num_experts] local tokens per expert
        tokens_per_expert_2d: [ep_size, num_experts] token distribution matrix
        router_probs: [num_tokens, num_experts] full softmax probabilities (for backward)
        top_indices: [num_tokens, top_k] top-k expert indices (for backward)
        router_logits: [num_tokens, num_experts] router logits (for debugging)
    """
    ep_size = ep_group.size()
    my_rank = ep_group.rank()
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[-1]
    device = hidden_states.device

    # Step 1: Compute router logits
    # Detach weight to avoid retaining computation graph across iterations
    router_logits = torch.matmul(hidden_states.float(), router_weight.detach().float())

    # Step 2: Softmax and top-k selection
    router_probs = F.softmax(router_logits, dim=-1)
    top_probs, top_indices = torch.topk(router_probs, k=top_k, dim=-1)

    # Step 3: Normalize top-k probabilities
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

    # Step 4: Expand tokens - each token is replicated top_k times
    expanded_tokens = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)
    expanded_probs = top_probs.reshape(-1)
    expanded_expert_indices = top_indices.reshape(-1)

    # Step 5: Sort by expert index (stable sort to ensure determinism)
    sorted_indices = torch.argsort(expanded_expert_indices, stable=True)
    permuted_tokens = expanded_tokens[sorted_indices]
    permuted_probs = expanded_probs[sorted_indices]
    sorted_expert_indices = expanded_expert_indices[sorted_indices]

    # Step 6: Count tokens per expert
    tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=num_experts)

    # Step 7: Compute input_splits and output_splits for AllToAll
    experts_per_rank = num_experts // ep_size

    input_splits = torch.zeros(ep_size, dtype=torch.int64, device=device)
    for i in range(ep_size):
        start_expert = i * experts_per_rank
        end_expert = start_expert + experts_per_rank
        input_splits[i] = tokens_per_expert[start_expert:end_expert].sum()

    # AllGather to get all ranks' input_splits
    all_input_splits = [torch.zeros_like(input_splits) for _ in range(ep_size)]
    dist.all_gather(all_input_splits, input_splits, group=ep_group)
    all_input_splits = torch.stack(all_input_splits)  # [ep_size, ep_size]

    # output_splits[i] = rank i's tokens destined for my experts
    output_splits = all_input_splits[:, my_rank].clone()

    # Step 8: Compute restore indices for combining results
    restore_indices = torch.argsort(sorted_indices)

    # Step 9: AllGather tokens_per_expert to get 2D distribution matrix
    all_tokens_per_expert_list = [torch.zeros_like(tokens_per_expert) for _ in range(ep_size)]
    dist.all_gather(all_tokens_per_expert_list, tokens_per_expert, group=ep_group)
    tokens_per_expert_2d = torch.stack(all_tokens_per_expert_list)  # [ep_size, num_experts]

    return (permuted_tokens, permuted_probs, restore_indices, sorted_indices,
            input_splits, output_splits, tokens_per_expert, tokens_per_expert_2d,
            router_probs, top_indices, router_logits)


# =============================================================================
# Helper Functions
# =============================================================================

def compute_fc1_act_per_source(
    tokens: torch.Tensor,
    w1: torch.Tensor,
    activation_func,
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    source_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute FC1+activation for tokens from a single source rank.
    Also returns the pre-activation FC1 output for backward.

    Args:
        tokens: [num_tokens, hidden] tokens from source_rank
        w1: [num_local_experts, hidden, ffn_hidden]
        activation_func: activation function
        num_local_experts: number of local experts
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]
        source_rank: source rank of data

    Returns:
        act_output: [num_tokens, ffn_hidden] - activation output
        fc1_output: [num_tokens, ffn_hidden] - pre-activation FC1 output (for backward)
    """
    device = tokens.device
    dtype = tokens.dtype
    ffn_hidden = w1.shape[-1]

    if tokens.numel() == 0:
        return (torch.empty(0, ffn_hidden, dtype=dtype, device=device),
                torch.empty(0, ffn_hidden, dtype=dtype, device=device))

    if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
        # Convert to CPU list once to avoid multiple .item() calls causing CUDA sync
        tokens_cpu = num_global_tokens_per_local_expert[0, source_rank].cpu().tolist()

        act_output = torch.zeros(tokens.shape[0], ffn_hidden, dtype=dtype, device=device)
        fc1_output = torch.zeros(tokens.shape[0], ffn_hidden, dtype=dtype, device=device)

        offset = 0
        for exp_idx in range(num_local_experts):
            n_tok = tokens_cpu[exp_idx]
            if n_tok > 0:
                exp_tokens = tokens[offset:offset + n_tok]
                exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                fc1_output[offset:offset + n_tok] = exp_fc1
                act_output[offset:offset + n_tok] = activation_func(exp_fc1)
                offset += n_tok

        return act_output, fc1_output
    else:
        fc1 = torch.matmul(tokens, w1[0])
        return activation_func(fc1), fc1


def compute_fc2_per_source(
    act: torch.Tensor,
    w2: torch.Tensor,
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    source_rank: int,
) -> torch.Tensor:
    """
    Compute FC2 for activation values from a single source rank.

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
        # Convert to CPU list once to avoid multiple .item() calls causing CUDA sync
        tokens_cpu = num_global_tokens_per_local_expert[0, source_rank].cpu().tolist()

        fc2_output = torch.zeros(act.shape[0], hidden_size, dtype=act.dtype, device=device)

        offset = 0
        for exp_idx in range(num_local_experts):
            n_tok = tokens_cpu[exp_idx]
            if n_tok > 0:
                exp_act = act[offset:offset + n_tok]
                fc2_output[offset:offset + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                offset += n_tok

        return fc2_output
    else:
        return torch.matmul(act, w2[0])


def merge_tokens_expert_major(
    local_tokens: torch.Tensor,
    all_peer_tokens: torch.Tensor,
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    my_rank: int,
    ep_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Merge local and peer tokens into expert-major order.

    Args:
        local_tokens: [local_count, hidden_size] local tokens (in expert order)
        all_peer_tokens: [peer_count, hidden_size] all peer tokens (cat in rank order)
        num_local_experts: Number of local experts
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]
        my_rank: Current rank
        ep_size: Expert parallel world size
        device: Torch device

    Returns:
        all_expert_tokens: [total, hidden_size] expert-major order
        all_tokens_per_expert: [num_local_experts] token count per expert
    """
    hidden_size = local_tokens.shape[-1] if local_tokens.numel() > 0 else all_peer_tokens.shape[-1]
    dtype = local_tokens.dtype if local_tokens.numel() > 0 else all_peer_tokens.dtype

    # Convert to CPU list once to avoid multiple .item() calls causing CUDA sync
    tokens_cpu = num_global_tokens_per_local_expert[0].cpu().tolist()  # [ep_size, num_local_experts]

    # Compute total token count per expert
    all_tokens_per_expert = []
    for exp_idx in range(num_local_experts):
        total = sum(tokens_cpu[rank][exp_idx] for rank in range(ep_size))
        all_tokens_per_expert.append(total)

    total_tokens = sum(all_tokens_per_expert)

    if total_tokens == 0:
        return (torch.empty(0, hidden_size, dtype=dtype, device=device),
                all_tokens_per_expert)

    all_expert_tokens = torch.zeros(total_tokens, hidden_size, dtype=dtype, device=device)

    # Precompute start offset for each rank in all_peer_tokens
    peer_rank_offsets = {}
    offset = 0
    for rank in range(ep_size):
        if rank == my_rank:
            continue
        peer_rank_offsets[rank] = offset
        for exp_idx in range(num_local_experts):
            offset += tokens_cpu[rank][exp_idx]

    # Fill in expert-major order
    write_offset = 0
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            n_tok = tokens_cpu[rank][exp_idx]
            if n_tok == 0:
                continue

            if rank == my_rank:
                # Extract from local (local is in expert order)
                local_exp_offset = sum(tokens_cpu[my_rank][e] for e in range(exp_idx))
                all_expert_tokens[write_offset:write_offset + n_tok] = \
                    local_tokens[local_exp_offset:local_exp_offset + n_tok]
            else:
                # Extract from peer (peer is cat in rank order, each rank in expert order)
                peer_base = peer_rank_offsets[rank]
                peer_exp_offset = sum(tokens_cpu[rank][e] for e in range(exp_idx))
                src_offset = peer_base + peer_exp_offset
                all_expert_tokens[write_offset:write_offset + n_tok] = \
                    all_peer_tokens[src_offset:src_offset + n_tok]

            write_offset += n_tok

    return all_expert_tokens, all_tokens_per_expert


def precompute_backward_sort_indices(
    num_local_experts: int,
    ep_size: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Precompute sort indices needed for backward pass.

    Args:
        num_local_experts: Number of local experts
        ep_size: Expert parallel world size
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]
        device: Torch device

    Returns:
        Dictionary containing:
        - split_sizes_rank_major: chunk sizes in rank-major order
        - sorted_idxs_rank_to_exp: indices for rank-major -> expert-major
        - split_sizes_exp_major: chunk sizes in expert-major order
        - sorted_idxs_exp_to_rank: indices for expert-major -> rank-major
    """
    # Convert to CPU list once to avoid multiple .item() calls causing CUDA sync
    tokens_cpu = num_global_tokens_per_local_expert[0].cpu().tolist()  # [ep_size, num_local_experts]

    # rank-major chunk sizes: [R0_E0, R0_E1, R1_E0, R1_E1, ...]
    split_sizes_rank_major = []
    for rank in range(ep_size):
        for exp_idx in range(num_local_experts):
            split_sizes_rank_major.append(tokens_cpu[rank][exp_idx])

    # rank-major -> expert-major indices
    sorted_idxs_rank_to_exp = []
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            sorted_idxs_rank_to_exp.append(rank * num_local_experts + exp_idx)

    # expert-major chunk sizes: [E0_R0, E0_R1, ..., E1_R0, E1_R1, ...]
    split_sizes_exp_major = []
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            split_sizes_exp_major.append(tokens_cpu[rank][exp_idx])

    # expert-major -> rank-major indices
    sorted_idxs_exp_to_rank = []
    for rank in range(ep_size):
        for exp_idx in range(num_local_experts):
            sorted_idxs_exp_to_rank.append(exp_idx * ep_size + rank)

    return {
        'split_sizes_rank_major': torch.tensor(split_sizes_rank_major, dtype=torch.int64, device=device),
        'sorted_idxs_rank_to_exp': torch.tensor(sorted_idxs_rank_to_exp, dtype=torch.int64, device=device),
        'split_sizes_exp_major': torch.tensor(split_sizes_exp_major, dtype=torch.int64, device=device),
        'sorted_idxs_exp_to_rank': torch.tensor(sorted_idxs_exp_to_rank, dtype=torch.int64, device=device),
    }


# =============================================================================
# Phase 1: Dispatch + FC1 with P2P Overlap
# =============================================================================

def dispatch_fc1_p2p_forward(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    input_splits: List[int],
    output_splits: List[int],
    ep_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    activation_func,
    num_local_experts: int,
    num_global_tokens_per_local_expert: Optional[torch.Tensor],
    needs_backward: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor],
           List[int], Dict[int, int]]:
    """
    Dispatch phase with P2P overlap: parallel FC1+Act computation with P2P communication.

    Pipeline:
        Round 0: Start P2P_0, compute local FC1 + Act
        Round i: req.wait(P2P_{i-1}), start P2P_i, compute recv_{i-1} FC1 + Act
        Final:   req.wait(last round), compute last FC1 + Act

    Note: FC1 outputs are NOT saved during forward. They will be recomputed during
    backward to save memory copy overhead (~2.5ms savings).

    Args:
        tokens: [num_tokens, hidden] input tokens (sorted by expert)
        weight1: [hidden, ffn_hidden * num_local_experts] FC1 weight
        input_splits: [ep_size] token count to send to each rank
        output_splits: [ep_size] token count to receive from each rank
        ep_group: Expert Parallel process group
        overlap_ctx: P2P overlap context
        activation_func: Activation function
        num_local_experts: Number of local experts
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]
        needs_backward: Whether backward is needed (unused, kept for API compatibility)

    Returns:
        local_tokens: [local_count, hidden] local tokens
        local_act: [local_count, ffn_hidden] local activation output
        recv_act_results: Dict[partner -> act_tensor] - activation outputs from peers
        recv_buffers: Dict[partner -> token_tensor] - received tokens from peers
        partners: List of partner ranks
        recv_offsets: Dict of partner -> offset in buffer
    """
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    device = tokens.device
    dtype = tokens.dtype
    hidden_size = tokens.shape[-1]

    # Weight dimensions - weight1 is already 3D: [num_local_experts, hidden, ffn_hidden]
    ffn_hidden = weight1.shape[-1]
    w1 = weight1  # No permute needed - already in correct shape

    # Compute offsets
    input_offsets = [0]
    for s in input_splits:
        input_offsets.append(input_offsets[-1] + s)

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    local_count = input_splits[my_rank]
    local_start = input_offsets[my_rank]

    # Convert to CPU list once to avoid multiple .item() calls causing CUDA sync
    tokens_cpu = None
    if num_global_tokens_per_local_expert is not None:
        tokens_cpu = num_global_tokens_per_local_expert[0].cpu().tolist()  # [ep_size, num_local_experts]

    # Compute local token count per expert
    local_tokens_per_expert = None
    if tokens_cpu is not None:
        local_tokens_per_expert = [tokens_cpu[my_rank][exp_idx] for exp_idx in range(num_local_experts)]

    # Get Round-Robin scheduled partners
    partners = []
    for round_idx in range(overlap_ctx.num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner != -1:
            partners.append(partner)

    # Extract local tokens (no clone needed - data will be copied in merge_tokens_expert_major anyway)
    local_tokens = tokens[local_start:local_start + local_count] if local_count > 0 else torch.empty(0, hidden_size, dtype=dtype, device=device)

    # Prepare send data (token slices)
    send_chunks = {}
    for partner in partners:
        if input_splits[partner] > 0:
            # Slice along dim 0 of contiguous tensor is already contiguous - no need for .contiguous()
            send_chunks[partner] = tokens[input_offsets[partner]:input_offsets[partner+1]]

    # Prepare receive buffers (by partner order)
    recv_buffers = {}
    for partner in partners:
        recv_size = output_splits[partner]
        if recv_size > 0:
            recv_buffers[partner] = torch.empty(recv_size, hidden_size, dtype=dtype, device=device)

    # Pre-allocate Act buffer (FC1 is no longer saved - will be recomputed in backward)
    total_recv = sum(output_splits)
    all_recv_act_buffer = torch.empty(total_recv, ffn_hidden, dtype=dtype, device=device) if total_recv > 0 else None

    # Compute each partner's offset in buffer
    recv_offsets = {}
    offset = 0
    for i in range(ep_size):
        if i != my_rank:
            recv_offsets[i] = offset
            offset += output_splits[i]

    # =========================================================================
    # Dispatch Phase Pipeline with delayed req.wait()
    # Key insight: Start P2P_i first, then wait for P2P_{i-1} to complete
    # This ensures P2P_i runs in background while we process P2P_{i-1}'s data
    # =========================================================================
    prev_partner = None
    prev_reqs = []
    recv_act_results = {}
    local_act = None

    for round_idx, partner in enumerate(partners):
        # 1. Start current round P2P (async, returns immediately)
        with torch.cuda.stream(comm_stream):
            if round_idx == 0:
                comm_stream.wait_stream(default_stream)  # Wait for send_chunks preparation
            p2p_ops = []
            if partner in recv_buffers:
                p2p_ops.append(dist.P2POp(dist.irecv, recv_buffers[partner], partner, group=ep_group))
            if partner in send_chunks:
                p2p_ops.append(dist.P2POp(dist.isend, send_chunks[partner], partner, group=ep_group))
            curr_reqs = dist.batch_isend_irecv(p2p_ops) if p2p_ops else []

        # 2. Wait for PREVIOUS round's P2P to complete (current round runs in background)
        if round_idx > 0:
            for req in prev_reqs:
                req.wait()

        # 3. Compute FC1 + Act (overlaps with current round's P2P)
        if round_idx == 0:
            # First round: compute local FC1 + Act (parallel with P2P_0)
            if local_count > 0 and local_tokens_per_expert is not None:
                local_act = torch.zeros(local_count, ffn_hidden, dtype=dtype, device=device)
                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = local_tokens_per_expert[exp_idx]
                    if n_tok > 0:
                        exp_tokens = local_tokens[start:start + n_tok]
                        exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                        local_act[start:start + n_tok] = activation_func(exp_fc1)
                        start += n_tok
            elif local_count > 0:
                exp_fc1 = torch.matmul(local_tokens, w1[0])
                local_act = activation_func(exp_fc1)
        else:
            # Use previous round's received data (now guaranteed complete)
            if prev_partner in recv_buffers:
                recv_data = recv_buffers[prev_partner]
                recv_count = recv_data.shape[0]
                buf_offset = recv_offsets[prev_partner]
                recv_act = all_recv_act_buffer[buf_offset:buf_offset + recv_count]
                if num_local_experts > 1 and tokens_cpu is not None:
                    offset_inner = 0
                    for exp_idx in range(num_local_experts):
                        n_tok = tokens_cpu[prev_partner][exp_idx]
                        if n_tok > 0:
                            exp_tokens = recv_data[offset_inner:offset_inner + n_tok]
                            exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                            recv_act[offset_inner:offset_inner + n_tok] = activation_func(exp_fc1)
                            offset_inner += n_tok
                else:
                    exp_fc1 = torch.matmul(recv_data, w1[0])
                    recv_act.copy_(activation_func(exp_fc1))
                recv_act_results[prev_partner] = recv_act

        prev_partner = partner
        prev_reqs = curr_reqs

    # Process last round: wait for last P2P and compute
    if len(partners) > 0:
        for req in prev_reqs:
            req.wait()

        if prev_partner in recv_buffers:
            recv_data = recv_buffers[prev_partner]
            recv_count = recv_data.shape[0]
            buf_offset = recv_offsets[prev_partner]
            recv_act = all_recv_act_buffer[buf_offset:buf_offset + recv_count]
            if num_local_experts > 1 and tokens_cpu is not None:
                offset_inner = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_cpu[prev_partner][exp_idx]
                    if n_tok > 0:
                        exp_tokens = recv_data[offset_inner:offset_inner + n_tok]
                        exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                        recv_act[offset_inner:offset_inner + n_tok] = activation_func(exp_fc1)
                        offset_inner += n_tok
            else:
                exp_fc1 = torch.matmul(recv_data, w1[0])
                recv_act.copy_(activation_func(exp_fc1))
            recv_act_results[prev_partner] = recv_act

    return (local_tokens, local_act, recv_act_results, recv_buffers, partners, recv_offsets)


# =============================================================================
# Phase 2: FC2 + Combine with P2P Overlap
# =============================================================================

def fc2_combine_p2p_forward(
    local_tokens: torch.Tensor,
    local_act: torch.Tensor,
    recv_act_results: Dict[int, torch.Tensor],
    recv_buffers: Dict[int, torch.Tensor],
    weight2: torch.Tensor,
    input_splits: List[int],
    output_splits: List[int],
    ep_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    num_local_experts: int,
    num_global_tokens_per_local_expert: Optional[torch.Tensor],
    partners: List[int],
    needs_backward: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], Dict[str, torch.Tensor]]:
    """
    FC2 + Combine phase with P2P overlap: parallel FC2 computation with P2P communication.

    Also performs merge tokens and precompute backward indices BEFORE req.wait() to overlap
    with the final Combine P2P communication.

    Note: FC1 outputs are NOT saved. They will be recomputed during backward.

    Pipeline:
        Round -1: Compute first peer's FC2
        Round i:  event.synchronize(), start P2P_i, compute next peer's FC2
        Final:    Compute local FC2 (parallel with last P2P)
        Before wait: Merge tokens + precompute indices (parallel with P2P)

    Args:
        local_tokens: [local_count, hidden] local tokens
        local_act: [local_count, ffn_hidden] local activation output
        recv_act_results: Dict[partner -> act_tensor] - activation outputs from peers
        recv_buffers: Dict[partner -> token_tensor] - received tokens from peers
        weight2: [ffn_hidden * num_local_experts, hidden] FC2 weight
        input_splits: [ep_size] token count to send to each rank
        output_splits: [ep_size] token count to receive from each rank
        ep_group: Expert Parallel process group
        overlap_ctx: P2P overlap context
        num_local_experts: Number of local experts
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]
        partners: List of partner ranks
        needs_backward: Whether backward is needed (skip merge/precompute if False)

    Returns:
        combined_output: [total_tokens, hidden] final output
        local_fc2: [local_count, hidden] local FC2 output
        all_expert_tokens: [total_recv, hidden] all tokens (expert-major order)
        all_tokens_per_expert: List of token counts per expert
        backward_indices: Dictionary of precomputed indices for backward
    """
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    device = local_tokens.device
    dtype = local_tokens.dtype
    hidden_size = weight2.shape[-1]

    # Weight dimensions - weight2 is already 3D: [num_local_experts, ffn_hidden, hidden]
    ffn_hidden = weight2.shape[1]
    w2 = weight2  # No permute needed - already in correct shape

    # Compute offsets
    input_offsets = [0]
    for s in input_splits:
        input_offsets.append(input_offsets[-1] + s)

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    local_count = input_splits[my_rank]
    local_start = input_offsets[my_rank]

    # Convert to CPU list once to avoid multiple .item() calls causing CUDA sync
    tokens_cpu = None
    if num_global_tokens_per_local_expert is not None:
        tokens_cpu = num_global_tokens_per_local_expert[0].cpu().tolist()  # [ep_size, num_local_experts]

    # Compute local token count per expert
    local_tokens_per_expert = None
    if tokens_cpu is not None:
        local_tokens_per_expert = [tokens_cpu[my_rank][exp_idx] for exp_idx in range(num_local_experts)]

    # =========================================================================
    # Combine Phase Pipeline
    # =========================================================================
    total_output = sum(input_splits)
    combined_output = torch.empty(total_output, hidden_size, dtype=dtype, device=device)

    peer_fc2_results = {}
    local_fc2 = None

    # Reuse single event from context to reduce overhead
    fc2_event = overlap_ctx.data_ready_event
    has_pending_fc2 = False

    # Round -1: Pre-compute first peer's FC2
    if len(partners) > 0:
        first_partner = partners[0]
        if first_partner in recv_act_results:
            recv_act = recv_act_results[first_partner]
            if num_local_experts > 1 and tokens_cpu is not None:
                peer_fc2 = torch.zeros(recv_act.shape[0], hidden_size, dtype=dtype, device=device)
                offset = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_cpu[first_partner][exp_idx]
                    if n_tok > 0:
                        exp_act = recv_act[offset:offset + n_tok]
                        peer_fc2[offset:offset + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                        offset += n_tok
            else:
                peer_fc2 = torch.matmul(recv_act, w2[0])
            peer_fc2_results[first_partner] = peer_fc2
            # Record event on default_stream: FC2 computation done
            fc2_event.record(default_stream)
            has_pending_fc2 = True

    # Pipeline loop
    all_combine_reqs = []
    for round_idx, partner in enumerate(partners):
        # Start P2P (send FC2 result to partner, receive from partner)
        # GPU sync: comm_stream waits for FC2 computation to complete before sending
        with torch.cuda.stream(comm_stream):
            # Key: comm_stream waits for default_stream's FC2 to complete
            if has_pending_fc2:
                comm_stream.wait_event(fc2_event)

            p2p_ops = []
            # Receive: FC2 result from partner
            recv_size = input_splits[partner]
            if recv_size > 0:
                recv_chunk = combined_output[input_offsets[partner]:input_offsets[partner+1]]
                p2p_ops.append(dist.P2POp(dist.irecv, recv_chunk, partner, group=ep_group))
            # Send: my FC2 result to partner
            if partner in peer_fc2_results:
                p2p_ops.append(dist.P2POp(dist.isend, peer_fc2_results[partner], partner, group=ep_group))
            reqs = dist.batch_isend_irecv(p2p_ops) if p2p_ops else []
            all_combine_reqs.extend(reqs)

        # Parallel with current P2P: compute next round FC2 or local FC2
        has_pending_fc2 = False  # Reset for next iteration
        if round_idx + 1 < len(partners):
            next_partner = partners[round_idx + 1]
            if next_partner in recv_act_results:
                recv_act = recv_act_results[next_partner]
                if num_local_experts > 1 and tokens_cpu is not None:
                    peer_fc2 = torch.zeros(recv_act.shape[0], hidden_size, dtype=dtype, device=device)
                    offset = 0
                    for exp_idx in range(num_local_experts):
                        n_tok = tokens_cpu[next_partner][exp_idx]
                        if n_tok > 0:
                            exp_act = recv_act[offset:offset + n_tok]
                            peer_fc2[offset:offset + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                            offset += n_tok
                else:
                    peer_fc2 = torch.matmul(recv_act, w2[0])
                peer_fc2_results[next_partner] = peer_fc2
                fc2_event.record(default_stream)
                has_pending_fc2 = True
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

    # =========================================================================
    # Merge tokens and precompute indices BEFORE req.wait()
    # (parallel with last Combine P2P communication)
    # Note: FC1 is NOT saved - will be recomputed during backward
    # =========================================================================
    all_expert_tokens = None
    all_tokens_per_expert = []
    backward_indices = {}

    if needs_backward:
        # Merge tokens for backward (expert-major order) - FC1 will be recomputed
        total_recv = sum(output_splits)
        if total_recv > 0:
            all_peer_tokens_list = []
            for i in range(ep_size):
                if i == my_rank:
                    continue
                if i in recv_buffers:
                    all_peer_tokens_list.append(recv_buffers[i])
            all_peer_tokens = torch.cat(all_peer_tokens_list, dim=0) if all_peer_tokens_list else torch.empty(0, hidden_size, dtype=dtype, device=device)
        else:
            all_peer_tokens = torch.empty(0, hidden_size, dtype=dtype, device=device)

        if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
            all_expert_tokens, all_tokens_per_expert = merge_tokens_expert_major(
                local_tokens, all_peer_tokens,
                num_local_experts, num_global_tokens_per_local_expert,
                my_rank, ep_size, device
            )
        else:
            all_expert_tokens = torch.cat([local_tokens, all_peer_tokens], dim=0) if all_peer_tokens.numel() > 0 else local_tokens
            all_tokens_per_expert = [all_expert_tokens.shape[0]]

        # Precompute backward sort indices
        if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
            backward_indices = precompute_backward_sort_indices(
                num_local_experts, ep_size, num_global_tokens_per_local_expert, device
            )

    # =========================================================================
    # Wait for all Combine P2P to complete
    # =========================================================================
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

    # Write local result to combined_output
    if local_fc2 is not None:
        combined_output[local_start:local_start + local_count] = local_fc2

    return combined_output, local_fc2, all_expert_tokens, all_tokens_per_expert, backward_indices


__all__ = [
    # Router forward
    'router_forward',
    # Helper functions
    'compute_fc1_act_per_source',
    'compute_fc2_per_source',
    'merge_tokens_expert_major',
    'precompute_backward_sort_indices',
    # Phase 1: Dispatch + FC1
    'dispatch_fc1_p2p_forward',
    # Phase 2: FC2 + Combine
    'fc2_combine_p2p_forward',
]

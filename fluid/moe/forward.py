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
from fluid.core.nvtx import nvtx_range, nvtx_range_push, nvtx_range_pop, nvtx_mark, Colors

# =============================================================================
# Grouped GEMM Support (optional, for efficient expert computation)
# =============================================================================
try:
    import grouped_gemm as gg
    GROUPED_GEMM_AVAILABLE = True
except ImportError:
    gg = None
    GROUPED_GEMM_AVAILABLE = False


def grouped_fc1_act(tokens: torch.Tensor, w1: torch.Tensor,
                    tokens_per_expert: torch.Tensor, activation_func) -> torch.Tensor:
    """Compute FC1 + activation for all experts. Uses grouped GEMM if available.

    Args:
        tokens: [total_tokens, hidden] input tokens
        w1: [num_experts, hidden, ffn_hidden] FC1 weights
        tokens_per_expert: 1D CPU int64 tensor of token counts per expert
        activation_func: Activation function
    """
    if tokens.shape[0] == 0:
        return torch.empty(0, w1.shape[-1], dtype=tokens.dtype, device=tokens.device)

    num_experts = w1.shape[0]
    if GROUPED_GEMM_AVAILABLE and num_experts > 1:
        return activation_func(gg.ops.gmm(tokens, w1, tokens_per_expert, trans_b=False))
    elif num_experts == 1:
        return activation_func(torch.matmul(tokens, w1[0]))
    else:
        # Fallback: for loop over experts
        output = torch.empty(tokens.shape[0], w1.shape[-1], dtype=tokens.dtype, device=tokens.device)
        offset = 0
        for i, n in enumerate(tokens_per_expert.tolist()):
            if n > 0:
                output[offset:offset+n] = activation_func(torch.matmul(tokens[offset:offset+n], w1[i]))
                offset += n
        return output


def grouped_fc2(act: torch.Tensor, w2: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """Compute FC2 for all experts. Uses grouped GEMM if available.

    Args:
        act: [total_tokens, ffn_hidden] activation values
        w2: [num_experts, ffn_hidden, hidden] FC2 weights
        tokens_per_expert: 1D CPU int64 tensor of token counts per expert
    """
    if act.shape[0] == 0:
        return torch.empty(0, w2.shape[-1], dtype=act.dtype, device=act.device)

    num_experts = w2.shape[0]
    if GROUPED_GEMM_AVAILABLE and num_experts > 1:
        return gg.ops.gmm(act, w2, tokens_per_expert, trans_b=False)
    elif num_experts == 1:
        return torch.matmul(act, w2[0])
    else:
        # Fallback: for loop over experts
        output = torch.empty(act.shape[0], w2.shape[-1], dtype=act.dtype, device=act.device)
        offset = 0
        for i, n in enumerate(tokens_per_expert.tolist()):
            if n > 0:
                output[offset:offset+n] = torch.matmul(act[offset:offset+n], w2[i])
                offset += n
        return output


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
           torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Router forward computation: token-to-expert assignment with top-k selection.

    This function computes:
    1. Router logits via linear projection
    2. Softmax probabilities
    3. Top-k expert selection
    4. Token permutation by expert assignment
    5. AllToAll split sizes computation

    Note: tokens_per_expert_2d AllGather has been removed. Each rank's tokens_per_expert
    is now piggybacked on P2P communication in dispatch_fc1_p2p_forward.

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
        tokens_per_expert: [num_experts] local tokens per expert (full, not 2D)
        router_probs: [num_tokens, num_experts] full softmax probabilities (for backward)
        top_indices: [num_tokens, top_k] top-k expert indices (for backward)
        router_logits: [num_tokens, num_experts] router logits (for debugging)
    """
    nvtx_range_push("router_forward")

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

    # Note: AllGather for tokens_per_expert is removed - metadata is now piggybacked on P2P
    # Each rank will receive peer's tokens_per_expert along with token data in dispatch_fc1_p2p_forward

    nvtx_range_pop()
    return (permuted_tokens, permuted_probs, restore_indices, sorted_indices,
            input_splits, output_splits, tokens_per_expert,
            router_probs, top_indices, router_logits)


# =============================================================================
# Helper Functions
# =============================================================================

def merge_tokens_expert_major(
    local_tokens: torch.Tensor,
    all_peer_tokens: torch.Tensor,
    num_local_experts: int,
    tokens_cpu: torch.Tensor,
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
        tokens_cpu: [ep_size, num_local_experts] CPU int64 tensor of token counts
        my_rank: Current rank
        ep_size: Expert parallel world size
        device: Torch device

    Returns:
        all_expert_tokens: [total, hidden_size] expert-major order
        all_tokens_per_expert: [num_local_experts] token count per expert
    """
    hidden_size = local_tokens.shape[-1] if local_tokens.numel() > 0 else all_peer_tokens.shape[-1]
    dtype = local_tokens.dtype if local_tokens.numel() > 0 else all_peer_tokens.dtype

    # Convert to list once for efficient iteration
    tokens_list = tokens_cpu.tolist()  # [ep_size][num_local_experts]

    # Compute total token count per expert
    all_tokens_per_expert = []
    for exp_idx in range(num_local_experts):
        total = sum(tokens_list[rank][exp_idx] for rank in range(ep_size))
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
            offset += tokens_list[rank][exp_idx]

    # Fill in expert-major order
    write_offset = 0
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            n_tok = tokens_list[rank][exp_idx]
            if n_tok == 0:
                continue

            if rank == my_rank:
                # Extract from local (local is in expert order)
                local_exp_offset = sum(tokens_list[my_rank][e] for e in range(exp_idx))
                all_expert_tokens[write_offset:write_offset + n_tok] = \
                    local_tokens[local_exp_offset:local_exp_offset + n_tok]
            else:
                # Extract from peer (peer is cat in rank order, each rank in expert order)
                peer_base = peer_rank_offsets[rank]
                peer_exp_offset = sum(tokens_list[rank][e] for e in range(exp_idx))
                src_offset = peer_base + peer_exp_offset
                all_expert_tokens[write_offset:write_offset + n_tok] = \
                    all_peer_tokens[src_offset:src_offset + n_tok]

            write_offset += n_tok

    return all_expert_tokens, all_tokens_per_expert


def precompute_backward_sort_indices(
    num_local_experts: int,
    ep_size: int,
    tokens_cpu: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Precompute sort indices needed for backward pass.

    Args:
        num_local_experts: Number of local experts
        ep_size: Expert parallel world size
        tokens_cpu: [ep_size, num_local_experts] CPU int64 tensor of token counts
        device: Torch device

    Returns:
        Dictionary containing:
        - split_sizes_rank_major: chunk sizes in rank-major order
        - sorted_idxs_rank_to_exp: indices for rank-major -> expert-major
        - split_sizes_exp_major: chunk sizes in expert-major order
        - sorted_idxs_exp_to_rank: indices for expert-major -> rank-major
    """
    # Convert to list once for efficient iteration
    tokens_list = tokens_cpu.tolist()  # [ep_size][num_local_experts]

    # rank-major chunk sizes: [R0_E0, R0_E1, R1_E0, R1_E1, ...]
    split_sizes_rank_major = []
    for rank in range(ep_size):
        for exp_idx in range(num_local_experts):
            split_sizes_rank_major.append(tokens_list[rank][exp_idx])

    # rank-major -> expert-major indices
    sorted_idxs_rank_to_exp = []
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            sorted_idxs_rank_to_exp.append(rank * num_local_experts + exp_idx)

    # expert-major chunk sizes: [E0_R0, E0_R1, ..., E1_R0, E1_R1, ...]
    split_sizes_exp_major = []
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            split_sizes_exp_major.append(tokens_list[rank][exp_idx])

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
    tokens_per_expert: torch.Tensor,
    needs_backward: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor],
           List[int], Dict[int, int], torch.Tensor]:
    """
    Dispatch phase with P2P overlap: parallel FC1+Act computation with P2P communication.

    Pipeline:
        Round 0: Start P2P_0, compute local FC1 + Act
        Round i: req.wait(P2P_{i-1}), start P2P_i, compute recv_{i-1} FC1 + Act
        Final:   req.wait(last round), compute last FC1 + Act

    P2P Metadata Piggybacking:
        Each P2P message includes a metadata row containing tokens_per_expert info.
        This eliminates the need for a separate AllGather operation.

    Note: FC1 outputs are NOT saved during forward. They will be recomputed during
    backward to save memory copy overhead (~2.5ms savings).

    Args:
        tokens: [num_tokens, hidden] input tokens (sorted by expert)
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight
        input_splits: [ep_size] token count to send to each rank
        output_splits: [ep_size] token count to receive from each rank
        ep_group: Expert Parallel process group
        overlap_ctx: P2P overlap context
        activation_func: Activation function
        num_local_experts: Number of local experts
        tokens_per_expert: [num_global_experts] local tokens per expert from router
        needs_backward: Whether backward is needed (unused, kept for API compatibility)

    Returns:
        local_tokens: [local_count, hidden] local tokens
        local_act: [local_count, ffn_hidden] local activation output
        recv_act_results: Dict[partner -> act_tensor] - activation outputs from peers
        recv_buffers: Dict[partner -> token_tensor] - received tokens from peers (without metadata row)
        partners: List of partner ranks
        recv_offsets: Dict of partner -> offset in buffer
        tokens_cpu: [ep_size, num_local_experts] CPU int64 tensor built from P2P metadata
    """
    nvtx_range_push("dispatch_fc1_p2p")
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

    # Extract local tokens_per_expert (for my local experts)
    # tokens_per_expert[my_rank * num_local_experts : (my_rank + 1) * num_local_experts]
    local_tokens_per_expert = tokens_per_expert[my_rank * num_local_experts : (my_rank + 1) * num_local_experts]
    local_tokens_per_expert_cpu = local_tokens_per_expert.to(dtype=torch.int64).cpu()

    # Initialize tokens_cpu tensor to collect metadata from all ranks
    # tokens_cpu[rank] = tokens_per_expert for that rank's tokens going to my local experts
    tokens_cpu = torch.zeros(ep_size, num_local_experts, dtype=torch.int64)
    tokens_cpu[my_rank] = local_tokens_per_expert_cpu

    # Get Round-Robin scheduled partners
    partners = []
    for round_idx in range(overlap_ctx.num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner != -1:
            partners.append(partner)

    # Extract local tokens (no clone needed - data will be copied in merge_tokens_expert_major anyway)
    local_tokens = tokens[local_start:local_start + local_count] if local_count > 0 else torch.empty(0, hidden_size, dtype=dtype, device=device)

    # Prepare send data with metadata row
    # Metadata encoding: Use int32 view to exactly preserve integer values
    # Each int32 value maps to 2 float16 values (or 1 float32/bfloat16 value)
    # We store metadata as int32 values reinterpreted as the dtype's raw bits
    send_chunks = {}
    send_buffers_with_metadata = {}
    for partner in partners:
        if input_splits[partner] > 0:
            token_chunk = tokens[input_offsets[partner]:input_offsets[partner+1]]
            # Metadata: my tokens_per_expert for partner's local experts
            partner_metadata = tokens_per_expert[partner * num_local_experts : (partner + 1) * num_local_experts]
            # Create metadata row - encode as int32, then view as dtype
            metadata_int32 = partner_metadata.to(torch.int32).to(device)
            # Pad to required size (need enough space for num_local_experts int32 values)
            # int32 is 4 bytes, float16 is 2 bytes, so we need 2*num_local_experts float16 elements
            # For bfloat16/float32, we need different calculations
            element_size = torch.tensor([], dtype=dtype).element_size()
            int32_as_elements = 4 // element_size  # How many dtype elements per int32
            metadata_elements = num_local_experts * int32_as_elements
            metadata_row = torch.zeros(1, hidden_size, dtype=dtype, device=device)
            # View int32 as dtype and copy
            metadata_as_dtype = metadata_int32.view(torch.int8).view(dtype)
            metadata_row[0, :metadata_elements] = metadata_as_dtype
            # Concatenate metadata + tokens
            send_buffers_with_metadata[partner] = torch.cat([metadata_row, token_chunk], dim=0)
            send_chunks[partner] = send_buffers_with_metadata[partner]

    # Prepare receive buffers with extra row for metadata
    recv_buffers_with_metadata = {}
    recv_buffers = {}  # Will store token data without metadata
    for partner in partners:
        recv_size = output_splits[partner]
        if recv_size > 0:
            # +1 row for metadata
            recv_buffers_with_metadata[partner] = torch.empty(recv_size + 1, hidden_size, dtype=dtype, device=device)

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

    # Compute metadata layout (same as encoding)
    element_size = torch.tensor([], dtype=dtype).element_size()
    int32_as_elements = 4 // element_size
    metadata_elements = num_local_experts * int32_as_elements

    def extract_metadata_and_tokens(partner):
        """Extract metadata from received buffer and update tokens_cpu."""
        if partner in recv_buffers_with_metadata:
            full_buffer = recv_buffers_with_metadata[partner]
            # First row is metadata encoded as int32 viewed as dtype
            metadata_as_dtype = full_buffer[0, :metadata_elements]
            # View back as int32
            metadata_int32 = metadata_as_dtype.view(torch.int8).view(torch.int32)
            tokens_cpu[partner] = metadata_int32[:num_local_experts].to(torch.int64).cpu()
            # Rest is token data
            recv_buffers[partner] = full_buffer[1:]

    for round_idx, partner in enumerate(partners):
        # 1. Start current round P2P (async, returns immediately)
        nvtx_range_push(f"dispatch_p2p_R{round_idx}")
        with torch.cuda.stream(comm_stream):
            if round_idx == 0:
                comm_stream.wait_stream(default_stream)  # Wait for send_chunks preparation
            p2p_ops = []
            if partner in recv_buffers_with_metadata:
                p2p_ops.append(dist.P2POp(dist.irecv, recv_buffers_with_metadata[partner], partner, group=ep_group))
            if partner in send_chunks:
                p2p_ops.append(dist.P2POp(dist.isend, send_chunks[partner], partner, group=ep_group))
            curr_reqs = dist.batch_isend_irecv(p2p_ops) if p2p_ops else []
        nvtx_range_pop()

        # 2. Wait for PREVIOUS round's P2P to complete (current round runs in background)
        if round_idx > 0:
            nvtx_range_push(f"dispatch_wait_R{round_idx-1}")
            for req in prev_reqs:
                req.wait()
            nvtx_range_pop()

        # 3. Extract metadata and compute FC1 + Act (overlaps with current round's P2P)
        nvtx_range_push(f"fc1_compute_R{round_idx}")
        if round_idx == 0:
            # First round: compute local FC1 + Act (parallel with P2P_0)
            if local_count > 0:
                local_act = grouped_fc1_act(local_tokens, w1, local_tokens_per_expert_cpu, activation_func)
        elif prev_partner is not None:
            # Extract metadata from previous round's received data
            extract_metadata_and_tokens(prev_partner)
            if prev_partner in recv_buffers:
                # Compute previous round's received data (now guaranteed complete)
                recv_data = recv_buffers[prev_partner]
                buf_offset = recv_offsets[prev_partner]
                recv_act = all_recv_act_buffer[buf_offset:buf_offset + recv_data.shape[0]]
                recv_act.copy_(grouped_fc1_act(recv_data, w1, tokens_cpu[prev_partner], activation_func))
                recv_act_results[prev_partner] = recv_act
        nvtx_range_pop()

        prev_partner = partner
        prev_reqs = curr_reqs

    # Process last round: wait for last P2P and compute
    if len(partners) > 0:
        nvtx_range_push("dispatch_wait_last")
        for req in prev_reqs:
            req.wait()
        nvtx_range_pop()

        nvtx_range_push("fc1_compute_last")
        if prev_partner is not None:
            extract_metadata_and_tokens(prev_partner)
            if prev_partner in recv_buffers:
                recv_data = recv_buffers[prev_partner]
                buf_offset = recv_offsets[prev_partner]
                recv_act = all_recv_act_buffer[buf_offset:buf_offset + recv_data.shape[0]]
                recv_act.copy_(grouped_fc1_act(recv_data, w1, tokens_cpu[prev_partner], activation_func))
                recv_act_results[prev_partner] = recv_act
        nvtx_range_pop()

    nvtx_range_pop()  # dispatch_fc1_p2p
    return (local_tokens, local_act, recv_act_results, recv_buffers, partners, recv_offsets, tokens_cpu)


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
    partners: List[int],
    tokens_cpu: torch.Tensor,
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
        weight2: [num_local_experts, ffn_hidden, hidden] FC2 weight
        input_splits: [ep_size] token count to send to each rank
        output_splits: [ep_size] token count to receive from each rank
        ep_group: Expert Parallel process group
        overlap_ctx: P2P overlap context
        num_local_experts: Number of local experts
        partners: List of partner ranks
        tokens_cpu: [ep_size, num_local_experts] CPU int64 tensor from dispatch_fc1
        needs_backward: Whether backward is needed (skip merge/precompute if False)

    Returns:
        combined_output: [total_tokens, hidden] final output
        local_fc2: [local_count, hidden] local FC2 output
        all_expert_tokens: [total_recv, hidden] all tokens (expert-major order)
        all_tokens_per_expert: List of token counts per expert
        backward_indices: Dictionary of precomputed indices for backward
    """
    nvtx_range_push("fc2_combine_p2p")
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

    # local_tokens_per_expert is a 1D CPU tensor for grouped_gemm
    local_tokens_per_expert = tokens_cpu[my_rank] if tokens_cpu is not None else None

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
            nvtx_range_push("fc2_compute_first")
            peer_fc2_results[first_partner] = grouped_fc2(
                recv_act_results[first_partner], w2, tokens_cpu[first_partner])
            fc2_event.record(default_stream)
            has_pending_fc2 = True
            nvtx_range_pop()

    # Pipeline loop
    all_combine_reqs = []
    for round_idx, partner in enumerate(partners):
        # Start P2P (send FC2 result to partner, receive from partner)
        nvtx_range_push(f"combine_p2p_R{round_idx}")
        with torch.cuda.stream(comm_stream):
            if has_pending_fc2:
                comm_stream.wait_event(fc2_event)
            p2p_ops = []
            if input_splits[partner] > 0:
                p2p_ops.append(dist.P2POp(dist.irecv,
                    combined_output[input_offsets[partner]:input_offsets[partner+1]], partner, group=ep_group))
            if partner in peer_fc2_results:
                p2p_ops.append(dist.P2POp(dist.isend, peer_fc2_results[partner], partner, group=ep_group))
            if p2p_ops:
                all_combine_reqs.extend(dist.batch_isend_irecv(p2p_ops))
        nvtx_range_pop()

        # Parallel with current P2P: compute next round FC2 or local FC2
        has_pending_fc2 = False
        if round_idx + 1 < len(partners):
            next_partner = partners[round_idx + 1]
            if next_partner in recv_act_results:
                nvtx_range_push(f"fc2_compute_R{round_idx+1}")
                peer_fc2_results[next_partner] = grouped_fc2(
                    recv_act_results[next_partner], w2, tokens_cpu[next_partner])
                fc2_event.record(default_stream)
                has_pending_fc2 = True
                nvtx_range_pop()
        elif local_act is not None:
            # Last round: compute local FC2 (parallel with last P2P)
            nvtx_range_push("fc2_compute_local")
            local_fc2 = grouped_fc2(local_act, w2, local_tokens_per_expert)
            nvtx_range_pop()

    # =========================================================================
    # Merge tokens and precompute indices BEFORE req.wait()
    # =========================================================================
    nvtx_range_push("merge_precompute")
    all_expert_tokens = None
    all_tokens_per_expert = []
    backward_indices = {}

    if needs_backward and tokens_cpu is not None:
        # Collect peer tokens
        all_peer_tokens = torch.cat(
            [recv_buffers[i] for i in range(ep_size) if i != my_rank and i in recv_buffers],
            dim=0
        ) if recv_buffers else torch.empty(0, hidden_size, dtype=dtype, device=device)

        if num_local_experts > 1:
            all_expert_tokens, all_tokens_per_expert = merge_tokens_expert_major(
                local_tokens, all_peer_tokens, num_local_experts,
                tokens_cpu, my_rank, ep_size, device)
            backward_indices = precompute_backward_sort_indices(
                num_local_experts, ep_size, tokens_cpu, device)
        else:
            all_expert_tokens = torch.cat([local_tokens, all_peer_tokens], dim=0) if all_peer_tokens.numel() > 0 else local_tokens
            all_tokens_per_expert = [all_expert_tokens.shape[0]]
    nvtx_range_pop()

    # =========================================================================
    # Wait for all Combine P2P to complete
    # =========================================================================
    nvtx_range_push("combine_wait_all")
    for req in all_combine_reqs:
        req.wait()
    nvtx_range_pop()

    # Handle no partners case (ep_size=1): compute local FC2
    if len(partners) == 0 and local_act is not None:
        local_fc2 = grouped_fc2(local_act, w2, local_tokens_per_expert)

    # Write local result to combined_output
    if local_fc2 is not None:
        combined_output[local_start:local_start + local_count] = local_fc2

    nvtx_range_pop()  # fc2_combine_p2p
    return combined_output, local_fc2, all_expert_tokens, all_tokens_per_expert, backward_indices


__all__ = [
    'router_forward',
    'merge_tokens_expert_major',
    'precompute_backward_sort_indices',
    'dispatch_fc1_p2p_forward',
    'fc2_combine_p2p_forward',
]

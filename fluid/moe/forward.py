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

import math

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Tuple, Optional, Dict

from fluid.core.comm import MultiCardOverlapContext
from fluid.core.nvtx import nvtx_range, nvtx_range_push, nvtx_range_pop
from fluid.core.triton_kernels import permute_by_row_idx

try:
    from grouped_gemm.backend import gmm as _cutlass_gmm
    _HAS_GROUPED_GEMM = True
except Exception:
    _cutlass_gmm = None
    _HAS_GROUPED_GEMM = False

_LAYOUT_IDX_CACHE = {}


def _to_batch_sizes(tokens_per_expert) -> torch.Tensor:
    """Convert tokens_per_expert to CPU int64 tensor for grouped_gemm backend."""
    if torch.is_tensor(tokens_per_expert):
        if tokens_per_expert.dtype == torch.int64 and tokens_per_expert.device.type == "cpu":
            return tokens_per_expert
        return tokens_per_expert.to(dtype=torch.int64, device="cpu")
    return torch.tensor(tokens_per_expert, dtype=torch.int64)


def _grouped_gemm_or_none(
    A: torch.Tensor,
    B: torch.Tensor,
    tokens_per_expert,
    trans_a: bool = False,
    trans_b: bool = False,
) -> Optional[torch.Tensor]:
    """Try grouped_gemm backend, return None when unavailable."""
    if (not _HAS_GROUPED_GEMM) or (not A.is_cuda):
        return None
    batch_sizes = _to_batch_sizes(tokens_per_expert)
    return _cutlass_gmm(A, B, batch_sizes, trans_a=trans_a, trans_b=trans_b)


def grouped_fc1_act(tokens: torch.Tensor, w1: torch.Tensor,
                    tokens_per_expert: torch.Tensor, activation_func,
                    return_fc1: bool = False):
    """Compute FC1 + activation for all experts.

    Args:
        tokens: [total_tokens, hidden] input tokens
        w1: [num_experts, hidden, ffn_hidden] FC1 weights
        tokens_per_expert: 1D CPU int64 tensor of token counts per expert
        activation_func: Activation function
        return_fc1: If True, return (fc1, act); otherwise return act only.
    """
    if tokens.shape[0] == 0:
        empty = torch.empty(0, w1.shape[-1], dtype=tokens.dtype, device=tokens.device)
        return (empty, empty) if return_fc1 else empty

    num_experts = w1.shape[0]
    if num_experts == 1:
        fc1 = torch.matmul(tokens, w1[0])
        if return_fc1:
            return fc1, activation_func(fc1)
        return activation_func(fc1)

    # Multi-expert path: prefer grouped GEMM backend, fallback to for-loop.
    fc1 = _grouped_gemm_or_none(tokens, w1, tokens_per_expert, trans_b=False)
    if fc1 is None:
        fc1 = torch.empty(tokens.shape[0], w1.shape[-1], dtype=tokens.dtype, device=tokens.device)
        counts = tokens_per_expert.tolist() if torch.is_tensor(tokens_per_expert) else tokens_per_expert
        offset = 0
        for i, n in enumerate(counts):
            if n > 0:
                torch.mm(tokens[offset:offset+n], w1[i], out=fc1[offset:offset+n])
                offset += n
    if return_fc1:
        return fc1, activation_func(fc1)
    return activation_func(fc1)


def grouped_fc2(act: torch.Tensor, w2: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """Compute FC2 for all experts.

    Args:
        act: [total_tokens, ffn_hidden] activation values
        w2: [num_experts, ffn_hidden, hidden] FC2 weights
        tokens_per_expert: 1D CPU int64 tensor of token counts per expert
    """
    if act.shape[0] == 0:
        return torch.empty(0, w2.shape[-1], dtype=act.dtype, device=act.device)

    num_experts = w2.shape[0]
    if num_experts == 1:
        return torch.matmul(act, w2[0])

    # Multi-expert path: prefer grouped GEMM backend, fallback to for-loop.
    output = _grouped_gemm_or_none(act, w2, tokens_per_expert, trans_b=False)
    if output is not None:
        return output

    output = torch.empty(act.shape[0], w2.shape[-1], dtype=act.dtype, device=act.device)
    counts = tokens_per_expert.tolist() if torch.is_tensor(tokens_per_expert) else tokens_per_expert
    offset = 0
    for i, n in enumerate(counts):
        if n > 0:
            torch.mm(act[offset:offset+n], w2[i], out=output[offset:offset+n])
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
    capacity_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Router forward: token-to-expert assignment with top-k selection and capacity dropping.

    Args:
        hidden_states: [num_tokens, hidden_size] input tokens
        router_weight: [hidden_size, num_experts] router weight matrix
        num_experts: Total number of experts across all ranks
        top_k: Number of experts each token is sent to
        ep_group: Expert Parallel process group
        capacity_factor: Expert capacity = ceil(num_tokens * top_k / num_experts * capacity_factor)

    Returns:
        permuted_tokens: [num_kept, hidden_size] tokens sorted by expert (after capacity drop)
        permuted_probs: [num_kept] routing probabilities
        sorted_indices: [num_kept] positions in expanded [N*top_k] space (= kept_expanded_indices)
        token_ids: [num_kept] original token index (= sorted_indices // top_k, precomputed)
        input_splits: [ep_size] token count to send to each rank
        output_splits: [ep_size] token count to receive from each rank
        tokens_per_expert: [num_experts] clamped token counts per expert
        router_probs: [num_tokens, num_experts] full softmax probabilities (for backward)
        top_probs: [num_tokens, top_k] normalized top-k probabilities (for backward)
        top_indices: [num_tokens, top_k] top-k expert indices (for backward)
    """
    nvtx_range_push("router_forward")

    ep_size = ep_group.size()
    my_rank = ep_group.rank()
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[-1]
    device = hidden_states.device

    # Step 1: Router logits
    router_logits = torch.matmul(hidden_states.float(), router_weight.detach().float())

    # Step 2: Softmax + top-k
    router_probs = F.softmax(router_logits, dim=-1)
    top_probs, top_indices = torch.topk(router_probs, k=top_k, dim=-1)
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

    # Step 3: Flatten expert indices + sort by expert (stable for determinism)
    expanded_expert_indices = top_indices.reshape(-1)
    sorted_indices = torch.argsort(expanded_expert_indices, stable=True)
    sorted_expert_indices = expanded_expert_indices[sorted_indices]

    # Step 4: Count tokens per expert
    tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=num_experts)

    # Step 5: Capacity dropping — vectorized (no Python loop)
    expert_capacity = int(math.ceil(num_tokens * top_k / num_experts * capacity_factor))
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
    torch.cumsum(tokens_per_expert, dim=0, out=offsets[1:])
    within_expert_pos = torch.arange(num_tokens * top_k, device=device) - offsets[sorted_expert_indices]
    keep_mask = within_expert_pos < expert_capacity

    sorted_indices = sorted_indices[keep_mask]
    tokens_per_expert = tokens_per_expert.clamp(max=expert_capacity)

    # Step 6: Derive permuted_tokens/probs from sorted_indices directly
    # (avoids materialising [N*K, H] expanded_tokens intermediate)
    token_ids = sorted_indices // top_k
    permuted_tokens = hidden_states.index_select(0, token_ids)
    permuted_probs = top_probs.reshape(-1).index_select(0, sorted_indices)

    # Step 7: Compute AllToAll splits — vectorized
    experts_per_rank = num_experts // ep_size
    input_splits = tokens_per_expert.view(ep_size, experts_per_rank).sum(dim=1)

    all_input_splits = [torch.zeros_like(input_splits) for _ in range(ep_size)]
    dist.all_gather(all_input_splits, input_splits, group=ep_group)
    all_input_splits = torch.stack(all_input_splits)
    output_splits = all_input_splits[:, my_rank].clone()

    nvtx_range_pop()
    return (permuted_tokens, permuted_probs, sorted_indices, token_ids,
            input_splits, output_splits, tokens_per_expert,
            router_probs, top_probs, top_indices)


# =============================================================================
# Helper Functions
# =============================================================================

def merge_tokens_expert_major(
    local_tokens: torch.Tensor,
    all_peer_tokens: torch.Tensor,
    num_local_experts: int,
    tokens_list,
    my_rank: int,
    ep_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    """Merge tokens to expert-major with one rank-major concat + one index_select."""
    hidden_size = local_tokens.shape[-1] if local_tokens.numel() > 0 else all_peer_tokens.shape[-1]
    tok_dtype = local_tokens.dtype if local_tokens.numel() > 0 else all_peer_tokens.dtype

    if torch.is_tensor(tokens_list):
        tokens_2d = tokens_list.to(device=device, dtype=torch.int64)
    else:
        tokens_2d = torch.as_tensor(tokens_list, dtype=torch.int64, device=device)
    all_tokens_per_expert_t = tokens_2d.sum(dim=0)
    all_tokens_per_expert = all_tokens_per_expert_t.tolist()
    total_tokens = int(all_tokens_per_expert_t.sum().item())

    if total_tokens == 0:
        return (
            torch.empty(0, hidden_size, dtype=tok_dtype, device=device),
            all_tokens_per_expert,
        )

    # 1) Build rank-major tokens: [R0(E0..E{e-1}), R1(...), ...]
    rank_counts = tokens_2d.sum(dim=1).tolist()
    expected_local = rank_counts[my_rank]
    expected_peer = total_tokens - expected_local
    if local_tokens.shape[0] != expected_local:
        raise RuntimeError(
            f"merge_tokens_expert_major local shape mismatch: "
            f"expected {expected_local}, got tokens={local_tokens.shape[0]}"
        )
    if all_peer_tokens.shape[0] != expected_peer:
        raise RuntimeError(
            f"merge_tokens_expert_major peer shape mismatch: "
            f"expected {expected_peer}, got tokens={all_peer_tokens.shape[0]}"
        )

    tokens_parts = []
    peer_offset = 0
    for rank in range(ep_size):
        n_tok = rank_counts[rank]
        if n_tok == 0:
            continue
        if rank == my_rank:
            tokens_parts.append(local_tokens)
        else:
            tokens_parts.append(all_peer_tokens[peer_offset:peer_offset + n_tok])
            peer_offset += n_tok

    if len(tokens_parts) == 1:
        rank_major_tokens = tokens_parts[0]
    else:
        rank_major_tokens = torch.cat(tokens_parts, dim=0)

    # 2) Build row permutation for rank-major -> expert-major once.
    split_sizes_rank_major = tokens_2d.reshape(-1)
    sorted_idxs_rank_to_exp, _ = _get_cached_layout_indices(
        num_local_experts=num_local_experts,
        ep_size=ep_size,
        device=device,
    )
    row_idx_rank_to_exp = _build_row_reorder_index(
        split_sizes=split_sizes_rank_major,
        sorted_chunk_indices=sorted_idxs_rank_to_exp,
        device=device,
    )

    # 3) Single reorder via Triton gather (avoids intermediate allocation).
    all_expert_tokens = torch.empty(total_tokens, hidden_size, dtype=tok_dtype, device=device)
    permute_by_row_idx(rank_major_tokens, row_idx_rank_to_exp, all_expert_tokens)
    return all_expert_tokens, all_tokens_per_expert


def _build_row_reorder_index(
    split_sizes: torch.Tensor,
    sorted_chunk_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build row permutation index from chunk-size permutation without Python nested loops."""
    if split_sizes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    split_sizes = split_sizes.to(device=device, dtype=torch.int64)
    sorted_chunk_indices = sorted_chunk_indices.to(device=device, dtype=torch.int64)
    ordered_sizes = split_sizes.index_select(0, sorted_chunk_indices)
    total_rows = int(ordered_sizes.sum().item())
    if total_rows == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    # Per-chunk start offsets in source (rank-major) and destination (expert-major) layouts.
    src_offsets = torch.cumsum(split_sizes, dim=0) - split_sizes
    dst_offsets = torch.cumsum(ordered_sizes, dim=0) - ordered_sizes
    ordered_src_offsets = src_offsets.index_select(0, sorted_chunk_indices)

    ordered_chunk_ids = torch.repeat_interleave(
        torch.arange(ordered_sizes.numel(), dtype=torch.int64, device=device),
        ordered_sizes,
    )
    row_src_base = ordered_src_offsets.index_select(0, ordered_chunk_ids)
    row_dst_base = dst_offsets.index_select(0, ordered_chunk_ids)
    dst_rows = torch.arange(total_rows, dtype=torch.int64, device=device)
    return row_src_base + (dst_rows - row_dst_base)


def _get_cached_layout_indices(
    num_local_experts: int,
    ep_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get cached layout permutation indices on target device."""
    dev_key = (device.type, device.index)
    key = (num_local_experts, ep_size, dev_key)
    cached = _LAYOUT_IDX_CACHE.get(key, None)
    if cached is not None:
        return cached

    # rank-major: [R0_E0, R0_E1, ..., R1_E0, ...]
    # rank->exp permutation: [R0_E0, R1_E0, ..., R0_E1, R1_E1, ...]
    rank_major = torch.arange(ep_size * num_local_experts, dtype=torch.int64, device=device).view(ep_size, num_local_experts)
    sorted_idxs_rank_to_exp = rank_major.transpose(0, 1).reshape(-1)

    # exp-major -> rank-major permutation.
    # exp-major chunk order index is [E0_R0, E0_R1, ..., E1_R0, E1_R1, ...]
    # rank-major expects [R0_E0, R0_E1, ..., R1_E0, ...]
    exp_major = torch.arange(ep_size * num_local_experts, dtype=torch.int64, device=device).view(num_local_experts, ep_size)
    sorted_idxs_exp_to_rank = exp_major.transpose(0, 1).reshape(-1)

    _LAYOUT_IDX_CACHE[key] = (sorted_idxs_rank_to_exp, sorted_idxs_exp_to_rank)
    return _LAYOUT_IDX_CACHE[key]


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
    # Tensorized construction (CPU tensor -> device tensor), avoiding Python loops.
    split_sizes_rank_major = tokens_cpu.reshape(-1).to(device=device, dtype=torch.int64)
    split_sizes_exp_major = tokens_cpu.transpose(0, 1).reshape(-1).to(device=device, dtype=torch.int64)
    sorted_idxs_rank_to_exp, sorted_idxs_exp_to_rank = _get_cached_layout_indices(
        num_local_experts=num_local_experts,
        ep_size=ep_size,
        device=device,
    )

    return {
        'split_sizes_rank_major': split_sizes_rank_major,
        'sorted_idxs_rank_to_exp': sorted_idxs_rank_to_exp,
        'split_sizes_exp_major': split_sizes_exp_major,
        'sorted_idxs_exp_to_rank': sorted_idxs_exp_to_rank,
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

    FC1 pre-activation values are NOT saved; backward recomputes them to save memory.

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
    # Build local-to-global rank mapping for P2P ops
    global_ranks = dist.get_process_group_ranks(ep_group)
    device = tokens.device
    dtype = tokens.dtype
    hidden_size = tokens.shape[-1]
    element_size = torch.finfo(dtype).bits // 8

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
            # Move compact int32 metadata to CPU first, then cast to int64 on CPU.
            tokens_cpu[partner] = metadata_int32[:num_local_experts].cpu().to(torch.int64)
            # Rest is token data
            recv_buffers[partner] = full_buffer[1:]

    for round_idx, partner in enumerate(partners):
        # 1. Start current round P2P (async, returns immediately)
        nvtx_range_push(f"dispatch_p2p_R{round_idx}")
        with torch.cuda.stream(comm_stream):
            if round_idx == 0:
                comm_stream.wait_stream(default_stream)  # Wait for send_chunks preparation
            p2p_ops = []
            global_partner = global_ranks[partner]
            if partner in recv_buffers_with_metadata:
                p2p_ops.append(dist.P2POp(dist.irecv, recv_buffers_with_metadata[partner], global_partner, group=ep_group))
            if partner in send_chunks:
                p2p_ops.append(dist.P2POp(dist.isend, send_chunks[partner], global_partner, group=ep_group))
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
                recv_act = grouped_fc1_act(recv_data, w1, tokens_cpu[prev_partner], activation_func)
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
                recv_act = grouped_fc1_act(recv_data, w1, tokens_cpu[prev_partner], activation_func)
                recv_act_results[prev_partner] = recv_act
        nvtx_range_pop()

    nvtx_range_pop()  # dispatch_fc1_p2p
    return (
        local_tokens, local_act, recv_act_results, recv_buffers,
        partners, recv_offsets, tokens_cpu
    )


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
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[torch.Tensor], Optional[list], Optional[dict]]:
    """
    FC2 + Combine phase with P2P overlap.

    Pipeline:
        Round -1: Compute first peer's FC2
        Round i:  event.synchronize(), start P2P_i, compute next peer's FC2
        Final:    Compute local FC2 + merge/sort for backward (parallel with last P2P)

    Returns:
        combined_output, local_fc2,
        all_expert_tokens (expert-major), all_tokens_per_expert, backward_indices
    """
    nvtx_range_push("fc2_combine_p2p")
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    # Build local-to-global rank mapping for P2P ops
    global_ranks = dist.get_process_group_ranks(ep_group)
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

    # local_tokens_per_expert is a 1D CPU tensor of per-expert token counts
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
            global_partner = global_ranks[partner]
            if input_splits[partner] > 0:
                p2p_ops.append(dist.P2POp(dist.irecv,
                    combined_output[input_offsets[partner]:input_offsets[partner+1]], global_partner, group=ep_group))
            if partner in peer_fc2_results:
                p2p_ops.append(dist.P2POp(dist.isend, peer_fc2_results[partner], global_partner, group=ep_group))
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
    # Merge tokens to expert-major + precompute sort indices (overlap with last P2P)
    # recv_buffers are from dispatch phase and fully available.
    # =========================================================================
    all_expert_tokens = None
    all_tokens_per_expert = None
    backward_indices = None
    if needs_backward and tokens_cpu is not None:
        nvtx_range_push("merge_sort_fwd")
        # Build peer tokens concat
        peer_tokens = [recv_buffers[i] for i in range(ep_size) if i != my_rank and i in recv_buffers]
        if len(peer_tokens) == 0:
            all_peer_tokens = torch.empty(0, hidden_size, dtype=dtype, device=device)
        elif len(peer_tokens) == 1:
            all_peer_tokens = peer_tokens[0]
        else:
            all_peer_tokens = torch.cat(peer_tokens, dim=0)

        all_expert_tokens, all_tokens_per_expert = merge_tokens_expert_major(
            local_tokens, all_peer_tokens,
            num_local_experts, tokens_cpu, my_rank, ep_size, device,
        )
        backward_indices = precompute_backward_sort_indices(
            num_local_experts=num_local_experts, ep_size=ep_size,
            tokens_cpu=tokens_cpu, device=device,
        )
        backward_indices['row_idx_rank_to_exp'] = _build_row_reorder_index(
            backward_indices['split_sizes_rank_major'],
            backward_indices['sorted_idxs_rank_to_exp'], device,
        )
        backward_indices['row_idx_exp_to_rank'] = _build_row_reorder_index(
            backward_indices['split_sizes_exp_major'],
            backward_indices['sorted_idxs_exp_to_rank'], device,
        )
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

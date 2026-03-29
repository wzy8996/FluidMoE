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
from fluid.core.nvtx import nvtx_range_push, nvtx_range_pop

try:
    from grouped_gemm.backend import gmm as _cutlass_gmm
    _HAS_GROUPED_GEMM = True
except Exception:
    _cutlass_gmm = None
    _HAS_GROUPED_GEMM = False

try:
    from transformer_engine.pytorch.triton.permutation import (
        make_row_id_map as _te_make_row_id_map,
        permute_with_mask_map as _te_permute_with_mask_map,
    )
    _HAS_TE_PERMUTE = True
except ImportError:
    _te_make_row_id_map = None
    _te_permute_with_mask_map = None
    _HAS_TE_PERMUTE = False

_FUSED_ACT_PROBS_CACHE = {}

_LAYOUT_IDX_CACHE = {}
_PADDED_ROW_IDX_CACHE = {}
_GROUP_RANKS_CACHE = {}
_P2P_BUFFER_CACHE = {}


def _get_p2p_buffer(tag: str, rows: int, cols: int, dtype: torch.dtype, device: torch.device):
    """Grow-only persistent P2P buffer. Reuses memory across iterations,
    avoids torch.empty allocation and NCCL re-registration overhead."""
    key = (tag, cols, str(dtype), device.type, device.index if device.index is not None else -1)
    buf = _P2P_BUFFER_CACHE.get(key)
    if buf is None or buf.shape[0] < rows:
        buf = torch.empty(max(rows, 1), cols, dtype=dtype, device=device)
        _P2P_BUFFER_CACHE[key] = buf
    return buf[:rows]


def _get_group_ranks(group: dist.ProcessGroup):
    """Cache dist.get_process_group_ranks() to avoid repeated Python list allocation."""
    gid = id(group)
    cached = _GROUP_RANKS_CACHE.get(gid)
    if cached is not None:
        return cached
    ranks = dist.get_process_group_ranks(group)
    _GROUP_RANKS_CACHE[gid] = ranks
    return ranks


def _get_padded_row_indices(
    nle: int, ep_size: int, cap: int, device: torch.device,
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """Cached row reorder indices for uniform padding case (no GPU-CPU sync)."""
    key = (nle, ep_size, cap, device.type,
           device.index if device.index is not None else -1)
    cached = _PADDED_ROW_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    sorted_r2e, sorted_e2r = _get_cached_layout_indices(nle, ep_size, device)
    split_sizes = torch.full((ep_size * nle,), cap, dtype=torch.int64, device=device)
    row_idx_r2e = _build_row_reorder_index(split_sizes, sorted_r2e, device)
    row_idx_e2r = _build_row_reorder_index(split_sizes, sorted_e2r, device)
    _PADDED_ROW_IDX_CACHE[key] = (row_idx_r2e, row_idx_e2r)
    return row_idx_r2e, row_idx_e2r


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


def _fused_gelu_with_probs(x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return (torch.nn.functional.gelu(x) * probs).to(dtype)


def _fused_silu_with_probs(x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return (torch.nn.functional.silu(x) * probs).to(dtype)


# Pre-compile with torch.jit.script (ahead-of-time, no tracing, no hang risk)
try:
    _fused_gelu_with_probs = torch.jit.script(_fused_gelu_with_probs)
    _fused_silu_with_probs = torch.jit.script(_fused_silu_with_probs)
    _JIT_SCRIPT_OK = True
except Exception:
    _JIT_SCRIPT_OK = False


def _get_fused_act_with_probs(activation_func):
    """Return a fused activation*probs kernel.

    Uses torch.jit.script for known activations (gelu, silu) to avoid the
    torch.compile tracing hang in distributed P2P settings. Falls back to
    unfused for unknown activations.
    """
    key = id(activation_func)
    if key not in _FUSED_ACT_PROBS_CACHE:
        if _JIT_SCRIPT_OK:
            from fluid.core.te_ops import te_gelu
            if activation_func is te_gelu or activation_func is torch.nn.functional.gelu:
                _FUSED_ACT_PROBS_CACHE[key] = _fused_gelu_with_probs
                return _fused_gelu_with_probs
            if activation_func is torch.nn.functional.silu:
                _FUSED_ACT_PROBS_CACHE[key] = _fused_silu_with_probs
                return _fused_silu_with_probs
        # Fallback: unfused
        def unfused_act_with_probs(x, probs):
            dtype = x.dtype
            return (activation_func(x) * probs).to(dtype)
        _FUSED_ACT_PROBS_CACHE[key] = unfused_act_with_probs
    return _FUSED_ACT_PROBS_CACHE[key]


def grouped_fc1_act(tokens: torch.Tensor, w1: torch.Tensor,
                    tokens_per_expert: torch.Tensor, activation_func,
                    return_fc1: bool = False,
                    probs: Optional[torch.Tensor] = None):
    """Compute FC1 + activation (+ optional probs weighting) for all experts.

    When ``probs`` is provided, the output is ``activation(FC1(tokens)) * probs``
    (Megatron-aligned: probs applied between activation and FC2, fused via @jit_fuser).

    Args:
        tokens: [total_tokens, hidden] input tokens
        w1: [num_experts, hidden, ffn_hidden] FC1 weights
        tokens_per_expert: 1D CPU int64 tensor of token counts per expert
        activation_func: Activation function
        return_fc1: If True, return (fc1, act_weighted); otherwise return act_weighted only.
        probs: Optional [total_tokens] routing probabilities.
    """
    if tokens.shape[0] == 0:
        empty = torch.empty(0, w1.shape[-1], dtype=tokens.dtype, device=tokens.device)
        return (empty, empty) if return_fc1 else empty

    num_experts = w1.shape[0]
    if num_experts == 1:
        fc1 = torch.matmul(tokens, w1[0])
        if probs is not None:
            act = _get_fused_act_with_probs(activation_func)(fc1, probs.unsqueeze(-1))
        else:
            act = activation_func(fc1)
        if return_fc1:
            return fc1, act
        return act

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
    if probs is not None:
        act = _get_fused_act_with_probs(activation_func)(fc1, probs.unsqueeze(-1))
    else:
        act = activation_func(fc1)
    if return_fc1:
        return fc1, act
    return act


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Router forward: token-to-expert assignment with top-k selection and capacity dropping.
    Follows Megatron's routing format: routing_map [T, E] + routing_probs [T, E].

    Args:
        hidden_states: [num_tokens, hidden_size] input tokens
        router_weight: [hidden_size, num_experts] router weight matrix
        num_experts: Total number of experts across all ranks
        top_k: Number of experts each token is sent to
        ep_group: Expert Parallel process group
        capacity_factor: Expert capacity = ceil(num_tokens * top_k / num_experts * capacity_factor)

    Returns:
        permuted_tokens: [num_permuted, hidden_size] tokens sorted by expert
        permuted_probs: [num_permuted] routing probabilities (1D, expert-major order)
        sorted_indices: [num_permuted] original token indices (0..T-1, expert-major order)
        input_splits: [ep_size] token count to send to each rank
        output_splits: [ep_size] token count to receive from each rank
        tokens_per_expert: [num_experts] token counts per expert (after capacity drop)
        top_indices: [num_tokens, top_k] selected expert indices per token (for backward)
        top_probs: [num_tokens, top_k] softmax(top_k_logits) probabilities (for backward)
        row_id_map: [num_tokens, 2*num_experts+1] TE permute map (None if TE unavailable)
    """
    nvtx_range_push("router_forward")

    ep_size = ep_group.size()
    my_rank = ep_group.rank()
    num_tokens = hidden_states.shape[0]
    device = hidden_states.device

    # Step 1: Router logits
    # Matmul in input dtype matching Megatron (bf16 when wrapped by Float16Module)
    router_logits = torch.matmul(hidden_states, router_weight.detach().to(hidden_states.dtype))

    # Step 2: Top-k + softmax → routing_map [T, E] + routing_probs [T, E]
    # Aligned with Megatron post-softmax mode (default): topk first, then softmax on top-k.
    top_logits, top_indices = torch.topk(router_logits, k=top_k, dim=-1)
    top_probs = F.softmax(top_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
    routing_probs = torch.zeros_like(router_logits).scatter(1, top_indices, top_probs)
    routing_map = torch.zeros_like(router_logits).int().scatter(1, top_indices, 1).bool()

    # Step 3: Capacity dropping (probs policy, aligned with Megatron default)
    if capacity_factor > 0:
        expert_capacity = int(math.ceil(num_tokens * top_k / num_experts * capacity_factor))
        # For each expert column, keep top-capacity tokens by probability
        if expert_capacity < num_tokens:
            _, capacity_indices = torch.topk(
                routing_probs, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1.0).bool()
        else:
            capacity_mask = torch.ones_like(routing_probs).bool()
        # Keep only tokens that are both selected AND within capacity
        routing_map = torch.logical_and(routing_map, capacity_mask)
        routing_probs = routing_probs * routing_map.float()

    # Step 4: Permute tokens (Megatron-aligned)
    hidden_size = hidden_states.shape[1]

    if capacity_factor > 0 and _HAS_TE_PERMUTE:
        # ---- Megatron fused path: argsort padding + TE permute ----
        routing_map_int_T = routing_map.to(torch.int8).T.contiguous()  # [E, T]
        sorted_per_expert = routing_map_int_T.argsort(
            dim=-1, descending=True, stable=True
        )[:, :expert_capacity].contiguous()  # [E, capacity]
        sorted_indices = sorted_per_expert.reshape(-1)

        padded_routing_map = torch.zeros(
            num_tokens, num_experts, dtype=torch.bool, device=device)
        expert_ids_expand = torch.arange(
            num_experts, device=device
        ).unsqueeze(1).expand(-1, expert_capacity).reshape(-1)
        padded_routing_map[sorted_indices, expert_ids_expand] = True

        num_out_tokens = num_experts * expert_capacity
        tokens_per_expert = torch.full(
            (num_experts,), expert_capacity, dtype=torch.int64, device=device)

        with torch.no_grad():
            row_id_map = _te_make_row_id_map(
                padded_routing_map.int(), num_tokens, num_experts)
            permuted_tokens, _, permuted_probs = _te_permute_with_mask_map(
                hidden_states, row_id_map, routing_probs, None,
                num_tokens, num_experts, num_out_tokens, hidden_size, None,
            )

        # Compute splits (uniform in padded mode)
        experts_per_rank = num_experts // ep_size
        input_splits = tokens_per_expert.view(ep_size, experts_per_rank).sum(dim=1)
        output_splits = input_splits  # symmetric in padded mode
    else:
        # ---- Fallback: PyTorch ops (non-fused or non-padded) ----
        tokens_per_expert = routing_map.int().sum(dim=0).long()  # [E]
        routing_map_T = routing_map.bool().T.contiguous()  # [E, T]
        token_indices = torch.arange(
            num_tokens, device=device).unsqueeze(0).expand(num_experts, -1)
        sorted_indices = token_indices.masked_select(routing_map_T)
        permuted_probs = routing_probs.T.contiguous().masked_select(routing_map_T)
        permuted_tokens = hidden_states.index_select(0, sorted_indices)
        row_id_map = None  # not available in fallback path

        experts_per_rank = num_experts // ep_size
        input_splits = tokens_per_expert.view(ep_size, experts_per_rank).sum(dim=1)
        if capacity_factor > 0:
            output_splits = input_splits
        else:
            all_input_splits = [torch.zeros_like(input_splits) for _ in range(ep_size)]
            dist.all_gather(all_input_splits, input_splits, group=ep_group)
            all_input_splits = torch.stack(all_input_splits)
            output_splits = all_input_splits[:, my_rank].clone()

    nvtx_range_pop()
    return (permuted_tokens, permuted_probs, sorted_indices,
            input_splits, output_splits, tokens_per_expert,
            top_indices, top_probs, row_id_map)


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

    # 3) Single reorder via index_select gather.
    all_expert_tokens = torch.empty(total_tokens, hidden_size, dtype=tok_dtype, device=device)
    torch.index_select(rank_major_tokens, 0, row_idx_rank_to_exp, out=all_expert_tokens)
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
    pre_tokens_cpu: Optional[torch.Tensor] = None,
    permuted_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor],
           List[int], Dict[int, int], torch.Tensor,
           Optional[torch.Tensor], Optional[Dict[int, torch.Tensor]]]:
    """
    Dispatch phase with P2P overlap: parallel FC1+Act computation with P2P communication.

    Pipeline:
        Round 0: Start P2P_0, compute local FC1 + Act
        Round i: req.wait(P2P_{i-1}), start P2P_i, compute recv_{i-1} FC1 + Act
        Final:   req.wait(last round), compute last FC1 + Act

    P2P Metadata Piggybacking (skipped when pre_tokens_cpu is provided):
        Each P2P message includes a metadata row containing tokens_per_expert info.
        This eliminates the need for a separate AllGather operation.
        When pre_tokens_cpu is given (e.g. padding with known uniform metadata),
        metadata is skipped — no extra P2P row, no torch.cat, no .cpu() syncs.

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
        pre_tokens_cpu: Optional [ep_size, nle] CPU int64 tensor. When provided,
            metadata piggybacking is skipped (no .cpu() syncs, no torch.cat).
        permuted_probs: Optional [num_tokens] routing probabilities (Megatron-aligned:
            probs applied between activation and FC2). Sent via P2P alongside tokens.

    Returns:
        local_tokens: [local_count, hidden] local tokens
        local_act: [local_count, ffn_hidden] local activation output (probs-weighted if permuted_probs given)
        recv_act_results: Dict[partner -> act_tensor] - activation outputs from peers
        recv_buffers: Dict[partner -> token_tensor] - received tokens from peers
        partners: List of partner ranks
        recv_offsets: Dict of partner -> offset in buffer
        tokens_cpu: [ep_size, num_local_experts] CPU int64 tensor
        local_probs: Optional [local_count] local routing probs (None if permuted_probs is None)
        recv_probs: Optional Dict[partner -> probs_tensor] (None if permuted_probs is None)
    """
    nvtx_range_push("dispatch_fc1_p2p")
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    # Build local-to-global rank mapping for P2P ops (cached)
    global_ranks = _get_group_ranks(ep_group)
    device = tokens.device
    dtype = tokens.dtype
    hidden_size = tokens.shape[-1]
    element_size = torch.finfo(dtype).bits // 8
    use_metadata = pre_tokens_cpu is None  # skip piggybacking when metadata is known

    # Compute offsets
    input_offsets = [0]
    for s in input_splits:
        input_offsets.append(input_offsets[-1] + s)

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    local_count = input_splits[my_rank]
    local_start = input_offsets[my_rank]

    if use_metadata:
        # Extract local tokens_per_expert (GPU -> CPU sync required)
        local_tokens_per_expert = tokens_per_expert[my_rank * num_local_experts : (my_rank + 1) * num_local_experts]
        local_tokens_per_expert_cpu = local_tokens_per_expert.to(dtype=torch.int64).cpu()
        tokens_cpu = torch.zeros(ep_size, num_local_experts, dtype=torch.int64)
        tokens_cpu[my_rank] = local_tokens_per_expert_cpu
    else:
        # Metadata known a priori (e.g. padding) — no GPU-CPU sync needed
        tokens_cpu = pre_tokens_cpu
        local_tokens_per_expert_cpu = tokens_cpu[my_rank]

    # Get Round-Robin scheduled partners
    partners = []
    for round_idx in range(overlap_ctx.num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner != -1:
            partners.append(partner)

    # Extract local tokens (no clone needed - data will be copied in merge_tokens_expert_major anyway)
    local_tokens = tokens[local_start:local_start + local_count] if local_count > 0 else torch.empty(0, hidden_size, dtype=dtype, device=device)

    # Extract local probs (Megatron-aligned: sent via P2P for act * probs weighting)
    if permuted_probs is not None:
        local_probs = permuted_probs[local_start:local_start + local_count] if local_count > 0 else torch.empty(0, dtype=permuted_probs.dtype, device=device)
        probs_dtype = permuted_probs.dtype
    else:
        local_probs = None
        probs_dtype = None

    # Prepare send data
    # Build per-partner send/recv data as lists (avoid dict lookups in loop)
    # Payload fusion: tokens + probs packed into one tensor [N, H+1] to halve NCCL ops
    has_probs = permuted_probs is not None
    send_chunk_list = []       # None if no data to send
    recv_meta_list = []        # None if no data to recv (1D flat buffer)
    recv_buffers = {}          # populated after metadata extraction
    recv_probs_list = []       # extracted from flat recv buffer

    if use_metadata:
        int32_as_elements = 4 // element_size
        metadata_elements = num_local_experts * int32_as_elements
        # metadata occupies 1 row worth of elements in the flat buffer
        meta_flat_elems = hidden_size

    for partner in partners:
        n_send = input_splits[partner]
        recv_size = output_splits[partner]

        # Send buffer: 1D flat [meta_elems? + N*H + N?] — tokens then probs
        if n_send > 0:
            token_chunk = tokens[input_offsets[partner]:input_offsets[partner+1]]
            parts = []
            if use_metadata:
                partner_metadata = tokens_per_expert[partner * num_local_experts : (partner + 1) * num_local_experts]
                metadata_int32 = partner_metadata.to(torch.int32).to(device)
                metadata_row = torch.zeros(meta_flat_elems, dtype=dtype, device=device)
                metadata_as_dtype = metadata_int32.view(torch.int8).view(dtype)
                metadata_row[:metadata_elements] = metadata_as_dtype
                parts.append(metadata_row)
            parts.append(token_chunk.reshape(-1))
            if has_probs:
                probs_chunk = permuted_probs[input_offsets[partner]:input_offsets[partner+1]]
                parts.append(probs_chunk.to(dtype))
            send_chunk_list.append(torch.cat(parts))
        else:
            send_chunk_list.append(None)

        # Recv buffer: 1D flat, persistent
        if recv_size > 0:
            meta_elems = meta_flat_elems if use_metadata else 0
            total_elems = meta_elems + recv_size * hidden_size + (recv_size if has_probs else 0)
            recv_meta_list.append(_get_p2p_buffer(
                f"dispatch_recv_{partner}", total_elems, 1, dtype, device).squeeze(-1))
        else:
            recv_meta_list.append(None)
        recv_probs_list.append(None)

    # Compute each partner's offset in buffer
    recv_offsets = {}
    offset = 0
    for i in range(ep_size):
        if i != my_rank:
            recv_offsets[i] = offset
            offset += output_splits[i]

    # =========================================================================
    # Dispatch Phase Pipeline with event-based sync (no CPU blocking)
    # Key insight: use a per-round event on comm_stream to hand completed P2P
    # data to default_stream, while still overlapping P2P_i with FC1 for
    # round i-1. The event must be recorded only after NCCL completion.
    # =========================================================================
    prev_idx = -1
    all_reqs = []
    # Pre-compute per-round data to avoid dict/function-call overhead in loop
    _global_partners = [global_ranks[p] for p in partners]
    _dispatch_events = [overlap_ctx.get_round_event("moe_dispatch", i) for i in range(len(partners))]
    p2p_events = []
    recv_act_results = {}
    local_act = None

    def extract_metadata_and_tokens(idx):
        """Extract metadata + split flat buffer into tokens and probs (zero-copy views)."""
        partner = partners[idx]
        flat = recv_meta_list[idx]
        if flat is not None:
            off = 0
            if use_metadata:
                metadata_as_dtype = flat[:metadata_elements]
                metadata_int32 = metadata_as_dtype.view(torch.int8).view(torch.int32)
                tokens_cpu[partner] = metadata_int32[:num_local_experts].cpu().to(torch.int64)
                off = meta_flat_elems
            n = output_splits[partner]
            # tokens: view as [N, H] — contiguous because flat buffer is contiguous
            recv_buffers[partner] = flat[off:off + n * hidden_size].view(n, hidden_size)
            if has_probs:
                probs_off = off + n * hidden_size
                recv_probs_list[idx] = flat[probs_off:probs_off + n].to(probs_dtype)

    for round_idx, partner in enumerate(partners):
        nvtx_range_push(f"dispatch_p2p_R{round_idx}")
        with torch.cuda.stream(comm_stream):
            if round_idx == 0:
                comm_stream.wait_stream(default_stream)
            # Fused payload: tokens+probs in one tensor → 2 NCCL ops instead of 4
            p2p_ops = []
            gp = _global_partners[round_idx]
            if recv_meta_list[round_idx] is not None:
                recv_meta_list[round_idx].record_stream(comm_stream)
                p2p_ops.append(dist.P2POp(dist.irecv, recv_meta_list[round_idx], gp, group=ep_group))
            if send_chunk_list[round_idx] is not None:
                send_chunk_list[round_idx].record_stream(comm_stream)
                p2p_ops.append(dist.P2POp(dist.isend, send_chunk_list[round_idx], gp, group=ep_group))
            curr_reqs = dist.batch_isend_irecv(p2p_ops) if p2p_ops else []
            all_reqs.extend(curr_reqs)
            if curr_reqs:
                curr_reqs[-1].wait()
            evt = _dispatch_events[round_idx]
            evt.record(comm_stream)
            p2p_events.append(evt)
        nvtx_range_pop()

        # 2. Compute FC1 + Act (overlaps with current round's P2P, no CPU blocking)
        nvtx_range_push(f"fc1_compute_R{round_idx}")
        if round_idx == 0:
            if local_count > 0:
                local_act = grouped_fc1_act(
                    local_tokens, weight1, local_tokens_per_expert_cpu,
                    activation_func, probs=local_probs)
        elif prev_idx >= 0:
            default_stream.wait_event(p2p_events[round_idx - 1])
            extract_metadata_and_tokens(prev_idx)
            prev_p = partners[prev_idx]
            if prev_p in recv_buffers:
                recv_data = recv_buffers[prev_p]
                partner_probs = recv_probs_list[prev_idx]
                recv_act = grouped_fc1_act(
                    recv_data, weight1, tokens_cpu[prev_p],
                    activation_func, probs=partner_probs)
                recv_act_results[prev_p] = recv_act
        nvtx_range_pop()

        prev_idx = round_idx

    # Process last round
    if len(partners) > 0:
        nvtx_range_push("fc1_compute_last")
        default_stream.wait_event(p2p_events[-1])
        last_idx = len(partners) - 1
        extract_metadata_and_tokens(last_idx)
        last_p = partners[last_idx]
        if last_p in recv_buffers:
            recv_data = recv_buffers[last_p]
            partner_probs = recv_probs_list[last_idx]
            recv_act = grouped_fc1_act(
                recv_data, weight1, tokens_cpu[last_p],
                activation_func, probs=partner_probs)
            recv_act_results[last_p] = recv_act
        nvtx_range_pop()

    # Make default_stream itself observe NCCL completion before returning.
    # This second wait is required in addition to the comm_stream-side wait
    # above: ProcessGroupNCCL tracks allocator safety against the user-facing
    # streams that call wait()/synchronize().
    if all_reqs:
        all_reqs[-1].wait()

    nvtx_range_pop()  # dispatch_fc1_p2p
    return (
        local_tokens, local_act, recv_act_results, recv_buffers,
        partners, recv_offsets, tokens_cpu,
        local_probs,
        {partners[i]: recv_probs_list[i] for i in range(len(partners)) if recv_probs_list[i] is not None} if permuted_probs is not None else None,
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
    local_probs: Optional[torch.Tensor] = None,
    recv_probs: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[torch.Tensor], Optional[list], Optional[dict],
           Optional[torch.Tensor]]:
    """
    FC2 + Combine phase with P2P overlap.

    Pipeline:
        Round -1: Compute first peer's FC2
        Round i:  event.synchronize(), start P2P_i, compute next peer's FC2
        Final:    Compute local FC2 + merge/sort for backward (parallel with last P2P)

    Returns:
        combined_output, local_fc2,
        all_expert_tokens (expert-major), all_tokens_per_expert, backward_indices,
        all_expert_probs (expert-major, for backward probs gradient; None if local_probs is None)
    """
    nvtx_range_push("fc2_combine_p2p")
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    # Build local-to-global rank mapping for P2P ops (cached)
    global_ranks = _get_group_ranks(ep_group)
    device = local_tokens.device
    dtype = local_tokens.dtype
    hidden_size = weight2.shape[-1]

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
    combined_output = _get_p2p_buffer("combine_output", total_output, hidden_size, dtype, device)

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
                recv_act_results[first_partner], weight2, tokens_cpu[first_partner])
            fc2_event.record(default_stream)
            has_pending_fc2 = True
            nvtx_range_pop()

    # Pipeline loop
    # Pre-compute per-round data
    _combine_global_partners = [global_ranks[p] for p in partners]
    _combine_recv_slices = []
    for i, partner in enumerate(partners):
        if input_splits[partner] > 0:
            _combine_recv_slices.append(combined_output[input_offsets[partner]:input_offsets[partner+1]])
        else:
            _combine_recv_slices.append(None)

    all_combine_reqs = []
    for round_idx, partner in enumerate(partners):
        nvtx_range_push(f"combine_p2p_R{round_idx}")
        with torch.cuda.stream(comm_stream):
            if has_pending_fc2:
                comm_stream.wait_event(fc2_event)
            p2p_ops = []
            gp = _combine_global_partners[round_idx]
            recv_slice = _combine_recv_slices[round_idx]
            if recv_slice is not None:
                recv_slice.record_stream(comm_stream)
                p2p_ops.append(dist.P2POp(dist.irecv, recv_slice, gp, group=ep_group))
            if partner in peer_fc2_results:
                peer_fc2_results[partner].record_stream(comm_stream)
                p2p_ops.append(dist.P2POp(dist.isend, peer_fc2_results[partner], gp, group=ep_group))
            if p2p_ops:
                curr_reqs = dist.batch_isend_irecv(p2p_ops)
                all_combine_reqs.extend(curr_reqs)
                if curr_reqs:
                    curr_reqs[-1].wait()
        nvtx_range_pop()

        # Parallel with current P2P: compute next round FC2 or local FC2
        has_pending_fc2 = False
        if round_idx + 1 < len(partners):
            next_partner = partners[round_idx + 1]
            if next_partner in recv_act_results:
                nvtx_range_push(f"fc2_compute_R{round_idx+1}")
                peer_fc2_results[next_partner] = grouped_fc2(
                    recv_act_results[next_partner], weight2, tokens_cpu[next_partner])
                fc2_event.record(default_stream)
                has_pending_fc2 = True
                nvtx_range_pop()
        elif local_act is not None:
            # Last round: compute local FC2 (parallel with last P2P)
            nvtx_range_push("fc2_compute_local")
            local_fc2 = grouped_fc2(local_act, weight2, local_tokens_per_expert)
            nvtx_range_pop()

    # =========================================================================
    # Merge tokens to expert-major + precompute sort indices (overlap with last P2P)
    # recv_buffers are from dispatch phase and fully available.
    # =========================================================================
    all_expert_tokens = None
    all_tokens_per_expert = None
    backward_indices = None
    all_expert_probs = None
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

        # Build peer probs concat (Megatron-aligned: probs in expert-major for backward)
        if local_probs is not None and recv_probs is not None:
            peer_probs_list = [recv_probs[i] for i in range(ep_size)
                               if i != my_rank and i in recv_probs]
            if len(peer_probs_list) == 0:
                all_peer_probs = torch.empty(0, dtype=local_probs.dtype, device=device)
            elif len(peer_probs_list) == 1:
                all_peer_probs = peer_probs_list[0]
            else:
                all_peer_probs = torch.cat(peer_probs_list, dim=0)

        # Check for uniform padding case (CPU tensor — no GPU sync)
        _cap = int(tokens_cpu.view(-1)[0])
        _is_padded = bool((tokens_cpu == _cap).all())

        if _is_padded and _cap > 0:
            # Fast path: all indices are deterministic and cached.
            # Eliminates GPU-CPU syncs from _build_row_reorder_index and merge.
            row_idx_r2e, row_idx_e2r = _get_padded_row_indices(
                num_local_experts, ep_size, _cap, device)

            n_per_rank = num_local_experts * _cap
            total_tokens_merge = ep_size * n_per_rank

            # Build rank-major tokens
            rank_parts = []
            peer_offset = 0
            for rank in range(ep_size):
                if rank == my_rank:
                    rank_parts.append(local_tokens)
                else:
                    rank_parts.append(all_peer_tokens[peer_offset:peer_offset + n_per_rank])
                    peer_offset += n_per_rank
            rank_major = torch.cat(rank_parts, dim=0) if len(rank_parts) > 1 else rank_parts[0]

            # Reorder to expert-major using cached index
            all_expert_tokens = torch.empty(total_tokens_merge, hidden_size, dtype=dtype, device=device)
            torch.index_select(rank_major, 0, row_idx_r2e, out=all_expert_tokens)

            # Merge probs to expert-major (same layout as tokens)
            if local_probs is not None and recv_probs is not None:
                rank_probs_parts = []
                peer_probs_offset = 0
                for rank in range(ep_size):
                    if rank == my_rank:
                        rank_probs_parts.append(local_probs)
                    else:
                        rank_probs_parts.append(
                            all_peer_probs[peer_probs_offset:peer_probs_offset + n_per_rank])
                        peer_probs_offset += n_per_rank
                rank_major_probs = torch.cat(rank_probs_parts, dim=0) if len(rank_probs_parts) > 1 else rank_probs_parts[0]
                all_expert_probs = rank_major_probs[row_idx_r2e]

            all_tokens_per_expert = [ep_size * _cap] * num_local_experts

            # Backward indices from cache (no _build_row_reorder_index calls)
            sorted_r2e, sorted_e2r = _get_cached_layout_indices(
                num_local_experts, ep_size, device)
            split_uniform = torch.full(
                (ep_size * num_local_experts,), _cap, dtype=torch.int64, device=device)
            backward_indices = {
                'split_sizes_rank_major': split_uniform,
                'sorted_idxs_rank_to_exp': sorted_r2e,
                'split_sizes_exp_major': split_uniform,
                'sorted_idxs_exp_to_rank': sorted_e2r,
                'row_idx_rank_to_exp': row_idx_r2e,
                'row_idx_exp_to_rank': row_idx_e2r,
            }
        else:
            # General path: non-uniform splits, compute from scratch
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
            # Merge probs using same layout indices (general path)
            if local_probs is not None and recv_probs is not None:
                rank_probs_parts = []
                peer_probs_offset = 0
                for rank in range(ep_size):
                    if rank == my_rank:
                        rank_probs_parts.append(local_probs)
                    else:
                        rank_probs_parts.append(
                            all_peer_probs[peer_probs_offset:peer_probs_offset + int(tokens_cpu[rank].sum().item())])
                        peer_probs_offset += int(tokens_cpu[rank].sum().item())
                rank_major_probs = torch.cat(rank_probs_parts, dim=0) if len(rank_probs_parts) > 1 else rank_probs_parts[0]
                all_expert_probs = rank_major_probs[backward_indices['row_idx_rank_to_exp']]
        nvtx_range_pop()

    # Same rationale as dispatch_fc1_p2p: make default_stream observe NCCL
    # completion before local writes read from combined_output or the function
    # returns and drops temporary tensors.
    nvtx_range_push("combine_wait_all")
    if all_combine_reqs:
        all_combine_reqs[-1].wait()
    nvtx_range_pop()

    # Handle no partners case (ep_size=1): compute local FC2
    if len(partners) == 0 and local_act is not None:
        local_fc2 = grouped_fc2(local_act, weight2, local_tokens_per_expert)

    # Write local result to combined_output
    if local_fc2 is not None:
        combined_output[local_start:local_start + local_count] = local_fc2

    nvtx_range_pop()  # fc2_combine_p2p
    return combined_output, local_fc2, all_expert_tokens, all_tokens_per_expert, backward_indices, all_expert_probs


def pad_moe_dispatch(
    permuted_tokens: torch.Tensor,
    permuted_probs: torch.Tensor,
    sorted_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    cap_per_rank: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Pad dispatched tokens so every expert has exactly cap_per_rank tokens.

    Tokens beyond cap_per_rank are truncated; experts with fewer tokens are
    zero-padded.  Returns a boolean ``real_mask`` so the caller can distinguish
    real tokens from padding in the backward pass.

    Fully vectorized — no Python loop, no per-expert GPU-CPU sync.

    Args:
        permuted_tokens: [num_permuted, hidden_size] expert-major ordered tokens
        permuted_probs: [num_permuted] routing probabilities
        sorted_indices: [num_permuted] original token indices (0..T-1)
        tokens_per_expert: [num_experts] token counts per expert
        cap_per_rank: capacity per expert (uniform)
        num_experts: total number of experts

    Returns:
        padded_tokens, padded_probs, padded_sorted_indices,
        padded_tokens_per_expert, real_mask
    """
    device = permuted_tokens.device
    hidden_size = permuted_tokens.shape[1]
    num_real = permuted_tokens.shape[0]
    total_padded = num_experts * cap_per_rank

    padded_tokens = torch.zeros(total_padded, hidden_size,
                                dtype=permuted_tokens.dtype, device=device)
    padded_probs = torch.zeros(total_padded, dtype=permuted_probs.dtype,
                               device=device)
    padded_sorted_indices = torch.zeros(total_padded, dtype=sorted_indices.dtype,
                                        device=device)
    real_mask = torch.zeros(total_padded, dtype=torch.bool, device=device)

    if num_real > 0:
        tpe = tokens_per_expert[:num_experts].to(dtype=torch.int64, device=device)
        n_copy = tpe.clamp(max=cap_per_rank)

        # Expert boundaries in compact layout
        cum = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
        torch.cumsum(n_copy, dim=0, out=cum[1:])

        # Per-token expert id via searchsorted
        row_ids = torch.arange(num_real, dtype=torch.int64, device=device)
        expert_ids = torch.searchsorted(cum[1:], row_ids, right=True)
        within_pos = row_ids - cum[:-1][expert_ids]

        # Destination in padded buffer: expert_id * cap_per_rank + within_pos
        dst_idx = expert_ids * cap_per_rank + within_pos

        padded_tokens.index_copy_(0, dst_idx, permuted_tokens)
        padded_probs.index_copy_(0, dst_idx, permuted_probs)
        padded_sorted_indices.index_copy_(0, dst_idx, sorted_indices)
        real_mask[dst_idx] = True

    padded_tpe = torch.full((num_experts,), cap_per_rank,
                            dtype=tokens_per_expert.dtype, device=device)
    return (padded_tokens, padded_probs, padded_sorted_indices,
            padded_tpe, real_mask)


__all__ = [
    'router_forward',
    'pad_moe_dispatch',
    'merge_tokens_expert_major',
    'precompute_backward_sort_indices',
    'dispatch_fc1_p2p_forward',
    'fc2_combine_p2p_forward',
]

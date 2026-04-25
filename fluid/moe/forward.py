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
from fluid.core.p2p_backend import get_p2p_backend
from fluid.core.python_profile import profile_section
from fluid.core.te_ops import te_gelu

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


class EPPlan:
    """Lightweight pre-computed static plan for padded MoE dispatch/combine.

    Caches only Python values and event references — zero GPU buffer allocation.
    The single GPU tensor (tokens_per_expert_gpu) is a few dozen bytes.
    """
    __slots__ = (
        'ep_size', 'my_rank', 'num_local_experts', 'expert_capacity',
        'S', 'input_splits_list', 'output_splits_list', 'input_offsets',
        'pre_tokens_cpu', 'local_tokens_per_expert_cpu',
        'local_count', 'local_start',
        'partners', 'global_partners', 'num_rounds',
        'dispatch_events', 'combine_events',
        'partner_to_index',
        'recv_total_elems', 'recv_offsets',
        'tokens_per_expert_gpu', 'all_tokens_per_expert',
        'expert_ids_gpu',
    )

    def __init__(self, num_tokens, num_experts, top_k, capacity_factor,
                 ep_group, overlap_ctx, device, hidden_size):
        self.ep_size = ep_group.size()
        self.my_rank = ep_group.rank()
        self.num_local_experts = num_experts // self.ep_size

        self.expert_capacity = int(math.ceil(
            num_tokens * top_k / num_experts * capacity_factor))
        self.S = self.num_local_experts * self.expert_capacity

        self.input_splits_list = [self.S] * self.ep_size
        self.output_splits_list = [self.S] * self.ep_size
        self.input_offsets = [i * self.S for i in range(self.ep_size + 1)]

        self.pre_tokens_cpu = torch.full(
            (self.ep_size, self.num_local_experts),
            self.expert_capacity, dtype=torch.int64)
        self.local_tokens_per_expert_cpu = self.pre_tokens_cpu[self.my_rank]

        self.local_count = self.S
        self.local_start = self.my_rank * self.S

        self.num_rounds = overlap_ctx.num_rounds
        self.partners = []
        for r in range(self.num_rounds):
            p = overlap_ctx.get_partner(self.my_rank, r)
            if p != -1:
                self.partners.append(p)
        global_ranks = _get_group_ranks(ep_group)
        self.global_partners = [global_ranks[p] for p in self.partners]

        self.dispatch_events = [
            overlap_ctx.get_round_event("moe_dispatch", i)
            for i in range(len(self.partners))]
        self.combine_events = [
            overlap_ctx.get_round_event("moe_combine", i)
            for i in range(len(self.partners))]
        self.partner_to_index = {partner: idx for idx, partner in enumerate(self.partners)}

        # Pre-computed recv buffer sizes (fused payload: tokens + probs)
        self.recv_total_elems = []
        for p in self.partners:
            n = self.S
            total = n * hidden_size + n  # tokens + probs
            self.recv_total_elems.append(total)

        self.recv_offsets = {}
        offset = 0
        for i in range(self.ep_size):
            if i != self.my_rank:
                self.recv_offsets[i] = offset
                offset += self.S

        # Only GPU tensor: tiny (num_experts int64 ≈ few dozen bytes)
        self.tokens_per_expert_gpu = torch.full(
            (num_experts,), self.expert_capacity, dtype=torch.int64, device=device)
        self.all_tokens_per_expert = [self.ep_size * self.expert_capacity] * self.num_local_experts

        # Padded expert_ids for router_backward: [0]*cap + [1]*cap + ...
        # Using a Python int in repeat_interleave keeps the op fully async
        # (avoids GPU-tensor sync in the scheduler's dW window).
        self.expert_ids_gpu = torch.arange(
            num_experts, device=device
        ).repeat_interleave(self.expert_capacity)


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


def _to_batch_sizes_list(tokens_per_expert):
    if torch.is_tensor(tokens_per_expert):
        if tokens_per_expert.device.type == "cpu":
            return tokens_per_expert.tolist()
        return tokens_per_expert.to(dtype=torch.int64, device="cpu").tolist()
    return tokens_per_expert


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
        counts = _to_batch_sizes_list(tokens_per_expert)
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
    ep_plan: Optional['EPPlan'] = None,
    router_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor]]:
    """
    Router forward: token-to-expert assignment with top-k selection and capacity dropping.
    Follows Megatron's routing format: routing_map [T, E] + routing_probs [T, E].

    Returns:
        permuted_tokens: [num_permuted, hidden_size] tokens sorted by expert
        permuted_probs: [num_permuted] routing probabilities (1D, expert-major order)
        sorted_indices: [num_permuted] original token indices (expert-major order)
        tokens_per_expert: [num_experts] token counts per expert (after capacity drop)
        top_indices: [num_tokens, top_k] selected expert indices per token (for backward)
        top_probs: [num_tokens, top_k] softmax(top_k_logits) probabilities (for backward)
        router_logits: [num_tokens, num_experts] pre-softmax logits in router_dtype
            (needed for aux/z loss backward)
        row_id_map: [num_tokens, 2*num_experts+1] TE permute map (None if TE unavailable)
    """
    nvtx_range_push("router_forward")
    assert capacity_factor > 0, "router_forward only supports fixed-capacity padded mode"
    assert ep_plan is not None, "router_forward requires a pre-built EPPlan"

    num_tokens = hidden_states.shape[0]
    device = hidden_states.device

    # Step 1: Router logits. Compute in router_dtype (fp32 for stability, else bf16).
    if router_dtype != hidden_states.dtype:
        router_logits = torch.matmul(
            hidden_states.to(router_dtype),
            router_weight.detach().to(router_dtype))
    else:
        router_logits = torch.matmul(hidden_states, router_weight.detach())

    # Step 2: Top-k + softmax → routing_map [T, E] + routing_probs [T, E]
    # Aligned with Megatron post-softmax mode (default): topk first, then softmax on top-k.
    top_logits, top_indices = torch.topk(router_logits, k=top_k, dim=-1)
    top_probs = F.softmax(top_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
    routing_probs = torch.zeros_like(router_logits).scatter(1, top_indices, top_probs)
    routing_map = torch.zeros_like(router_logits).int().scatter(1, top_indices, 1).bool()

    # Step 3: Capacity dropping (probs policy, aligned with Megatron default)
    expert_capacity = int(math.ceil(num_tokens * top_k / num_experts * capacity_factor))
    if expert_capacity < num_tokens:
        _, capacity_indices = torch.topk(
            routing_probs, k=expert_capacity, dim=0, sorted=False)
        capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1.0).bool()
    else:
        capacity_mask = torch.ones_like(routing_probs).bool()
    routing_map = torch.logical_and(routing_map, capacity_mask)
    routing_probs = routing_probs * routing_map.float()

    # Step 4: Permute tokens (padded path only)
    hidden_size = hidden_states.shape[1]

    # Build padded_routing_map: each expert has exactly expert_capacity True entries.
    # Argsort (real-first, padding from non-routed) selects which tokens belong
    # to each expert's slots; padded_routing_map marks them.
    routing_map_int_T = routing_map.to(torch.int8).T.contiguous()  # [E, T]
    sorted_per_expert = routing_map_int_T.argsort(
        dim=-1, descending=True, stable=True
    )[:, :expert_capacity].contiguous()  # [E, capacity]
    expert_ids_expand = ep_plan.expert_ids_gpu  # [E*cap], const, cached in EPPlan
    padded_routing_map = torch.zeros(
        num_tokens, num_experts, dtype=torch.bool, device=device)
    padded_routing_map[sorted_per_expert.reshape(-1), expert_ids_expand] = True

    # Sorted_indices in TE-compatible ordering: token ids ASCENDING within
    # each expert's slots. Compute via sort(sorted_per_expert) (fully async,
    # shape known) instead of torch.nonzero (implicit CPU-GPU sync on
    # variable-length output shape).
    sorted_indices = sorted_per_expert.sort(dim=-1).values.reshape(-1)

    num_out_tokens = num_experts * expert_capacity
    tokens_per_expert = ep_plan.tokens_per_expert_gpu

    if _HAS_TE_PERMUTE:
        # Fused TE kernel path: permute tokens + gather probs in one call.
        # TE writes zeros for padded slots; downstream MoE multiplies by
        # probs (=0 at padded slots), so padded contributions are 0 as expected.
        with torch.no_grad():
            row_id_map = _te_make_row_id_map(
                padded_routing_map.int(), num_tokens, num_experts)
            permuted_tokens, _, permuted_probs = _te_permute_with_mask_map(
                hidden_states, row_id_map, routing_probs, None,
                num_tokens, num_experts, num_out_tokens, hidden_size, None,
            )
    else:
        # Fallback (no TE): gather with TE-compatible sorted_indices.
        permuted_tokens = hidden_states.index_select(0, sorted_indices)
        permuted_probs = routing_probs[sorted_indices, expert_ids_expand]
        row_id_map = None

    nvtx_range_pop()
    return (permuted_tokens, permuted_probs, sorted_indices,
            tokens_per_expert, top_indices, top_probs,
            router_logits, routing_map, row_id_map)


# =============================================================================
# Helper Functions
# =============================================================================

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


# =============================================================================
# Phase 1: Dispatch + FC1 with P2P Overlap
# =============================================================================

def dispatch_fc1_p2p_forward(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    ep_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    activation_func,
    num_local_experts: int,
    ep_plan: 'EPPlan',
    permuted_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor],
           List[int], Dict[int, int], torch.Tensor,
           Optional[torch.Tensor], Optional[Dict[int, torch.Tensor]]]:
    with profile_section("moe_fwd.dispatch_fc1_p2p"):
        return _dispatch_fc1_p2p_forward_impl(
            tokens, weight1, ep_group, overlap_ctx, activation_func,
            num_local_experts, ep_plan, permuted_probs=permuted_probs,
        )


def _dispatch_fc1_p2p_forward_impl(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    ep_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    activation_func,
    num_local_experts: int,
    ep_plan: 'EPPlan',
    permuted_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor],
           List[int], Dict[int, int], torch.Tensor,
           Optional[torch.Tensor], Optional[Dict[int, torch.Tensor]]]:
    """
    Dispatch phase with P2P overlap (padded mode, EPPlan required).

    Pipeline:
        Round 0: Start P2P_0, compute local FC1 + Act
        Round i: req.wait(P2P_{i-1}), start P2P_i, compute recv_{i-1} FC1 + Act
        Final:   req.wait(last round), compute last FC1 + Act

    Returns:
        local_tokens, local_act, recv_act_results, recv_buffer_views,
        partners, recv_offsets, tokens_cpu, local_probs, recv_probs
    """
    nvtx_range_push("dispatch_fc1_p2p")
    device = tokens.device
    dtype = tokens.dtype
    hidden_size = tokens.shape[-1]

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    # All static values pre-computed in EPPlan.
    input_splits = ep_plan.input_splits_list
    output_splits = ep_plan.output_splits_list
    input_offsets = ep_plan.input_offsets
    local_count = ep_plan.local_count
    local_start = ep_plan.local_start
    tokens_cpu = ep_plan.pre_tokens_cpu
    local_tokens_per_expert_cpu = ep_plan.local_tokens_per_expert_cpu
    partners = ep_plan.partners
    _global_partners = ep_plan.global_partners
    _dispatch_events = ep_plan.dispatch_events

    # Extract local tokens (no clone needed - data will be copied in fc2_combine anyway)
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
    _need_probs_cast = has_probs and permuted_probs.dtype != dtype
    send_chunk_list = []       # None if no data to send
    recv_meta_list = []        # None if no data to recv (1D flat buffer)
    recv_probs_list = []       # extracted from flat recv buffer

    p2p_backend = get_p2p_backend()

    for i, partner in enumerate(partners):
        n_send = input_splits[partner]
        recv_size = output_splits[partner]

        # Send buffer: 1D flat [N*H + N?] — tokens then probs (padded mode).
        if n_send > 0:
            token_chunk = tokens[input_offsets[partner]:input_offsets[partner+1]]
            total_elems = ep_plan.recv_total_elems[i]
            send_buf = _get_p2p_buffer(
                f"dispatch_send_{i}", total_elems, 1, dtype, device).squeeze(-1)
            send_buf[:n_send * hidden_size].copy_(token_chunk.reshape(-1))
            if has_probs:
                probs_chunk = permuted_probs[input_offsets[partner]:input_offsets[partner+1]]
                send_buf[n_send * hidden_size:].copy_(
                    probs_chunk.to(dtype) if _need_probs_cast else probs_chunk)
            send_chunk_list.append(send_buf)
        else:
            send_chunk_list.append(None)

        # Recv buffer: 1D flat, persistent (symmetric heap when NVSHMEM active)
        if recv_size > 0:
            total_elems = recv_size * hidden_size + (recv_size if has_probs else 0)
            recv_meta_list.append(p2p_backend.alloc_recv_buffer(
                f"dispatch_recv_{partner}", total_elems, dtype, device))
        else:
            recv_meta_list.append(None)
        recv_probs_list.append(None)

    recv_offsets = ep_plan.recv_offsets

    # =========================================================================
    # Dispatch Phase Pipeline with event-based sync (no CPU blocking)
    # Key insight: use a per-round event on comm_stream to hand completed P2P
    # data to default_stream, while still overlapping P2P_i with FC1 for
    # round i-1. The event must be recorded only after NCCL completion.
    # =========================================================================
    prev_idx = -1
    # all_reqs removed — P2P backend manages request lifecycle internally
    p2p_events = []
    recv_act_results = [None] * len(partners)
    recv_buffer_views = [None] * len(partners)
    local_act = None

    def extract_tokens(idx):
        """Split flat buffer into tokens and probs (zero-copy views)."""
        partner = partners[idx]
        flat = recv_meta_list[idx]
        if flat is not None:
            n = output_splits[partner]
            # tokens: view as [N, H] — contiguous because flat buffer is contiguous
            recv_buffer_views[idx] = flat[:n * hidden_size].view(n, hidden_size)
            if has_probs:
                probs_view = flat[n * hidden_size:n * hidden_size + n]
                recv_probs_list[idx] = probs_view.to(probs_dtype) if _need_probs_cast else probs_view

    _p2p = p2p_backend

    # Per-region CCE hook (r3_f = MoE dispatch). Start event on comm_stream
    # before first P2P; end event = last p2p event; waits registered per round.
    from fluid.core.scheduler import get_backward_scheduler as _get_sched
    from fluid.core import overhead_profiler as _oh_fwd
    _sched_f = _get_sched()
    _r3f_start = _sched_f.begin_fwd_comm(comm_stream) if len(partners) > 0 else None

    for round_idx, partner in enumerate(partners):
        nvtx_range_push(f"dispatch_p2p_R{round_idx}")
        _oh_tok = _oh_fwd.enter("overhead.fwd_tournament_round") if _oh_fwd.ENABLED else None
        try:
            with torch.cuda.stream(comm_stream):
                if round_idx == 0:
                    comm_stream.wait_stream(default_stream)
                evt = _dispatch_events[round_idx]
                _p2p.exchange(
                    send_buf=send_chunk_list[round_idx],
                    recv_buf=recv_meta_list[round_idx],
                    partner_global_rank=_global_partners[round_idx],
                    partner_local_rank=partners[round_idx],
                    group=ep_group, comm_stream=comm_stream, event=evt,
                )
                p2p_events.append(evt)
        finally:
            if _oh_tok is not None:
                _oh_fwd.exit(_oh_tok)
        nvtx_range_pop()

        # 2. Compute FC1 + Act (overlaps with current round's P2P, no CPU blocking)
        nvtx_range_push(f"fc1_compute_R{round_idx}")
        if round_idx == 0:
            if local_count > 0:
                local_act = grouped_fc1_act(
                    local_tokens, weight1, local_tokens_per_expert_cpu,
                    activation_func, probs=local_probs)
        elif prev_idx >= 0:
            _sched_f.record_fwd_wait('r3_f', p2p_events[round_idx - 1])
            extract_tokens(prev_idx)
            prev_p = partners[prev_idx]
            if recv_buffer_views[prev_idx] is not None:
                recv_data = recv_buffer_views[prev_idx]
                partner_probs = recv_probs_list[prev_idx]
                recv_act = grouped_fc1_act(
                    recv_data, weight1, tokens_cpu[prev_p],
                    activation_func, probs=partner_probs)
                recv_act_results[prev_idx] = recv_act
        nvtx_range_pop()

        prev_idx = round_idx

    # Process last round
    if len(partners) > 0:
        nvtx_range_push("fc1_compute_last")
        _sched_f.end_fwd_comm('r3_f', _r3f_start, comm_stream, end_evt=p2p_events[-1])
        _sched_f.record_fwd_wait('r3_f', p2p_events[-1])
        last_idx = len(partners) - 1
        extract_tokens(last_idx)
        last_p = partners[last_idx]
        if recv_buffer_views[last_idx] is not None:
            recv_data = recv_buffer_views[last_idx]
            partner_probs = recv_probs_list[last_idx]
            recv_act = grouped_fc1_act(
                recv_data, weight1, tokens_cpu[last_p],
                activation_func, probs=partner_probs)
            recv_act_results[last_idx] = recv_act
        nvtx_range_pop()

    if _p2p.needs_final_wait():
        _p2p.final_wait()

    recv_probs = ({partners[i]: recv_probs_list[i] for i in range(len(partners))
                   if recv_probs_list[i] is not None}
                  if permuted_probs is not None else None)

    nvtx_range_pop()  # dispatch_fc1_p2p
    return (
        local_tokens, local_act, recv_act_results, recv_buffer_views,
        partners, recv_offsets, tokens_cpu,
        local_probs,
        recv_probs,
    )


# =============================================================================
# Phase 2: FC2 + Combine with P2P Overlap
# =============================================================================

def fc2_combine_p2p_forward(
    local_tokens: torch.Tensor,
    local_act: torch.Tensor,
    recv_act_results: List[Optional[torch.Tensor]],
    recv_buffer_views: List[Optional[torch.Tensor]],
    weight2: torch.Tensor,
    ep_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    num_local_experts: int,
    ep_plan: 'EPPlan',
    needs_backward: bool = True,
    local_probs: Optional[torch.Tensor] = None,
    recv_probs: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[torch.Tensor], Optional[list], Optional[dict],
           Optional[torch.Tensor]]:
    with profile_section("moe_fwd.fc2_combine_p2p"):
        return _fc2_combine_p2p_forward_impl(
            local_tokens, local_act, recv_act_results, recv_buffer_views,
            weight2, ep_group, overlap_ctx, num_local_experts, ep_plan,
            needs_backward=needs_backward,
            local_probs=local_probs, recv_probs=recv_probs,
        )


def _fc2_combine_p2p_forward_impl(
    local_tokens: torch.Tensor,
    local_act: torch.Tensor,
    recv_act_results: List[Optional[torch.Tensor]],
    recv_buffer_views: List[Optional[torch.Tensor]],
    weight2: torch.Tensor,
    ep_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    num_local_experts: int,
    ep_plan: 'EPPlan',
    needs_backward: bool = True,
    local_probs: Optional[torch.Tensor] = None,
    recv_probs: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[torch.Tensor], Optional[list], Optional[dict],
           Optional[torch.Tensor]]:
    """
    FC2 + Combine phase with P2P overlap (padded mode, EPPlan required).

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
    device = local_tokens.device
    dtype = local_tokens.dtype
    hidden_size = weight2.shape[-1]

    # All static values pre-computed in EPPlan.
    ep_size = ep_plan.ep_size
    my_rank = ep_plan.my_rank
    input_splits = ep_plan.input_splits_list
    input_offsets = ep_plan.input_offsets
    local_count = ep_plan.local_count
    local_start = ep_plan.local_start
    local_tokens_per_expert = ep_plan.local_tokens_per_expert_cpu
    total_output = ep_plan.S * ep_size
    partner_to_index = ep_plan.partner_to_index
    partners = ep_plan.partners
    tokens_cpu = ep_plan.pre_tokens_cpu

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    # =========================================================================
    # Combine Phase Pipeline
    # =========================================================================
    p2p_backend = get_p2p_backend()
    combined_output = p2p_backend.alloc_recv_buffer(
        "combine_output", total_output * hidden_size, dtype, device
    ).view(total_output, hidden_size)

    peer_fc2_results = [None] * len(partners)
    local_fc2 = None

    # Reuse single event from context to reduce overhead
    fc2_event = overlap_ctx.data_ready_event
    has_pending_fc2 = False

    # Round -1: Pre-compute first peer's FC2
    if len(partners) > 0:
        first_partner = partners[0]
        if recv_act_results[0] is not None:
            nvtx_range_push("fc2_compute_first")
            peer_fc2_results[0] = grouped_fc2(
                recv_act_results[0], weight2, tokens_cpu[first_partner])
            fc2_event.record(default_stream)
            has_pending_fc2 = True
            nvtx_range_pop()

    # Pipeline loop
    _combine_global_partners = ep_plan.global_partners
    _combine_recv_slices = []
    for i, partner in enumerate(partners):
        if input_splits[partner] > 0:
            _combine_recv_slices.append(combined_output[input_offsets[partner]:input_offsets[partner+1]])
        else:
            _combine_recv_slices.append(None)

    _p2p_combine = p2p_backend

    # Per-region CCE hook (r4_f = MoE combine). combine does not publish per-round
    # events (p2p_backend handles sync via final_wait); record one r4_f comm pair
    # from first exchange to last exchange, and one wait pair around final_wait.
    from fluid.core.scheduler import get_backward_scheduler as _get_sched
    from fluid.core import overhead_profiler as _oh_fwd
    _sched_f = _get_sched()
    _r4f_start = _sched_f.begin_fwd_comm(comm_stream) if len(partners) > 0 else None
    _r4f_end_evt = None

    for round_idx, partner in enumerate(partners):
        nvtx_range_push(f"combine_p2p_R{round_idx}")
        _oh_tok = _oh_fwd.enter("overhead.fwd_tournament_round") if _oh_fwd.ENABLED else None
        try:
            with torch.cuda.stream(comm_stream):
                if has_pending_fc2:
                    comm_stream.wait_event(fc2_event)
                recv_slice = _combine_recv_slices[round_idx]
                if recv_slice is not None:
                    recv_slice.record_stream(comm_stream)
                send_buf = peer_fc2_results[round_idx]
                if send_buf is not None:
                    send_buf.record_stream(comm_stream)
                _p2p_combine.exchange(
                    send_buf=send_buf,
                    recv_buf=recv_slice,
                    partner_global_rank=_combine_global_partners[round_idx],
                    partner_local_rank=partners[round_idx],
                    group=ep_group, comm_stream=comm_stream, event=None,
                )
                # On the last round, record an end event on comm_stream for r4_f.
                if round_idx == len(partners) - 1 and _r4f_start is not None:
                    _r4f_end_evt = torch.cuda.Event(enable_timing=True)
                    _r4f_end_evt.record(comm_stream)
        finally:
            if _oh_tok is not None:
                _oh_fwd.exit(_oh_tok)
        nvtx_range_pop()

        # Parallel with current P2P: compute next round FC2 or local FC2
        has_pending_fc2 = False
        if round_idx + 1 < len(partners):
            next_idx = round_idx + 1
            if recv_act_results[next_idx] is not None:
                nvtx_range_push(f"fc2_compute_R{round_idx+1}")
                next_partner = partners[next_idx]
                peer_fc2_results[next_idx] = grouped_fc2(
                    recv_act_results[next_idx], weight2, tokens_cpu[next_partner])
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

        # Padded fast path: direct rank-segment copy, no intermediate cat.
        _cap = ep_plan.expert_capacity
        row_idx_r2e, row_idx_e2r = _get_padded_row_indices(
            num_local_experts, ep_size, _cap, device)

        n_per_rank = num_local_experts * _cap
        total_tokens_merge = ep_size * n_per_rank

        # Build rank-major by direct copy_ per rank (no cat)
        rank_major = torch.empty(total_tokens_merge, hidden_size, dtype=dtype, device=device)
        rank_major[my_rank * n_per_rank:(my_rank + 1) * n_per_rank] = local_tokens
        for p in partners:
            p_idx = partner_to_index[p]
            if recv_buffer_views[p_idx] is not None:
                rank_major[p * n_per_rank:(p + 1) * n_per_rank] = recv_buffer_views[p_idx]

        # Reorder to expert-major using cached index
        all_expert_tokens = torch.empty(total_tokens_merge, hidden_size, dtype=dtype, device=device)
        torch.index_select(rank_major, 0, row_idx_r2e, out=all_expert_tokens)

        # Merge probs to expert-major by direct copy_ per rank (no cat)
        if local_probs is not None and recv_probs is not None:
            rank_major_probs = torch.empty(total_tokens_merge, dtype=local_probs.dtype, device=device)
            rank_major_probs[my_rank * n_per_rank:(my_rank + 1) * n_per_rank] = local_probs
            for p in partners:
                if p in recv_probs:
                    rank_major_probs[p * n_per_rank:(p + 1) * n_per_rank] = recv_probs[p]
            all_expert_probs = rank_major_probs[row_idx_r2e]

        all_tokens_per_expert = ep_plan.all_tokens_per_expert

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
        nvtx_range_pop()

    nvtx_range_push("combine_wait_all")
    # Close r4_f comm pair on comm_stream, then bracket the NCCL final_wait
    # on default_stream (CPU block; subsequent default-stream kernels cannot
    # launch until final_wait returns).
    if _r4f_start is not None:
        _sched_f.end_fwd_comm('r4_f', _r4f_start, comm_stream, end_evt=_r4f_end_evt)
    if _p2p_combine.needs_final_wait():
        if _sched_f.comm_metrics_enabled and _r4f_end_evt is not None:
            w_s = torch.cuda.Event(enable_timing=True)
            w_e = torch.cuda.Event(enable_timing=True)
            w_s.record(default_stream)
            _p2p_combine.final_wait()
            w_e.record(default_stream)
            _sched_f._visible_wait_pairs.append(('r4_f', w_s, w_e))
        else:
            _p2p_combine.final_wait()
    elif _r4f_end_evt is not None:
        # No CPU final_wait but combined_output is consumed on default_stream,
        # so bracket a simple wait_event for the exposure estimate.
        _sched_f.record_fwd_wait('r4_f', _r4f_end_evt)
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
    'EPPlan',
    'router_forward',
    'pad_moe_dispatch',
    'dispatch_fc1_p2p_forward',
    'fc2_combine_p2p_forward',
]

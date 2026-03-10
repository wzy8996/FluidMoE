"""
Triton kernels for MoE hotspots.

permute_by_row_idx   — dst[i] = src[idx[i]], supports strided output
unpermute_by_row_idx — dst[idx[i]] = src[i], supports strided input
restore_topk_reduce  — fused index_select + view + sum
router_bwd_fused     — fused router backward (scatter + norm_bwd + softmax_bwd)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# permute_by_row_idx (gather):  dst[i, :] = src[idx[i], :]
#
# Supports strided dst (e.g. column-slice view of a larger tensor).
# src columns must be contiguous (stride[:, 1] == 1).
# ---------------------------------------------------------------------------

@triton.jit
def _row_gather_kernel(
    src_ptr, idx_ptr, dst_ptr,
    N, D,
    stride_src_row, stride_dst_row,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D

    src_row = tl.load(idx_ptr + row)
    vals = tl.load(src_ptr + src_row * stride_src_row + d_off, mask=mask)
    tl.store(dst_ptr + row * stride_dst_row + d_off, vals, mask=mask)


def permute_by_row_idx(
    src: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Gather rows: out[i, :] = src[indices[i], :].

    Supports strided *out* (e.g. a column-slice view of a larger tensor).
    *src* columns must be contiguous (stride[:, 1] == 1).

    Args:
        src:     [M, D]  source tensor (columns contiguous)
        indices: [N]     int64 row indices into src
        out:     [N, D]  destination (may be strided on dim-0)

    Returns:
        out (same object passed in)
    """
    N = indices.shape[0]
    D = src.shape[1]
    assert out.shape == (N, D), f"out shape {out.shape} != ({N}, {D})"
    assert src.stride(1) == 1, "src columns must be contiguous"
    assert out.stride(1) == 1, "out columns must be contiguous"

    if N == 0:
        return out

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_D = min(BLOCK_D, 2048)
    grid = (N, triton.cdiv(D, BLOCK_D))

    _row_gather_kernel[grid](
        src, indices, out,
        N, D,
        src.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D,
        num_warps=8,
    )
    return out


# backward compat alias
row_gather = permute_by_row_idx


# ---------------------------------------------------------------------------
# unpermute_by_row_idx (scatter):  dst[idx[i], :] = src[i, :]
#
# Supports strided src (e.g. column-slice view of a larger tensor).
# dst columns must be contiguous (stride[:, 1] == 1).
# NOTE: caller must ensure no duplicate indices (1-to-1 mapping).
# ---------------------------------------------------------------------------

@triton.jit
def _row_scatter_kernel(
    src_ptr, idx_ptr, dst_ptr,
    N, D,
    stride_src_row, stride_dst_row,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D

    dst_row = tl.load(idx_ptr + row)
    vals = tl.load(src_ptr + row * stride_src_row + d_off, mask=mask)
    tl.store(dst_ptr + dst_row * stride_dst_row + d_off, vals, mask=mask)


def unpermute_by_row_idx(
    src: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Scatter rows: out[indices[i], :] = src[i, :].

    Inverse of permute_by_row_idx. Caller must ensure indices are unique
    (permutation, not general scatter-add).

    Supports strided *src* (e.g. a column-slice view of a larger tensor).
    *out* columns must be contiguous (stride[:, 1] == 1).

    Args:
        src:     [N, D]  source tensor (may be strided on dim-0)
        indices: [N]     int64 destination row indices
        out:     [M, D]  destination (columns contiguous)

    Returns:
        out (same object passed in)
    """
    N = indices.shape[0]
    D = src.shape[1]
    assert out.shape[1] == D, f"out cols {out.shape[1]} != {D}"
    assert out.stride(1) == 1, "out columns must be contiguous"
    assert src.stride(1) == 1, "src columns must be contiguous"

    if N == 0:
        return out

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_D = min(BLOCK_D, 2048)
    grid = (N, triton.cdiv(D, BLOCK_D))

    _row_scatter_kernel[grid](
        src, indices, out,
        N, D,
        src.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D,
        num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# restore_topk_reduce
#
#   out[s, :] = sum_{k=0}^{top_k-1} src[indices[s * top_k + k], :]
#
# Replaces:
#   expanded = src.index_select(0, indices)   # [S*top_k, D] intermediate
#   out = expanded.view(S, top_k, D).sum(1)   # [S, D]
#
# The intermediate tensor is never materialised.
# ---------------------------------------------------------------------------

@triton.jit
def _restore_topk_reduce_kernel(
    src_ptr, idx_ptr, dst_ptr,
    S, top_k, D,
    stride_src_row, stride_dst_row,
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    s = tl.program_id(0)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in range(top_k):
        src_row = tl.load(idx_ptr + s * top_k + k)
        vals = tl.load(src_ptr + src_row * stride_src_row + d_off, mask=mask)
        acc += vals.to(tl.float32)

    tl.store(dst_ptr + s * stride_dst_row + d_off, acc.to(OUT_DTYPE), mask=mask)


def restore_topk_reduce(
    src: torch.Tensor,
    indices: torch.Tensor,
    num_tokens: int,
    top_k: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused index-gather + top-k reduce.

    Equivalent to:
        src.index_select(0, indices).view(num_tokens, top_k, D).sum(dim=1)
    but without materialising the [num_tokens * top_k, D] intermediate.

    Args:
        src:        [T, D]  source tensor
        indices:    [S * top_k]  gather indices (S = num_tokens)
        num_tokens: S
        top_k:      number of expert copies per token
        out:        optional [S, D] output buffer

    Returns:
        [S, D] reduced result
    """
    D = src.shape[1]
    S = num_tokens
    assert indices.shape[0] == S * top_k

    if out is None:
        out = torch.empty(S, D, dtype=src.dtype, device=src.device)
    assert out.shape == (S, D)

    if S == 0:
        return out

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_D = min(BLOCK_D, 2048)
    grid = (S, triton.cdiv(D, BLOCK_D))

    # Map torch dtype to triton constexpr dtype
    _DTYPE_MAP = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }
    out_tl_dtype = _DTYPE_MAP[out.dtype]

    _restore_topk_reduce_kernel[grid](
        src, indices, out,
        S, top_k, D,
        src.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D,
        OUT_DTYPE=out_tl_dtype,
        num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# router_bwd_fused — Fused router backward (steps 1-4)
#
# Fuses: scatter → normalize_bwd → topk_bwd (scatter_) → softmax_bwd
# into a single kernel.  The final matmul (grad_logits @ weight.T) stays
# in cuBLAS.
#
# Input per-row (token):
#   grad_permuted_probs[sorted_indices[j]] → scatter to (token, slot)
#   top_probs[token, :]       — normalized top-k probs
#   top_indices[token, :]     — which experts were selected
#   router_probs[token, :]    — full softmax probs
#
# Output per-row:
#   grad_router_logits[token, :] — gradient w.r.t. pre-softmax logits
# ---------------------------------------------------------------------------

@triton.jit
def _router_bwd_fused_kernel(
    # grad_top_probs: [N, top_k]  (pre-scattered by caller)
    grad_top_ptr,
    # top_probs: [N, top_k]
    top_probs_ptr,
    # top_indices: [N, top_k]
    top_indices_ptr,
    # router_probs: [N, E]
    router_probs_ptr,
    # output: grad_router_logits [N, E]
    out_ptr,
    N, top_k, E,
    stride_gt_row,         # grad_top_probs.stride(0)
    stride_tp_row,         # top_probs.stride(0)
    stride_ti_row,         # top_indices.stride(0)
    stride_rp_row,         # router_probs.stride(0)
    stride_out_row,        # out.stride(0)
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """2D grid: (token, e_tile).  Each program handles one token × one E-tile.

    sum(grad_router_probs * router_probs) is computed from top_k elements only
    (O(K) per token, redundant across tiles but avoids cross-tile reduction).
    """
    token = tl.program_id(0)
    e_tile = tl.program_id(1)
    e_start = e_tile * BLOCK_E
    e_off = e_start + tl.arange(0, BLOCK_E)
    e_mask = e_off < E

    # Vectorized load of top_k data for normalize backward scalars
    k_off = tl.arange(0, BLOCK_K)
    k_mask = k_off < top_k
    tp_vec = tl.load(top_probs_ptr + token * stride_tp_row + k_off, mask=k_mask, other=0.0).to(tl.float32)
    gt_vec = tl.load(grad_top_ptr + token * stride_gt_row + k_off, mask=k_mask, other=0.0).to(tl.float32)
    sum_raw = tl.maximum(tl.sum(tp_vec, axis=0), 1e-6)
    dot_gp = tl.sum(gt_vec * tp_vec, axis=0)

    # Per-element loop: scatter into this tile + compute sum_grp_rp from top_k
    base_gt = grad_top_ptr + token * stride_gt_row
    base_tp = top_probs_ptr + token * stride_tp_row
    base_ti = top_indices_ptr + token * stride_ti_row
    grad_rp = tl.zeros([BLOCK_E], dtype=tl.float32)
    sum_grp_rp = 0.0  # scalar: sum(grad_raw[k] * router_probs[top_indices[k]])
    for k in range(BLOCK_K):
        if k < top_k:
            gt_k = tl.load(base_gt + k).to(tl.float32)
            tp_k = tl.load(base_tp + k).to(tl.float32)
            idx = tl.load(base_ti + k)
            val = (gt_k - dot_gp * tp_k) / sum_raw
            # sum_grp_rp: load router_prob at expert idx (same across tiles)
            rp_k = tl.load(router_probs_ptr + token * stride_rp_row + idx).to(tl.float32)
            sum_grp_rp += val * rp_k
            # Scatter val into this tile's range
            grad_rp = tl.where(e_off == idx, grad_rp + val, grad_rp)

    # Softmax backward: grad_logits = rp * (grad_rp - sum_grp_rp)
    rp = tl.load(router_probs_ptr + token * stride_rp_row + e_off, mask=e_mask, other=0.0).to(tl.float32)
    grad_logits = rp * (grad_rp - sum_grp_rp)

    tl.store(out_ptr + token * stride_out_row + e_off, grad_logits.to(out_ptr.dtype.element_ty), mask=e_mask)


_ROUTER_BWD_TILE_E = 128  # per-tile E size (controls register pressure)


def router_bwd_fused(
    grad_top_probs: torch.Tensor,
    top_probs: torch.Tensor,
    top_indices: torch.Tensor,
    router_probs: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused router backward: normalize_bwd + topk_scatter + softmax_bwd.

    Uses a 2D Triton grid (N, cdiv(E, TILE_E)) so any number of experts
    is supported without register pressure issues.

    The scatter from grad_permuted_probs → grad_top_probs[N, top_k] is done
    by the caller (cheap, irregular pattern). This kernel fuses the remaining
    dense steps 2-4.

    Args:
        grad_top_probs: [N, top_k] scattered gradient (from grad_permuted_probs)
        top_probs:      [N, top_k] normalized top-k probabilities
        top_indices:    [N, top_k] expert indices
        router_probs:   [N, E] full softmax probabilities
        out:            optional [N, E] output buffer

    Returns:
        grad_router_logits: [N, E]
    """
    N, top_k = top_probs.shape
    E = router_probs.shape[1]

    if out is None:
        out = torch.empty(N, E, dtype=router_probs.dtype, device=router_probs.device)

    if N == 0:
        return out

    BLOCK_K = triton.next_power_of_2(top_k)
    BLOCK_E = min(triton.next_power_of_2(E), _ROUTER_BWD_TILE_E)
    grid = (N, triton.cdiv(E, BLOCK_E))

    _router_bwd_fused_kernel[grid](
        grad_top_probs,
        top_probs,
        top_indices,
        router_probs,
        out,
        N, top_k, E,
        grad_top_probs.stride(0),
        top_probs.stride(0),
        top_indices.stride(0),
        router_probs.stride(0),
        out.stride(0),
        BLOCK_K=BLOCK_K,
        BLOCK_E=BLOCK_E,
    )
    return out

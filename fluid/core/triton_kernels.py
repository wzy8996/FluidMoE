"""
Triton kernels for MoE hotspots.

permute_by_row_idx   — dst[i] = src[idx[i]], supports strided output
unpermute_by_row_idx — dst[idx[i]] = src[i], supports strided input
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

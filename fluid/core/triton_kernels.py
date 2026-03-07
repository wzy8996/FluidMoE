"""
Triton kernels for MoE backward hotspots.

P0: row_gather — fused index_select that supports strided output,
    eliminating the extra copy in R1 chunked pipeline.

P1: restore_topk_reduce — fused index_select + view + sum,
    eliminating the 128 MB intermediate tensor in R2 restore path.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# P0: row_gather  —  output[i, :] = input[indices[i], :]
#
# Unlike torch.index_select, this kernel writes correctly into a
# non-contiguous (strided) output tensor, so the caller can pass
# grad_all_fc2[:, h_start:h_end] directly and skip the extra copy.
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


def row_gather(
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
    BLOCK_D = min(BLOCK_D, 4096)
    grid = (N, triton.cdiv(D, BLOCK_D))

    _row_gather_kernel[grid](
        src, indices, out,
        N, D,
        src.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D,
    )
    return out


# ---------------------------------------------------------------------------
# P1: restore_topk_reduce
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
    BLOCK_D = min(BLOCK_D, 4096)
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
    )
    return out


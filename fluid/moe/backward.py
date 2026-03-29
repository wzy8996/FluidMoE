"""
MoE Backward Operations with AllToAll + Overlap

4 Overlap Regions (matching forward's 4 P2P overlap points):

  Region 1 (communication-first): Combine AllToAll → FC2 dx
    - Submit all AllToAll chunks first
    - As each chunk completes, compute FC2 dx partial (grad @ w2.T slice)
    - After all chunks: activation backward → grad_all_fc1

  Region 2 (compute-first): FC1 dx → Dispatch AllToAll
    - Compute FC1 dx in chunks (grad @ w1.T slice)
    - As each chunk completes, submit dispatch AllToAll for that chunk
    - During AllToAll: dW tasks, router backward, LN2 backward

  Region 3 (compute-first): Output Proj dX → sp2hp AllToAll  [in attention/backward.py]
  Region 4 (communication-first): hp2sp AllToAll → QKV dX    [in attention/backward.py]

Key functions:
- combine_fc2_backward: Region 1 - combine AllToAll + FC2 dx pipeline
- fc1_dispatch_backward: Region 2 - FC1 dx + dispatch AllToAll pipeline
- register_moe_dw_tasks: Register dW tasks for weight1 and weight2
"""

import os
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Dict

from fluid.core import _all_to_all
from fluid.core.scheduler import get_backward_scheduler
from fluid.core.nvtx import nvtx_range_push, nvtx_range_pop
from fluid.moe.forward import (
    _grouped_gemm_or_none,
    _get_cached_layout_indices, _build_row_reorder_index,
)
from fluid.core.te_ops import te_gelu, te_silu, te_dgelu, te_dsilu

_SKIP_ALL_DW = os.environ.get('FLUID_SKIP_ALL_DW', '0') == '1'
_SKIP_DISPATCH_DW = os.environ.get('FLUID_SKIP_DISPATCH_DW', '0') == '1'


class _BackwardBufferPool:
    """Simple per-device tensor pool for hot backward workspaces."""

    def __init__(self):
        self._buffers = {}

    def _key(self, tag: str, cols: int, dtype: torch.dtype, device: torch.device):
        return (tag, int(cols), str(dtype), device.type, int(device.index) if device.index is not None else -1)

    def get_2d(
        self,
        tag: str,
        rows: int,
        cols: int,
        dtype: torch.dtype,
        device: torch.device,
        zero: bool = False,
    ) -> torch.Tensor:
        """Return a [rows, cols] view from pooled storage (grow-only by rows)."""
        key = self._key(tag, cols, dtype, device)
        buf = self._buffers.get(key, None)
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty(rows, cols, dtype=dtype, device=device)
            self._buffers[key] = buf
        out = buf[:rows]
        if zero:
            out.zero_()
        return out


_BWD_POOL = _BackwardBufferPool()


_CHUNK_ROW_IDX_CACHE = {}


def _get_chunk_row_indices(
    epc: int, ep_size: int, cap: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cached layout-convert row indices for expert-group chunks.

    Returns (row_idx_rank_to_exp, row_idx_exp_to_rank) for a chunk
    with *epc* experts, *ep_size* ranks, *cap* tokens per expert-rank block.
    """
    key = (epc, ep_size, cap, device.type,
           device.index if device.index is not None else -1)
    cached = _CHUNK_ROW_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    sorted_r2e, sorted_e2r = _get_cached_layout_indices(epc, ep_size, device)
    split_sizes = torch.full((ep_size * epc,), cap, dtype=torch.int64, device=device)
    row_idx_r2e = _build_row_reorder_index(split_sizes, sorted_r2e, device)
    row_idx_e2r = _build_row_reorder_index(split_sizes, sorted_e2r, device)
    _CHUNK_ROW_IDX_CACHE[key] = (row_idx_r2e, row_idx_e2r)
    return row_idx_r2e, row_idx_e2r


_SCATTER_IDX_CACHE = {}


def _build_chunk_scatter_indices(
    nle: int, ep_size: int, cap: int, num_chunks: int, device: torch.device,
) -> List[torch.Tensor]:
    """Precompute per-chunk scatter index: chunk-expert-major → final expert-major.

    For chunk c, scatter_idx[j] gives the flat row in the final expert-major
    buffer [nle, ep_size, cap, D] where chunk-expert-major position j should go.

    Chunk-expert-major layout (chunk_total rows):
      expert e, rank r, token t  →  j = e * ep_size * cap_c + r * cap_c + t
    Final expert-major layout (total rows):
      expert e, rank r, token t  →  flat = e * ep_size * cap + r * cap + c * cap_c + t
    """
    key = (nle, ep_size, cap, num_chunks, device.type,
           device.index if device.index is not None else -1)
    cached = _SCATTER_IDX_CACHE.get(key)
    if cached is not None:
        return cached

    cap_c = cap // num_chunks
    chunk_total = nle * ep_size * cap_c

    # Vectorised: compute all j → flat mappings at once
    j = torch.arange(chunk_total, dtype=torch.int64, device=device)
    e = j // (ep_size * cap_c)
    rem = j % (ep_size * cap_c)
    r = rem // cap_c
    t = rem % cap_c
    # base = e * ep_size * cap + r * cap  (independent of c)
    base = e * (ep_size * cap) + r * cap

    indices = []
    for c in range(num_chunks):
        indices.append(base + c * cap_c + t)

    _SCATTER_IDX_CACHE[key] = indices
    return indices


def build_moe_chunk_config(
    num_local_experts: int,
    ep_size: int,
    cap: int,
    moe_combine_chunks: int,
    moe_dispatch_chunks: int,
    device: torch.device,
) -> dict:
    """Pre-compute static shapes for padded MoE backward chunking.

    With capacity padding, input_splits = output_splits = [S]*ep_size
    where S = num_local_experts * cap. All chunk shapes are deterministic,
    so validation and dynamic computation in backward can be skipped.

    R1 (combine) and R2 (dispatch) may use different chunk counts.
    """
    nle = num_local_experts
    S = nle * cap
    total = ep_size * S
    splits = [S] * ep_size

    cfg = dict(
        nle=nle, ep_size=ep_size, cap=cap,
        S=S, total=total, splits=splits,
    )

    # R1 (combine): prefer expert-dim chunking (zero scatter), fall back to cap-dim
    C1 = moe_combine_chunks
    if C1 > 1 and nle % C1 == 0:
        # Expert-dim chunking: split nle into C1 groups → contiguous output per chunk
        nle_c = nle // C1
        chunk_S = nle_c * cap
        row_r2e, _ = _get_chunk_row_indices(nle_c, ep_size, cap, device)
        cfg['r1'] = dict(
            mode='expert', num_chunks=C1, nle_c=nle_c,
            chunk_S=chunk_S, chunk_total=ep_size * chunk_S,
            chunk_splits=[chunk_S] * ep_size,
            chunk_tpe=[ep_size * cap] * nle_c,
            row_r2e=row_r2e,
        )
    elif C1 > 1 and cap % C1 == 0:
        # Cap-dim chunking: split cap → needs scatter
        cap_c = cap // C1
        chunk_S = nle * cap_c
        row_r2e, _ = _get_chunk_row_indices(nle, ep_size, cap_c, device)
        scatter_indices = _build_chunk_scatter_indices(
            nle, ep_size, cap, C1, device)
        cfg['r1'] = dict(
            mode='cap', num_chunks=C1, cap_c=cap_c,
            chunk_S=chunk_S, chunk_total=ep_size * chunk_S,
            chunk_splits=[chunk_S] * ep_size,
            chunk_tpe=[ep_size * cap_c] * nle,
            row_r2e=row_r2e,
            scatter_idx=scatter_indices,
        )
    else:
        cfg['r1'] = None

    # R2 (dispatch): prefer expert-dim chunking, fall back to cap-dim
    C2 = moe_dispatch_chunks
    if C2 > 1 and nle % C2 == 0:
        # Expert-dim: input is contiguous, reassembly uses strided copy
        nle_c = nle // C2
        chunk_S = nle_c * cap
        _, row_e2r = _get_chunk_row_indices(nle_c, ep_size, cap, device)
        cfg['r2'] = dict(
            mode='expert', num_chunks=C2, nle_c=nle_c,
            chunk_S=chunk_S, chunk_total=ep_size * chunk_S,
            chunk_splits=[chunk_S] * ep_size,
            chunk_tpe=[ep_size * cap] * nle_c,
            row_e2r=row_e2r,
        )
    elif C2 > 1 and cap % C2 == 0:
        # Cap-dim: needs scatter for reassembly
        cap_c = cap // C2
        chunk_S = nle * cap_c
        _, row_e2r = _get_chunk_row_indices(nle, ep_size, cap_c, device)
        chunk_total_r2 = ep_size * chunk_S
        j = torch.arange(chunk_total_r2, dtype=torch.int64, device=device)
        r = j // (nle * cap_c)
        rem = j % (nle * cap_c)
        e = rem // cap_c
        t = rem % cap_c
        base_r2 = r * (nle * cap) + e * cap
        r2_scatter = [base_r2 + c * cap_c + t for c in range(C2)]
        cfg['r2'] = dict(
            mode='cap', num_chunks=C2, cap_c=cap_c,
            chunk_S=chunk_S, chunk_total=chunk_total_r2,
            chunk_splits=[chunk_S] * ep_size,
            chunk_tpe=[ep_size * cap_c] * nle,
            row_e2r=row_e2r,
            scatter_idx=r2_scatter,
        )
    else:
        cfg['r2'] = None

    return cfg


def _should_keep_grad_all_fc2_stable() -> bool:
    """Whether grad_all_fc2 may outlive current region due to deferred dW execution."""
    return skip_dispatch_dw() or skip_all_dw()


def _reorder_chunks(
    tensor: torch.Tensor,
    row_idx: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reorder chunked rows by row index."""
    if out is None:
        return tensor.index_select(0, row_idx)
    torch.index_select(tensor, 0, row_idx, out=out)
    return out


def _maybe_execute_dw_tasks(scheduler, for_dispatch: bool = False) -> None:
    """Execute deferred dW tasks if not skipped by current policy flags."""
    if for_dispatch:
        should_skip = skip_dispatch_dw() or skip_all_dw()
    else:
        should_skip = skip_all_dw()
    if should_skip:
        return
    nvtx_range_push("dw_tasks")
    scheduler.execute_dw_tasks()
    nvtx_range_pop()


def _post_combine_single_alltoall(
    grad_combined_recv: torch.Tensor,
    backward_indices: Dict[str, torch.Tensor],
    weight2: torch.Tensor,
    all_tokens_per_expert: List[int],
    all_fc1: torch.Tensor,
    activation_func,
    total_recv: int,
    hidden_size: int,
    ffn_hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    all_expert_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Common post-processing for non-chunked combine: layout -> fc2 dx -> probs bwd -> act bwd."""
    nvtx_range_push("layout_convert")
    row_idx_rank_to_exp = backward_indices['row_idx_rank_to_exp']
    if _should_keep_grad_all_fc2_stable():
        grad_all_fc2_out = torch.empty(total_recv, hidden_size, dtype=dtype, device=device)
    else:
        grad_all_fc2_out = _BWD_POOL.get_2d(
            tag="combine_grad_all_fc2",
            rows=total_recv,
            cols=hidden_size,
            dtype=dtype,
            device=device,
        )
    grad_all_fc2 = _reorder_chunks(
        grad_combined_recv,
        row_idx=row_idx_rank_to_exp,
        out=grad_all_fc2_out,
    )
    nvtx_range_pop()

    nvtx_range_push("fc2_dx")
    grad_exp_act = _BWD_POOL.get_2d(
        tag="combine_grad_exp_act",
        rows=total_recv,
        cols=ffn_hidden,
        dtype=dtype,
        device=device,
    )
    grouped_fc2_dx(grad_all_fc2, weight2, all_tokens_per_expert, out=grad_exp_act)
    nvtx_range_pop()

    # Probs backward: grad_exp_act is grad w.r.t. act_weighted = act * probs
    grad_probs = None
    if all_expert_probs is not None:
        nvtx_range_push("probs_backward")
        fc1_detached = all_fc1.detach()
        act_pre_probs = activation_func(fc1_detached)
        grad_probs = (grad_exp_act * act_pre_probs).sum(dim=-1)
        grad_exp_act = grad_exp_act * all_expert_probs.unsqueeze(-1).to(grad_exp_act.dtype)
        nvtx_range_pop()

    nvtx_range_push("act_backward")
    grad_all_fc1, act_output = _activation_backward(grad_exp_act, all_fc1, activation_func)
    nvtx_range_pop()

    # For w2 dW: act_output should be act_weighted = act * probs (the actual FC2 input)
    if all_expert_probs is not None:
        act_output = act_output * all_expert_probs.unsqueeze(-1).to(act_output.dtype)

    return grad_all_fc1, act_output, grad_all_fc2, grad_probs


def skip_all_dw() -> bool:
    """Check if all dW tasks should be skipped (cached)."""
    return _SKIP_ALL_DW


def skip_dispatch_dw() -> bool:
    """Check if dispatch dW tasks should be skipped (cached)."""
    return _SKIP_DISPATCH_DW


def _activation_backward(
    grad_exp_act: torch.Tensor,
    all_fc1: torch.Tensor,
    activation_func,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Activation backward: use TE fused kernels when possible, else autograd."""
    fc1_detached = all_fc1.detach()
    act_output = activation_func(fc1_detached)

    if activation_func is te_gelu:
        grad_all_fc1 = te_dgelu(grad_exp_act, fc1_detached)
    elif activation_func is te_silu:
        grad_all_fc1 = te_dsilu(grad_exp_act, fc1_detached)
    else:
        # Fallback: autograd for unknown activations
        with torch.enable_grad():
            fc1_with_grad = fc1_detached.requires_grad_(True)
            act_out = activation_func(fc1_with_grad)
            grad_all_fc1, = torch.autograd.grad(act_out, fc1_with_grad, grad_exp_act, retain_graph=False)

    return grad_all_fc1, act_output


def grouped_fc2_dx(
    grad_fc2: torch.Tensor,
    w2: torch.Tensor,
    tokens_per_expert: List[int],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute FC2 dx for all experts: grad_fc2 @ w2.T

    Args:
        grad_fc2: [total_tokens, hidden] gradient w.r.t FC2 output
        w2: [num_experts, ffn_hidden, hidden] FC2 weights
        tokens_per_expert: token counts per expert

    Returns:
        grad_exp_act: [total_tokens, ffn_hidden] gradient w.r.t activation output
    """
    if grad_fc2.shape[0] == 0:
        return torch.empty(0, w2.shape[1], dtype=grad_fc2.dtype, device=grad_fc2.device)

    num_experts = w2.shape[0]
    ffn_hidden = w2.shape[1]
    if num_experts == 1:
        if out is None:
            return torch.matmul(grad_fc2, w2[0].t())
        torch.mm(grad_fc2, w2[0].t(), out=out)
        return out

    # Try GroupGEMM: grad_fc2 @ w2.T  (trans_b=True)
    gmm_result = _grouped_gemm_or_none(grad_fc2, w2, tokens_per_expert, trans_b=True)
    if gmm_result is not None:
        if out is not None:
            out.copy_(gmm_result)
            return out
        return gmm_result

    # Fallback: per-expert loop
    if out is None:
        out = torch.empty(grad_fc2.shape[0], ffn_hidden, dtype=grad_fc2.dtype, device=grad_fc2.device)
    offset = 0
    for i, n in enumerate(tokens_per_expert):
        n = int(n)
        if n > 0:
            torch.mm(grad_fc2[offset:offset + n], w2[i].t(), out=out[offset:offset + n])
            offset += n
    return out


def grouped_fc1_dx(grad_fc1: torch.Tensor, w1: torch.Tensor,
                   tokens_per_expert: List[int]) -> torch.Tensor:
    """Compute FC1 dx for all experts: grad_fc1 @ w1.T

    Args:
        grad_fc1: [total_tokens, ffn_hidden] gradient w.r.t FC1 output
        w1: [num_experts, hidden, ffn_hidden] FC1 weights
        tokens_per_expert: token counts per expert

    Returns:
        grad_tokens: [total_tokens, hidden] gradient w.r.t input tokens
    """
    if grad_fc1.shape[0] == 0:
        return torch.empty(0, w1.shape[1], dtype=grad_fc1.dtype, device=grad_fc1.device)

    num_experts = w1.shape[0]
    hidden_size = w1.shape[1]
    if num_experts == 1:
        return torch.matmul(grad_fc1, w1[0].t())

    # Try GroupGEMM: grad_fc1 @ w1.T  (trans_b=True)
    gmm_result = _grouped_gemm_or_none(grad_fc1, w1, tokens_per_expert, trans_b=True)
    if gmm_result is not None:
        return gmm_result

    # Fallback: per-expert loop
    out = torch.empty(grad_fc1.shape[0], hidden_size, dtype=grad_fc1.dtype, device=grad_fc1.device)
    offset = 0
    for i, n in enumerate(tokens_per_expert):
        n = int(n)
        if n > 0:
            torch.mm(grad_fc1[offset:offset + n], w1[i].t(), out=out[offset:offset + n])
            offset += n
    return out


# =============================================================================
# dW Task Registration
# =============================================================================

def register_moe_dw_tasks(
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    all_expert_tokens: torch.Tensor,
    act_output: torch.Tensor,
    grad_all_fc2: torch.Tensor,
    grad_all_fc1: torch.Tensor,
    num_local_experts: int,
    all_tokens_per_expert: List[int],
    layer_id: int,
    orig_weight1: torch.Tensor,
    orig_weight2: torch.Tensor,
    needs_ar: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Register dW tasks for weight1 and weight2 to execute during AllToAll communication.

    This should be called BEFORE the Dispatch AllToAll to allow dW computation
    to overlap with communication.

    Args:
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight (3D view)
        weight2: [num_local_experts, ffn_hidden, hidden] FC2 weight (3D view)
        all_expert_tokens: [total_recv, hidden] all tokens (expert-major order)
        act_output: [total_recv, ffn_hidden] activation output (from expert_backward)
        grad_all_fc2: [total_recv, hidden] gradient w.r.t. FC2 output
        grad_all_fc1: [total_recv, ffn_hidden] gradient w.r.t. FC1 output
        num_local_experts: Number of local experts
        all_tokens_per_expert: Token count per expert
        layer_id: Layer ID for task naming
        orig_weight1: Original weight1 tensor (2D) for gradient assignment
        orig_weight2: Original weight2 tensor (2D) for gradient assignment

    Returns:
        (grad_weight1, grad_weight2) if scheduler disabled, else (None, None)
        Note: gradients are returned in 2D format matching orig_weight shapes
    """
    scheduler = get_backward_scheduler()



    # Get dimensions
    ffn_hidden = weight1.shape[-1]
    hidden_size = weight2.shape[-1]

    if scheduler.is_enabled():
        # Scheduler enabled: register dW tasks for execution during later AllToAll
        num_local_experts_saved = num_local_experts
        all_tokens_per_expert_saved = all_tokens_per_expert
        grad_all_fc2_saved = grad_all_fc2.detach()
        grad_all_fc1_saved = grad_all_fc1.detach()
        act_output_saved = act_output.detach()
        all_expert_tokens_saved = all_expert_tokens.detach()

        def compute_dw_weight2():
            # dW2 = act_output.T @ grad_fc2  (trans_a=True)
            gmm = _grouped_gemm_or_none(act_output_saved, grad_all_fc2_saved,
                                        all_tokens_per_expert_saved, trans_a=True)
            if gmm is not None:
                return gmm
            grad_w2 = torch.zeros(num_local_experts_saved, ffn_hidden, hidden_size,
                                  dtype=act_output_saved.dtype, device=act_output_saved.device)
            offset = 0
            for i, n in enumerate(all_tokens_per_expert_saved):
                n = int(n)
                if n > 0:
                    torch.mm(act_output_saved[offset:offset + n].t(),
                             grad_all_fc2_saved[offset:offset + n], out=grad_w2[i])
                    offset += n
            return grad_w2

        def compute_dw_weight1():
            # dW1 = expert_tokens.T @ grad_fc1  (trans_a=True)
            gmm = _grouped_gemm_or_none(all_expert_tokens_saved, grad_all_fc1_saved,
                                        all_tokens_per_expert_saved, trans_a=True)
            if gmm is not None:
                return gmm
            grad_w1 = torch.zeros(num_local_experts_saved, hidden_size, ffn_hidden,
                                  dtype=all_expert_tokens_saved.dtype, device=all_expert_tokens_saved.device)
            offset = 0
            for i, n in enumerate(all_tokens_per_expert_saved):
                n = int(n)
                if n > 0:
                    torch.mm(all_expert_tokens_saved[offset:offset + n].t(),
                             grad_all_fc1_saved[offset:offset + n], out=grad_w1[i])
                    offset += n
            return grad_w1

        scheduler.register_dw_task(
            layer_name=f"moe_weight2_L{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight2,
            weight_param=orig_weight2,
            needs_ar=needs_ar,
        )
        scheduler.register_dw_task(
            layer_name=f"moe_weight1_L{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight1,
            weight_param=orig_weight1,
            needs_ar=needs_ar,
        )
        return None, None
    else:
        # Scheduler disabled: compute dW directly
        # dW2 = act_output.T @ grad_fc2  (trans_a=True)
        grad_w2_3d = _grouped_gemm_or_none(act_output, grad_all_fc2,
                                           all_tokens_per_expert, trans_a=True)
        # dW1 = expert_tokens.T @ grad_fc1  (trans_a=True)
        grad_w1_3d = _grouped_gemm_or_none(all_expert_tokens, grad_all_fc1,
                                           all_tokens_per_expert, trans_a=True)
        if grad_w2_3d is None or grad_w1_3d is None:
            # Fallback: per-expert loop
            grad_w2_3d = torch.zeros(num_local_experts, ffn_hidden, hidden_size,
                                     dtype=act_output.dtype, device=act_output.device)
            grad_w1_3d = torch.zeros(num_local_experts, hidden_size, ffn_hidden,
                                     dtype=all_expert_tokens.dtype, device=all_expert_tokens.device)
            offset = 0
            for i, n in enumerate(all_tokens_per_expert):
                n = int(n)
                if n > 0:
                    torch.mm(act_output[offset:offset + n].t(),
                             grad_all_fc2[offset:offset + n], out=grad_w2_3d[i])
                    torch.mm(all_expert_tokens[offset:offset + n].t(),
                             grad_all_fc1[offset:offset + n], out=grad_w1_3d[i])
                    offset += n
        return grad_w1_3d, grad_w2_3d


def recompute_fc1_gemm(
    all_expert_tokens: torch.Tensor,
    all_tokens_per_expert: List[int],
    weight1: torch.Tensor,
) -> torch.Tensor:
    """Recompute FC1 GEMM only (no activation), overlaps with R1 AllToAll."""
    total = all_expert_tokens.shape[0]
    ffn = weight1.shape[-1]
    dtype = all_expert_tokens.dtype
    device = all_expert_tokens.device
    if total == 0:
        return torch.empty(0, ffn, dtype=dtype, device=device)
    all_fc1 = _BWD_POOL.get_2d("fc1_recompute", total, ffn, dtype, device)
    if weight1.shape[0] == 1:
        torch.mm(all_expert_tokens, weight1[0], out=all_fc1)
    else:
        # Use GroupedGEMM (same as Megatron): tokens @ w1 (trans_b=False)
        gmm_result = _grouped_gemm_or_none(
            all_expert_tokens, weight1, all_tokens_per_expert, trans_b=False)
        if gmm_result is not None:
            all_fc1.copy_(gmm_result)
        else:
            # Fallback: per-expert loop
            offset = 0
            for i, n in enumerate(all_tokens_per_expert):
                n = int(n)
                if n > 0:
                    torch.mm(all_expert_tokens[offset:offset + n], weight1[i],
                             out=all_fc1[offset:offset + n])
                    offset += n
    return all_fc1


# =============================================================================
# Region 1: Combine AllToAll → FC2 dx (Communication-First Pipeline)
# =============================================================================

def combine_fc2_backward(
    grad_output: torch.Tensor,
    input_splits_list: List[int],
    output_splits_list: List[int],
    ep_group,
    layer_id: int,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    activation_func,
    num_local_experts: int,
    num_chunks: int = 1,
    # Pre-computed merge results from forward
    all_expert_tokens: Optional[torch.Tensor] = None,
    all_tokens_per_expert: Optional[List[int]] = None,
    backward_indices: Optional[Dict[str, torch.Tensor]] = None,
    # Pre-computed static chunk config (from build_moe_chunk_config)
    chunk_config: Optional[dict] = None,
    # Megatron-aligned: probs applied inside expert computation
    all_expert_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, List[int], Dict[str, torch.Tensor],
           Optional[torch.Tensor]]:
    """
    Region 1: Combine AllToAll overlap FC2 dx (communication-first pipeline).

    Pipeline (token-dim chunking, C chunks of cap/C tokens per expert per rank):
      1. Rearrange grad to chunk-first layout (split token dim)
      2. Submit all AllToAll chunks (uniform S/C per rank)
      3. FC1 GEMM recompute overlap with AllToAll
      4. dW tasks overlap with AllToAll
      5. Per chunk: wait → layout convert → FC2 dx (full weights, fewer tokens)
      6. Activation backward → grad_all_fc1

    comm_stream:    |A2A_0|A2A_1|...|A2A_{C-1}|
    default_stream: |rearrange+submit|fc1_recompute+dW|w0+lc0+fc2dx0|w1+lc1+fc2dx1|...|act_bwd|

    Args:
        grad_output: [total_output, hidden] gradient w.r.t. output
        input_splits_list: AllToAll input split sizes
        output_splits_list: AllToAll output split sizes
        ep_group: Expert parallel process group
        layer_id: Layer ID for debugging
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight (for recompute)
        weight2: [num_local_experts, ffn_hidden, hidden] FC2 weight
        activation_func: Activation function
        num_local_experts: Number of local experts
        num_chunks: Number of chunks for token dimension (requires uniform splits)
        all_expert_tokens: [total_recv, hidden] pre-merged expert-major tokens (from forward)
        all_tokens_per_expert: token counts per local expert (from forward)
        backward_indices: pre-computed layout convert indices (from forward)

    Returns:
        grad_all_fc1: [total_recv, ffn_hidden] gradient w.r.t. FC1 output
        act_output: [total_recv, ffn_hidden] activation output (detached; probs-weighted if applicable)
        all_fc1: [total_recv, ffn_hidden] FC1 pre-activation (expert-major)
        grad_all_fc2: [total_recv, hidden] gradient after AllToAll + layout convert (for dW)
        all_expert_tokens: [total_recv, hidden] expert-major tokens
        all_tokens_per_expert: token counts per local expert
        backward_indices: layout convert indices for region 1/2
        grad_expert_probs: Optional [total_recv] gradient w.r.t. probs in expert-major order
    """
    nvtx_range_push("combine_fc2_backward")
    scheduler = get_backward_scheduler()
    device = grad_output.device
    dtype = grad_output.dtype
    hidden_size = grad_output.shape[1]
    ffn_hidden = weight2.shape[1]
    total_output = grad_output.shape[0]
    total_recv = sum(output_splits_list)

    # Validate token-dim chunking: use pre-computed config or check dynamically
    _r1_cfg = chunk_config.get('r1') if chunk_config is not None else None
    if _r1_cfg is not None:
        num_chunks = _r1_cfg['num_chunks']  # pre-validated
    elif num_chunks > 1:
        _is_uniform = (len(set(input_splits_list)) == 1 and
                       len(set(output_splits_list)) == 1)
        if not _is_uniform:
            num_chunks = 1
        elif (input_splits_list[0] % num_chunks != 0 or
              output_splits_list[0] % num_chunks != 0):
            num_chunks = 1
        else:
            _cap = input_splits_list[0] // num_local_experts
            # Accept if either expert-dim (nle % C) or cap-dim (cap % C) works
            if num_local_experts % num_chunks != 0 and _cap % num_chunks != 0:
                num_chunks = 1

    if all_expert_tokens is None:
        raise RuntimeError("combine_fc2_backward requires pre-computed all_expert_tokens from forward.")
    if all_tokens_per_expert is None:
        raise RuntimeError("combine_fc2_backward requires pre-computed all_tokens_per_expert from forward.")
    if backward_indices is None:
        raise RuntimeError("combine_fc2_backward requires pre-computed backward_indices from forward.")

    if not scheduler.is_enabled():
        # ---- Fallback: scheduler disabled, fully synchronous ----
        nvtx_range_push("combine_alltoall")
        grad_combined_recv = _all_to_all(
            grad_output.contiguous(),
            output_split_sizes=output_splits_list,
            input_split_sizes=input_splits_list,
            group=ep_group,
            debug_tag="moe_combine_sync",
        )
        nvtx_range_pop()

        all_fc1 = recompute_fc1_gemm(all_expert_tokens, all_tokens_per_expert, weight1)
        grad_all_fc1, act_output, grad_all_fc2, grad_probs = _post_combine_single_alltoall(
            grad_combined_recv=grad_combined_recv,
            backward_indices=backward_indices,
            weight2=weight2,
            all_tokens_per_expert=all_tokens_per_expert,
            all_fc1=all_fc1,
            activation_func=activation_func,
            total_recv=total_recv,
            hidden_size=hidden_size,
            ffn_hidden=ffn_hidden,
            dtype=dtype,
            device=device,
            all_expert_probs=all_expert_probs,
        )

        nvtx_range_pop()  # combine_fc2_backward
        return (
            grad_all_fc1, act_output.detach(), all_fc1, grad_all_fc2,
            all_expert_tokens, all_tokens_per_expert, backward_indices, grad_probs,
        )

    if num_chunks <= 1:
        # ---- C=1 with scheduler: async AllToAll + merge-precompute/dW overlap ----
        result_holder = [None]
        grad_output_contig = grad_output.contiguous()
        def do_alltoall():
            result_holder[0] = _all_to_all(
                grad_output_contig,
                output_split_sizes=output_splits_list,
                input_split_sizes=input_splits_list,
                group=ep_group,
                debug_tag="moe_combine_async_c1",
            )
            return result_holder[0]

        task_id = scheduler.submit_alltoall(do_alltoall)

        # FC1 GEMM recompute (overlaps with AllToAll; merge+sort done in forward)
        all_fc1 = recompute_fc1_gemm(all_expert_tokens, all_tokens_per_expert, weight1)
        _maybe_execute_dw_tasks(scheduler, for_dispatch=False)

        scheduler.wait_alltoall(task_id)
        grad_combined_recv = result_holder[0]
        grad_all_fc1, act_output, grad_all_fc2, grad_probs = _post_combine_single_alltoall(
            grad_combined_recv=grad_combined_recv,
            backward_indices=backward_indices,
            weight2=weight2,
            all_tokens_per_expert=all_tokens_per_expert,
            all_fc1=all_fc1,
            activation_func=activation_func,
            total_recv=total_recv,
            hidden_size=hidden_size,
            ffn_hidden=ffn_hidden,
            dtype=dtype,
            device=device,
            all_expert_probs=all_expert_probs,
        )

        nvtx_range_pop()  # combine_fc2_backward
        return (
            grad_all_fc1, act_output.detach(), all_fc1, grad_all_fc2,
            all_expert_tokens, all_tokens_per_expert, backward_indices, grad_probs,
        )

    # ========================================================================
    # Chunked communication-first pipeline
    # Two modes: expert-dim (zero scatter) or cap-dim (with scatter fallback)
    # Per-chunk overlap: wait → layout convert → FC2 dx
    # ========================================================================
    nvtx_range_push("combine_fc2_chunked")

    _r1_mode = _r1_cfg.get('mode', 'cap') if _r1_cfg is not None else 'cap'

    if _r1_cfg is not None:
        nle = chunk_config['nle']
        ep_size = chunk_config['ep_size']
        cap = chunk_config['cap']
        chunk_S_send = chunk_S_recv = _r1_cfg['chunk_S']
        chunk_total = _r1_cfg['chunk_total']
        chunk_send_splits = chunk_recv_splits = _r1_cfg['chunk_splits']
    else:
        nle = num_local_experts
        ep_size = len(input_splits_list)
        S_send = input_splits_list[0]
        S_recv = output_splits_list[0]
        cap = S_send // nle
        chunk_S_send = S_send // num_chunks
        chunk_S_recv = S_recv // num_chunks
        chunk_total = ep_size * chunk_S_recv
        chunk_send_splits = [chunk_S_send] * ep_size
        chunk_recv_splits = [chunk_S_recv] * ep_size
        # Determine mode dynamically
        if nle % num_chunks == 0:
            _r1_mode = 'expert'
        else:
            _r1_mode = 'cap'

    # Step 1: Rearrange to chunk-first layout
    nvtx_range_push("rearrange_submit")
    grad_output_contig = grad_output.contiguous()
    _combine_in_all = _BWD_POOL.get_2d(
        tag="combine_input_chunks_tok", rows=total_output, cols=hidden_size,
        dtype=dtype, device=device,
    )
    if _r1_mode == 'expert':
        # Expert-dim: [ep, nle, cap, H] = [ep, C, nle_c, cap, H] → [C, ep, nle_c, cap, H]
        nle_c = _r1_cfg['nle_c'] if _r1_cfg is not None else nle // num_chunks
        _combine_in_all.view(
            num_chunks, ep_size, nle_c, cap, hidden_size
        ).copy_(
            grad_output_contig.view(
                ep_size, num_chunks, nle_c, cap, hidden_size
            ).permute(1, 0, 2, 3, 4)
        )
    else:
        # Cap-dim: [ep, nle, cap, H] = [ep, nle, C, cap_c, H] → [C, ep, nle, cap_c, H]
        cap_c = _r1_cfg['cap_c'] if _r1_cfg is not None else cap // num_chunks
        _combine_in_all.view(
            num_chunks, ep_size, nle, cap_c, hidden_size
        ).copy_(
            grad_output_contig.view(
                ep_size, nle, num_chunks, cap_c, hidden_size
            ).permute(2, 0, 1, 3, 4)
        )
    _combine_out_all = _BWD_POOL.get_2d(
        tag="combine_output_chunks_tok", rows=total_recv, cols=hidden_size,
        dtype=dtype, device=device,
    )

    # Step 2: Submit all AllToAll chunks (batch: single stream switch)
    def _make_combine_a2a(ib, ob):
        def fn():
            ib.record_stream(torch.cuda.current_stream())
            ob.record_stream(torch.cuda.current_stream())
            dist.all_to_all_single(
                ob, ib, output_split_sizes=chunk_recv_splits,
                input_split_sizes=chunk_send_splits, group=ep_group,
            )
            return ob
        return fn

    comm_fns = []
    for c in range(num_chunks):
        off_s = c * ep_size * chunk_S_send
        off_r = c * chunk_total
        comm_fns.append(_make_combine_a2a(
            _combine_in_all[off_s:off_s + ep_size * chunk_S_send],
            _combine_out_all[off_r:off_r + chunk_total],
        ))
    task_ids = scheduler.submit_alltoall_batch(comm_fns)
    nvtx_range_pop()  # rearrange_submit

    # Step 3: FC1 GEMM recompute (overlaps with AllToAll)
    all_fc1 = recompute_fc1_gemm(all_expert_tokens, all_tokens_per_expert, weight1)

    # Step 4: dW tasks overlap with AllToAll
    _maybe_execute_dw_tasks(scheduler, for_dispatch=False)

    # Chunk layout index
    if _r1_cfg is not None:
        chunk_row_r2e = _r1_cfg['row_r2e']
        chunk_tpe = _r1_cfg['chunk_tpe']
    else:
        if _r1_mode == 'expert':
            chunk_row_r2e, _ = _get_chunk_row_indices(nle_c, ep_size, cap, device)
            chunk_tpe = [ep_size * cap] * nle_c
        else:
            chunk_row_r2e, _ = _get_chunk_row_indices(nle, ep_size, cap_c, device)
            chunk_tpe = [ep_size * cap_c] * nle

    # Allocate final expert-major buffers
    if _should_keep_grad_all_fc2_stable():
        grad_all_fc2 = torch.empty(total_recv, hidden_size, dtype=dtype, device=device)
    else:
        grad_all_fc2 = _BWD_POOL.get_2d(
            tag="combine_grad_all_fc2", rows=total_recv, cols=hidden_size,
            dtype=dtype, device=device,
        )
    grad_exp_act = _BWD_POOL.get_2d(
        tag="combine_grad_exp_act", rows=total_recv, cols=ffn_hidden,
        dtype=dtype, device=device,
    )

    # Step 5: Per-chunk compute
    if _r1_mode == 'expert':
        # Expert-dim: each chunk writes to a contiguous slice → zero scatter
        for c in range(num_chunks):
            is_last = (c == num_chunks - 1)
            nvtx_range_push(f"chunk_{c}")
            scheduler.wait_alltoall(task_ids[c], try_trickle=is_last)

            off = c * chunk_total
            chunk_recv = _combine_out_all[off:off + chunk_total]

            # R2E directly into final buffer (contiguous slice)
            fc2_slice = grad_all_fc2[off:off + chunk_total]
            torch.index_select(chunk_recv, 0, chunk_row_r2e, out=fc2_slice)

            # FC2 dx directly into final buffer (contiguous slice)
            act_slice = grad_exp_act[off:off + chunk_total]
            grouped_fc2_dx(fc2_slice, weight2[c * nle_c:(c + 1) * nle_c],
                           chunk_tpe, out=act_slice)
            nvtx_range_pop()
    else:
        # Cap-dim: needs scatter via precomputed indices
        _fc2_chunk_buf = _BWD_POOL.get_2d(
            tag="combine_fc2_chunk_buf", rows=chunk_total, cols=hidden_size,
            dtype=dtype, device=device,
        )
        _act_chunk_buf = _BWD_POOL.get_2d(
            tag="combine_act_chunk_buf", rows=chunk_total, cols=ffn_hidden,
            dtype=dtype, device=device,
        )
        if _r1_cfg is not None:
            scatter_indices = _r1_cfg['scatter_idx']
        else:
            scatter_indices = _build_chunk_scatter_indices(
                nle, ep_size, cap, num_chunks, device)

        for c in range(num_chunks):
            is_last = (c == num_chunks - 1)
            nvtx_range_push(f"chunk_{c}")
            scheduler.wait_alltoall(task_ids[c], try_trickle=is_last)

            off_r = c * chunk_total
            chunk_recv = _combine_out_all[off_r:off_r + chunk_total]
            scatter_c = scatter_indices[c]

            torch.index_select(chunk_recv, 0, chunk_row_r2e, out=_fc2_chunk_buf)
            grad_all_fc2.index_copy_(0, scatter_c, _fc2_chunk_buf)
            grouped_fc2_dx(_fc2_chunk_buf, weight2, chunk_tpe, out=_act_chunk_buf)
            grad_exp_act.index_copy_(0, scatter_c, _act_chunk_buf)
            nvtx_range_pop()

    # Step 6: Probs backward + activation backward on full expert-major tensor
    grad_probs = None
    if all_expert_probs is not None:
        nvtx_range_push("probs_backward")
        fc1_detached = all_fc1.detach()
        act_pre_probs = activation_func(fc1_detached)
        grad_probs = (grad_exp_act * act_pre_probs).sum(dim=-1)
        grad_exp_act = grad_exp_act * all_expert_probs.unsqueeze(-1).to(grad_exp_act.dtype)
        nvtx_range_pop()

    nvtx_range_push("act_backward")
    grad_all_fc1, act_output = _activation_backward(grad_exp_act, all_fc1, activation_func)
    nvtx_range_pop()

    # For w2 dW: act_output should be act_weighted = act * probs (the actual FC2 input)
    if all_expert_probs is not None:
        act_output = act_output * all_expert_probs.unsqueeze(-1).to(act_output.dtype)

    nvtx_range_pop()  # combine_fc2_chunked
    nvtx_range_pop()  # combine_fc2_backward
    return (
        grad_all_fc1, act_output.detach(), all_fc1, grad_all_fc2,
        all_expert_tokens, all_tokens_per_expert, backward_indices, grad_probs,
    )


# =============================================================================
# Region 2: FC1 dx → Dispatch AllToAll (Compute-First Pipeline)
# =============================================================================

def fc1_dispatch_backward(
    grad_all_fc1: torch.Tensor,
    weight1: torch.Tensor,
    num_local_experts: int,
    all_tokens_per_expert: List[int],
    row_idx_exp_to_rank: torch.Tensor,
    input_splits_list: List[int],
    output_splits_list: List[int],
    ep_group,
    layer_id: int = 0,
    num_chunks: int = 1,
    chunk_config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Region 2: FC1 dx + Dispatch AllToAll (compute-first pipeline).

    Pipeline (token-dim chunking, C chunks of cap/C tokens per expert per rank):
      1. Per chunk: lazy-extract from expert-major → FC1 dx → layout convert → submit AllToAll
      2. dW tasks overlap with remaining AllToAll
      3. Wait all chunks → reassemble grad_tokens

    default_stream: |extract0+fc1dx0+lc0+submit0|extract1+fc1dx1+...|dW|wait+reassemble|
    comm_stream:                           |A2A_0|A2A_1|...|A2A_{C-1}|

    Args:
        grad_all_fc1: [total_recv, ffn_hidden] gradient w.r.t. FC1 output
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight
        num_local_experts: Number of local experts
        all_tokens_per_expert: Token count per expert
        row_idx_exp_to_rank: Row-level reorder index (expert-major -> rank-major)
        input_splits_list: AllToAll input split sizes
        output_splits_list: AllToAll output split sizes
        ep_group: Expert parallel process group
        layer_id: Layer ID for debugging
        num_chunks: Number of chunks for token dimension (requires uniform splits)

    Returns:
        grad_tokens: [total_send, hidden] gradient w.r.t. input tokens (after AllToAll)
    """
    nvtx_range_push("fc1_dispatch_backward")
    scheduler = get_backward_scheduler()
    device = grad_all_fc1.device
    dtype = grad_all_fc1.dtype
    total_recv = grad_all_fc1.shape[0]
    hidden_size = weight1.shape[1]

    # Validate token-dim chunking: use pre-computed config or check dynamically
    _r2_cfg = chunk_config.get('r2') if chunk_config is not None else None
    if _r2_cfg is not None:
        num_chunks = _r2_cfg['num_chunks']  # pre-validated
    elif num_chunks > 1:
        _is_uniform = (len(set(input_splits_list)) == 1 and
                       len(set(output_splits_list)) == 1)
        if not _is_uniform:
            num_chunks = 1
        elif (input_splits_list[0] % num_chunks != 0 or
              output_splits_list[0] % num_chunks != 0):
            num_chunks = 1
        else:
            _cap = output_splits_list[0] // num_local_experts
            if num_local_experts % num_chunks != 0 and _cap % num_chunks != 0:
                num_chunks = 1

    total_send = sum(input_splits_list)

    if not scheduler.is_enabled() or num_chunks <= 1:
        # ---- Non-chunked path ----
        nvtx_range_push("fc1_dx")
        grad_all_tokens = grouped_fc1_dx(grad_all_fc1, weight1, all_tokens_per_expert)
        nvtx_range_pop()

        # Reorder expert-major -> rank-major
        nvtx_range_push("reorder")
        grad_dispatched_out = _BWD_POOL.get_2d(
            tag="dispatch_grad_dispatched",
            rows=total_recv,
            cols=hidden_size,
            dtype=dtype,
            device=device,
        )
        grad_dispatched = _reorder_chunks(
            grad_all_tokens,
            row_idx=row_idx_exp_to_rank,
            out=grad_dispatched_out,
        )
        nvtx_range_pop()

        # Dispatch AllToAll
        if scheduler.is_enabled():
            result_holder = [None]
            grad_dispatched_contig = grad_dispatched.contiguous()
            def do_alltoall():
                result_holder[0] = _all_to_all(
                    grad_dispatched_contig,
                    output_split_sizes=input_splits_list,
                    input_split_sizes=output_splits_list,
                    group=ep_group,
                    debug_tag="moe_dispatch_async_c1",
                )
                return result_holder[0]

            task_id = scheduler.submit_alltoall(do_alltoall)

            _maybe_execute_dw_tasks(scheduler, for_dispatch=True)

            scheduler.wait_alltoall(task_id)
            grad_tokens = result_holder[0]
        else:
            grad_tokens = _all_to_all(
                grad_dispatched.contiguous(),
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group,
                debug_tag="moe_dispatch_sync",
            )

        nvtx_range_pop()  # fc1_dispatch_backward
        return grad_tokens

    # ========================================================================
    # Chunked compute-first pipeline
    # Two modes: expert-dim (zero scatter/extract) or cap-dim (with scatter)
    # Per-chunk overlap: FC1 dx → layout convert → submit AllToAll
    # ========================================================================
    nvtx_range_push("fc1_dispatch_chunked")

    _r2_mode = _r2_cfg.get('mode', 'cap') if _r2_cfg is not None else 'cap'

    if _r2_cfg is not None:
        nle = chunk_config['nle']
        ep_size = chunk_config['ep_size']
        cap = chunk_config['cap']
        chunk_S_send = chunk_S_recv = _r2_cfg['chunk_S']
        chunk_total = _r2_cfg['chunk_total']
        chunk_send_splits = chunk_recv_splits = _r2_cfg['chunk_splits']
        chunk_row_e2r = _r2_cfg['row_e2r']
        chunk_tpe = _r2_cfg['chunk_tpe']
    else:
        nle = num_local_experts
        ep_size = len(input_splits_list)
        S_a2a_send = output_splits_list[0]
        S_a2a_recv = input_splits_list[0]
        cap = S_a2a_send // nle
        chunk_S_send = S_a2a_send // num_chunks
        chunk_S_recv = S_a2a_recv // num_chunks
        chunk_total = ep_size * chunk_S_send
        chunk_send_splits = [chunk_S_send] * ep_size
        chunk_recv_splits = [chunk_S_recv] * ep_size
        if nle % num_chunks == 0:
            _r2_mode = 'expert'
            nle_c = nle // num_chunks
            _, chunk_row_e2r = _get_chunk_row_indices(nle_c, ep_size, cap, device)
            chunk_tpe = [ep_size * cap] * nle_c
        else:
            _r2_mode = 'cap'
            cap_c = cap // num_chunks
            _, chunk_row_e2r = _get_chunk_row_indices(nle, ep_size, cap_c, device)
            chunk_tpe = [ep_size * cap_c] * nle
    ffn_hidden_size = grad_all_fc1.shape[1]

    # Allocate AllToAll buffers (per-chunk slices, no aliasing)
    _dispatch_in_all = _BWD_POOL.get_2d(
        tag="dispatch_input_chunks_tok", rows=total_recv, cols=hidden_size,
        dtype=dtype, device=device,
    )
    _dispatch_out_all = _BWD_POOL.get_2d(
        tag="dispatch_output_chunks_tok", rows=total_send, cols=hidden_size,
        dtype=dtype, device=device,
    )

    # Step 1+2: Per-chunk: extract → FC1 dx → layout convert → submit AllToAll
    task_ids = []

    def _make_dispatch_a2a(ib, ob):
        def fn():
            ib.record_stream(torch.cuda.current_stream())
            ob.record_stream(torch.cuda.current_stream())
            dist.all_to_all_single(
                ob, ib, output_split_sizes=chunk_recv_splits,
                input_split_sizes=chunk_send_splits, group=ep_group,
            )
            return ob
        return fn

    if _r2_mode == 'expert':
        nle_c = _r2_cfg['nle_c'] if _r2_cfg is not None else nle // num_chunks
        for c in range(num_chunks):
            nvtx_range_push(f"chunk_{c}")
            off = c * chunk_total

            # Expert-dim: input is already contiguous — no extraction needed!
            fc1_chunk = grad_all_fc1[off:off + chunk_total]

            # FC1 dx with chunk's weight slice
            chunk_grad_tokens = grouped_fc1_dx(
                fc1_chunk,
                weight1[c * nle_c:(c + 1) * nle_c],
                chunk_tpe,
            )

            # Expert-major → rank-major layout convert
            in_buf = _dispatch_in_all[off:off + chunk_total]
            torch.index_select(chunk_grad_tokens, 0, chunk_row_e2r, out=in_buf)

            # Submit AllToAll
            out_buf = _dispatch_out_all[off:off + chunk_total]
            task_ids.append(scheduler.submit_alltoall(
                _make_dispatch_a2a(in_buf, out_buf)
            ))
            nvtx_range_pop()
    else:
        cap_c = _r2_cfg['cap_c'] if _r2_cfg is not None else cap // num_chunks
        _fc1_chunk_buf = _BWD_POOL.get_2d(
            tag="dispatch_fc1_chunk_buf", rows=chunk_total, cols=ffn_hidden_size,
            dtype=dtype, device=device,
        )
        grad_all_fc1_5d = grad_all_fc1.view(nle, ep_size, cap, ffn_hidden_size)

        for c in range(num_chunks):
            nvtx_range_push(f"chunk_{c}")
            off = c * chunk_total

            # Lazy extraction: strided copy from expert-major
            _fc1_chunk_buf.view(nle, ep_size, cap_c, ffn_hidden_size).copy_(
                grad_all_fc1_5d[:, :, c * cap_c:(c + 1) * cap_c, :]
            )

            chunk_grad_tokens = grouped_fc1_dx(
                _fc1_chunk_buf, weight1, chunk_tpe,
            )

            in_buf = _dispatch_in_all[off:off + chunk_total]
            torch.index_select(chunk_grad_tokens, 0, chunk_row_e2r, out=in_buf)

            out_buf = _dispatch_out_all[off:off + chunk_total]
            task_ids.append(scheduler.submit_alltoall(
                _make_dispatch_a2a(in_buf, out_buf)
            ))
            nvtx_range_pop()

    # Step 3: dW tasks overlap with remaining AllToAll
    _maybe_execute_dw_tasks(scheduler, for_dispatch=True)

    # Step 4: Wait for last AllToAll (NCCL FIFO guarantees all prior chunks done),
    #         then batch reassemble all chunks.
    nvtx_range_push("wait_reassemble")

    grad_tokens = _BWD_POOL.get_2d(
        tag="dispatch_grad_tokens", rows=total_send, cols=hidden_size,
        dtype=dtype, device=device,
    )

    # Single wait on last chunk
    scheduler.wait_alltoall(task_ids[-1], num_tasks=len(task_ids), try_trickle=True)

    if _r2_mode == 'expert':
        # Expert-dim reassembly: strided copy [C, ep, nle_c, cap, H] → [ep, nle, cap, H]
        nle_c = _r2_cfg['nle_c'] if _r2_cfg is not None else nle // num_chunks
        grad_tokens_4d = grad_tokens.view(ep_size, nle, cap, hidden_size)
        for c in range(num_chunks):
            off = c * chunk_total
            grad_tokens_4d[:, c * nle_c:(c + 1) * nle_c, :, :].copy_(
                _dispatch_out_all[off:off + chunk_total].view(
                    ep_size, nle_c, cap, hidden_size)
            )
    else:
        # Cap-dim reassembly: scatter
        cap_c = _r2_cfg['cap_c'] if _r2_cfg is not None else cap // num_chunks
        if _r2_cfg is not None:
            r2_scatter_indices = _r2_cfg['scatter_idx']
        else:
            j = torch.arange(chunk_total, dtype=torch.int64, device=device)
            _r = j // (nle * cap_c)
            _rem = j % (nle * cap_c)
            _e = _rem // cap_c
            _t = _rem % cap_c
            _base = _r * (nle * cap) + _e * cap
            r2_scatter_indices = [_base + c * cap_c + _t for c in range(num_chunks)]
        for c in range(num_chunks):
            off = c * chunk_total
            grad_tokens.index_copy_(
                0, r2_scatter_indices[c],
                _dispatch_out_all[off:off + chunk_total],
            )
    nvtx_range_pop()

    nvtx_range_pop()  # fc1_dispatch_chunked
    nvtx_range_pop()  # fc1_dispatch_backward
    return grad_tokens



# =============================================================================
# Router Backward
# =============================================================================

def router_backward(
    grad_permuted_probs: torch.Tensor,
    sorted_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    top_indices: torch.Tensor,
    top_probs: torch.Tensor,
    router_weight: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute router backward: gradients through post-softmax routing and linear projection.
    Aligned with Megatron's default post-softmax mode: topk(logits) → softmax(top_k).

    Args:
        grad_permuted_probs: [num_real] gradient w.r.t. permuted probabilities
        sorted_indices: [num_real] original token indices (0..T-1, expert-major order)
        tokens_per_expert: [num_experts] token count per expert (pre-padding)
        top_indices: [num_tokens, top_k] selected expert indices per token
        top_probs: [num_tokens, top_k] softmax(top_k_logits) probabilities
        router_weight: [hidden_size, num_experts] router weight matrix
        num_tokens: Number of input tokens
        num_experts: Number of experts
        top_k: Number of experts per token
        dtype: Data type for output

    Returns:
        grad_hidden_from_router: [num_tokens, hidden_size] gradient w.r.t. hidden_states
        grad_router_logits: [num_tokens, num_experts] gradient w.r.t. router logits (for dW)
    """
    device = grad_permuted_probs.device

    # Step 1: Scatter grad from permuted [num_real] to [T, E]
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=device), tokens_per_expert
    )
    grad_routing_probs = torch.zeros(
        num_tokens, num_experts, dtype=grad_permuted_probs.dtype, device=device
    )
    grad_routing_probs[sorted_indices, expert_ids] = grad_permuted_probs

    # Step 2: Gather grad at top-k positions → [T, k]
    grad_top_probs = grad_routing_probs.gather(1, top_indices)

    # Step 3: Backward through softmax (only over top-k dimension)
    sum_grad = (grad_top_probs * top_probs).sum(dim=-1, keepdim=True)
    grad_top_logits = top_probs * (grad_top_probs - sum_grad)

    # Step 4: Scatter back to [T, E] (only top-k positions have gradients)
    grad_router_logits = torch.zeros(
        num_tokens, num_experts, dtype=grad_top_logits.dtype, device=device
    )
    grad_router_logits.scatter_(1, top_indices, grad_top_logits)

    # Step 5: Backward through router linear: logits = hidden @ weight
    # Matmul in grad dtype matching Megatron
    grad_hidden_from_router = torch.matmul(
        grad_router_logits, router_weight.to(grad_router_logits.dtype).t()
    )

    return grad_hidden_from_router, grad_router_logits


def register_router_dw_task(
    hidden_states: torch.Tensor,
    grad_router_logits: torch.Tensor,
    router_weight: torch.Tensor,
    layer_id: int,
) -> Optional[torch.Tensor]:
    """
    Register router weight gradient task or compute directly if scheduler disabled.

    The router weight gradient is computed as: grad_weight = hidden.T @ grad_logits

    Args:
        hidden_states: [num_tokens, hidden_size] input hidden states
        grad_router_logits: [num_tokens, num_experts] gradient w.r.t. router logits
        router_weight: [hidden_size, num_experts] original router weight (for gradient assignment)
        layer_id: Layer ID for task naming

    Returns:
        grad_router_weight: Gradient if scheduler disabled, else None (registered as task)
    """
    scheduler = get_backward_scheduler()

    if scheduler.is_enabled():
        hidden_saved = hidden_states.detach()
        grad_logits_saved = grad_router_logits.detach()

        def compute_router_dw():
            return torch.matmul(hidden_saved.t(), grad_logits_saved)

        scheduler.register_dw_task(
            layer_name=f"router_weight_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_router_dw,
            weight_param=router_weight,
        )
        return None
    else:
        grad_router_weight = torch.matmul(hidden_states.t(), grad_router_logits)
        return grad_router_weight


__all__ = [
    # MoE matmul helpers
    'grouped_fc2_dx',
    'grouped_fc1_dx',
    # Region 1: Combine AllToAll + FC2 dx
    'combine_fc2_backward',
    # Region 2: FC1 dx + Dispatch AllToAll
    'fc1_dispatch_backward',
    # dW registration
    'register_moe_dw_tasks',
    # Router backward
    'router_backward',
    'register_router_dw_task',
    # Static chunk config
    'build_moe_chunk_config',
]

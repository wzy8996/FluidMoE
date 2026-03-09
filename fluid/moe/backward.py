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
from fluid.core.triton_kernels import row_gather, split_columns_to_chunks
from fluid.core.scheduler import get_backward_scheduler
from fluid.core.nvtx import nvtx_range_push, nvtx_range_pop
from fluid.moe.forward import _grouped_gemm_or_none, _to_batch_sizes
from fluid.core.te_ops import te_gelu, te_silu, te_dgelu, te_dsilu

_SKIP_ALL_DW = os.environ.get('FLUID_SKIP_ALL_DW', '0') == '1'
_SKIP_DISPATCH_DW = os.environ.get('FLUID_SKIP_DISPATCH_DW', '0') == '1'
_DEBUG_SCHEDULER_REGISTER = os.environ.get('FLUID_DEBUG_SCHEDULER_REGISTER', '0') == '1'


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


def _nonzero_expert_ranges(tokens_per_expert: List[int]) -> List[Tuple[int, int, int]]:
    """Return [(expert_idx, row_start, row_end)] for experts with non-zero tokens."""
    ranges: List[Tuple[int, int, int]] = []
    offset = 0
    for i, n in enumerate(tokens_per_expert):
        n_i = int(n)
        if n_i > 0:
            end = offset + n_i
            ranges.append((i, offset, end))
            offset = end
    return ranges


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Common post-processing for non-chunked combine: layout -> fc2 dx -> act bwd."""
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

    nvtx_range_push("act_backward")
    grad_all_fc1, act_output = _activation_backward(grad_exp_act, all_fc1, activation_func)
    nvtx_range_pop()
    return grad_all_fc1, act_output, grad_all_fc2


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

    # Debug output
    if _DEBUG_SCHEDULER_REGISTER:
        print(f"[DEBUG] register_moe_dw_tasks(L{layer_id}): scheduler.is_enabled()={scheduler.is_enabled()}", flush=True)

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
        offset = 0
        for i, n in enumerate(all_tokens_per_expert):
            n = int(n)
            if n > 0:
                torch.mm(all_expert_tokens[offset:offset + n], weight1[i], out=all_fc1[offset:offset + n])
                offset += n
    return all_fc1


# =============================================================================
# Combine Backward AllToAll with Merge Precompute
# =============================================================================


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], Dict[str, torch.Tensor]]:
    """
    Region 1: Combine AllToAll overlap FC2 dx (communication-first pipeline).

    Pipeline:
      1. Prepare all input chunks (batch memory operations)
      2. Submit all AllToAll chunks (chunked along hidden_size)
      3. FC1 GEMM recompute overlap with AllToAll (merge+sort done in forward)
      4. dW tasks overlap with AllToAll
      5. For each chunk: wait → layout convert → FC2 dx partial (accumulate)
      6. After all chunks: activation backward → grad_all_fc1

    comm_stream:    |A2A_0|A2A_1|A2A_2|A2A_3|
    default_stream: |prep|fc1_recompute+dW|wait0+dx0|wait1+dx1|wait2+dx2|wait3+dx3|act_bwd|

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
        num_chunks: Number of chunks for hidden dimension
        all_expert_tokens: [total_recv, hidden] pre-merged expert-major tokens (from forward)
        all_tokens_per_expert: token counts per local expert (from forward)
        backward_indices: pre-computed layout convert indices (from forward)

    Returns:
        grad_all_fc1: [total_recv, ffn_hidden] gradient w.r.t. FC1 output
        act_output: [total_recv, ffn_hidden] activation output (detached)
        all_fc1: [total_recv, ffn_hidden] FC1 pre-activation (expert-major)
        grad_all_fc2: [total_recv, hidden] gradient after AllToAll + layout convert (for dW)
        all_expert_tokens: [total_recv, hidden] expert-major tokens
        all_tokens_per_expert: token counts per local expert
        backward_indices: layout convert indices for region 1/2
    """
    nvtx_range_push("combine_fc2_backward")
    scheduler = get_backward_scheduler()
    device = grad_output.device
    dtype = grad_output.dtype
    hidden_size = grad_output.shape[1]
    ffn_hidden = weight2.shape[1]
    total_output = grad_output.shape[0]
    total_recv = sum(output_splits_list)

    # Validate chunk size
    if num_chunks > 1 and hidden_size % num_chunks != 0:
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
        grad_all_fc1, act_output, grad_all_fc2 = _post_combine_single_alltoall(
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
        )

        nvtx_range_pop()  # combine_fc2_backward
        return (
            grad_all_fc1, act_output.detach(), all_fc1, grad_all_fc2,
            all_expert_tokens, all_tokens_per_expert, backward_indices,
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
        grad_all_fc1, act_output, grad_all_fc2 = _post_combine_single_alltoall(
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
        )

        nvtx_range_pop()  # combine_fc2_backward
        return (
            grad_all_fc1, act_output.detach(), all_fc1, grad_all_fc2,
            all_expert_tokens, all_tokens_per_expert, backward_indices,
        )

    # ========================================================================
    # Chunked communication-first pipeline
    # ========================================================================
    nvtx_range_push("combine_fc2_chunked")
    chunk_size = hidden_size // num_chunks

    # Step 1/2: prepare chunk input + submit AllToAll chunk immediately
    # (start communication earlier and avoid an extra full pass over chunks)
    grad_output_contig = grad_output.contiguous()
    task_ids = []
    chunk_results = [None] * num_chunks
    _combine_out_all = _BWD_POOL.get_2d(
        tag="combine_output_chunks",
        rows=num_chunks * total_recv,
        cols=chunk_size,
        dtype=dtype,
        device=device,
    )
    chunk_output_buffers = [_combine_out_all[i * total_recv:(i + 1) * total_recv] for i in range(num_chunks)]
    _combine_in_all = _BWD_POOL.get_2d(
        tag="combine_input_chunks",
        rows=num_chunks * total_output,
        cols=chunk_size,
        dtype=dtype,
        device=device,
    )
    chunk_input_buffers = [_combine_in_all[i * total_output:(i + 1) * total_output] for i in range(num_chunks)]

    def make_alltoall_fn(idx, input_buf, output_buf):
        def do_alltoall():
            dist.all_to_all_single(
                output_buf, input_buf,
                output_split_sizes=output_splits_list,
                input_split_sizes=input_splits_list,
                group=ep_group,
            )
            chunk_results[idx] = output_buf
            return output_buf
        return do_alltoall

    nvtx_range_push("prepare_submit_chunks")
    # Split all columns into contiguous chunks with a single Triton kernel,
    # then submit AllToAll for each chunk.
    split_columns_to_chunks(grad_output_contig, num_chunks, _combine_in_all)
    task_ids = []
    for chunk_idx in range(num_chunks):
        task_id = scheduler.submit_alltoall(
            make_alltoall_fn(chunk_idx, chunk_input_buffers[chunk_idx], chunk_output_buffers[chunk_idx])
        )
        task_ids.append(task_id)
    nvtx_range_pop()

    # Step 3: FC1 GEMM recompute (overlaps with AllToAll; merge+sort done in forward)
    all_fc1 = recompute_fc1_gemm(all_expert_tokens, all_tokens_per_expert, weight1)

    # Step 4: dW tasks overlap with AllToAll
    _maybe_execute_dw_tasks(scheduler, for_dispatch=False)

    # Step 5: For each chunk: wait → layout convert → FC2 dx partial
    row_idx_rank_to_exp = backward_indices['row_idx_rank_to_exp']
    grad_exp_act = _BWD_POOL.get_2d(
        tag="combine_grad_exp_act",
        rows=total_recv,
        cols=ffn_hidden,
        dtype=dtype,
        device=device,
    )
    if _should_keep_grad_all_fc2_stable():
        # dW tasks may be deferred to finish_batch; avoid pooled storage aliasing.
        grad_all_fc2 = torch.empty(total_recv, hidden_size, dtype=dtype, device=device)
    else:
        grad_all_fc2 = _BWD_POOL.get_2d(
            tag="combine_grad_all_fc2",
            rows=total_recv,
            cols=hidden_size,
            dtype=dtype,
            device=device,
        )

    expert_ranges = _nonzero_expert_ranges(all_tokens_per_expert)

    for chunk_idx in range(num_chunks):
        h_start = chunk_idx * chunk_size
        h_end = h_start + chunk_size

        # Wait for this chunk's AllToAll (only trickle AR on last chunk)
        nvtx_range_push(f"wait_a2a_{chunk_idx}")
        is_last = (chunk_idx == num_chunks - 1)
        scheduler.wait_alltoall(task_ids[chunk_idx], try_trickle=is_last)
        nvtx_range_pop()

        grad_recv_chunk = chunk_results[chunk_idx]  # [total_recv, chunk_size]

        # Layout convert: rank-major → expert-major (needed for dW)
        nvtx_range_push(f"layout_cvt_{chunk_idx}")
        row_gather(grad_recv_chunk, row_idx_rank_to_exp,
                   out=grad_all_fc2[:, h_start:h_end])
        nvtx_range_pop()

        # FC2 dx partial: accumulated across chunks
        nvtx_range_push(f"fc2_dx_{chunk_idx}")
        if chunk_idx == 0:
            for exp_idx, start, end in expert_ranges:
                torch.mm(
                    grad_all_fc2[start:end, h_start:h_end],
                    weight2[exp_idx, :, h_start:h_end].t(),
                    out=grad_exp_act[start:end],
                )
        else:
            for exp_idx, start, end in expert_ranges:
                grad_exp_act[start:end].addmm_(
                    grad_all_fc2[start:end, h_start:h_end],
                    weight2[exp_idx, :, h_start:h_end].t()
                )
        nvtx_range_pop()

    # Step 6: Activation backward (after all chunks complete)
    nvtx_range_push("act_backward")
    grad_all_fc1, act_output = _activation_backward(grad_exp_act, all_fc1, activation_func)
    nvtx_range_pop()

    nvtx_range_pop()  # combine_fc2_chunked
    nvtx_range_pop()  # combine_fc2_backward
    return (
        grad_all_fc1, act_output.detach(), all_fc1, grad_all_fc2,
        all_expert_tokens, all_tokens_per_expert, backward_indices,
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
) -> torch.Tensor:
    """
    Region 2: FC1 dx + Dispatch AllToAll (compute-first pipeline).

    Pipeline:
      1. Compute FC1 dx in chunks (grad_fc1 @ w1.T, chunked along hidden)
      2. As each chunk completes: reorder + submit dispatch AllToAll
      3. dW tasks overlap with final AllToAll
      4. Wait for all AllToAll to complete, gather results

    default_stream: |dx_0+reorder+submit|dx_1+reorder+submit|...|dW|wait|
    comm_stream:                       |A2A_0              |A2A_1|...|

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
        num_chunks: Number of chunks for hidden dimension

    Returns:
        grad_tokens: [total_send, hidden] gradient w.r.t. input tokens (after AllToAll)
    """
    nvtx_range_push("fc1_dispatch_backward")
    scheduler = get_backward_scheduler()
    device = grad_all_fc1.device
    dtype = grad_all_fc1.dtype
    total_recv = grad_all_fc1.shape[0]
    hidden_size = weight1.shape[1]

    # Validate chunk size
    if num_chunks > 1 and hidden_size % num_chunks != 0:
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
    # ========================================================================
    nvtx_range_push("fc1_dispatch_chunked")
    chunk_size = hidden_size // num_chunks
    grad_tokens = _BWD_POOL.get_2d(
        tag="dispatch_grad_tokens",
        rows=total_send,
        cols=hidden_size,
        dtype=dtype,
        device=device,
    )
    chunk_results = [None] * num_chunks
    task_ids = []
    expert_ranges = _nonzero_expert_ranges(all_tokens_per_expert)
    _dispatch_out_all = _BWD_POOL.get_2d(
        tag="dispatch_output_chunks",
        rows=num_chunks * total_send,
        cols=chunk_size,
        dtype=dtype,
        device=device,
    )
    chunk_output_buffers = [_dispatch_out_all[i * total_send:(i + 1) * total_send] for i in range(num_chunks)]
    _dispatch_dx_all = _BWD_POOL.get_2d(
        tag="dispatch_dx_chunks",
        rows=num_chunks * total_recv,
        cols=chunk_size,
        dtype=dtype,
        device=device,
    )
    dx_chunk_buffers = [_dispatch_dx_all[i * total_recv:(i + 1) * total_recv] for i in range(num_chunks)]
    _dispatch_reorder_all = _BWD_POOL.get_2d(
        tag="dispatch_reorder_chunks",
        rows=num_chunks * total_recv,
        cols=chunk_size,
        dtype=dtype,
        device=device,
    )
    reorder_out_buffers = [_dispatch_reorder_all[i * total_recv:(i + 1) * total_recv] for i in range(num_chunks)]

    def make_alltoall_fn(idx, input_buf, output_buf):
        def do_alltoall():
            dist.all_to_all_single(
                output_buf, input_buf,
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group,
            )
            chunk_results[idx] = output_buf
            return output_buf
        return do_alltoall

    for chunk_idx in range(num_chunks):
        h_start = chunk_idx * chunk_size
        h_end = h_start + chunk_size

        # Compute FC1 dx chunk: grad_fc1 @ w1[:, h_start:h_end, :].T
        nvtx_range_push(f"fc1_dx_{chunk_idx}")
        dx_chunk = dx_chunk_buffers[chunk_idx]
        for exp_idx, start, end in expert_ranges:
            torch.mm(
                grad_all_fc1[start:end],
                weight1[exp_idx, h_start:h_end, :].t(),
                out=dx_chunk[start:end],
            )
        nvtx_range_pop()

        # Reorder expert-major -> rank-major
        nvtx_range_push(f"reorder_{chunk_idx}")
        reordered = _reorder_chunks(
            dx_chunk,
            row_idx=row_idx_exp_to_rank,
            out=reorder_out_buffers[chunk_idx],
        )
        nvtx_range_pop()

        # Submit AllToAll
        nvtx_range_push(f"submit_a2a_{chunk_idx}")
        task_id = scheduler.submit_alltoall(
            make_alltoall_fn(chunk_idx, reordered, chunk_output_buffers[chunk_idx])
        )
        task_ids.append(task_id)
        nvtx_range_pop()

    # dW tasks during final AllToAll
    _maybe_execute_dw_tasks(scheduler, for_dispatch=True)

    # Wait and copy each chunk into grad_tokens immediately.
    # all_to_all_single needs contiguous output so the column-slice copy
    # is unavoidable, but interleaving wait+copy pipelines the copies
    # with remaining AllToAll completion.
    nvtx_range_push("wait_gather")
    for chunk_idx in range(num_chunks):
        scheduler.wait_alltoall(task_ids[chunk_idx])
        h_start = chunk_idx * chunk_size
        grad_tokens[:, h_start:h_start + chunk_size].copy_(chunk_results[chunk_idx])
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
    top_probs: torch.Tensor,
    router_probs: torch.Tensor,
    top_indices: torch.Tensor,
    router_weight: torch.Tensor,
    num_tokens: int,
    top_k: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute router backward: gradients through softmax, top-k, and linear projection.

    Args:
        grad_permuted_probs: [kept_tokens] gradient w.r.t. permuted probabilities
        sorted_indices: [kept_tokens] kept expanded indices (in [0, N*top_k))
        top_probs: [num_tokens, top_k] normalized top-k probabilities (from forward)
        router_probs: [num_tokens, num_experts] full softmax probabilities
        top_indices: [num_tokens, top_k] indices of top-k experts
        router_weight: [hidden_size, num_experts] router weight matrix
        num_tokens: Number of input tokens
        top_k: Number of experts per token
        dtype: Data type for output

    Returns:
        grad_hidden_from_router: [num_tokens, hidden_size] gradient w.r.t. hidden_states
        grad_router_logits: [num_tokens, num_experts] gradient w.r.t. router logits (for dW)
    """
    # Step 1: Scatter grad back to [N, top_k]
    device = grad_permuted_probs.device
    grad_top_probs = torch.zeros(num_tokens, top_k, dtype=grad_permuted_probs.dtype, device=device)
    token_ids = sorted_indices // top_k
    slot_ids = sorted_indices % top_k
    grad_top_probs[token_ids, slot_ids] = grad_permuted_probs
    top_probs_saved = top_probs  # already normalized in forward

    # Step 2: Backward through normalization
    # top_probs = raw_top_probs / sum(raw_top_probs)
    # grad_raw[i] = grad_top[i] / s - (grad_top · top_probs) * top_probs[i] / s
    grad_dot = (grad_top_probs * top_probs_saved).sum(dim=-1, keepdim=True)
    grad_raw_top_probs = (grad_top_probs - grad_dot * top_probs_saved) / top_probs_saved.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    # Step 3: Backward through top-k selection
    # top_probs, top_indices = torch.topk(router_probs, k=top_k, dim=-1)
    # grad_router_probs is zero except at top_indices positions
    grad_router_probs = torch.zeros_like(router_probs)
    grad_router_probs.scatter_(1, top_indices, grad_raw_top_probs)

    # Step 4: Backward through softmax
    # router_probs = softmax(router_logits)
    # grad_logits = router_probs * (grad_probs - sum(grad_probs * router_probs))
    sum_grad_probs = (grad_router_probs * router_probs).sum(dim=-1, keepdim=True)
    grad_router_logits = router_probs * (grad_router_probs - sum_grad_probs)

    # Step 5: Backward through router linear: logits = hidden @ weight
    # grad_hidden = grad_logits @ weight.T
    grad_hidden_from_router = torch.matmul(grad_router_logits.float(), router_weight.t().float()).to(dtype)

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
            grad_weight = torch.matmul(hidden_saved.t().float(), grad_logits_saved.float())
            return grad_weight

        scheduler.register_dw_task(
            layer_name=f"router_weight_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_router_dw,
            weight_param=router_weight,
        )
        return None
    else:
        grad_router_weight = torch.matmul(hidden_states.t().float(), grad_router_logits.float())
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
]

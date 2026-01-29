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

from fluid.core import _all_to_all, _sort_chunks_by_idxs
from fluid.core.scheduler import get_backward_scheduler
from fluid.core.nvtx import nvtx_range_push, nvtx_range_pop, nvtx_mark


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
    if os.environ.get('FLUID_DEBUG_SCHEDULER_REGISTER', '0') == '1':
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
        ffn_hidden_saved = ffn_hidden
        hidden_size_saved = hidden_size

        def compute_dw_weight2():
            # Compute gradients directly in 3D [E, ffn, hidden] - no permute needed
            # Weight shape: [E, ffn, hidden], gradient same shape
            device = grad_all_fc2_saved.device
            dtype = grad_all_fc2_saved.dtype
            grad_w2_3d = torch.zeros(num_local_experts_saved, ffn_hidden_saved, hidden_size_saved,
                                     dtype=dtype, device=device)
            start = 0
            for exp_idx in range(num_local_experts_saved):
                n_tok = all_tokens_per_expert_saved[exp_idx]
                if n_tok > 0:
                    grad_w2_3d[exp_idx] = torch.matmul(
                        act_output_saved[start:start+n_tok].t(),
                        grad_all_fc2_saved[start:start+n_tok]
                    )
                    start += n_tok
            return grad_w2_3d  # Return 3D directly

        def compute_dw_weight1():
            # Compute gradients directly in 3D [E, hidden, ffn] - no permute needed
            # Weight shape: [E, hidden, ffn], gradient same shape
            device = grad_all_fc1_saved.device
            dtype = grad_all_fc1_saved.dtype
            grad_w1_3d = torch.zeros(num_local_experts_saved, hidden_size_saved, ffn_hidden_saved,
                                     dtype=dtype, device=device)
            start = 0
            for exp_idx in range(num_local_experts_saved):
                n_tok = all_tokens_per_expert_saved[exp_idx]
                if n_tok > 0:
                    grad_w1_3d[exp_idx] = torch.matmul(
                        all_expert_tokens_saved[start:start+n_tok].t(),
                        grad_all_fc1_saved[start:start+n_tok]
                    )
                    start += n_tok
            return grad_w1_3d  # Return 3D directly

        scheduler.register_dw_task(
            layer_name=f"moe_weight2_L{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight2,
            priority=100,
            weight_param=orig_weight2,
            needs_ar=False,  # Expert params use EP, not DP
        )
        scheduler.register_dw_task(
            layer_name=f"moe_weight1_L{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight1,
            priority=99,
            weight_param=orig_weight1,
            needs_ar=False,  # Expert params use EP, not DP
        )
        return None, None
    else:
        # Scheduler disabled: compute dW directly in 3D (no permute needed)
        device = grad_all_fc2.device
        dtype = grad_all_fc2.dtype

        # Compute in 3D [E, ffn, hidden] for weight2
        grad_w2_3d = torch.zeros(num_local_experts, ffn_hidden, hidden_size, dtype=dtype, device=device)
        # Compute in 3D [E, hidden, ffn] for weight1
        grad_w1_3d = torch.zeros(num_local_experts, hidden_size, ffn_hidden, dtype=dtype, device=device)

        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                grad_w2_3d[exp_idx] = torch.matmul(
                    act_output[start:start+n_tok].t(),
                    grad_all_fc2[start:start+n_tok]
                )
                grad_w1_3d[exp_idx] = torch.matmul(
                    all_expert_tokens[start:start+n_tok].t(),
                    grad_all_fc1[start:start+n_tok]
                )
                start += n_tok

        # Return 3D gradients directly (weights are now 3D)
        return grad_w1_3d, grad_w2_3d


# =============================================================================
# FC1 Recomputation (for Activation Recomputation optimization)
# =============================================================================

def recompute_fc1(
    all_expert_tokens: torch.Tensor,
    weight1: torch.Tensor,
    num_local_experts: int,
    all_tokens_per_expert: List[int],
) -> torch.Tensor:
    """
    Recompute FC1 outputs from tokens and weight1.

    This is called during backward to avoid saving FC1 in forward,
    reducing memory copy overhead (~2.5ms savings in forward).

    The recomputation takes ~5ms but can be hidden by Combine AllToAll (~28ms).

    Args:
        all_expert_tokens: [total_recv, hidden] all tokens (expert-major order)
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight (3D view)
        num_local_experts: Number of local experts
        all_tokens_per_expert: Token count per expert

    Returns:
        all_fc1: [total_recv, ffn_hidden] FC1 outputs (expert-major order)
    """
    device = all_expert_tokens.device
    dtype = all_expert_tokens.dtype
    total_recv = all_expert_tokens.shape[0]
    ffn_hidden = weight1.shape[-1]

    if total_recv == 0:
        return torch.empty(0, ffn_hidden, dtype=dtype, device=device)

    all_fc1 = torch.zeros(total_recv, ffn_hidden, dtype=dtype, device=device)

    start = 0
    for exp_idx in range(num_local_experts):
        n_tok = all_tokens_per_expert[exp_idx]
        if n_tok > 0:
            # FC1 = tokens @ weight1
            all_fc1[start:start + n_tok] = torch.matmul(
                all_expert_tokens[start:start + n_tok],
                weight1[exp_idx]
            )
            start += n_tok

    return all_fc1


# =============================================================================
# Combine Backward AllToAll with FC1 Recomputation
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
    weight2: torch.Tensor,
    activation_func,
    num_local_experts: int,
    all_tokens_per_expert: List[int],
    num_chunks: int = 4,
    # For FC1 recompute
    all_expert_tokens: Optional[torch.Tensor] = None,
    weight1: Optional[torch.Tensor] = None,
    # Layout convert indices
    backward_indices: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Region 1: Combine AllToAll overlap FC2 dx (communication-first pipeline).

    Pipeline:
      1. Submit all AllToAll chunks (chunked along hidden_size)
      2. FC1 recompute + dW tasks overlap with AllToAll
      3. For each chunk: wait → layout convert → FC2 dx partial (accumulate)
      4. After all chunks: activation backward → grad_all_fc1

    comm_thread: |A2A_0|A2A_1|A2A_2|A2A_3|
    default:     |fc1_recomp+dW|wait0+dx0|wait1+dx1|wait2+dx2|wait3+dx3|act_bwd|

    Args:
        grad_output: [total_output, hidden] gradient w.r.t. output
        input_splits_list: AllToAll input split sizes
        output_splits_list: AllToAll output split sizes
        ep_group: Expert parallel process group
        layer_id: Layer ID for debugging
        weight2: [num_local_experts, ffn_hidden, hidden] FC2 weight
        activation_func: Activation function
        num_local_experts: Number of local experts
        all_tokens_per_expert: Token count per expert
        num_chunks: Number of chunks for hidden dimension
        all_expert_tokens: [total_recv, hidden] tokens for FC1 recomputation
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight
        backward_indices: Dict with layout convert indices

    Returns:
        grad_all_fc1: [total_recv, ffn_hidden] gradient w.r.t. FC1 output
        act_output: [total_recv, ffn_hidden] activation output (detached)
        all_fc1: [total_recv, ffn_hidden] recomputed FC1 or None
        grad_all_fc2: [total_recv, hidden] gradient after AllToAll + layout convert (for dW)
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

    all_fc1 = None

    if not scheduler.is_enabled() or num_chunks <= 1:
        # ---- Fallback: non-chunked path ----
        nvtx_range_push("combine_alltoall")
        grad_combined_recv = _all_to_all(
            grad_output.contiguous(),
            output_split_sizes=output_splits_list,
            input_split_sizes=input_splits_list,
            group=ep_group
        )
        nvtx_range_pop()

        # FC1 recompute
        if all_expert_tokens is not None and weight1 is not None:
            nvtx_range_push("fc1_recompute")
            all_fc1 = recompute_fc1(all_expert_tokens, weight1, num_local_experts, all_tokens_per_expert)
            nvtx_range_pop()

        # Layout convert: rank-major -> expert-major
        nvtx_range_push("layout_convert")
        if backward_indices is not None and 'split_sizes_rank_major' in backward_indices:
            grad_all_fc2 = _sort_chunks_by_idxs(
                grad_combined_recv,
                backward_indices['split_sizes_rank_major'],
                backward_indices['sorted_idxs_rank_to_exp'],
            )
        else:
            grad_all_fc2 = grad_combined_recv
        nvtx_range_pop()

        # FC2 dx: grad_fc2 @ w2.T → grad_exp_act
        nvtx_range_push("fc2_dx")
        grad_exp_act = torch.zeros(total_recv, ffn_hidden, dtype=dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                grad_exp_act[start:start+n_tok] = torch.matmul(
                    grad_all_fc2[start:start+n_tok], weight2[exp_idx].t()
                )
                start += n_tok
        nvtx_range_pop()

        # Activation backward
        nvtx_range_push("act_backward")
        if all_fc1 is None:
            all_fc1 = recompute_fc1(all_expert_tokens, weight1, num_local_experts, all_tokens_per_expert)
        with torch.enable_grad():
            fc1_with_grad = all_fc1.detach().requires_grad_(True)
            act_output = activation_func(fc1_with_grad)
            grad_all_fc1, = torch.autograd.grad(act_output, fc1_with_grad, grad_exp_act, retain_graph=False)
        nvtx_range_pop()

        nvtx_range_pop()  # combine_fc2_backward
        return grad_all_fc1, act_output.detach(), all_fc1, grad_all_fc2

    # ========================================================================
    # Chunked communication-first pipeline
    # ========================================================================
    nvtx_range_push("combine_fc2_chunked")
    chunk_size = hidden_size // num_chunks

    # Step 1: Submit all AllToAll chunks
    task_ids = []
    chunk_results = [None] * num_chunks

    grad_output_contig = grad_output.contiguous()
    for chunk_idx in range(num_chunks):
        h_start = chunk_idx * chunk_size
        h_end = h_start + chunk_size

        _chunk_idx = chunk_idx
        _input_chunk = grad_output_contig[:, h_start:h_end].contiguous()

        def make_alltoall_fn(idx, input_buf):
            def do_alltoall():
                result = _all_to_all(
                    input_buf,
                    output_split_sizes=output_splits_list,
                    input_split_sizes=input_splits_list,
                    group=ep_group
                )
                chunk_results[idx] = result
                return result
            return do_alltoall

        nvtx_range_push(f"submit_a2a_{chunk_idx}")
        task_id = scheduler.submit_alltoall(make_alltoall_fn(_chunk_idx, _input_chunk))
        task_ids.append(task_id)
        nvtx_range_pop()

    # Step 2: FC1 recompute + dW tasks overlap with AllToAll
    if all_expert_tokens is not None and weight1 is not None:
        nvtx_range_push("fc1_recompute")
        all_fc1 = recompute_fc1(all_expert_tokens, weight1, num_local_experts, all_tokens_per_expert)
        nvtx_range_pop()

    if not os.environ.get('FLUID_SKIP_ALL_DW', '0') == '1':
        nvtx_range_push("dw_tasks")
        scheduler.execute_dw_tasks()
        nvtx_range_pop()

    # Step 3: For each chunk: wait → layout convert → FC2 dx partial
    grad_exp_act = torch.zeros(total_recv, ffn_hidden, dtype=dtype, device=device)
    grad_all_fc2 = torch.empty(total_recv, hidden_size, dtype=dtype, device=device)

    for chunk_idx in range(num_chunks):
        h_start = chunk_idx * chunk_size
        h_end = h_start + chunk_size

        # Wait for this chunk's AllToAll
        nvtx_range_push(f"wait_a2a_{chunk_idx}")
        scheduler.wait_alltoall(task_ids[chunk_idx])
        nvtx_range_pop()

        grad_recv_chunk = chunk_results[chunk_idx]  # [total_recv, chunk_size]

        # Layout convert for this chunk
        nvtx_range_push(f"layout_cvt_{chunk_idx}")
        if backward_indices is not None and 'split_sizes_rank_major' in backward_indices:
            grad_fc2_chunk = _sort_chunks_by_idxs(
                grad_recv_chunk,
                backward_indices['split_sizes_rank_major'],
                backward_indices['sorted_idxs_rank_to_exp'],
            )
        else:
            grad_fc2_chunk = grad_recv_chunk
        nvtx_range_pop()

        # Save for dW computation
        grad_all_fc2[:, h_start:h_end] = grad_fc2_chunk

        # FC2 dx partial: grad_fc2_chunk @ w2[exp, :, h_start:h_end].T
        nvtx_range_push(f"fc2_dx_{chunk_idx}")
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                grad_exp_act[start:start+n_tok].addmm_(
                    grad_fc2_chunk[start:start+n_tok],
                    weight2[exp_idx, :, h_start:h_end].t()
                )
                start += n_tok
        nvtx_range_pop()

    # Step 4: Activation backward (after all chunks complete)
    nvtx_range_push("act_backward")
    if all_fc1 is None:
        all_fc1 = recompute_fc1(all_expert_tokens, weight1, num_local_experts, all_tokens_per_expert)
    with torch.enable_grad():
        fc1_with_grad = all_fc1.detach().requires_grad_(True)
        act_output = activation_func(fc1_with_grad)
        grad_all_fc1, = torch.autograd.grad(act_output, fc1_with_grad, grad_exp_act, retain_graph=False)
    nvtx_range_pop()

    nvtx_range_pop()  # combine_fc2_chunked
    nvtx_range_pop()  # combine_fc2_backward
    return grad_all_fc1, act_output.detach(), all_fc1, grad_all_fc2


# =============================================================================
# Region 2: FC1 dx → Dispatch AllToAll (Compute-First Pipeline)
# =============================================================================

def fc1_dispatch_backward(
    grad_all_fc1: torch.Tensor,
    weight1: torch.Tensor,
    num_local_experts: int,
    all_tokens_per_expert: List[int],
    split_sizes_exp_major: List[int],
    sorted_idxs_exp_to_rank: List[int],
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

    default:     |dx_0+reorder+submit|dx_1+reorder+submit|...|dW|wait|
    comm_thread:                     |A2A_0              |A2A_1|...|

    Args:
        grad_all_fc1: [total_recv, ffn_hidden] gradient w.r.t. FC1 output
        weight1: [num_local_experts, hidden, ffn_hidden] FC1 weight
        num_local_experts: Number of local experts
        all_tokens_per_expert: Token count per expert
        split_sizes_exp_major: Chunk sizes in expert-major order
        sorted_idxs_exp_to_rank: Indices for expert-major -> rank-major reorder
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

    # Convert to list if tensor
    if torch.is_tensor(split_sizes_exp_major):
        split_sizes_exp_major = split_sizes_exp_major.tolist()
    if torch.is_tensor(sorted_idxs_exp_to_rank):
        sorted_idxs_exp_to_rank = sorted_idxs_exp_to_rank.tolist()

    # Validate chunk size
    if num_chunks > 1 and hidden_size % num_chunks != 0:
        num_chunks = 1

    total_send = sum(input_splits_list)

    if not scheduler.is_enabled() or num_chunks <= 1:
        # ---- Non-chunked path ----
        nvtx_range_push("fc1_dx")
        grad_all_tokens = torch.zeros(total_recv, hidden_size, dtype=dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                grad_all_tokens[start:start+n_tok] = torch.matmul(
                    grad_all_fc1[start:start+n_tok], weight1[exp_idx].t()
                )
                start += n_tok
        nvtx_range_pop()

        # Reorder expert-major -> rank-major
        nvtx_range_push("reorder")
        grad_dispatched = _sort_chunks_by_idxs(
            grad_all_tokens, split_sizes_exp_major, sorted_idxs_exp_to_rank
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
                    group=ep_group
                )
                return result_holder[0]

            task_id = scheduler.submit_alltoall(do_alltoall)

            if not (os.environ.get('FLUID_SKIP_DISPATCH_DW', '0') == '1' or
                    os.environ.get('FLUID_SKIP_ALL_DW', '0') == '1'):
                nvtx_range_push("dw_tasks")
                scheduler.execute_dw_tasks()
                nvtx_range_pop()

            scheduler.wait_alltoall(task_id)
            grad_tokens = result_holder[0]
        else:
            grad_tokens = _all_to_all(
                grad_dispatched.contiguous(),
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group
            )

        nvtx_range_pop()  # fc1_dispatch_backward
        return grad_tokens

    # ========================================================================
    # Chunked compute-first pipeline
    # ========================================================================
    nvtx_range_push("fc1_dispatch_chunked")
    chunk_size = hidden_size // num_chunks
    grad_tokens = torch.empty(total_send, hidden_size, dtype=dtype, device=device)
    chunk_results = [None] * num_chunks
    task_ids = []

    for chunk_idx in range(num_chunks):
        h_start = chunk_idx * chunk_size
        h_end = h_start + chunk_size

        # Compute FC1 dx chunk: grad_fc1 @ w1[exp, h_start:h_end, :].T
        nvtx_range_push(f"fc1_dx_{chunk_idx}")
        dx_chunk = torch.zeros(total_recv, chunk_size, dtype=dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                dx_chunk[start:start+n_tok] = torch.matmul(
                    grad_all_fc1[start:start+n_tok],
                    weight1[exp_idx, h_start:h_end, :].t()
                )
                start += n_tok
        nvtx_range_pop()

        # Reorder expert-major -> rank-major
        nvtx_range_push(f"reorder_{chunk_idx}")
        reordered = _sort_chunks_by_idxs(
            dx_chunk, split_sizes_exp_major, sorted_idxs_exp_to_rank
        )
        nvtx_range_pop()

        # Submit AllToAll
        _chunk_idx = chunk_idx
        _input_buf = reordered.contiguous()

        def make_alltoall_fn(idx, input_buf, t_send, c_size):
            def do_alltoall():
                output_buf = torch.empty(t_send, c_size, dtype=dtype, device=device)
                dist.all_to_all_single(
                    output_buf, input_buf,
                    output_split_sizes=input_splits_list,
                    input_split_sizes=output_splits_list,
                    group=ep_group,
                )
                chunk_results[idx] = output_buf
                return output_buf
            return do_alltoall

        nvtx_range_push(f"submit_a2a_{chunk_idx}")
        task_id = scheduler.submit_alltoall(make_alltoall_fn(_chunk_idx, _input_buf, total_send, chunk_size))
        task_ids.append(task_id)
        nvtx_range_pop()

    # dW tasks during final AllToAll
    if not (os.environ.get('FLUID_SKIP_DISPATCH_DW', '0') == '1' or
            os.environ.get('FLUID_SKIP_ALL_DW', '0') == '1'):
        nvtx_range_push("dw_tasks")
        scheduler.execute_dw_tasks()
        nvtx_range_pop()

    # Wait for last AllToAll (FIFO guarantees all earlier ones done)
    nvtx_range_push("wait_alltoall")
    if task_ids:
        scheduler.wait_alltoall(task_ids[-1], num_tasks=len(task_ids))
    nvtx_range_pop()

    # Gather results into output
    nvtx_range_push("gather_results")
    for chunk_idx in range(num_chunks):
        h_start = chunk_idx * chunk_size
        h_end = h_start + chunk_size
        grad_tokens[:, h_start:h_end].copy_(chunk_results[chunk_idx])
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
    restore_indices: torch.Tensor,
    permuted_probs: torch.Tensor,
    router_probs: torch.Tensor,
    top_indices: torch.Tensor,
    router_weight: torch.Tensor,
    num_tokens: int,
    top_k: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute router backward: gradients through softmax, top-k, and linear projection.

    This computes:
    1. Gradient through prob permutation/sorting
    2. Gradient through prob normalization
    3. Gradient through top-k selection
    4. Gradient through softmax
    5. Gradient through router linear (input @ weight -> logits)

    Args:
        grad_permuted_probs: [num_tokens * top_k] gradient w.r.t. permuted probabilities
        sorted_indices: [num_tokens * top_k] indices used to sort tokens
        restore_indices: [num_tokens * top_k] indices to restore original order
        permuted_probs: [num_tokens * top_k] permuted routing probabilities
        router_probs: [num_tokens, num_experts] full router probabilities (after softmax)
        top_indices: [num_tokens, top_k] indices of top-k experts
        router_weight: [hidden_size, num_experts] router weight matrix
        num_tokens: Number of input tokens
        top_k: Number of experts per token
        dtype: Data type for output

    Returns:
        grad_hidden_from_router: [num_tokens, hidden_size] gradient w.r.t. hidden_states
        grad_router_logits: [num_tokens, num_experts] gradient w.r.t. router logits (for dW)
    """
    # Step 1: Backward through prob permutation
    # permuted_probs = expanded_probs[sorted_indices]
    # expanded_probs = top_probs.reshape(-1)
    grad_expanded_probs = torch.zeros_like(grad_permuted_probs)
    grad_expanded_probs[sorted_indices] = grad_permuted_probs
    grad_top_probs = grad_expanded_probs.view(num_tokens, top_k)

    # Step 2: Backward through normalization
    # top_probs = raw_top_probs / sum(raw_top_probs)
    # grad_raw[i] = grad_top[i] / s - (grad_top · top_probs) * top_probs[i] / s
    top_probs_saved = permuted_probs[restore_indices].view(num_tokens, top_k)
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
            priority=0,  # Low priority, execute after other dW tasks
            weight_param=router_weight,
        )
        return None
    else:
        grad_router_weight = torch.matmul(hidden_states.t().float(), grad_router_logits.float())
        return grad_router_weight


__all__ = [
    # Region 1: Combine AllToAll + FC2 dx
    'combine_fc2_backward',
    # Region 2: FC1 dx + Dispatch AllToAll
    'fc1_dispatch_backward',
    # dW registration
    'register_moe_dw_tasks',
    'recompute_fc1',
    # Router backward
    'router_backward',
    'register_router_dw_task',
]

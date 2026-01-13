"""
Fluid versions of tensor parallel communication primitives with global scheduler integration

This module provides drop-in replacements for standard AlltoAll operations
that integrate with the global backward scheduler for fine-grained overlap.

Key optimization (v0.6):
- Chunked dX + AllToAll pipeline
- dX computation is split into chunks
- Each dX chunk completes -> its AllToAll starts immediately
- Subsequent dX chunks overlap with previous AllToAll
- Only the last chunk's AllToAll needs dW to fill the gap

Key optimization (v0.8): dW-AllToAll Overlap with PyTorch streams
- dW executes truly in parallel with AllToAll
- Timeline:
    default_stream:  |=== dX ===| event |=== dW ===|
                                    ↓
    comm_stream:               wait |=== AllToAll ===| event
                                                          ↓
    default_stream:                                  wait → continue
"""

import torch
import torch.distributed as dist
import os
from typing import List, Optional, Tuple, Dict


def _all_to_all(
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    group
) -> torch.Tensor:
    """
    Direct call to PyTorch all_to_all_single (bypass Megatron wrapper)

    Args:
        input: Input tensor
        output_split_sizes: Size of each output chunk (None for equal split)
        input_split_sizes: Size of each input chunk (None for equal split)
        group: Process group

    Returns:
        Output tensor after all-to-all
    """
    world_size = group.size()
    if world_size == 1:
        return input.clone()

    input = input.contiguous()

    if output_split_sizes is None:
        # Equal split
        output = torch.empty_like(input)
    else:
        # Unequal split (all2all-v)
        output = input.new_empty(
            size=[sum(output_split_sizes)] + list(input.size()[1:]),
            dtype=input.dtype,
            device=input.device,
        )

    dist.all_to_all_single(
        output, input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output

# Chunking configuration for dX + AllToAll pipeline
# Set via environment variable, default to 1 (no chunking)
#
# v0.7: FC1-only chunking
# Only chunk FC1 backward (not FC2 or activation) since FC1 is immediately
# followed by dispatch AllToAll. This allows:
# - FC2 backward: grouped_gemm (efficient, not chunked)
# - Activation backward: element-wise (fast, not chunked)
# - FC1 backward: chunked, with each chunk immediately followed by dispatch AllToAll
#
# Timeline:
#   |-- FC2_bwd --|-- Act_bwd --|-- FC1_c0 --|-- FC1_c1 --|
#                                |--- A2A_c0 ---|--- A2A_c1 ---|
#   FC1_c1 overlaps with A2A_c0, saving AllToAll time
DX_NUM_CHUNKS = int(os.environ.get('FLUID_DX_NUM_CHUNKS', '1'))

import threading
_dispatch_a2a_done = threading.local()

def set_dispatch_alltoall_done(done: bool = True):
    """Mark that dispatch AllToAll was handled in expert backward."""
    _dispatch_a2a_done.done = done

def get_dispatch_alltoall_done() -> bool:
    """Check if dispatch AllToAll was already done."""
    return getattr(_dispatch_a2a_done, 'done', False)

def clear_dispatch_alltoall_done():
    """Clear the dispatch AllToAll done flag."""
    _dispatch_a2a_done.done = False


class _FluidAllToAll(torch.autograd.Function):
    """
    AlltoAll with global backward scheduler integration and chunking support

    Forward: Same as standard _AllToAll
    Backward:
    - When num_chunks > 1: Splits grad into chunks, pipelines AllToAll
    - Triggers global scheduler to launch dW tasks during last chunk's communication

    Chunked backward timeline:
    |-- A2A_chunk0 --|-- A2A_chunk1 --|-- ... --|-- A2A_last --| <- dW here

    Note: This chunks the AllToAll itself, not the upstream dX computation.
    For true dX+AllToAll pipelining, use _FluidFusedDispatchExpert which fuses
    dX computation with AllToAll dispatch.
    """

    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes, comm_type="unknown"):
        """
        Forward function - same as standard AlltoAll

        Args:
            comm_type: "ep" for Expert Parallel, "ulysses" for Context Parallel
        """
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.comm_type = comm_type
        ctx.input_shape = input.shape

        # Use standard _AllToAll apply (not forward directly)
        return _all_to_all(input, output_split_sizes, input_split_sizes, group)

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Backward with chunked AllToAll and dW overlap

        When FLUID_DX_NUM_CHUNKS > 1:
        - Split grad_output into chunks
        - Launch each chunk's AllToAll asynchronously on comm_stream
        - On last chunk, trigger dW execution for overlap
        - Reassemble results maintaining correct ordering

        When TRUE ASYNC is enabled (ASYNC=1):
        - Use stream-based dW-AllToAll overlap for better GPU utilization
        """
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        grad = grad_output[0]
        num_chunks = DX_NUM_CHUNKS

        # Get split sizes for backward (swap input/output)
        # In backward: what was output becomes input, what was input becomes output
        backward_output_splits = ctx.input_split_sizes  # Where we send TO
        backward_input_splits = ctx.output_split_sizes  # Where we receive FROM

        # If no chunking or splits are None, use non-chunked path
        if num_chunks <= 1 or backward_output_splits is None or backward_input_splits is None:
            # Use dW-AllToAll overlap for all AllToAll (MoE dispatch and CP attention)
            if backward_output_splits is not None and backward_input_splits is not None:
                # ============================================================
                # TRUE ASYNC: dW-AllToAll Overlap with PyTorch streams
                # ============================================================
                # NOTE: We use PyTorch streams and events instead of C++ async_alltoall
                # because async_alltoall uses a separate NCCL communicator (WORLD group)
                # while the dispatch AllToAll uses the EP group communicator.
                #
                # Timeline:
                #   default_stream:  |=== dX ===| event |=== dW ===|
                #                                   ↓
                #   comm_stream:                wait |=== AllToAll ===| event
                #                                                         ↓
                #   default_stream:                                   wait → continue
                # ============================================================
                import time
                debug_timing = os.environ.get('FLUID_DEBUG_BACKWARD_TIMING', '0') == '1'
                if debug_timing:
                    torch.cuda.synchronize()
                    t_start = time.perf_counter()
                    print(f"[TRUE ASYNC dW-A2A] Starting, grad shape: {grad.shape}", flush=True)

                # Convert split sizes to list of ints
                send_splits = [int(s.item() if torch.is_tensor(s) else s) for s in backward_output_splits]
                recv_splits = [int(s.item() if torch.is_tensor(s) else s) for s in backward_input_splits]

                # Step 1: Launch AllToAll on comm_stream (non-blocking from default stream)
                # Use wait_stream instead of events to match ASYNC=0 behavior
                comm_stream = scheduler.comm_stream

                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_stream(scheduler.default_stream)
                    result = _all_to_all(
                        grad.contiguous(),
                        send_splits,  # backward: send to original senders
                        recv_splits,  # backward: recv from original receivers
                        ctx.group
                    )

                if debug_timing:
                    torch.cuda.synchronize()
                    t_after_launch = time.perf_counter()
                    print(f"[TRUE ASYNC dW-A2A] AllToAll launched: {(t_after_launch-t_start)*1000:.2f} ms", flush=True)

                # Step 2: Execute dW tasks NOW (parallel with AllToAll)
                # This is the key: dW runs on default_stream while AllToAll runs on comm_stream
                dw_executed = scheduler._execute_all_dw_tasks_sync()

                if debug_timing:
                    torch.cuda.synchronize()
                    t_after_dw = time.perf_counter()
                    print(f"[TRUE ASYNC dW-A2A] dW executed ({dw_executed} tasks): {(t_after_dw-t_after_launch)*1000:.2f} ms", flush=True)

                # Step 3: Wait for AllToAll to complete
                scheduler.default_stream.wait_stream(comm_stream)

                if debug_timing:
                    torch.cuda.synchronize()
                    t_end = time.perf_counter()
                    print(f"[TRUE ASYNC dW-A2A] Total: {(t_end-t_start)*1000:.2f} ms", flush=True)

                return (None, result, None, None, None)

            # Fallback: splits are None (shouldn't happen in normal usage)
            result = _all_to_all(grad, None, None, ctx.group)
            return (None, result, None, None, None)

        # ============================================================
        # Chunked AllToAll backward
        # Key: Split grad by the same proportions that AllToAll uses
        # Data is organized by source rank: [rank0's data][rank1's data]...
        # ============================================================

        ep_size = len(backward_input_splits)
        total_tokens = grad.shape[0]

        # Compute start indices for each source rank's data in grad
        rank_starts = [0]
        for split in backward_input_splits:
            rank_starts.append(rank_starts[-1] + split)

        alltoall_results = []

        for chunk_idx in range(num_chunks):
            # ============================================================
            # Extract this chunk's data from grad
            # chunk contains: [rank0's chunk portion][rank1's chunk portion]...
            # ============================================================
            chunk_parts = []
            chunk_backward_input_splits = []  # How much we receive from each rank for this chunk
            chunk_backward_output_splits = []  # How much we send to each rank for this chunk

            for src_rank in range(ep_size):
                rank_tokens = backward_input_splits[src_rank]
                if rank_tokens == 0:
                    chunk_backward_input_splits.append(0)
                    continue

                # Calculate chunk boundaries for this source rank
                chunk_size = rank_tokens // num_chunks
                remainder = rank_tokens % num_chunks

                chunk_start_in_rank = chunk_idx * chunk_size + min(chunk_idx, remainder)
                if chunk_idx < remainder:
                    this_chunk_size = chunk_size + 1
                else:
                    this_chunk_size = chunk_size
                chunk_end_in_rank = chunk_start_in_rank + this_chunk_size

                if this_chunk_size == 0:
                    chunk_backward_input_splits.append(0)
                    continue

                chunk_backward_input_splits.append(this_chunk_size)

                # Global indices in grad tensor
                global_start = rank_starts[src_rank] + chunk_start_in_rank
                global_end = rank_starts[src_rank] + chunk_end_in_rank

                chunk_parts.append(grad[global_start:global_end])

            # Compute chunk_backward_output_splits (how much we send to each dest rank)
            for dest_rank in range(ep_size):
                dest_tokens = backward_output_splits[dest_rank]
                if dest_tokens == 0:
                    chunk_backward_output_splits.append(0)
                    continue
                chunk_size = dest_tokens // num_chunks
                remainder = dest_tokens % num_chunks
                if chunk_idx < remainder:
                    this_chunk_size = chunk_size + 1
                else:
                    this_chunk_size = chunk_size
                chunk_backward_output_splits.append(this_chunk_size)

            # Concatenate chunk parts
            if chunk_parts:
                chunk_grad = torch.cat(chunk_parts, dim=0)
            else:
                chunk_grad = torch.zeros(0, grad.shape[-1], dtype=grad.dtype, device=grad.device)

            # ============================================================
            # Launch AllToAll for this chunk on comm_stream
            # ============================================================
            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                chunk_result = _all_to_all(
                    chunk_grad.contiguous(),
                    chunk_backward_output_splits,
                    chunk_backward_input_splits,
                    ctx.group,
                )
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)

            alltoall_results.append(chunk_result)

            # For the last chunk, set up dW overlap
            if chunk_idx == num_chunks - 1:
                scheduler.set_alltoall_end_event(event)
                scheduler.on_alltoall_start(comm_type=ctx.comm_type)

        # ============================================================
        # Execute dW tasks while AllToAll is running on comm_stream
        # This is the key dW-AllToAll overlap for chunked path
        # ============================================================
        dw_executed = scheduler._execute_all_dw_tasks_sync()

        # ============================================================
        # Wait and reassemble results
        # Results are ordered by dest rank: [dest0's data][dest1's data]...
        # Need to interleave chunks correctly
        # ============================================================
        scheduler.default_stream.wait_stream(scheduler.comm_stream)

        # Collect per-dest-rank chunks
        dest_rank_chunks = [[] for _ in range(ep_size)]
        for chunk_idx in range(num_chunks):
            chunk_result = alltoall_results[chunk_idx]
            offset = 0
            for dest_rank in range(ep_size):
                dest_tokens = backward_output_splits[dest_rank]
                if dest_tokens == 0:
                    continue
                chunk_size = dest_tokens // num_chunks
                remainder = dest_tokens % num_chunks
                if chunk_idx < remainder:
                    this_chunk_size = chunk_size + 1
                else:
                    this_chunk_size = chunk_size
                if this_chunk_size > 0:
                    dest_rank_chunks[dest_rank].append(chunk_result[offset:offset+this_chunk_size])
                    offset += this_chunk_size

        # Concatenate chunks for each dest rank, then concatenate dest ranks
        final_parts = []
        for dest_rank in range(ep_size):
            if dest_rank_chunks[dest_rank]:
                final_parts.append(torch.cat(dest_rank_chunks[dest_rank], dim=0))

        if final_parts:
            result = torch.cat(final_parts, dim=0)
        else:
            result = torch.zeros(sum(backward_output_splits), grad.shape[-1],
                               dtype=grad.dtype, device=grad.device)

        return (None, result, None, None, None)


def fluid_all_to_all(
    input: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    output_split_sizes: list = None,
    input_split_sizes: list = None,
    comm_type: str = "unknown",
) -> torch.Tensor:
    """
    Fluid AlltoAll with global scheduler integration

    Args:
        input: Input tensor
        group: Process group for communication
        output_split_sizes: Split sizes for output (optional)
        input_split_sizes: Split sizes for input (optional)
        comm_type: Type of communication ("ep", "ulysses", or "unknown")

    Returns:
        Output tensor after AlltoAll
    """
    return _FluidAllToAll.apply(group, input, output_split_sizes, input_split_sizes, comm_type)


def fluid_all_to_all_sp2hp(input_: torch.Tensor, group=None) -> torch.Tensor:
    """
    Ulysses SP: Sequence Parallel -> Head Parallel

    Shape: [seq/CP, batch, heads, dim] -> [seq, batch, heads/CP, dim]
    """
    from megatron.core.utils import get_tensor_model_parallel_group_if_none

    group = get_tensor_model_parallel_group_if_none(group)
    cp = group.size()
    seq_local, batch, heads, dim = input_.shape

    # Rearrange: split heads, move CP to front, flatten
    x = input_.view(seq_local, batch, cp, heads // cp, dim)
    x = x.permute(2, 0, 1, 3, 4).contiguous()
    x = x.view(seq_local * cp, -1)

    # AllToAll communication
    output = fluid_all_to_all(
        x, group,
        output_split_sizes=[seq_local] * cp,
        input_split_sizes=[seq_local] * cp,
        comm_type="ulysses"
    )

    # Reshape to output format
    return output.view(seq_local * cp, batch, heads // cp, dim)


def fluid_all_to_all_hp2sp(input_: torch.Tensor, group=None) -> torch.Tensor:
    """
    Ulysses SP: Head Parallel -> Sequence Parallel (reverse of sp2hp)

    Shape: [seq, batch, heads/CP, dim] -> [seq/CP, batch, heads, dim]
    """
    from megatron.core.utils import get_tensor_model_parallel_group_if_none

    group = get_tensor_model_parallel_group_if_none(group)
    cp = group.size()
    seq, batch, heads_local, dim = input_.shape
    seq_local = seq // cp

    # Flatten to 2D
    x = input_.view(seq, batch * heads_local * dim)

    # AllToAll communication
    output = fluid_all_to_all(
        x, group,
        output_split_sizes=[seq_local] * cp,
        input_split_sizes=[seq_local] * cp,
        comm_type="ulysses"
    )

    # Rearrange: unflatten, permute, merge heads
    output = output.view(cp, seq_local, batch, heads_local, dim)
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    return output.view(seq_local, batch, heads_local * cp, dim)


def fluid_all_to_all_qkv_sp2hp_batched(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    group=None
) -> tuple:
    """
    Batched QKV sp2hp AllToAll (merge 3 separate AllToAll into 1)

    Merges Q, K, V into a single AllToAll communication to reduce
    communication overhead and create longer communication window for dW overlap.

    Shape:
        Input:  q, k, v: [seq/CP, batch, heads, dim]
        Output: q, k, v: [seq, batch, heads/CP, dim]
    """
    from megatron.core.utils import get_tensor_model_parallel_group_if_none

    group = get_tensor_model_parallel_group_if_none(group)
    cp = group.size()

    if cp == 1:
        return query, key, value

    seq_local, batch, heads, dim = query.shape
    heads_local = heads // cp

    # Concatenate Q, K, V along head dimension: [seq_local, B, heads, D] x 3 -> [seq_local, B, 3*heads, D]
    qkv_batched = torch.cat([query, key, value], dim=2)

    # Single sp2hp AllToAll for batched QKV
    qkv_sp = fluid_all_to_all_sp2hp(qkv_batched, group)

    # Split back to Q, K, V: [seq, B, 3*heads_local, D] -> 3 x [seq, B, heads_local, D]
    # IMPORTANT: Make contiguous to avoid "view not compatible" errors in later operations
    q_sp = qkv_sp[:, :, :heads_local, :].contiguous()
    k_sp = qkv_sp[:, :, heads_local:2*heads_local, :].contiguous()
    v_sp = qkv_sp[:, :, 2*heads_local:, :].contiguous()

    return q_sp, k_sp, v_sp


def fluid_all_to_all_mixed_qkv_sp2hp(
    mixed_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Direct sp2hp AllToAll on mixed QKV tensor without pre-split.

    Optimized for Baseline mode to avoid redundant split->concat->split operations.

    Args:
        mixed_qkv: [seq_local, batch, q_proj_size + 2*kv_proj_size]
                   where q_proj_size = num_heads * head_dim
                         kv_proj_size = num_kv_heads * head_dim
        num_heads: Total number of query heads (before CP partitioning)
        num_kv_heads: Total number of key/value heads (before CP partitioning)
        head_dim: Dimension per head
        group: Communication group

    Returns:
        query: [seq_full, batch, num_heads/CP, head_dim]
        key: [seq_full, batch, num_kv_heads/CP, head_dim]
        value: [seq_full, batch, num_kv_heads/CP, head_dim]
    """
    from megatron.core.utils import get_tensor_model_parallel_group_if_none

    group = get_tensor_model_parallel_group_if_none(group)
    cp = group.size()

    if cp == 1:
        # No CP, just reshape and split locally
        seq_local, batch, _ = mixed_qkv.shape
        # Reshape to [seq, batch, num_groups, (q_per_group + 2) * head_dim]
        num_groups = num_kv_heads
        q_per_group = num_heads // num_kv_heads
        mixed_qkv_reshaped = mixed_qkv.view(
            seq_local, batch, num_groups, (q_per_group + 2) * head_dim
        )
        # Split Q, K, V
        q_size = q_per_group * head_dim
        query, key, value = torch.split(
            mixed_qkv_reshaped, [q_size, head_dim, head_dim], dim=3
        )
        # Reshape query to [seq, batch, num_heads, head_dim]
        query = query.reshape(seq_local, batch, num_heads, head_dim)
        return query, key, value

    # ===== With CP: Do AllToAll then split =====
    # IMPORTANT: Megatron uses interleaved QKV layout: [Q0,K0,V0, Q1,K1,V1, ...]
    # Each "group" contains (q_per_group + 2) heads: Q heads for this group + K + V
    seq_local, batch, _ = mixed_qkv.shape

    num_groups = num_kv_heads  # Number of KV groups (= num_heads for MHA)
    q_per_group = num_heads // num_kv_heads  # Q heads per group (= 1 for MHA)
    group_size = (q_per_group + 2) * head_dim  # Size per group in output dimension

    # View as groups: [seq_local, B, num_groups, group_size]
    mixed_qkv_grouped = mixed_qkv.view(seq_local, batch, num_groups, group_size)

    # AllToAll sp2hp on groups: [seq_local, B, num_groups, group_size] -> [seq_full, B, num_groups/CP, group_size]
    # sp2hp splits the "groups" dimension and gathers sequence
    mixed_qkv_sp = fluid_all_to_all_sp2hp(mixed_qkv_grouped, group)

    # Now split Q, K, V from each group
    # mixed_qkv_sp: [seq_full, B, groups_local, group_size]
    groups_local = num_groups // cp
    q_size_per_group = q_per_group * head_dim

    # Split each group into [Q, K, V]
    query, key, value = torch.split(
        mixed_qkv_sp, [q_size_per_group, head_dim, head_dim], dim=3
    )
    # query: [seq_full, B, groups_local, q_per_group * head_dim]
    # key/value: [seq_full, B, groups_local, head_dim]

    # Reshape query to have proper head dimension
    heads_local = num_heads // cp
    query = query.reshape(query.size(0), query.size(1), heads_local, head_dim)

    # key and value already have shape [seq_full, B, kv_heads_local, head_dim]
    # but need to be contiguous
    key = key.contiguous()
    value = value.contiguous()

    return query, key, value


def fluid_all_to_all_qkv_hp2sp_batched(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    group=None
) -> tuple:
    """
    Batched QKV hp2sp AllToAll (merge 3 separate AllToAll into 1)

    Merges Q, K, V into a single AllToAll communication to reduce
    communication overhead and create longer communication window for dW overlap.

    Shape:
        Input:  q, k, v: [seq, batch, heads/CP, dim]
        Output: q, k, v: [seq/CP, batch, heads, dim]
    """
    from megatron.core.utils import get_tensor_model_parallel_group_if_none

    group = get_tensor_model_parallel_group_if_none(group)
    cp = group.size()

    if cp == 1:
        return query, key, value

    seq, batch, heads_local, dim = query.shape
    heads = heads_local * cp

    # Concatenate Q, K, V along head dimension: [seq, B, heads_local, D] x 3 -> [seq, B, 3*heads_local, D]
    qkv_batched = torch.cat([query, key, value], dim=2)

    # Single hp2sp AllToAll for batched QKV
    qkv_hp = fluid_all_to_all_hp2sp(qkv_batched, group)

    # Split back to Q, K, V: [seq_local, B, 3*heads, D] -> 3 x [seq_local, B, heads, D]
    # IMPORTANT: Make contiguous to avoid "view not compatible" errors in later operations
    q_hp = qkv_hp[:, :, :heads, :].contiguous()
    k_hp = qkv_hp[:, :, heads:2*heads, :].contiguous()
    v_hp = qkv_hp[:, :, 2*heads:, :].contiguous()

    return q_hp, k_hp, v_hp


# ============================================================
# Pipelined Q/K/V sp2hp AllToAll with compute overlap
# ============================================================
# Stream cache for pipelined AllToAll
_pipelined_comm_stream_cache = {}

def _get_pipelined_comm_stream(device):
    """Get or create communication stream for pipelined AllToAll"""
    if device not in _pipelined_comm_stream_cache:
        _pipelined_comm_stream_cache[device] = torch.cuda.Stream(device=device)
    return _pipelined_comm_stream_cache[device]


class _PipelinedSp2HpQKV(torch.autograd.Function):
    """
    Pipelined sp2hp AllToAll for Q, K, V with compute-communication overlap.

    Forward:
    - Pipeline Q, K, V AllToAll operations
    - Overlap K's AllToAll with Q's reshape, V's AllToAll with K's reshape

    Backward:
    - Reverse hp2sp AllToAll for gradients (using standard serial path)
    """

    @staticmethod
    def forward(ctx, query, key, value, group):
        """
        Forward: Pipelined sp2hp AllToAll for Q, K, V

        Args:
            query, key, value: [seq/CP, batch, heads, dim]
            group: Process group for AllToAll

        Returns:
            query_hp, key_hp, value_hp: [seq, batch, heads/CP, dim]
        """
        ctx.group = group
        cp = group.size()

        device = query.device
        default_stream = torch.cuda.current_stream(device)
        comm_stream = _get_pipelined_comm_stream(device)

        seq_local, batch, heads, dim = query.shape
        heads_local = heads // cp

        ctx.seq_local = seq_local
        ctx.batch = batch
        ctx.heads = heads
        ctx.heads_local = heads_local
        ctx.dim = dim
        ctx.cp = cp

        # Debug timing
        debug_timing = os.environ.get('FLUID_DEBUG_FORWARD_TIMING', '0') == '1'
        if debug_timing:
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_q_done = torch.cuda.Event(enable_timing=True)
            ev_k_done = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record(default_stream)

        # Helper to prepare input for AllToAll
        def prepare_for_a2a(x):
            """[seq/CP, B, H, D] -> [seq*CP, flat] ready for AllToAll"""
            x = x.view(seq_local, batch, cp, heads_local, dim)
            x = x.permute(2, 0, 1, 3, 4).contiguous()
            return x.view(seq_local * cp, -1)

        # Helper to reshape output from AllToAll
        def reshape_from_a2a(x):
            """[seq*CP, flat] -> [seq, B, H/CP, D]"""
            return x.view(seq_local * cp, batch, heads_local, dim)

        # Step 1: Prepare Q for AllToAll and do Q AllToAll
        q_prepared = prepare_for_a2a(query)
        q_output = torch.empty_like(q_prepared)
        dist.all_to_all_single(
            q_output, q_prepared,
            output_split_sizes=[seq_local] * cp,
            input_split_sizes=[seq_local] * cp,
            group=group,
        )

        # Step 2: Start K AllToAll on comm_stream
        k_prepared = prepare_for_a2a(key)
        k_output = torch.empty_like(k_prepared)
        comm_stream.wait_stream(default_stream)
        with torch.cuda.stream(comm_stream):
            k_handle = dist.all_to_all_single(
                k_output, k_prepared,
                output_split_sizes=[seq_local] * cp,
                input_split_sizes=[seq_local] * cp,
                group=group,
                async_op=True,
            )

        # Step 3: Reshape Q output (overlaps with K AllToAll)
        query_hp = reshape_from_a2a(q_output)

        if debug_timing:
            ev_q_done.record(default_stream)

        # Step 4: Wait for K AllToAll, start V AllToAll on comm_stream
        k_handle.wait()
        default_stream.wait_stream(comm_stream)

        v_prepared = prepare_for_a2a(value)
        v_output = torch.empty_like(v_prepared)
        comm_stream.wait_stream(default_stream)
        with torch.cuda.stream(comm_stream):
            v_handle = dist.all_to_all_single(
                v_output, v_prepared,
                output_split_sizes=[seq_local] * cp,
                input_split_sizes=[seq_local] * cp,
                group=group,
                async_op=True,
            )

        # Step 5: Reshape K output (overlaps with V AllToAll)
        key_hp = reshape_from_a2a(k_output)

        if debug_timing:
            ev_k_done.record(default_stream)

        # Step 6: Wait for V AllToAll and reshape
        v_handle.wait()
        default_stream.wait_stream(comm_stream)
        value_hp = reshape_from_a2a(v_output)

        if debug_timing:
            ev_end.record(default_stream)
            torch.cuda.synchronize()
            rank = group.rank()
            if rank == 0:
                t_q = ev_start.elapsed_time(ev_q_done)
                t_k = ev_q_done.elapsed_time(ev_k_done)
                t_v = ev_k_done.elapsed_time(ev_end)
                t_total = ev_start.elapsed_time(ev_end)
                print(f"[Pipelined sp2hp] Q: {t_q:.2f}ms, K: {t_k:.2f}ms, V: {t_v:.2f}ms, Total: {t_total:.2f}ms")

        return query_hp, key_hp, value_hp

    @staticmethod
    def backward(ctx, grad_query_hp, grad_key_hp, grad_value_hp):
        """
        Backward: hp2sp AllToAll for gradients (reverse of sp2hp)

        Input grads: [seq, batch, heads/CP, dim]
        Output grads: [seq/CP, batch, heads, dim]
        """
        group = ctx.group
        seq_local = ctx.seq_local
        batch = ctx.batch
        heads = ctx.heads
        heads_local = ctx.heads_local
        dim = ctx.dim
        cp = ctx.cp
        seq = seq_local * cp

        # hp2sp AllToAll for each gradient
        # [seq, B, H/CP, D] -> [seq/CP, B, H, D]
        def hp2sp_alltoall(grad):
            """Reverse of sp2hp: hp2sp AllToAll"""
            # Flatten to 2D
            x = grad.contiguous().view(seq, batch * heads_local * dim)

            # AllToAll
            output = torch.empty_like(x)
            dist.all_to_all_single(
                output, x,
                output_split_sizes=[seq_local] * cp,
                input_split_sizes=[seq_local] * cp,
                group=group,
            )

            # Reshape: [cp, seq_local, batch, heads_local, dim] -> [seq_local, batch, heads, dim]
            output = output.view(cp, seq_local, batch, heads_local, dim)
            output = output.permute(1, 2, 0, 3, 4).contiguous()
            return output.view(seq_local, batch, heads, dim)

        grad_query = hp2sp_alltoall(grad_query_hp)
        grad_key = hp2sp_alltoall(grad_key_hp)
        grad_value = hp2sp_alltoall(grad_value_hp)

        return grad_query, grad_key, grad_value, None


def fluid_pipelined_sp2hp_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    group=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pipelined sp2hp AllToAll for Q, K, V with compute-communication overlap.

    Timeline:
    default_stream: [Q permute]─[Q A2A]─[Q reshape]─────────[K reshape]─────────[V reshape]
    comm_stream:                        └─[K A2A]──────────┘└─[V A2A]──────────┘

    Overlap:
    - K's AllToAll overlaps with Q's output reshape
    - V's AllToAll overlaps with K's output reshape

    Shape: [seq/CP, batch, heads, dim] -> [seq, batch, heads/CP, dim] for each Q, K, V

    Test results show ~9.4% improvement over serial AllToAll in forward.
    """
    from megatron.core.utils import get_tensor_model_parallel_group_if_none

    group = get_tensor_model_parallel_group_if_none(group)
    cp = group.size()

    # If CP=1, no communication needed
    if cp == 1:
        return query, key, value

    return _PipelinedSp2HpQKV.apply(query, key, value, group)


def fluid_pipelined_sp2hp_with_qk_matmul(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    group,
    softmax_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pipelined sp2hp AllToAll with TRUE compute-communication overlap.

    Key insight: V's AllToAll overlaps with Q@K^T computation!

    Timeline:
    default_stream: [Q A2A]─[K A2A]─[Q/K reshape]─[Q@K^T (BIG!)]─[V reshape]
    comm_stream:                                  └─[V A2A]──────┘

    This achieves real overlap because Q@K^T is the dominant computation
    and it doesn't need V at all!

    Args:
        query, key, value: [seq/CP, batch, heads, dim]
        group: Process group
        softmax_scale: Scale factor for attention scores

    Returns:
        attention_scores: [batch, heads/CP, seq, seq] - scaled Q@K^T result
        value_hp: [seq, batch, heads/CP, dim] - V after AllToAll
        query_hp: [seq, batch, heads/CP, dim] - Q after AllToAll (for backward)
    """
    cp = group.size()
    if cp == 1:
        # No communication needed, just return reshaped tensors
        # But we still compute Q@K^T
        seq, batch, heads, dim = query.shape
        # [seq, batch, heads, dim] -> [batch, heads, seq, dim]
        q_t = query.permute(1, 2, 0, 3).contiguous()
        k_t = key.permute(1, 2, 3, 0).contiguous()  # [batch, heads, dim, seq]
        attention_scores = torch.matmul(q_t, k_t) * softmax_scale
        return attention_scores, value, query

    device = query.device
    default_stream = torch.cuda.current_stream(device)
    comm_stream = _get_pipelined_comm_stream(device)

    seq_local, batch, heads, dim = query.shape
    heads_local = heads // cp
    seq = seq_local * cp

    debug_timing = os.environ.get('FLUID_DEBUG_FORWARD_TIMING', '0') == '1'
    if debug_timing:
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_qk_a2a_done = torch.cuda.Event(enable_timing=True)
        ev_qk_matmul_done = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record(default_stream)

    # Helper functions
    def prepare_for_a2a(x):
        """[seq/CP, B, H, D] -> [seq*CP, flat] ready for AllToAll"""
        x = x.view(seq_local, batch, cp, heads_local, dim)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        return x.view(seq * 1, -1)  # seq_local * cp

    def reshape_from_a2a(x):
        """[seq*CP, flat] -> [seq, B, H/CP, D]"""
        return x.view(seq, batch, heads_local, dim)

    # Step 1: Q AllToAll (synchronous)
    q_prepared = prepare_for_a2a(query)
    q_output = torch.empty_like(q_prepared)
    dist.all_to_all_single(
        q_output, q_prepared,
        output_split_sizes=[seq_local] * cp,
        input_split_sizes=[seq_local] * cp,
        group=group,
    )

    # Step 2: K AllToAll (synchronous)
    k_prepared = prepare_for_a2a(key)
    k_output = torch.empty_like(k_prepared)
    dist.all_to_all_single(
        k_output, k_prepared,
        output_split_sizes=[seq_local] * cp,
        input_split_sizes=[seq_local] * cp,
        group=group,
    )

    if debug_timing:
        ev_qk_a2a_done.record(default_stream)

    # Step 3: Start V AllToAll on comm_stream (ASYNC!)
    v_prepared = prepare_for_a2a(value)
    v_output = torch.empty_like(v_prepared)
    comm_stream.wait_stream(default_stream)
    with torch.cuda.stream(comm_stream):
        v_handle = dist.all_to_all_single(
            v_output, v_prepared,
            output_split_sizes=[seq_local] * cp,
            input_split_sizes=[seq_local] * cp,
            group=group,
            async_op=True,
        )

    # Step 4: Reshape Q, K and compute Q@K^T (overlaps with V AllToAll!)
    # This is the KEY computation that overlaps with V's AllToAll
    query_hp = reshape_from_a2a(q_output)  # [seq, B, H/CP, D]
    key_hp = reshape_from_a2a(k_output)    # [seq, B, H/CP, D]

    # [seq, B, H/CP, D] -> [B, H/CP, seq, D]
    q_t = query_hp.permute(1, 2, 0, 3).contiguous()
    # [seq, B, H/CP, D] -> [B, H/CP, D, seq]
    k_t = key_hp.permute(1, 2, 3, 0).contiguous()

    # Q @ K^T: [B, H/CP, seq, seq] - THIS IS BIG and overlaps with V AllToAll!
    attention_scores = torch.matmul(q_t, k_t) * softmax_scale

    if debug_timing:
        ev_qk_matmul_done.record(default_stream)

    # Step 5: Wait for V AllToAll and reshape
    v_handle.wait()
    default_stream.wait_stream(comm_stream)
    value_hp = reshape_from_a2a(v_output)  # [seq, B, H/CP, D]

    if debug_timing:
        ev_end.record(default_stream)
        torch.cuda.synchronize()
        rank = group.rank()
        if rank == 0:
            t_qk_a2a = ev_start.elapsed_time(ev_qk_a2a_done)
            t_qk_matmul = ev_qk_a2a_done.elapsed_time(ev_qk_matmul_done)
            t_v_wait = ev_qk_matmul_done.elapsed_time(ev_end)
            t_total = ev_start.elapsed_time(ev_end)
            print(f"[Pipelined sp2hp+QK] Q+K A2A: {t_qk_a2a:.2f}ms, "
                  f"Q@K^T (overlap V): {t_qk_matmul:.2f}ms, "
                  f"V wait: {t_v_wait:.2f}ms, Total: {t_total:.2f}ms")

    return attention_scores, value_hp, query_hp


def _all_to_all_sp2hp_forward(input_: torch.Tensor, group) -> torch.Tensor:
    """
    Forward-only sp2hp AllToAll (no autograd).
    Used in backward passes to reverse hp2sp.

    Shape: [seq/CP, batch, heads, dim] -> [seq, batch, heads/CP, dim]
    """
    cp = group.size()
    seq_local, batch, heads, dim = input_.shape

    # Rearrange: split heads, move CP to front, flatten (ensure contiguous)
    x = input_.contiguous().view(seq_local, batch, cp, heads // cp, dim)
    x = x.permute(2, 0, 1, 3, 4).contiguous()
    x = x.view(seq_local * cp, -1)

    # AllToAll communication (no grad)
    output = torch.empty_like(x)
    dist.all_to_all_single(
        output, x,
        output_split_sizes=[seq_local] * cp,
        input_split_sizes=[seq_local] * cp,
        group=group,
    )

    # Reshape to output format
    return output.view(seq_local * cp, batch, heads // cp, dim)


def _all_to_all_hp2sp_forward(input_: torch.Tensor, group) -> torch.Tensor:
    """
    Forward-only hp2sp AllToAll (no autograd).
    Used in backward passes to reverse sp2hp.

    Shape: [seq, batch, heads/CP, dim] -> [seq/CP, batch, heads, dim]
    """
    cp = group.size()
    seq, batch, heads_local, dim = input_.shape
    seq_local = seq // cp

    # Flatten to 2D (ensure contiguous for view)
    x = input_.contiguous().view(seq, batch * heads_local * dim)

    # AllToAll communication (no grad)
    output = torch.empty_like(x)
    dist.all_to_all_single(
        output, x,
        output_split_sizes=[seq_local] * cp,
        input_split_sizes=[seq_local] * cp,
        group=group,
    )

    # Rearrange: unflatten, permute, merge heads
    output = output.view(cp, seq_local, batch, heads_local, dim)
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    return output.view(seq_local, batch, heads_local * cp, dim)


def _to_list(splits):
    """Convert tensor splits to list"""
    return splits.tolist() if isinstance(splits, torch.Tensor) else splits


def fluid_all_to_all_moe_dispatch(
    input: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """
    MoE Token Dispatch via AllToAll

    Shape: [num_local_tokens, hidden] -> [num_global_tokens, hidden]

    Note: Uses "moe_dispatch" comm_type which may trigger chunked backward
    when FLUID_DX_NUM_CHUNKS > 1
    """
    return fluid_all_to_all(
        input, group,
        output_split_sizes=_to_list(output_splits),
        input_split_sizes=_to_list(input_splits),
        comm_type="moe_dispatch",
    )


def fluid_all_to_all_moe_dispatch_probs(
    input: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """
    MoE Probs Dispatch via AllToAll (separate from tokens)

    Shape: [num_local_tokens, 1] -> [num_global_tokens, 1]

    Note: Uses "moe_dispatch_probs" comm_type which will NOT be chunked.
    Probs backward is handled separately from tokens.
    """
    return fluid_all_to_all(
        input, group,
        output_split_sizes=_to_list(output_splits),
        input_split_sizes=_to_list(input_splits),
        comm_type="moe_dispatch_probs",
    )


def fluid_all_to_all_moe_combine(
    input: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """
    MoE Token Combine via AllToAll (reverse of dispatch)

    Shape: [num_global_tokens, hidden] -> [num_local_tokens, hidden]
    """
    return fluid_all_to_all(
        input, group,
        output_split_sizes=_to_list(output_splits),
        input_split_sizes=_to_list(input_splits),
        comm_type="moe_combine",
    )


# ============================================================
# Chunked dX + AllToAll Pipeline
# ============================================================

def get_dx_num_chunks() -> int:
    """Get the configured number of dX chunks."""
    return DX_NUM_CHUNKS


# ============================================================
# Fused dX + AllToAll for Ulysses (True Pipeline)
# ============================================================
# These functions fuse the dX computation (reshape/permute) with AllToAll
# to enable true dX + AllToAll pipelining in backward:
#
# Timeline:
# |-- dX_chunk0 --|-- dX_chunk1 --|-- dX_chunk2 --|-- dX_chunk3 --|
#                |--- A2A_0 ----|
#                               |--- A2A_1 ----|
#                                              |--- A2A_2 ----|
#                                                             |--- A2A_3 ----| <- dW here
# ============================================================


class _FluidFusedSp2HpAllToAll(torch.autograd.Function):
    """
    Fused Sequence Parallel -> Head Parallel AllToAll with chunked dX + AllToAll pipeline

    Forward: Same as fluid_all_to_all_sp2hp
    Backward: True dX + AllToAll pipeline
        - dX is the reverse reshape/permute operations
        - Chunked: each chunk's dX computed, then AllToAll starts immediately
    """

    @staticmethod
    def forward(ctx, input_, group, seq_local, batch, heads, dim, cp):
        """
        Forward: [seq/CP, batch, heads, dim] -> [seq, batch, heads/CP, dim]
        """
        ctx.group = group
        ctx.seq_local = seq_local
        ctx.batch = batch
        ctx.heads = heads
        ctx.dim = dim
        ctx.cp = cp

        # Rearrange: split heads, move CP to front, flatten
        x = input_.view(seq_local, batch, cp, heads // cp, dim)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(seq_local * cp, -1)

        # AllToAll communication
        output = _all_to_all(x, [seq_local] * cp, [seq_local] * cp, group)

        # Reshape to output format
        return output.view(seq_local * cp, batch, heads // cp, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward with chunked dX + AllToAll pipeline

        dX computation = reshape + permute (reverse of forward's reshape)
        """
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        group = ctx.group
        seq_local = ctx.seq_local
        batch = ctx.batch
        heads = ctx.heads
        dim = ctx.dim
        cp = ctx.cp
        num_chunks = DX_NUM_CHUNKS

        # grad_output: [seq, batch, heads/CP, dim]
        seq = seq_local * cp
        heads_local = heads // cp

        if num_chunks <= 1:
            # Original non-chunked path
            # Flatten
            grad = grad_output.view(seq, batch * heads_local * dim)

            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                result = _all_to_all(grad, [seq_local] * cp, [seq_local] * cp, group)
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)
                scheduler.set_alltoall_end_event(event)

            scheduler.on_alltoall_start(comm_type="ulysses")
            scheduler.default_stream.wait_stream(scheduler.comm_stream)

            # Reverse reshape: [cp, seq_local, ...] -> [seq_local, batch, heads, dim]
            result = result.view(cp, seq_local, batch, heads_local, dim)
            result = result.permute(1, 2, 0, 3, 4).contiguous()
            grad_input = result.view(seq_local, batch, heads, dim)

            return (grad_input, None, None, None, None, None, None)

        # ============================================================
        # Chunked dX + AllToAll Pipeline
        # Split along sequence dimension
        # ============================================================
        chunk_size = seq // num_chunks
        remainder = seq % num_chunks

        alltoall_results = []

        for chunk_idx in range(num_chunks):
            # Calculate chunk boundaries
            chunk_start = chunk_idx * chunk_size + min(chunk_idx, remainder)
            if chunk_idx < remainder:
                this_chunk_size = chunk_size + 1
            else:
                this_chunk_size = chunk_size
            chunk_end = chunk_start + this_chunk_size

            # ============================================================
            # Step 1: Compute dX for this chunk (flatten)
            # ============================================================
            grad_chunk = grad_output[chunk_start:chunk_end]  # [chunk_size, batch, heads_local, dim]
            grad_chunk_flat = grad_chunk.reshape(this_chunk_size, batch * heads_local * dim)

            # Compute per-rank split for this chunk
            chunk_splits = []
            for rank in range(cp):
                rank_chunk_size = this_chunk_size // cp
                if rank < this_chunk_size % cp:
                    rank_chunk_size += 1
                chunk_splits.append(rank_chunk_size)

            # ============================================================
            # Step 2: Launch AllToAll for this chunk
            # ============================================================
            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                chunk_result = _all_to_all(
                    grad_chunk_flat.contiguous(),
                    chunk_splits,
                    chunk_splits,
                    group,
                )
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)

            alltoall_results.append((chunk_result, this_chunk_size, chunk_splits))

            # For the last chunk, set up dW overlap
            if chunk_idx == num_chunks - 1:
                scheduler.set_alltoall_end_event(event)
                scheduler.on_alltoall_start(comm_type="ulysses")

        # ============================================================
        # Step 3: Wait and reassemble results
        # ============================================================
        scheduler.default_stream.wait_stream(scheduler.comm_stream)

        # Reassemble and apply reverse permute
        result_parts = []
        for chunk_result, this_chunk_size, chunk_splits in alltoall_results:
            # Reshape chunk result
            chunk_seq_local = this_chunk_size // cp
            if this_chunk_size % cp != 0:
                # Handle non-divisible case
                chunk_reshaped = chunk_result.view(-1, batch, heads_local, dim)
            else:
                chunk_reshaped = chunk_result.view(cp, chunk_seq_local, batch, heads_local, dim)
                chunk_reshaped = chunk_reshaped.permute(1, 2, 0, 3, 4).contiguous()
                chunk_reshaped = chunk_reshaped.view(chunk_seq_local, batch, heads, dim)
            result_parts.append(chunk_reshaped)

        grad_input = torch.cat(result_parts, dim=0)

        return (grad_input, None, None, None, None, None, None)


class _FluidFusedHp2SpAllToAll(torch.autograd.Function):
    """
    Fused Head Parallel -> Sequence Parallel AllToAll with chunked dX + AllToAll pipeline

    Forward: Same as fluid_all_to_all_hp2sp
    Backward: True dX + AllToAll pipeline
    """

    @staticmethod
    def forward(ctx, input_, group, seq, batch, heads_local, dim, cp):
        """
        Forward: [seq, batch, heads/CP, dim] -> [seq/CP, batch, heads, dim]
        """
        ctx.group = group
        ctx.seq = seq
        ctx.batch = batch
        ctx.heads_local = heads_local
        ctx.dim = dim
        ctx.cp = cp

        seq_local = seq // cp

        # Flatten to 2D
        x = input_.view(seq, batch * heads_local * dim)

        # AllToAll communication
        output = _all_to_all(x, [seq_local] * cp, [seq_local] * cp, group)

        # Rearrange: unflatten, permute, merge heads
        output = output.view(cp, seq_local, batch, heads_local, dim)
        output = output.permute(1, 2, 0, 3, 4).contiguous()
        return output.view(seq_local, batch, heads_local * cp, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward with chunked dX + AllToAll pipeline

        dX computation = reverse of forward's reshape/permute
        """
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        group = ctx.group
        seq = ctx.seq
        batch = ctx.batch
        heads_local = ctx.heads_local
        dim = ctx.dim
        cp = ctx.cp
        num_chunks = DX_NUM_CHUNKS

        # grad_output: [seq/CP, batch, heads, dim]
        seq_local = seq // cp
        heads = heads_local * cp

        if num_chunks <= 1:
            # Original non-chunked path
            # Reverse reshape: [seq_local, batch, heads, dim] -> [cp, seq_local, batch, heads_local, dim]
            grad = grad_output.view(seq_local, batch, cp, heads_local, dim)
            grad = grad.permute(2, 0, 1, 3, 4).contiguous()
            grad = grad.view(seq, -1)  # [seq, batch * heads_local * dim]

            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                result = _all_to_all(grad, [seq_local] * cp, [seq_local] * cp, group)
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)
                scheduler.set_alltoall_end_event(event)

            scheduler.on_alltoall_start(comm_type="ulysses")
            scheduler.default_stream.wait_stream(scheduler.comm_stream)

            # Reshape to output format
            grad_input = result.view(seq, batch, heads_local, dim)

            return (grad_input, None, None, None, None, None, None)

        # ============================================================
        # Chunked dX + AllToAll Pipeline
        # Split along seq_local dimension
        # ============================================================
        chunk_size = seq_local // num_chunks
        remainder = seq_local % num_chunks

        alltoall_results = []

        for chunk_idx in range(num_chunks):
            # Calculate chunk boundaries
            chunk_start = chunk_idx * chunk_size + min(chunk_idx, remainder)
            if chunk_idx < remainder:
                this_chunk_size = chunk_size + 1
            else:
                this_chunk_size = chunk_size
            chunk_end = chunk_start + this_chunk_size

            # ============================================================
            # Step 1: Compute dX for this chunk (reverse reshape/permute)
            # ============================================================
            grad_chunk = grad_output[chunk_start:chunk_end]  # [chunk_size, batch, heads, dim]

            # Reverse reshape: [chunk_size, batch, heads, dim] -> [cp, chunk_size, batch, heads_local, dim]
            grad_chunk = grad_chunk.view(this_chunk_size, batch, cp, heads_local, dim)
            grad_chunk = grad_chunk.permute(2, 0, 1, 3, 4).contiguous()
            grad_chunk_flat = grad_chunk.view(cp * this_chunk_size, batch * heads_local * dim)

            # Compute per-rank split for this chunk
            chunk_splits = [this_chunk_size] * cp

            # ============================================================
            # Step 2: Launch AllToAll for this chunk
            # ============================================================
            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                chunk_result = _all_to_all(
                    grad_chunk_flat.contiguous(),
                    chunk_splits,
                    chunk_splits,
                    group,
                )
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)

            alltoall_results.append((chunk_result, this_chunk_size))

            # For the last chunk, set up dW overlap
            if chunk_idx == num_chunks - 1:
                scheduler.set_alltoall_end_event(event)
                scheduler.on_alltoall_start(comm_type="ulysses")

        # ============================================================
        # Step 3: Wait and reassemble results
        # ============================================================
        scheduler.default_stream.wait_stream(scheduler.comm_stream)

        # Reassemble
        result_parts = []
        for chunk_result, this_chunk_size in alltoall_results:
            # Reshape chunk result: [cp * chunk_size, batch * heads_local * dim] -> [chunk_size * cp, batch, heads_local, dim]
            chunk_reshaped = chunk_result.view(cp * this_chunk_size, batch, heads_local, dim)
            result_parts.append(chunk_reshaped)

        grad_input = torch.cat(result_parts, dim=0)

        return (grad_input, None, None, None, None, None, None)


def fluid_fused_all_to_all_sp2hp(input_: torch.Tensor, group=None) -> torch.Tensor:
    """
    Fused Ulysses SP: Sequence Parallel -> Head Parallel with true dX + AllToAll pipeline

    Shape: [seq/CP, batch, heads, dim] -> [seq, batch, heads/CP, dim]

    This version fuses the dX computation with AllToAll in backward,
    enabling true dX + AllToAll pipelining.
    """
    from megatron.core.utils import get_tensor_model_parallel_group_if_none

    group = get_tensor_model_parallel_group_if_none(group)
    cp = group.size()
    seq_local, batch, heads, dim = input_.shape

    return _FluidFusedSp2HpAllToAll.apply(input_, group, seq_local, batch, heads, dim, cp)


class _FluidFusedCombineUnpermute(torch.autograd.Function):
    """
    Fused MoE Combine AllToAll + Unpermute with chunked dX + AllToAll pipeline

    Forward:
        1. AllToAll combine
        2. Unpermute to restore original token order

    Backward: True dX + AllToAll pipeline
        - dX is computed from permute backward (index_select)
        - Chunked: each chunk's dX computed, then AllToAll starts immediately
    """

    @staticmethod
    def forward(ctx, hidden_states, output_splits, input_splits, group, permutation_map, restore_shape,
                probs, routing_map):
        """
        Forward: AllToAll combine + unpermute (with optional probs multiplication)
        """
        ctx.group = group
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.permutation_map = permutation_map
        ctx.restore_shape = restore_shape
        ctx.probs = probs
        ctx.routing_map = routing_map

        output_splits_list = output_splits.tolist() if hasattr(output_splits, 'tolist') else list(output_splits)
        input_splits_list = input_splits.tolist() if hasattr(input_splits, 'tolist') else list(input_splits)

        # AllToAll combine
        combined = _all_to_all(hidden_states, output_splits_list, input_splits_list, group)

        # Apply probs before unpermute (Megatron standard behavior)
        permuted_probs = None
        if probs is not None and routing_map is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
            combined = combined * permuted_probs.unsqueeze(-1)
            ctx.permuted_probs = permuted_probs
        else:
            ctx.permuted_probs = None

        # Unpermute: scatter_add to restore original token order
        _, hidden = restore_shape
        output = torch.zeros(restore_shape, dtype=combined.dtype, device=combined.device)
        output.scatter_add_(0, permutation_map.unsqueeze(1).expand(-1, hidden), combined)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward with chunked dX + AllToAll pipeline

        dX computation = permute backward (index_select)

        Key insight for MoE combine backward:
        - The permutation_map reorders tokens by expert
        - After permute backward, data is ordered as: [tokens for rank0][tokens for rank1]...
        - We can chunk this data by rank, enabling true dX + AllToAll pipeline

        Timeline:
        |-- permute_backward (full) --|-- chunk0 A2A --|-- chunk1 A2A --| <- dW here
        """
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        group = ctx.group
        output_splits = ctx.output_splits
        input_splits = ctx.input_splits
        permutation_map = ctx.permutation_map
        permuted_probs = ctx.permuted_probs
        num_chunks = DX_NUM_CHUNKS

        output_splits_list = output_splits.tolist() if hasattr(output_splits, 'tolist') else list(output_splits)
        input_splits_list = input_splits.tolist() if hasattr(input_splits, 'tolist') else list(input_splits)
        ep_size = len(input_splits_list)

        # Backward AllToAll splits (swap input/output)
        backward_output_splits = input_splits_list  # Where we send TO
        backward_input_splits = output_splits_list  # Where we receive FROM

        if num_chunks <= 1:
            # Original non-chunked path
            # Permute backward: index_select
            grad_combined = grad_output.index_select(0, permutation_map)

            # Apply probs in backward (same as forward: multiply by permuted_probs)
            if permuted_probs is not None:
                grad_combined = grad_combined * permuted_probs.unsqueeze(-1)

            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                result = _all_to_all(grad_combined, backward_output_splits, backward_input_splits, group)
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)
                scheduler.set_alltoall_end_event(event)

            scheduler.on_alltoall_start(comm_type="moe_combine")
            dw_executed = scheduler._execute_all_dw_tasks_sync()
            scheduler.default_stream.wait_stream(scheduler.comm_stream)

            return (result, None, None, None, None, None, None, None)

        # ============================================================
        # Hidden Dimension Chunked dX + AllToAll Pipeline (FlowMoE-style)
        #
        # Key insight: Instead of chunking tokens (which is hard due to
        # permutation_map ordering), we chunk along the hidden dimension.
        #
        # Each chunk:
        #   1. Computes index_select for partial hidden dims
        #   2. Records event when data is ready
        #   3. Launches AllToAll that waits only for that event
        #
        # This enables TRUE dX-AllToAll pipelining:
        #   default: [idx0] ev0 [idx1] ev1 [idx2] ev2 ...
        #                   ↓        ↓        ↓
        #   comm:     wait ev0 [A2A0] wait ev1 [A2A1] ...
        #
        # idx1 overlaps with A2A0, idx2 overlaps with A2A1, etc.
        # All token splits remain the same; only hidden dim is chunked.
        # ============================================================

        hidden_size = grad_output.shape[-1]
        chunk_size = hidden_size // num_chunks
        remainder = hidden_size % num_chunks

        alltoall_results = []
        chunk_ready_events = []  # Events to signal when each chunk's data is ready

        for chunk_idx in range(num_chunks):
            # Calculate hidden dimension boundaries for this chunk
            h_start = chunk_idx * chunk_size + min(chunk_idx, remainder)
            if chunk_idx < remainder:
                this_chunk_hidden = chunk_size + 1
            else:
                this_chunk_hidden = chunk_size
            h_end = h_start + this_chunk_hidden

            # ============================================================
            # Step 1: Compute dX for this hidden chunk (on default_stream)
            # index_select on partial hidden dimension
            # ============================================================
            grad_chunk = grad_output[:, h_start:h_end].index_select(0, permutation_map)
            # shape: [num_permuted_tokens, this_chunk_hidden]

            # Apply probs in backward (same as forward)
            if permuted_probs is not None:
                grad_chunk = grad_chunk * permuted_probs.unsqueeze(-1)

            # Make contiguous while still on default_stream
            grad_chunk = grad_chunk.contiguous()

            # Record event: this chunk's data is now ready
            chunk_ready_event = torch.cuda.Event()
            chunk_ready_event.record(scheduler.default_stream)
            chunk_ready_events.append(chunk_ready_event)

            # ============================================================
            # Step 2: Launch AllToAll for this chunk on comm_stream
            # Only wait for THIS chunk's event, not all previous compute
            # ============================================================
            with torch.cuda.stream(scheduler.comm_stream):
                # Wait only for this chunk's data, allowing overlap with next chunk's compute
                chunk_ready_event.wait(scheduler.comm_stream)
                chunk_result = _all_to_all(
                    grad_chunk,
                    backward_output_splits,  # Same token splits
                    backward_input_splits,   # Same token splits
                    group,
                )
                comm_done_event = torch.cuda.Event()
                comm_done_event.record(scheduler.comm_stream)

            alltoall_results.append(chunk_result)

            # For the last chunk, set up dW overlap
            if chunk_idx == num_chunks - 1:
                scheduler.set_alltoall_end_event(comm_done_event)
                scheduler.on_alltoall_start(comm_type="moe_combine")

        # ============================================================
        # Execute dW tasks while AllToAll is running on comm_stream
        # This is the key dW-AllToAll overlap for chunked path
        # ============================================================
        dw_executed = scheduler._execute_all_dw_tasks_sync()

        # ============================================================
        # Wait and reassemble results
        # Simply concatenate along hidden dimension
        # ============================================================
        scheduler.default_stream.wait_stream(scheduler.comm_stream)

        # Concatenate chunks along hidden dimension
        result = torch.cat(alltoall_results, dim=-1)

        return (result, None, None, None, None, None, None, None)


def fluid_fused_combine_unpermute(
    hidden_states: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    permutation_map: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fused MoE Combine + Unpermute with true dX + AllToAll pipeline

    Shape: [num_global_tokens, hidden] -> [num_tokens, hidden]

    This version fuses AllToAll combine with unpermute in backward,
    enabling true dX + AllToAll pipelining.

    Args:
        probs: [num_tokens, topk] - routing probabilities (optional)
        routing_map: [num_tokens, num_experts] - token to expert mapping (optional)
    """
    return _FluidFusedCombineUnpermute.apply(
        hidden_states, output_splits, input_splits, group, permutation_map, restore_shape,
        probs, routing_map
    )


def fluid_fused_all_to_all_hp2sp(input_: torch.Tensor, group=None) -> torch.Tensor:
    """
    Fused Ulysses SP: Head Parallel -> Sequence Parallel with true dX + AllToAll pipeline

    Shape: [seq, batch, heads/CP, dim] -> [seq/CP, batch, heads, dim]

    This version fuses the dX computation with AllToAll in backward,
    enabling true dX + AllToAll pipelining.
    """
    from megatron.core.utils import get_tensor_model_parallel_group_if_none

    group = get_tensor_model_parallel_group_if_none(group)
    cp = group.size()
    seq, batch, heads_local, dim = input_.shape

    return _FluidFusedHp2SpAllToAll.apply(input_, group, seq, batch, heads_local, dim, cp)


# ============================================================
# Fused hp2sp AllToAll + Linear Projection (Output Projection)
# ============================================================

class _FusedHp2SpLinearProj(torch.autograd.Function):
    """
    Fused: hp2sp AllToAll + Linear Projection with chunked backward

    Forward:
        1. hp2sp AllToAll: [seq, batch, heads/CP, dim] -> [seq/CP, batch, heads, dim]
        2. reshape to [seq/CP, batch, hidden]
        3. Linear: output = input @ weight.T + bias

    Backward (chunked dX GEMM + AllToAll pipeline):
        For each chunk along seq/CP dimension:
            1. Linear backward GEMM for this chunk: dX_chunk = grad_chunk @ weight
            2. reshape for AllToAll
            3. Launch sp2hp AllToAll (hp2sp backward) for this chunk on comm_stream

    Timeline:
    |-- Linear GEMM chunk0 --|-- A2A launch --|
                             |-- Linear GEMM chunk1 --|-- A2A launch --|
                                                                       |-- wait & dW --|
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, cp_group, layer_name, layer_id,
                seq, batch, heads_local, dim, cp):
        """
        Forward: hp2sp AllToAll + Linear

        Args:
            input_: [seq, batch, heads/CP, dim] - from core attention
            weight: [hidden, hidden_per_partition] - output projection weight
            bias: [hidden] or None - output projection bias
        """
        ctx.cp_group = cp_group
        ctx.layer_name = layer_name
        ctx.layer_id = layer_id
        ctx.seq = seq
        ctx.batch = batch
        ctx.heads_local = heads_local  # heads/CP
        ctx.dim = dim
        ctx.cp = cp

        seq_local = seq // cp
        heads = heads_local * cp
        hidden = heads * dim
        ctx.seq_local = seq_local
        ctx.heads = heads
        ctx.hidden = hidden

        # Step 1: hp2sp AllToAll
        # [seq, batch, heads/CP, dim] -> flatten -> AllToAll -> [seq/CP, batch, heads, dim]
        x = input_.view(seq, batch * heads_local * dim)
        result = _all_to_all(x, [seq_local] * cp, [seq_local] * cp, cp_group)

        # Reshape: [seq/CP, batch * heads * dim] -> [seq/CP, batch, hidden]
        linear_input = result.view(seq_local, batch, hidden)

        # Save for backward (before Linear)
        ctx.save_for_backward(linear_input, weight)
        ctx.use_bias = bias is not None

        # Step 2: Linear projection
        # Note: bias is NOT added here (skip_bias_add=True pattern)
        # Bias is returned separately and added in the residual connection
        output = torch.matmul(linear_input, weight.t())
        ctx.bias = bias  # Save for dBias computation in backward

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward with chunked Linear GEMM + sp2hp AllToAll (hp2sp backward) pipeline

        grad_output: [seq/CP, batch, hidden]

        Key insight: Linear backward GEMM is compute-intensive, AllToAll is communication.
        We can chunk the GEMM and pipeline it with AllToAll.
        """
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        linear_input, weight = ctx.saved_tensors
        bias = ctx.bias
        cp_group = ctx.cp_group
        seq_local = ctx.seq_local
        batch = ctx.batch
        heads_local = ctx.heads_local
        dim = ctx.dim
        cp = ctx.cp
        seq = ctx.seq
        heads = ctx.heads
        hidden = ctx.hidden
        num_chunks = DX_NUM_CHUNKS

        if num_chunks <= 1:
            # ============================================================
            # Non-chunked path
            # ============================================================
            # Step 1: Linear backward GEMM
            grad_linear_input = torch.matmul(grad_output, weight)

            # Step 2: Reshape for AllToAll
            # [seq/CP, batch, hidden] -> [seq/CP, batch, heads, dim] -> flatten
            grad_reshaped = grad_linear_input.view(seq_local, batch, heads, dim)
            grad_flat = grad_reshaped.view(seq_local, batch * heads * dim)

            # Step 3: sp2hp AllToAll (hp2sp backward)
            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                result = _all_to_all(grad_flat, [seq_local] * cp, [seq_local] * cp, cp_group)
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)
                scheduler.set_alltoall_end_event(event)

            scheduler.on_alltoall_start(comm_type="attn_hp2sp_bwd")
            scheduler.default_stream.wait_stream(scheduler.comm_stream)

            # Reshape: [seq, batch * heads/CP * dim] -> [seq, batch, heads/CP, dim]
            grad_input = result.view(seq, batch, heads_local, dim)

            # Register dW
            grad_output_saved = grad_output.detach()
            linear_input_saved = linear_input.detach()

            def compute_dw():
                input_2d = linear_input_saved.view(-1, linear_input_saved.shape[-1])
                grad_2d = grad_output_saved.view(-1, grad_output_saved.shape[-1])
                return torch.matmul(grad_2d.t(), input_2d)

            def compute_dbias():
                if ctx.use_bias:
                    return grad_output_saved.sum(dim=[0, 1])
                return None

            scheduler.register_dw_task(
                layer_name=f"{ctx.layer_name}_weight",
                layer_id=ctx.layer_id,
                compute_fn=compute_dw,
                priority=100,
                weight_param=weight,
            )

            if ctx.use_bias:
                scheduler.register_dw_task(
                    layer_name=f"{ctx.layer_name}_bias",
                    layer_id=ctx.layer_id,
                    compute_fn=compute_dbias,
                    priority=99,
                    weight_param=bias,
                )

            return (grad_input, None, None, None, None, None, None, None, None, None, None)

        # ============================================================
        # Chunked Linear GEMM + sp2hp AllToAll Pipeline
        # ============================================================
        chunk_size = seq_local // num_chunks
        remainder = seq_local % num_chunks

        alltoall_results = []

        for chunk_idx in range(num_chunks):
            # Calculate chunk boundaries
            chunk_start = chunk_idx * chunk_size + min(chunk_idx, remainder)
            if chunk_idx < remainder:
                this_chunk_size = chunk_size + 1
            else:
                this_chunk_size = chunk_size
            chunk_end = chunk_start + this_chunk_size

            # ============================================================
            # Step 1: Linear backward GEMM for this chunk
            # ============================================================
            grad_chunk = grad_output[chunk_start:chunk_end]  # [chunk_size, batch, hidden]
            grad_linear_input_chunk = torch.matmul(grad_chunk, weight)

            # ============================================================
            # Step 2: Reshape for sp2hp AllToAll (hp2sp backward)
            # Same rearrangement as fluid_all_to_all_sp2hp forward
            # ============================================================
            # [chunk_size, batch, hidden] -> [chunk_size, batch, heads, dim]
            grad_reshaped = grad_linear_input_chunk.view(this_chunk_size, batch, heads, dim)
            # Split heads by cp: [chunk_size, batch, cp, heads_local, dim]
            grad_reshaped = grad_reshaped.view(this_chunk_size, batch, cp, heads_local, dim)
            # Permute: [cp, chunk_size, batch, heads_local, dim]
            grad_reshaped = grad_reshaped.permute(2, 0, 1, 3, 4).contiguous()
            # Flatten: [chunk_size * cp, batch * heads_local * dim]
            grad_flat = grad_reshaped.view(this_chunk_size * cp, batch * heads_local * dim)

            # Record event when GEMM + reshape chunk is done
            gemm_done = torch.cuda.Event()
            gemm_done.record()

            # ============================================================
            # Step 3: Launch sp2hp AllToAll for this chunk
            # ============================================================
            # Each rank sends this_chunk_size rows to each other rank
            chunk_splits = [this_chunk_size] * cp

            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_event(gemm_done)
                chunk_result = _all_to_all(
                    grad_flat,
                    chunk_splits,
                    chunk_splits,
                    cp_group,
                )
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)

            # For the last chunk, set up dW overlap
            if chunk_idx == num_chunks - 1:
                scheduler.set_alltoall_end_event(event)
                scheduler.on_alltoall_start(comm_type="attn_hp2sp_bwd")

            # Result shape: [this_chunk_size * cp, batch * heads_local * dim]
            alltoall_results.append((chunk_result, this_chunk_size * cp))

        # ============================================================
        # Step 4: Wait for all AllToAll and reassemble
        # ============================================================
        scheduler.default_stream.wait_stream(scheduler.comm_stream)

        # Reassemble grad_input
        # After AllToAll, each chunk has shape [chunk_size * cp, batch * heads_local * dim]
        # Need to reshape to [chunk_size * cp, batch, heads_local, dim]
        grad_input_parts = []
        for chunk_result, result_seq_len in alltoall_results:
            # Reshape: [chunk_size * cp, batch * heads_local * dim] -> [chunk_size * cp, batch, heads_local, dim]
            grad_chunk = chunk_result.view(result_seq_len, batch, heads_local, dim)
            grad_input_parts.append(grad_chunk)

        grad_input = torch.cat(grad_input_parts, dim=0)

        # Register dW
        grad_output_saved = grad_output.detach()
        linear_input_saved = linear_input.detach()

        def compute_dw():
            input_2d = linear_input_saved.view(-1, linear_input_saved.shape[-1])
            grad_2d = grad_output_saved.view(-1, grad_output_saved.shape[-1])
            return torch.matmul(grad_2d.t(), input_2d)

        def compute_dbias():
            if ctx.use_bias:
                return grad_output_saved.sum(dim=[0, 1])
            return None

        scheduler.register_dw_task(
            layer_name=f"{ctx.layer_name}_weight",
            layer_id=ctx.layer_id,
            compute_fn=compute_dw,
            priority=100,
            weight_param=weight,
        )

        if ctx.use_bias:
            scheduler.register_dw_task(
                layer_name=f"{ctx.layer_name}_bias",
                layer_id=ctx.layer_id,
                compute_fn=compute_dbias,
                priority=99,
                weight_param=bias,
            )

        return (grad_input, None, None, None, None, None, None, None, None, None, None)


def fluid_fused_hp2sp_linear_proj(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cp_group,
    layer_name: str,
    layer_id: int,
) -> torch.Tensor:
    """
    Fused hp2sp AllToAll + Linear Projection with chunked backward

    Shape: [seq, batch, heads/CP, dim] -> hp2sp -> Linear -> [seq/CP, batch, hidden]

    Forward: hp2sp AllToAll + Linear (output projection)
    Backward: Linear GEMM (chunked) + sp2hp AllToAll (pipelined)

    This enables true dX GEMM + AllToAll pipelining in attention backward.
    """
    seq, batch, heads_local, dim = input_.shape
    cp = cp_group.size()

    return _FusedHp2SpLinearProj.apply(
        input_, weight, bias, cp_group, layer_name, layer_id,
        seq, batch, heads_local, dim, cp
    )


# ============================================================
# Fused sp2hp AllToAll (x3) + Core Attention
# ============================================================

class _FusedSp2HpCoreAttention(torch.autograd.Function):
    """
    Fused: sp2hp AllToAll (x3 for Q, K, V) + Core Attention with chunked backward

    Forward:
        1. sp2hp AllToAll for Q, K, V: each [seq/CP, batch, heads, dim] -> [seq, batch, heads/CP, dim]
        2. Core Attention: (Q', K', V') -> context

    Backward (chunked dQ, dK, dV + AllToAll pipeline):
        1. Core Attention backward: grad_context -> dQ', dK', dV' (FlashAttn, atomic)
        2. For each chunk:
            - Slice dQ', dK', dV' chunk
            - Launch hp2sp AllToAll (sp2hp backward) for each on comm_stream

    Timeline:
    |-- FlashAttn backward (atomic) --|
                                      |-- dQ chunk0 A2A --|-- dQ chunk1 A2A --|
                                      |-- dK chunk0 A2A --|-- dK chunk1 A2A --|
                                      |-- dV chunk0 A2A --|-- dV chunk1 A2A --|
    """

    @staticmethod
    def forward(ctx, query, key, value, cp_group, core_attention_fn, attention_mask,
                attn_mask_type, packed_seq_params, seq_local, batch, heads, dim, cp):
        """
        Forward: sp2hp AllToAll (x3) + Core Attention

        Args:
            query, key, value: [seq/CP, batch, heads, dim]
            cp_group: context parallel group
            core_attention_fn: the core attention module's forward function
            attention_mask, attn_mask_type, packed_seq_params: attention params
        """
        ctx.cp_group = cp_group
        ctx.seq_local = seq_local
        ctx.batch = batch
        ctx.heads = heads
        ctx.dim = dim
        ctx.cp = cp

        seq = seq_local * cp
        heads_local = heads // cp
        ctx.seq = seq
        ctx.heads_local = heads_local

        # Step 1: sp2hp AllToAll for Q, K, V
        # [seq/CP, batch, heads, dim] -> [seq, batch, heads/CP, dim]
        def sp2hp_alltoall(x):
            x_flat = x.view(seq_local, batch * heads * dim)
            result = _all_to_all(x_flat, [seq_local] * cp, [seq_local] * cp, cp_group)
            return result.view(seq, batch, heads_local, dim)

        query_hp = sp2hp_alltoall(query)
        key_hp = sp2hp_alltoall(key)
        value_hp = sp2hp_alltoall(value)

        # Save for backward
        ctx.save_for_backward(query_hp, key_hp, value_hp)
        ctx.core_attention_fn = core_attention_fn
        ctx.attention_mask = attention_mask
        ctx.attn_mask_type = attn_mask_type
        ctx.packed_seq_params = packed_seq_params

        # Step 2: Core Attention
        # FlashAttention expects [seq, batch, heads/CP, dim]
        context = core_attention_fn(
            query_hp, key_hp, value_hp,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )

        return context

    @staticmethod
    def backward(ctx, grad_context):
        """
        Backward with TRUE dX-AllToAll pipelining via head chunking

        Key insight: FlashAttention backward computes each head independently.
        We can compute backward for a subset of heads, then immediately start
        AllToAll for those heads while computing the next subset.

        Timeline (num_chunks=2):
        |-- FA_bwd heads[0:h/2] --|-- FA_bwd heads[h/2:h] --|
                                  |-- A2A(dQ0,dK0,dV0) --|-- A2A(dQ1,dK1,dV1) --|
        """
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        query_hp, key_hp, value_hp = ctx.saved_tensors
        cp_group = ctx.cp_group
        seq_local = ctx.seq_local
        batch = ctx.batch
        heads = ctx.heads
        dim = ctx.dim
        cp = ctx.cp
        seq = ctx.seq
        heads_local = ctx.heads_local
        num_chunks = DX_NUM_CHUNKS

        if num_chunks <= 1:
            # ============================================================
            # Non-chunked path: Full FlashAttention backward then AllToAll
            # ============================================================
            # Compute full attention backward
            query_hp.requires_grad_(True)
            key_hp.requires_grad_(True)
            value_hp.requires_grad_(True)

            with torch.enable_grad():
                context_recompute = ctx.core_attention_fn(
                    query_hp, key_hp, value_hp,
                    attention_mask=ctx.attention_mask,
                    attn_mask_type=ctx.attn_mask_type,
                    packed_seq_params=ctx.packed_seq_params,
                )
                context_recompute.backward(grad_context)

            grad_query_hp = query_hp.grad.detach()
            grad_key_hp = key_hp.grad.detach()
            grad_value_hp = value_hp.grad.detach()
            query_hp.grad = None
            key_hp.grad = None
            value_hp.grad = None

            # Launch AllToAll for all three on comm_stream
            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                q_flat = grad_query_hp.view(seq, batch * heads_local * dim)
                grad_query = _all_to_all(q_flat, [seq_local] * cp, [seq_local] * cp, cp_group)
                k_flat = grad_key_hp.view(seq, batch * heads_local * dim)
                grad_key = _all_to_all(k_flat, [seq_local] * cp, [seq_local] * cp, cp_group)
                v_flat = grad_value_hp.view(seq, batch * heads_local * dim)
                grad_value = _all_to_all(v_flat, [seq_local] * cp, [seq_local] * cp, cp_group)

                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)
                scheduler.set_alltoall_end_event(event)

            scheduler.on_alltoall_start(comm_type="attn_sp2hp_bwd")
            scheduler.default_stream.wait_stream(scheduler.comm_stream)

            grad_query = grad_query.view(seq_local, batch, heads, dim)
            grad_key = grad_key.view(seq_local, batch, heads, dim)
            grad_value = grad_value.view(seq_local, batch, heads, dim)

            return (grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None, None, None)

        # ============================================================
        # TRUE PIPELINED: Head-chunked FlashAttention backward + AllToAll
        #
        # Key insight: Each head in attention is computed independently.
        # We compute backward for a subset of heads, then immediately
        # launch AllToAll while computing the next subset.
        #
        # Timeline:
        # default:  |-- FA_bwd[h0:h1] --|-- FA_bwd[h1:h2] --|-- ...
        #                              ↓ (event)
        # comm:                   wait |-- A2A chunk0 --|-- A2A chunk1 --|
        #
        # Edge case: If heads_local < num_chunks (e.g., 1 head per GPU),
        # we can't chunk by heads. Fall back to sequence-based chunking
        # or reduce num_chunks to match heads_local.
        # ============================================================

        # Adjust num_chunks if we don't have enough heads
        effective_chunks = min(num_chunks, heads_local)
        if effective_chunks < num_chunks:
            import os
            if os.environ.get('FLUID_DEBUG_FORWARD_TIMING', '0') == '1':
                print(f"[Attn Backward] heads_local={heads_local} < num_chunks={num_chunks}, "
                      f"reducing to {effective_chunks} chunks", flush=True)

        if effective_chunks <= 1:
            # Can't chunk by heads, fall back to non-chunked path
            # (reuse the non-chunked code above)
            query_hp.requires_grad_(True)
            key_hp.requires_grad_(True)
            value_hp.requires_grad_(True)

            with torch.enable_grad():
                context_recompute = ctx.core_attention_fn(
                    query_hp, key_hp, value_hp,
                    attention_mask=ctx.attention_mask,
                    attn_mask_type=ctx.attn_mask_type,
                    packed_seq_params=ctx.packed_seq_params,
                )
                context_recompute.backward(grad_context)

            grad_query_hp = query_hp.grad.detach()
            grad_key_hp = key_hp.grad.detach()
            grad_value_hp = value_hp.grad.detach()
            query_hp.grad = None
            key_hp.grad = None
            value_hp.grad = None

            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                q_flat = grad_query_hp.view(seq, batch * heads_local * dim)
                grad_query = _all_to_all(q_flat, [seq_local] * cp, [seq_local] * cp, cp_group)
                k_flat = grad_key_hp.view(seq, batch * heads_local * dim)
                grad_key = _all_to_all(k_flat, [seq_local] * cp, [seq_local] * cp, cp_group)
                v_flat = grad_value_hp.view(seq, batch * heads_local * dim)
                grad_value = _all_to_all(v_flat, [seq_local] * cp, [seq_local] * cp, cp_group)
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)
                scheduler.set_alltoall_end_event(event)

            scheduler.on_alltoall_start(comm_type="attn_sp2hp_bwd")
            scheduler.default_stream.wait_stream(scheduler.comm_stream)

            grad_query = grad_query.view(seq_local, batch, heads, dim)
            grad_key = grad_key.view(seq_local, batch, heads, dim)
            grad_value = grad_value.view(seq_local, batch, heads, dim)

            return (grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None, None, None)

        heads_per_chunk = heads_local // effective_chunks
        heads_remainder = heads_local % effective_chunks

        # Prepare output tensors
        grad_query_hp = torch.empty_like(query_hp)
        grad_key_hp = torch.empty_like(key_hp)
        grad_value_hp = torch.empty_like(value_hp)

        q_results, k_results, v_results = [], [], []

        for chunk_idx in range(effective_chunks):
            # Calculate head range for this chunk
            h_start = chunk_idx * heads_per_chunk + min(chunk_idx, heads_remainder)
            if chunk_idx < heads_remainder:
                this_heads = heads_per_chunk + 1
            else:
                this_heads = heads_per_chunk
            h_end = h_start + this_heads

            if this_heads == 0:
                continue

            # ============================================================
            # Step 1: FlashAttention backward for this head chunk
            # ============================================================
            # Slice Q, K, V by heads: [seq, batch, heads_local, dim] -> [seq, batch, this_heads, dim]
            q_chunk = query_hp[:, :, h_start:h_end, :].detach().requires_grad_(True)
            k_chunk = key_hp[:, :, h_start:h_end, :].detach().requires_grad_(True)
            v_chunk = value_hp[:, :, h_start:h_end, :].detach().requires_grad_(True)

            # Slice grad_context by heads
            # grad_context shape: [seq, batch, heads_local * dim] or [seq, batch, hidden_per_cp]
            grad_context_chunk = grad_context[:, :, h_start * dim : h_end * dim]

            # Run attention backward for this head chunk
            with torch.enable_grad():
                context_chunk = ctx.core_attention_fn(
                    q_chunk, k_chunk, v_chunk,
                    attention_mask=ctx.attention_mask,
                    attn_mask_type=ctx.attn_mask_type,
                    packed_seq_params=ctx.packed_seq_params,
                )
                context_chunk.backward(grad_context_chunk)

            # Get gradients
            dQ_chunk = q_chunk.grad.detach()  # [seq, batch, this_heads, dim]
            dK_chunk = k_chunk.grad.detach()
            dV_chunk = v_chunk.grad.detach()

            # Store in full grad tensors
            grad_query_hp[:, :, h_start:h_end, :] = dQ_chunk
            grad_key_hp[:, :, h_start:h_end, :] = dK_chunk
            grad_value_hp[:, :, h_start:h_end, :] = dV_chunk

            # Clear grads
            q_chunk.grad = None
            k_chunk.grad = None
            v_chunk.grad = None

            # Record event when this chunk's backward is done
            chunk_done = torch.cuda.Event()
            chunk_done.record()

            # ============================================================
            # Step 2: Launch AllToAll for this head chunk on comm_stream
            # This runs in parallel with next chunk's backward computation
            # ============================================================
            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_event(chunk_done)

                # For hp2sp AllToAll, we need to rearrange by heads
                # Each chunk has this_heads heads, need to convert to SP layout
                # The AllToAll will gather heads from all ranks
                # dQ_chunk: [seq, batch, this_heads, dim]
                # After hp2sp: [seq_local, batch, this_heads * cp, dim]

                # Flatten for AllToAll: [seq, batch * this_heads * dim]
                q_flat = dQ_chunk.reshape(seq, batch * this_heads * dim)
                k_flat = dK_chunk.reshape(seq, batch * this_heads * dim)
                v_flat = dV_chunk.reshape(seq, batch * this_heads * dim)

                # AllToAll splits
                chunk_splits = [seq_local] * cp

                q_result = _all_to_all(q_flat.contiguous(), chunk_splits, chunk_splits, cp_group)
                k_result = _all_to_all(k_flat.contiguous(), chunk_splits, chunk_splits, cp_group)
                v_result = _all_to_all(v_flat.contiguous(), chunk_splits, chunk_splits, cp_group)

                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)

            if chunk_idx == effective_chunks - 1:
                scheduler.set_alltoall_end_event(event)
                scheduler.on_alltoall_start(comm_type="attn_sp2hp_bwd")

            q_results.append((q_result, this_heads))
            k_results.append((k_result, this_heads))
            v_results.append((v_result, this_heads))

        # ============================================================
        # Step 3: Wait and reassemble results
        # ============================================================
        scheduler.default_stream.wait_stream(scheduler.comm_stream)

        # Reassemble by heads
        # Each result is [seq_local * cp, batch * this_heads * dim]
        # Need to reshape to [seq_local, batch, this_heads * cp, dim]
        def reassemble_by_heads(results):
            parts = []
            for result, this_heads in results:
                # result: [seq_local * cp, batch * this_heads * dim]
                # Reshape: [seq_local, cp, batch, this_heads, dim]
                reshaped = result.view(cp, seq_local, batch, this_heads, dim)
                # Permute to [seq_local, batch, cp, this_heads, dim]
                reshaped = reshaped.permute(1, 2, 0, 3, 4).contiguous()
                # Merge cp and this_heads: [seq_local, batch, this_heads * cp, dim]
                reshaped = reshaped.view(seq_local, batch, this_heads * cp, dim)
                parts.append(reshaped)
            # Concatenate along heads dimension
            return torch.cat(parts, dim=2)

        grad_query = reassemble_by_heads(q_results)
        grad_key = reassemble_by_heads(k_results)
        grad_value = reassemble_by_heads(v_results)

        return (grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None, None, None)


def fluid_fused_sp2hp_core_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cp_group,
    core_attention_fn,
    attention_mask=None,
    attn_mask_type=None,
    packed_seq_params=None,
) -> torch.Tensor:
    """
    Fused sp2hp AllToAll (x3) + Core Attention with chunked backward

    Shape:
        Q, K, V: [seq/CP, batch, heads, dim]
        -> sp2hp AllToAll (x3)
        -> Q', K', V': [seq, batch, heads/CP, dim]
        -> Core Attention
        -> context: [seq, batch, heads/CP * dim]

    Backward: Core Attention backward (atomic) + hp2sp AllToAll (chunked pipeline)
    """
    seq_local, batch, heads, dim = query.shape
    cp = cp_group.size()

    return _FusedSp2HpCoreAttention.apply(
        query, key, value, cp_group, core_attention_fn, attention_mask,
        attn_mask_type, packed_seq_params, seq_local, batch, heads, dim, cp
    )


class CommQueue:
    """
    Communication Queue for async AllToAll operations

    Manages async AllToAll submissions with proper stream synchronization.
    Allows submitting multiple AllToAll operations and waiting for all to complete.
    """

    def __init__(self, comm_stream: torch.cuda.Stream):
        self.comm_stream = comm_stream
        self.default_stream = torch.cuda.current_stream()
        self.pending_ops = []  # List of (result_tensor, done_event)

    def submit_alltoall(
        self,
        input_tensor: torch.Tensor,
        output_splits: List[int],
        input_splits: List[int],
        group: torch.distributed.ProcessGroup,
    ) -> None:
        """Submit an AllToAll operation asynchronously"""
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_stream(self.default_stream)
            result = _all_to_all(
                input_tensor.contiguous(),
                output_splits,
                input_splits,
                group,
            )
            done_event = torch.cuda.Event()
            done_event.record(self.comm_stream)
        self.pending_ops.append((result, done_event))

    def wait_all(self) -> List[torch.Tensor]:
        """Wait for all pending AllToAll operations and return results"""
        results = []
        for result, done_event in self.pending_ops:
            done_event.wait(self.default_stream)
            results.append(result)
        self.pending_ops = []
        return results


class AsyncP2PAttentionHandle:
    """Handle for tracking async P2P attention communication per peer"""
    def __init__(self, peer_rank: int, send_work, recv_work, recv_buffer: torch.Tensor,
                 recv_shape: Tuple[int, ...]):
        self.peer_rank = peer_rank
        self.send_work = send_work
        self.recv_work = recv_work
        self.recv_buffer = recv_buffer
        self.recv_shape = recv_shape  # Expected shape after reshape
        self._completed = False

    def is_completed(self) -> bool:
        """Check if recv from this peer is complete (non-blocking)"""
        if self._completed:
            return True
        if self.recv_work is not None and self.recv_work.is_completed():
            self._completed = True
            return True
        return False

    def wait(self) -> torch.Tensor:
        """Wait for recv to complete and return the data with correct shape"""
        if self.recv_work is not None:
            self.recv_work.wait()
        if self.send_work is not None:
            self.send_work.wait()
        self._completed = True
        # Reshape to expected shape
        if self.recv_buffer.numel() > 0:
            return self.recv_buffer.view(self.recv_shape)
        return self.recv_buffer


def async_p2p_attention_sp2hp_start(
    qkv_remote: torch.Tensor,  # [seq/CP, B, remote_qkv_dim] - concatenated Q,K,V for remote heads
    seq_local: int,
    batch_size: int,
    local_qkv_dim: int,  # Size of Q+K+V for local heads
    cp_group: torch.distributed.ProcessGroup,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> Tuple['_Sp2hpHandle', torch.cuda.Event]:
    """
    Start async P2P for sp2hp: send remote_heads QKV (single message), recv remote_seq QKV

    Optimized version: sends concatenated Q+K+V as ONE message instead of 3.

    For CP=2: Only 2 P2P ops (1 send + 1 recv) instead of 6.

    Args:
        qkv_remote: Concatenated Q,K,V for remote heads [seq/CP, B, remote_dim]
        seq_local: Local sequence length (seq/CP)
        batch_size: Batch size
        local_qkv_dim: Q+K+V dimension for local heads (what we'll receive)
        cp_group: Context parallel process group
        comm_stream: Optional CUDA stream for P2P operations

    Returns:
        handle: Single handle for all P2P (to wait)
        event: CUDA event recorded after P2P started
    """
    cp_size = cp_group.size()
    my_rank = cp_group.rank()

    if cp_size == 1:
        return None, None

    device = qkv_remote.device
    dtype = qkv_remote.dtype

    # For CP=2: simple 1-to-1 exchange
    # Each rank sends its remote_heads QKV, receives the other's remote_seq QKV

    p2p_ops = []
    recv_buffers = []

    # Send buffer - flatten for P2P
    send_tensor = qkv_remote.contiguous().view(-1)

    for peer_rank in range(cp_size):
        if peer_rank == my_rank:
            continue

        # Send: my local_seq with remote_heads (what peer needs)
        p2p_ops.append(dist.P2POp(dist.isend, send_tensor, peer_rank, group=cp_group))

        # Recv: peer's local_seq with my local_heads
        recv_size = seq_local * batch_size * local_qkv_dim
        recv_buffer = torch.empty(recv_size, dtype=dtype, device=device)
        recv_buffers.append((peer_rank, recv_buffer))
        p2p_ops.append(dist.P2POp(dist.irecv, recv_buffer, peer_rank, group=cp_group))

    # Execute P2P operations
    event = None
    if comm_stream is not None:
        with torch.cuda.stream(comm_stream):
            if p2p_ops:
                reqs = dist.batch_isend_irecv(p2p_ops)
            else:
                reqs = []
        event = torch.cuda.Event()
        event.record(comm_stream)
    else:
        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
        else:
            reqs = []

    handle = _Sp2hpHandle(
        reqs=reqs,
        recv_buffers=recv_buffers,
        recv_shape=(seq_local, batch_size, local_qkv_dim),
    )

    return handle, event


class _Sp2hpHandle:
    """Handle for sp2hp P2P communication (optimized: single message for Q+K+V)"""
    def __init__(self, reqs: List, recv_buffers: List[Tuple[int, torch.Tensor]], recv_shape: Tuple[int, ...]):
        self.reqs = reqs
        self.recv_buffers = recv_buffers  # [(peer_rank, buffer), ...]
        self.recv_shape = recv_shape
        self._completed = False

    def wait(self) -> List[Tuple[int, torch.Tensor]]:
        """Wait and return received data from all peers"""
        # Wait for all P2P to complete
        for req in self.reqs:
            if req is not None:
                req.wait()
        self._completed = True

        # Return received buffers reshaped
        results = []
        for peer_rank, buf in self.recv_buffers:
            results.append((peer_rank, buf.view(self.recv_shape)))
        return results


def async_p2p_attention_hp2sp_start(
    attn_output_remote_seq: torch.Tensor,  # [remote_seq, B, H/CP, D]
    seq_local: int,
    batch_size: int,
    heads_local: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[List[AsyncP2PAttentionHandle], torch.cuda.Event]:
    """
    Start async P2P for hp2sp: send remote_seq attention output, recv remote_heads

    In hp2sp, each rank:
    - Has attention output for full_seq with LOCAL heads
    - Needs output for local_seq with ALL heads
    - Sends: remote_seq portions (for each remote rank's local_seq)
    - Receives: local_seq with remote_heads (from each remote rank)

    Args:
        attn_output_remote_seq: Attention output for remote sequence portions
                               [remote_seq, B, H/CP, D] where remote_seq = seq * (CP-1) / CP
        seq_local: Local sequence length (seq/CP)
        batch_size: Batch size
        heads_local: Number of local heads (H/CP)
        head_dim: Head dimension
        cp_group: Context parallel process group
        comm_stream: Optional CUDA stream for P2P operations

    Returns:
        handles: List of handles for each peer
        event: CUDA event recorded after P2P started
    """
    cp_size = cp_group.size()
    my_rank = cp_group.rank()

    if cp_size == 1:
        return [], None

    device = attn_output_remote_seq.device
    dtype = attn_output_remote_seq.dtype

    p2p_ops = []
    recv_buffers = {}
    send_data_cache = {}

    # Each peer's remote_seq portion is seq_local
    peer_idx = 0
    for peer_rank in range(cp_size):
        if peer_rank == my_rank:
            continue

        # Send: remote_seq portion for this peer (their local_seq, my heads)
        seq_start = peer_idx * seq_local
        seq_end = seq_start + seq_local
        send_tensor = attn_output_remote_seq[seq_start:seq_end].contiguous()
        send_data_cache[peer_rank] = send_tensor
        p2p_ops.append(dist.P2POp(dist.isend, send_tensor, peer_rank, group=cp_group))

        # Recv: my local_seq, peer's heads
        recv_size = seq_local * batch_size * heads_local * head_dim
        recv_buffer = torch.empty(recv_size, dtype=dtype, device=device)
        recv_buffers[peer_rank] = recv_buffer
        p2p_ops.append(dist.P2POp(dist.irecv, recv_buffer, peer_rank, group=cp_group))

        peer_idx += 1

    # Execute P2P operations
    event = None
    if comm_stream is not None:
        with torch.cuda.stream(comm_stream):
            if p2p_ops:
                reqs = dist.batch_isend_irecv(p2p_ops)
            else:
                reqs = []
        event = torch.cuda.Event()
        event.record(comm_stream)
    else:
        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
        else:
            reqs = []

    # Create handles
    handles = []
    req_idx = 0
    peer_idx = 0

    for peer_rank in range(cp_size):
        if peer_rank == my_rank:
            continue

        send_work = reqs[req_idx] if req_idx < len(reqs) else None
        req_idx += 1
        recv_work = reqs[req_idx] if req_idx < len(reqs) else None
        req_idx += 1

        handle = AsyncP2PAttentionHandle(
            peer_rank=peer_rank,
            send_work=send_work,
            recv_work=recv_work,
            recv_buffer=recv_buffers.get(peer_rank, torch.empty(0, dtype=dtype, device=device)),
            recv_shape=(seq_local, batch_size, heads_local, head_dim),
        )
        handles.append(handle)
        peer_idx += 1

    return handles, event

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
"""

import torch
import os
from megatron.core.tensor_parallel.mappings import _AllToAll
from typing import List, Optional, Tuple

# Chunking configuration for dX + AllToAll pipeline
# Set via environment variable, default to 1 (no chunking)
#
# EXPERIMENTAL FEATURE - NOT RECOMMENDED FOR PRODUCTION:
#
# Why chunking doesn't help (and can hurt):
# 1. grouped_gemm has ~0.4ms fixed overhead per call, regardless of input size
# 2. Chunking dX computation (e.g., 4 chunks) means 8x more grouped_gemm calls
#    (2 calls per chunk × 4 chunks vs 2 calls total)
# 3. This adds ~2.4ms overhead per MoE layer (6 extra calls × 0.4ms)
#
# Measured results (8K tokens, 2x4090 PCIe):
# - FLUID_DX_NUM_CHUNKS=1: ~596ms/iter (97.5% dW overlap)
# - FLUID_DX_NUM_CHUNKS=2: ~620ms/iter (+24ms overhead)
# - FLUID_DX_NUM_CHUNKS=4: ~645ms/iter (+49ms overhead)
#
# The default dW overlap strategy (without dX chunking) already achieves
# 97.5% overlap ratio by scheduling dW tasks during AllToAll communication.
# Additional dX chunking provides no extra benefit.
#
# Future optimization: Move chunking logic to CUDA kernel level to avoid
# per-call overhead.
DX_NUM_CHUNKS = int(os.environ.get('FLUID_DX_NUM_CHUNKS', '1'))


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
        return _AllToAll.apply(group, input, output_split_sizes, input_split_sizes)

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Backward with chunked AllToAll and dW overlap

        When FLUID_DX_NUM_CHUNKS > 1:
        - Split grad_output into chunks
        - Launch each chunk's AllToAll asynchronously on comm_stream
        - On last chunk, trigger dW execution for overlap
        - Reassemble results maintaining correct ordering
        """
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        grad = grad_output[0]
        num_chunks = DX_NUM_CHUNKS

        # Get split sizes for backward (swap input/output)
        # In backward: what was output becomes input, what was input becomes output
        backward_output_splits = ctx.input_split_sizes  # Where we send TO
        backward_input_splits = ctx.output_split_sizes  # Where we receive FROM

        # If no chunking or splits are None, use original path
        if num_chunks <= 1 or backward_output_splits is None or backward_input_splits is None:
            # Original non-chunked path
            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                result = _AllToAll.apply(
                    ctx.group, grad, backward_output_splits, backward_input_splits
                )
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)
                scheduler.set_alltoall_end_event(event)

            scheduler.on_alltoall_start(comm_type=ctx.comm_type)
            scheduler.default_stream.wait_stream(scheduler.comm_stream)

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
                chunk_result = _AllToAll.apply(
                    ctx.group,
                    chunk_grad.contiguous(),
                    chunk_backward_output_splits,
                    chunk_backward_input_splits,
                )
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)

            alltoall_results.append(chunk_result)

            # For the last chunk, set up dW overlap
            if chunk_idx == num_chunks - 1:
                scheduler.set_alltoall_end_event(event)
                scheduler.on_alltoall_start(comm_type=ctx.comm_type)

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

class ChunkedAllToAllManager:
    """
    Manages chunked AllToAll for dX + communication pipeline.

    This class enables overlapping dX computation with AllToAll communication
    by splitting dX into chunks and starting each chunk's AllToAll as soon
    as that chunk is computed.

    Timeline:
    |-- dX_chunk1 --|-- dX_chunk2 --|-- dX_chunk3 --|-- dX_chunk4 --|
                   |--- A2A_1 ----|
                                  |--- A2A_2 ----|
                                                 |--- A2A_3 ----|
                                                                |--- A2A_4 ---| <- dW here

    Usage:
        manager = ChunkedAllToAllManager(num_chunks=4, group=ep_group, ...)
        for i in range(num_chunks):
            dx_chunk = compute_dx_chunk(i)  # On default stream
            manager.submit_chunk(i, dx_chunk)  # Starts async AllToAll
        result = manager.collect_results()  # Wait and concat
    """

    def __init__(
        self,
        num_chunks: int,
        group: torch.distributed.ProcessGroup,
        output_splits: List[int],
        input_splits: List[int],
        comm_type: str = "moe_dispatch",
    ):
        """
        Initialize the chunked AllToAll manager.

        Args:
            num_chunks: Number of chunks to split dX into
            group: Process group for AllToAll
            output_splits: Total output split sizes (will be proportionally divided)
            input_splits: Total input split sizes (will be proportionally divided)
            comm_type: Type of communication for scheduler
        """
        self.num_chunks = num_chunks
        self.group = group
        self.total_output_splits = output_splits
        self.total_input_splits = input_splits
        self.comm_type = comm_type

        # Results storage
        self.chunk_results: List[Optional[torch.Tensor]] = [None] * num_chunks
        self.chunk_events: List[Optional[torch.cuda.Event]] = [None] * num_chunks
        self.chunks_submitted = 0

        # Get scheduler
        from fluid.scheduler import get_backward_scheduler
        self.scheduler = get_backward_scheduler()

    def _compute_chunk_splits(
        self,
        total_splits: List[int],
        chunk_idx: int,
        chunk_size: int,
        total_size: int,
    ) -> List[int]:
        """Compute proportional splits for a chunk."""
        if total_splits is None:
            return None

        # Proportional allocation
        ratio = chunk_size / total_size
        chunk_splits = [max(1, int(s * ratio)) for s in total_splits]

        # Adjust to match exact chunk size
        split_sum = sum(chunk_splits)
        if split_sum != chunk_size:
            diff = chunk_size - split_sum
            chunk_splits[-1] += diff

        return chunk_splits

    def submit_chunk(
        self,
        chunk_idx: int,
        chunk_data: torch.Tensor,
        total_tokens: int,
    ):
        """
        Submit a chunk for async AllToAll.

        Args:
            chunk_idx: Index of this chunk (0 to num_chunks-1)
            chunk_data: The computed dX chunk tensor
            total_tokens: Total number of tokens across all chunks
        """
        chunk_size = chunk_data.shape[0]

        # Compute splits for this chunk
        chunk_output_splits = self._compute_chunk_splits(
            self.total_output_splits, chunk_idx, chunk_size, total_tokens
        )
        chunk_input_splits = self._compute_chunk_splits(
            self.total_input_splits, chunk_idx, chunk_size, total_tokens
        )

        # Launch AllToAll on comm_stream (async)
        with torch.cuda.stream(self.scheduler.comm_stream):
            self.scheduler.comm_stream.wait_stream(self.scheduler.default_stream)
            result = _AllToAll.apply(
                self.group,
                chunk_data.contiguous(),
                chunk_output_splits,
                chunk_input_splits,
            )
            event = torch.cuda.Event()
            event.record(self.scheduler.comm_stream)

        self.chunk_results[chunk_idx] = result
        self.chunk_events[chunk_idx] = event
        self.chunks_submitted += 1

        # For the last chunk, set up dW overlap
        if chunk_idx == self.num_chunks - 1:
            self.scheduler.set_alltoall_end_event(event)
            # Trigger dW execution while last chunk's AllToAll is running
            self.scheduler.on_alltoall_start(comm_type=self.comm_type)

    def collect_results(self) -> torch.Tensor:
        """
        Wait for all chunks and concatenate results.

        Returns:
            Concatenated result tensor
        """
        # Wait for all comm to complete
        self.scheduler.default_stream.wait_stream(self.scheduler.comm_stream)

        # Concatenate results
        return torch.cat([r for r in self.chunk_results if r is not None], dim=0)


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
        output = _AllToAll.apply(group, x, [seq_local] * cp, [seq_local] * cp)

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
                result = _AllToAll.apply(group, grad, [seq_local] * cp, [seq_local] * cp)
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
                chunk_result = _AllToAll.apply(
                    group,
                    grad_chunk_flat.contiguous(),
                    chunk_splits,
                    chunk_splits,
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
        output = _AllToAll.apply(group, x, [seq_local] * cp, [seq_local] * cp)

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
                result = _AllToAll.apply(group, grad, [seq_local] * cp, [seq_local] * cp)
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
                chunk_result = _AllToAll.apply(
                    group,
                    grad_chunk_flat.contiguous(),
                    chunk_splits,
                    chunk_splits,
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
    def forward(ctx, hidden_states, output_splits, input_splits, group, permutation_map, restore_shape):
        """
        Forward: AllToAll combine + unpermute
        """
        ctx.group = group
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.permutation_map = permutation_map
        ctx.restore_shape = restore_shape

        output_splits_list = output_splits.tolist() if hasattr(output_splits, 'tolist') else list(output_splits)
        input_splits_list = input_splits.tolist() if hasattr(input_splits, 'tolist') else list(input_splits)

        # AllToAll combine
        combined = _AllToAll.apply(group, hidden_states, output_splits_list, input_splits_list)

        # Unpermute
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

            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                result = _AllToAll.apply(group, grad_combined, backward_output_splits, backward_input_splits)
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)
                scheduler.set_alltoall_end_event(event)

            scheduler.on_alltoall_start(comm_type="moe_combine")
            scheduler.default_stream.wait_stream(scheduler.comm_stream)

            return (result, None, None, None, None, None)

        # ============================================================
        # Chunked AllToAll Pipeline for MoE Combine Backward
        #
        # Step 1: Compute full permute backward (index_select)
        # Step 2: Chunk the result by rank and pipeline AllToAll
        #
        # Note: Unlike Ulysses where dX itself is chunked, here we do full
        # permute backward first, then chunk the AllToAll. This is because
        # the permutation_map doesn't have a simple structure that allows
        # chunked index_select to produce rank-ordered output.
        # ============================================================

        # Step 1: Full permute backward
        grad_combined = grad_output.index_select(0, permutation_map)

        # Data is now ordered by: [tokens for rank0][tokens for rank1]...
        # backward_input_splits tells us how many tokens go to each rank

        # Compute rank boundaries
        rank_starts = [0]
        for split in backward_input_splits:
            rank_starts.append(rank_starts[-1] + split)

        alltoall_results = []

        for chunk_idx in range(num_chunks):
            # ============================================================
            # Extract this chunk's data
            # Each chunk contains proportional data from ALL ranks
            # ============================================================
            chunk_parts = []
            chunk_backward_input_splits = []
            chunk_backward_output_splits = []

            for src_rank in range(ep_size):
                rank_tokens = backward_input_splits[src_rank]
                if rank_tokens == 0:
                    chunk_backward_input_splits.append(0)
                    continue

                # Calculate chunk boundaries for this rank's data
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

                # Global indices in grad_combined
                global_start = rank_starts[src_rank] + chunk_start_in_rank
                global_end = rank_starts[src_rank] + chunk_end_in_rank

                chunk_parts.append(grad_combined[global_start:global_end])

            # Compute output splits for this chunk
            for dest_rank in range(ep_size):
                dest_tokens = backward_output_splits[dest_rank]
                if dest_tokens == 0:
                    chunk_backward_output_splits.append(0)
                    continue
                dest_chunk_size = dest_tokens // num_chunks
                dest_remainder = dest_tokens % num_chunks
                if chunk_idx < dest_remainder:
                    dest_this_chunk = dest_chunk_size + 1
                else:
                    dest_this_chunk = dest_chunk_size
                chunk_backward_output_splits.append(dest_this_chunk)

            # Concatenate chunk parts
            if chunk_parts:
                chunk_grad = torch.cat(chunk_parts, dim=0)
            else:
                chunk_grad = torch.zeros(0, grad_output.shape[-1],
                                        dtype=grad_output.dtype, device=grad_output.device)

            # ============================================================
            # Launch AllToAll for this chunk
            # ============================================================
            with torch.cuda.stream(scheduler.comm_stream):
                scheduler.comm_stream.wait_stream(scheduler.default_stream)
                chunk_result = _AllToAll.apply(
                    group,
                    chunk_grad.contiguous(),
                    chunk_backward_output_splits,
                    chunk_backward_input_splits,
                )
                event = torch.cuda.Event()
                event.record(scheduler.comm_stream)

            alltoall_results.append((chunk_result, chunk_backward_output_splits))

            # For the last chunk, set up dW overlap
            if chunk_idx == num_chunks - 1:
                scheduler.set_alltoall_end_event(event)
                scheduler.on_alltoall_start(comm_type="moe_combine")

        # ============================================================
        # Wait and reassemble results
        # ============================================================
        scheduler.default_stream.wait_stream(scheduler.comm_stream)

        # Reassemble: collect per-dest-rank chunks
        dest_rank_chunks = [[] for _ in range(ep_size)]
        for chunk_idx, (chunk_result, chunk_splits) in enumerate(alltoall_results):
            offset = 0
            for dest_rank in range(ep_size):
                split_size = chunk_splits[dest_rank]
                if split_size > 0:
                    dest_rank_chunks[dest_rank].append(chunk_result[offset:offset+split_size])
                    offset += split_size

        # Concatenate chunks for each dest rank
        final_parts = []
        for dest_rank in range(ep_size):
            if dest_rank_chunks[dest_rank]:
                final_parts.append(torch.cat(dest_rank_chunks[dest_rank], dim=0))

        if final_parts:
            result = torch.cat(final_parts, dim=0)
        else:
            result = torch.zeros(sum(backward_output_splits), grad_output.shape[-1],
                               dtype=grad_output.dtype, device=grad_output.device)

        return (result, None, None, None, None, None)


def fluid_fused_combine_unpermute(
    hidden_states: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    permutation_map: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """
    Fused MoE Combine + Unpermute with true dX + AllToAll pipeline

    Shape: [num_global_tokens, hidden] -> [num_tokens, hidden]

    This version fuses AllToAll combine with unpermute in backward,
    enabling true dX + AllToAll pipelining.
    """
    return _FluidFusedCombineUnpermute.apply(
        hidden_states, output_splits, input_splits, group, permutation_map, restore_shape
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

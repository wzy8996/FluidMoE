"""
Fluid versions of tensor parallel communication primitives with global scheduler integration

This module provides drop-in replacements for standard AlltoAll operations
that integrate with the global backward scheduler for fine-grained overlap.
"""

import torch
from megatron.core.tensor_parallel.mappings import _AllToAll


class _FluidAllToAll(torch.autograd.Function):
    """
    AlltoAll with global backward scheduler integration

    Forward: Same as standard _AllToAll
    Backward: Triggers global scheduler to launch dW tasks during communication
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

        # Use standard _AllToAll apply (not forward directly)
        return _AllToAll.apply(group, input, output_split_sizes, input_split_sizes)

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward with Fluid scheduler for communication-computation overlap"""
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        # Launch AlltoAll on comm_stream (async, uses network)
        with torch.cuda.stream(scheduler.comm_stream):
            scheduler.comm_stream.wait_stream(scheduler.default_stream)
            result = _AllToAll.apply(
                ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes
            )
            event = torch.cuda.Event()
            event.record(scheduler.comm_stream)
            scheduler.set_alltoall_end_event(event)

        # Trigger dW execution on default_stream (parallel with AlltoAll)
        scheduler.on_alltoall_start(comm_type=ctx.comm_type)
        scheduler.default_stream.wait_stream(scheduler.comm_stream)

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
    """
    return fluid_all_to_all(
        input, group,
        output_split_sizes=_to_list(output_splits),
        input_split_sizes=_to_list(input_splits),
        comm_type="moe_dispatch",
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

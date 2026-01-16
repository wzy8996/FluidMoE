"""
Basic AllToAll Communication Primitives

This module provides low-level AllToAll communication functions used by
MoE (Expert Parallel) and Attention (Context Parallel) modules.
"""

import torch
import torch.distributed as dist
from typing import List, Optional


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


def _all_to_all_sp2hp_forward(input_: torch.Tensor, group) -> torch.Tensor:
    """
    Forward-only sp2hp AllToAll (no autograd).
    Used in Attention for sequence parallel to head parallel conversion.

    Shape: [seq/CP, batch, heads, dim] -> [seq, batch, heads/CP, dim]

    Args:
        input_: Input tensor [seq_local, batch, heads, dim]
        group: Context parallel process group

    Returns:
        Output tensor [seq_full, batch, heads_local, dim]
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
    Used in Attention for head parallel to sequence parallel conversion.

    Shape: [seq, batch, heads/CP, dim] -> [seq/CP, batch, heads, dim]

    Args:
        input_: Input tensor [seq_full, batch, heads_local, dim]
        group: Context parallel process group

    Returns:
        Output tensor [seq_local, batch, heads, dim]
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


def _sort_chunks_by_idxs(input_tensor, split_sizes, sorted_idxs):
    """
    Sort chunks of input tensor by indices.
    Used in MoE for reordering tokens between rank-major and expert-major layouts.

    Args:
        input_tensor: [total_tokens, hidden] tensor
        split_sizes: list or tensor of chunk sizes (can contain zeros)
        sorted_idxs: list or tensor of new order indices

    Returns:
        Reordered tensor
    """
    if input_tensor.numel() == 0:
        return input_tensor

    # Convert to list (avoid GPU sync)
    if torch.is_tensor(split_sizes):
        split_sizes = split_sizes.tolist()
    if torch.is_tensor(sorted_idxs):
        sorted_idxs = sorted_idxs.tolist()

    # Direct split and cat (Megatron style, simple and efficient)
    chunks = torch.split(input_tensor, split_sizes, dim=0)
    return torch.cat([chunks[i] for i in sorted_idxs], dim=0)

"""
MoE Layer - Complete Autograd Functions with Routing

This module provides complete MoE autograd functions that combine:
- Forward: Router + P2P overlap for Dispatch+FC1 and FC2+Combine phases
- Backward: Router backward + Expert backward with AllToAll + chunked compute overlap + dW scheduling

Key classes:
- MoEP2PChunkedFunction: Full autograd function with routing + P2P forward + chunked AllToAll backward
- MoELayer: High-level nn.Module wrapper

Design principles:
- Router computation integrated with dW scheduling
- Forward uses P2P overlap for compute-communication overlap
- Backward uses AllToAll with chunked compute overlap
- Memory-efficient: save FC1 outputs instead of full activation
- dW tasks are registered and executed during AllToAll communication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, List

from fluid.core.forward_comm import MultiCardOverlapContext
from fluid.core import _sort_chunks_by_idxs
from fluid.core.scheduler import get_backward_scheduler

# Forward operations
from fluid.moe.forward import (
    router_forward,
    dispatch_fc1_p2p_forward,
    fc2_combine_p2p_forward,
)

# Backward operations
from fluid.moe.backward import (
    combine_backward,
    expert_backward,
    register_moe_dw_tasks,
    dispatch_backward,
    router_backward,
    register_router_dw_task,
)


class MoEP2PChunkedFunction(torch.autograd.Function):
    """
    Complete MoE autograd function with routing integrated:
    - Forward: Router + P2P overlap for Dispatch+FC1 and FC2+Combine phases
    - Backward: Router backward + Expert backward with AllToAll + chunked compute overlap + dW scheduling

    This is the main entry point for expert-parallel MoE computation.

    Forward timeline:
        Router: Compute logits -> softmax -> top_k selection
        Dispatch+FC1 phase: Each round computes FC1+Act for partner while P2P runs
        FC2+Combine phase: Each round computes FC2 while Combine P2P runs
        Apply probs and restore order

    Backward timeline:
        Step 1: Gradient through prob weighting and restore
        Step 2: Combine AllToAll (with dW overlap from previous layer)
        Step 3: Convert layout rank-major -> expert-major
        Step 4: Expert backward (compute grad_fc1, optionally grad_tokens)
        Step 5: Register dW tasks for weight1/weight2
        Step 6: Dispatch AllToAll with optional chunked dX overlap
        Step 7: Router backward (grad_hidden from permute + router dX)
        Step 8: Register router dW task
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        ep_group: dist.ProcessGroup,
        overlap_ctx: MultiCardOverlapContext,
        layer_id: int,
        num_experts: int,
        top_k: int,
        activation_func,
        num_chunks: int,
    ) -> torch.Tensor:
        """
        Forward pass with routing and P2P overlap.

        Args:
            hidden_states: [num_tokens, hidden_size] input tokens
            router_weight: [hidden_size, num_experts] router weight matrix
            weight1: [hidden, ffn_hidden * num_local_experts] FC1 weight
            weight2: [ffn_hidden * num_local_experts, hidden] FC2 weight
            ep_group: Expert Parallel process group
            overlap_ctx: P2P overlap context
            layer_id: Layer ID for dW task naming
            num_experts: Total number of experts
            top_k: Number of experts each token is sent to
            activation_func: Activation function
            num_chunks: Number of chunks for backward

        Returns:
            output: [num_tokens, hidden] final output
        """
        needs_grad = hidden_states.requires_grad
        ctx.needs_grad = needs_grad

        ep_size = ep_group.size()
        my_rank = ep_group.rank()
        num_tokens = hidden_states.shape[0]
        dtype = hidden_states.dtype
        hidden_size = hidden_states.shape[-1]
        num_local_experts = num_experts // ep_size

        total_ffn_hidden = weight1.shape[-1]
        ffn_hidden = total_ffn_hidden // num_local_experts

        # Detach weights for forward computation
        weight1_detached = weight1.detach()
        weight2_detached = weight2.detach()

        # =====================================================================
        # Step 1: Router computation (using router_forward from forward.py)
        # =====================================================================
        (permuted_tokens, permuted_probs, restore_indices, sorted_indices,
         input_splits, output_splits, tokens_per_expert, tokens_per_expert_2d,
         router_probs, top_indices, router_logits) = router_forward(
            hidden_states, router_weight, num_experts, top_k, ep_group
        )

        # Compute num_global_tokens_per_local_expert for expert computation
        local_start_expert = my_rank * num_local_experts
        num_global_tokens_per_local_expert = tokens_per_expert_2d[
            :, local_start_expert:local_start_expert + num_local_experts
        ].unsqueeze(0)

        input_splits_list = input_splits.tolist()
        output_splits_list = output_splits.tolist()

        # =====================================================================
        # Step 2: Dispatch + FC1 with P2P overlap
        # Note: FC1 is NOT saved - will be recomputed during backward
        # =====================================================================
        (local_tokens, local_act, recv_act_results, recv_buffers,
         partners, recv_offsets) = dispatch_fc1_p2p_forward(
            permuted_tokens, weight1_detached, input_splits_list, output_splits_list,
            ep_group, overlap_ctx, activation_func, num_local_experts,
            num_global_tokens_per_local_expert,
            needs_backward=needs_grad,
        )

        # =====================================================================
        # Step 3: FC2 + Combine with P2P overlap
        # =====================================================================
        (combined_output, local_fc2, all_expert_tokens,
         all_tokens_per_expert_list_out, backward_indices) = fc2_combine_p2p_forward(
            local_tokens, local_act, recv_act_results, recv_buffers,
            weight2_detached, input_splits_list, output_splits_list,
            ep_group, overlap_ctx, num_local_experts,
            num_global_tokens_per_local_expert, partners,
            needs_backward=needs_grad,
        )

        # =====================================================================
        # Step 4: Apply probs and restore order
        # =====================================================================
        # Apply routing probabilities
        weighted_output = combined_output * permuted_probs.unsqueeze(-1).to(combined_output.dtype)

        # Restore original order
        restored_output = weighted_output[restore_indices]

        # Sum over top_k experts
        output = restored_output.view(num_tokens, top_k, hidden_size).sum(dim=1)

        # =====================================================================
        # Save for backward
        # =====================================================================
        ctx.ep_group = ep_group
        ctx.activation_func = activation_func
        ctx.layer_id = layer_id
        ctx.num_local_experts = num_local_experts
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        ctx.input_splits_list = input_splits_list
        ctx.output_splits_list = output_splits_list
        ctx.ffn_hidden = ffn_hidden
        ctx.num_chunks = num_chunks

        if needs_grad:
            # Save tensors for backward
            # Note: all_fc1 is NOT saved - will be recomputed during backward
            ctx.save_for_backward(
                hidden_states,           # For router backward
                router_weight.detach(),  # For router backward
                all_expert_tokens,       # For expert backward + FC1 recomputation
                combined_output,         # For prob weighting backward
                permuted_probs,          # For prob weighting backward
                restore_indices,         # For restore backward
                sorted_indices,          # For permute backward
            )
            # Save original weights for gradient assignment
            ctx._orig_router_weight = router_weight
            ctx._orig_weight1 = weight1
            ctx._orig_weight2 = weight2
            ctx._weight1_detached = weight1_detached
            ctx._weight2_detached = weight2_detached
            ctx.backward_indices = backward_indices
            ctx.all_tokens_per_expert = all_tokens_per_expert_list_out
            # For router backward
            ctx.router_probs = router_probs.detach()
            ctx.top_indices = top_indices.detach()

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass with router and expert gradients.

        Args:
            grad_output: [num_tokens, hidden] gradient w.r.t. output

        Returns:
            Gradients for all inputs (most are None for non-tensor inputs)
        """
        if not ctx.needs_grad:
            return (None,) * 11

        (hidden_states, router_weight, all_expert_tokens,
         combined_output, permuted_probs, restore_indices, sorted_indices) = ctx.saved_tensors

        orig_router_weight = ctx._orig_router_weight
        orig_weight1 = ctx._orig_weight1
        orig_weight2 = ctx._orig_weight2
        weight1_2d = ctx._weight1_detached
        weight2_2d = ctx._weight2_detached
        backward_indices = ctx.backward_indices

        ep_group = ctx.ep_group
        activation_func = ctx.activation_func
        layer_id = ctx.layer_id
        num_local_experts = ctx.num_local_experts
        num_experts = ctx.num_experts
        top_k = ctx.top_k
        all_tokens_per_expert = ctx.all_tokens_per_expert
        input_splits_list = ctx.input_splits_list
        output_splits_list = ctx.output_splits_list
        num_chunks = ctx.num_chunks
        ffn_hidden = ctx.ffn_hidden
        hidden_size = weight2_2d.shape[-1]
        num_tokens = grad_output.shape[0]
        device = grad_output.device
        dtype = hidden_states.dtype

        router_probs = ctx.router_probs
        top_indices = ctx.top_indices

        # View weights to 3D for computation
        # weight1_2d: [hidden, ffn * E] -> view to [hidden, ffn, E] -> permute to [E, hidden, ffn]
        # weight2_2d: [ffn * E, hidden] -> view to [ffn, hidden, E] -> permute to [E, ffn, hidden]
        weight1 = weight1_2d.view(hidden_size, ffn_hidden, num_local_experts).permute(2, 0, 1).contiguous()
        weight2 = weight2_2d.view(ffn_hidden, hidden_size, num_local_experts).permute(2, 0, 1).contiguous()

        scheduler = get_backward_scheduler()

        # Use num_chunks directly (consistent with Attention layer)
        # num_chunks=1 means no chunking, num_chunks>1 means chunked dX + AllToAll overlap
        num_chunks_actual = num_chunks if scheduler.is_enabled() else 1

        # =====================================================================
        # Step 1: Backward through sum and restore
        # =====================================================================
        # grad_output: [num_tokens, hidden]
        # -> grad_restored: [num_tokens * top_k, hidden]
        grad_restored = grad_output.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)

        # Reverse restore: grad_weighted = grad_restored[inverse_restore_indices]
        # restore_indices maps permuted -> original, so we need inverse
        inverse_restore_indices = torch.argsort(restore_indices)
        grad_weighted = grad_restored[inverse_restore_indices]

        # =====================================================================
        # Step 2: Backward through prob weighting
        # =====================================================================
        # weighted_output = combined_output * permuted_probs
        # grad_combined = grad_weighted * permuted_probs
        # grad_permuted_probs = (grad_weighted * combined_output).sum(dim=-1)
        grad_combined = grad_weighted * permuted_probs.unsqueeze(-1).to(grad_weighted.dtype)
        grad_permuted_probs = (grad_weighted * combined_output.to(grad_weighted.dtype)).sum(dim=-1)

        # =====================================================================
        # Step 3: Combine Backward AllToAll with dW overlap + FC1 recomputation
        # FC1 is recomputed during AllToAll to save forward memory copy overhead
        # =====================================================================
        grad_combined_recv, all_fc1 = combine_backward(
            grad_combined, input_splits_list, output_splits_list, ep_group, layer_id,
            all_expert_tokens=all_expert_tokens,
            weight1=weight1,
            num_local_experts=num_local_experts,
            all_tokens_per_expert=all_tokens_per_expert,
        )

        # =====================================================================
        # Step 4: Convert layout: rank-major -> expert-major
        # =====================================================================
        if 'split_sizes_rank_major' in backward_indices:
            grad_all_fc2 = _sort_chunks_by_idxs(
                grad_combined_recv,
                backward_indices['split_sizes_rank_major'],
                backward_indices['sorted_idxs_rank_to_exp'],
            )
        else:
            grad_all_fc2 = grad_combined_recv

        # =====================================================================
        # Step 5: Expert backward computation
        # =====================================================================
        compute_dx = (num_chunks_actual == 1)
        grad_all_fc1, act_output, grad_all_tokens = expert_backward(
            grad_all_fc2, all_fc1, weight1, weight2,
            activation_func, num_local_experts, all_tokens_per_expert,
            compute_dx=compute_dx
        )

        # =====================================================================
        # Step 6: Register dW tasks for expert weights
        # =====================================================================
        grad_w1, grad_w2 = register_moe_dw_tasks(
            weight1, weight2, all_expert_tokens, act_output,
            grad_all_fc2, grad_all_fc1,
            num_local_experts, all_tokens_per_expert, layer_id,
            orig_weight1, orig_weight2
        )

        # =====================================================================
        # Step 7: Dispatch Backward AllToAll with optional chunked dX overlap
        # =====================================================================
        split_sizes_exp_major = backward_indices.get('split_sizes_exp_major', all_tokens_per_expert)
        sorted_idxs_exp_to_rank = backward_indices.get('sorted_idxs_exp_to_rank', list(range(len(all_tokens_per_expert))))

        grad_permuted_tokens = dispatch_backward(
            grad_all_fc1,
            weight1,
            split_sizes_exp_major,
            sorted_idxs_exp_to_rank,
            all_tokens_per_expert,
            input_splits_list,
            output_splits_list,
            ep_group,
            layer_id=layer_id,
            num_chunks=num_chunks_actual,
            grad_all_tokens=grad_all_tokens,
            comm_stream=scheduler.comm_stream if scheduler.is_enabled() else None,
        )

        # =====================================================================
        # Step 8: Backward through sort/permute to get grad_expanded_tokens
        # =====================================================================
        # permuted_tokens = expanded_tokens[sorted_indices]
        # So: grad_expanded_tokens[sorted_indices] = grad_permuted_tokens
        grad_expanded_tokens = torch.zeros_like(grad_permuted_tokens)
        grad_expanded_tokens[sorted_indices] = grad_permuted_tokens

        # =====================================================================
        # Step 9: Backward through expand to get grad_hidden_states
        # =====================================================================
        # expanded_tokens = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)
        # Reverse: sum over top_k copies
        grad_hidden_from_tokens = grad_expanded_tokens.view(num_tokens, top_k, hidden_size).sum(dim=1)

        # =====================================================================
        # Step 10: Router backward - compute grad through softmax and logits
        # =====================================================================
        grad_hidden_from_router, grad_router_logits = router_backward(
            grad_permuted_probs=grad_permuted_probs,
            sorted_indices=sorted_indices,
            restore_indices=restore_indices,
            permuted_probs=permuted_probs,
            router_probs=router_probs,
            top_indices=top_indices,
            router_weight=router_weight,
            num_tokens=num_tokens,
            top_k=top_k,
            dtype=dtype,
        )

        # =====================================================================
        # Step 11: Combine gradients for hidden_states
        # =====================================================================
        grad_hidden_states = grad_hidden_from_tokens + grad_hidden_from_router

        # =====================================================================
        # Step 12: Register router dW task
        # =====================================================================
        grad_router_weight = register_router_dw_task(
            hidden_states=hidden_states,
            grad_router_logits=grad_router_logits,
            router_weight=orig_router_weight,
            layer_id=layer_id,
        )

        # Return gradients in same order as forward inputs
        return (
            grad_hidden_states,   # hidden_states
            grad_router_weight,   # router_weight
            grad_w1,              # weight1
            grad_w2,              # weight2
            None,                 # ep_group
            None,                 # overlap_ctx
            None,                 # layer_id
            None,                 # num_experts
            None,                 # top_k
            None,                 # activation_func
            None,                 # num_chunks
        )


def moe_p2p_chunked(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    ep_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    layer_id: int = 0,
    num_experts: int = 4,
    top_k: int = 2,
    activation_func=F.silu,
    num_chunks: int = 4,
) -> torch.Tensor:
    """
    Expert-parallel MoE with routing, P2P forward overlap and chunked AllToAll backward.

    This is the main API for using the optimized MoE layer with integrated routing.

    Args:
        hidden_states: [num_tokens, hidden] input tokens
        router_weight: [hidden_size, num_experts] router weight matrix
        weight1: [hidden, ffn_hidden * num_local_experts] FC1 weight
        weight2: [ffn_hidden * num_local_experts, hidden] FC2 weight
        ep_group: Expert Parallel process group
        overlap_ctx: P2P overlap context
        layer_id: Layer ID for dW task naming
        num_experts: Total number of experts
        top_k: Number of experts each token is sent to
        activation_func: Activation function
        num_chunks: Number of chunks for backward overlap

    Returns:
        output: [num_tokens, hidden] final output

    Example:
        >>> # Setup
        >>> ep_group = dist.new_group(ranks=[0, 1, 2, 3])
        >>> overlap_ctx = MultiCardOverlapContext(ep_group)
        >>>
        >>> # Forward
        >>> output = moe_p2p_chunked(
        ...     hidden_states, router_weight, weight1, weight2,
        ...     ep_group, overlap_ctx,
        ...     layer_id=0, num_experts=8, top_k=2,
        ...     activation_func=F.gelu, num_chunks=4
        ... )
        >>>
        >>> # Backward (automatically uses chunked AllToAll overlap)
        >>> loss = output.sum()
        >>> loss.backward()
    """
    return MoEP2PChunkedFunction.apply(
        hidden_states, router_weight, weight1, weight2,
        ep_group, overlap_ctx, layer_id,
        num_experts, top_k, activation_func, num_chunks
    )


class MoELayer(nn.Module):
    """
    High-level MoE layer module with routing, P2P forward and chunked AllToAll backward.

    This module wraps the low-level autograd function with proper weight initialization
    and context management.

    Args:
        hidden_size: Model hidden dimension
        ffn_hidden_size: FFN hidden dimension
        num_experts: Total number of experts
        num_local_experts: Number of experts per rank
        top_k: Number of experts each token is sent to
        ep_group: Expert parallel process group
        layer_id: Layer ID for dW task naming
        num_chunks: Number of chunks for backward overlap
        activation_func: Activation function

    Example:
        >>> layer = MoELayer(
        ...     hidden_size=4096,
        ...     ffn_hidden_size=11008,
        ...     num_experts=8,
        ...     num_local_experts=2,
        ...     top_k=2,
        ...     ep_group=ep_group,
        ...     layer_id=0,
        ... )
        >>> output = layer(hidden_states)
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_experts: int,
        num_local_experts: int,
        top_k: int,
        ep_group: dist.ProcessGroup,
        layer_id: int = 0,
        num_chunks: int = 4,
        activation_func=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.top_k = top_k
        self.ep_group = ep_group
        self.layer_id = layer_id
        self.num_chunks = num_chunks
        self.activation_func = activation_func if activation_func is not None else F.gelu

        # Router weight
        self.router_weight = nn.Parameter(torch.empty(hidden_size, num_experts))

        # Expert weights (Megatron-compatible 2D layout)
        self.weight1 = nn.Parameter(torch.empty(hidden_size, ffn_hidden_size * num_local_experts))
        self.weight2 = nn.Parameter(torch.empty(ffn_hidden_size * num_local_experts, hidden_size))

        # P2P overlap context
        ep_size = ep_group.size()
        device = torch.device(f'cuda:{dist.get_rank(ep_group)}')
        self.overlap_ctx = MultiCardOverlapContext(device, ep_size, ep_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.router_weight)
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [num_tokens, hidden] input tokens

        Returns:
            output: [num_tokens, hidden] final output
        """
        return moe_p2p_chunked(
            hidden_states,
            self.router_weight,
            self.weight1,
            self.weight2,
            self.ep_group,
            self.overlap_ctx,
            self.layer_id,
            self.num_experts,
            self.top_k,
            self.activation_func,
            self.num_chunks,
        )

    def extra_repr(self) -> str:
        return (
            f'hidden_size={self.hidden_size}, '
            f'ffn_hidden_size={self.ffn_hidden_size}, '
            f'num_experts={self.num_experts}, '
            f'num_local_experts={self.num_local_experts}, '
            f'top_k={self.top_k}, '
            f'num_chunks={self.num_chunks}'
        )


__all__ = [
    'MoEP2PChunkedFunction',
    'moe_p2p_chunked',
    'MoELayer',
]

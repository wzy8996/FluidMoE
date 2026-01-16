"""
MoE Router with dW Scheduling Support

This module implements the router (gating network) for MoE layers with
lazy dW registration for computation-communication overlap.

The router computes which experts each token should be sent to, and
registers its weight gradient computation as a low-priority dW task
to be executed during subsequent AllToAll communications.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, List

from fluid.core.scheduler import get_backward_scheduler


class _RouterFunction(torch.autograd.Function):
    """
    Router autograd function with dW overlap support.

    Forward: Computes router logits = input @ weight
    Backward:
        - Immediately computes dX (on critical path)
        - Registers dW as low-priority task for later execution
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, layer_id: int) -> torch.Tensor:
        """
        Forward pass: Linear projection for routing.

        Args:
            input: [num_tokens, hidden_size]
            weight: [hidden_size, num_experts]
            layer_id: Layer index for dW task naming

        Returns:
            logits: [num_tokens, num_experts]
        """
        ctx.save_for_backward(input, weight.detach())
        ctx._orig_weight = weight  # Keep reference for gradient assignment
        ctx.layer_id = layer_id

        # Router forward: input @ weight
        # Detach weight to avoid retaining computation graph across iterations
        # (we compute dW manually in backward, so no need for autograd to track weight)
        logits = torch.matmul(input.float(), weight.detach().float())
        return logits

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass: Compute dX immediately, register dW for later.

        Args:
            grad_output: [num_tokens, num_experts]

        Returns:
            grad_input: [num_tokens, hidden_size]
        """
        input, weight = ctx.saved_tensors
        orig_weight = ctx._orig_weight
        scheduler = get_backward_scheduler()

        # === CRITICAL PATH: Compute dX immediately ===
        # grad_input = grad_output @ weight.T
        # [num_tokens, num_experts] @ [num_experts, hidden_size] = [num_tokens, hidden_size]
        grad_input = torch.matmul(grad_output.float(), weight.t().float())

        # === LAZY REGISTRATION: Register dW for overlap ===
        grad_output_saved = grad_output.detach()
        input_saved = input.detach()

        def compute_dw():
            # grad_weight = input.T @ grad_output
            # [hidden_size, num_tokens] @ [num_tokens, num_experts] = [hidden_size, num_experts]
            grad_weight = torch.matmul(input_saved.t().float(), grad_output_saved.float())
            return grad_weight

        scheduler.register_dw_task(
            layer_name=f"router_weight_layer{ctx.layer_id}",
            layer_id=ctx.layer_id,
            compute_fn=compute_dw,
            priority=0,  # Low priority, execute after other dW tasks
            weight_param=orig_weight,
        )

        return grad_input.to(input.dtype), None, None


def compute_routing(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    num_experts: int,
    top_k: int,
    ep_group,
    layer_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute MoE routing: token-to-expert assignment with top-k selection.

    This function:
    1. Computes router logits via _RouterFunction (registers dW)
    2. Applies softmax and top-k selection
    3. Sorts tokens by expert assignment
    4. Computes AllToAll split sizes

    Args:
        hidden_states: [num_tokens, hidden_size] input tokens
        router_weight: [hidden_size, num_experts] router weight matrix
        num_experts: Total number of experts across all ranks
        top_k: Number of experts each token is sent to
        ep_group: Expert Parallel process group
        layer_id: Layer index for dW task naming

    Returns:
        permuted_tokens: Tokens sorted by expert assignment
        input_splits: Number of tokens to send to each rank
        output_splits: Number of tokens to receive from each rank
        permuted_probs: Routing probabilities (sorted)
        restore_indices: Indices to restore original token order
        tokens_per_expert: Local tokens per expert count
        global_tokens_per_expert: Global tokens per expert count
        tokens_per_expert_2d: [ep_size, num_experts] token distribution matrix
    """
    ep_size = ep_group.size()
    my_rank = ep_group.rank()
    num_tokens = hidden_states.shape[0]
    device = hidden_states.device

    # Compute router logits (registers dW task via _RouterFunction)
    router_logits = _RouterFunction.apply(hidden_states, router_weight, layer_id)

    # Top-k selection
    router_probs = F.softmax(router_logits, dim=-1)
    top_probs, top_indices = torch.topk(router_probs, k=top_k, dim=-1)

    # Normalize top-k probabilities
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

    # Expand: each token is replicated top_k times
    expanded_tokens = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_states.shape[-1])
    expanded_probs = top_probs.reshape(-1)
    expanded_expert_indices = top_indices.reshape(-1)

    # Sort by expert index
    sorted_indices = torch.argsort(expanded_expert_indices)
    permuted_tokens = expanded_tokens[sorted_indices]
    permuted_probs = expanded_probs[sorted_indices]
    sorted_expert_indices = expanded_expert_indices[sorted_indices]

    # Count tokens per expert
    tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=num_experts)

    # Compute input_splits and output_splits for AllToAll
    experts_per_rank = num_experts // ep_size

    input_splits = torch.zeros(ep_size, dtype=torch.int64, device=device)
    for i in range(ep_size):
        start_expert = i * experts_per_rank
        end_expert = start_expert + experts_per_rank
        input_splits[i] = tokens_per_expert[start_expert:end_expert].sum()

    # AllGather to get all ranks' input_splits
    all_input_splits = [torch.zeros_like(input_splits) for _ in range(ep_size)]
    dist.all_gather(all_input_splits, input_splits, group=ep_group)
    all_input_splits = torch.stack(all_input_splits)  # [ep_size, ep_size]

    # output_splits[i] = rank i's tokens destined for my experts
    output_splits = all_input_splits[:, my_rank].clone()

    # Save restore indices for combining results
    restore_indices = torch.argsort(sorted_indices)

    # AllGather tokens_per_expert to get 2D distribution matrix
    all_tokens_per_expert = [torch.zeros_like(tokens_per_expert) for _ in range(ep_size)]
    dist.all_gather(all_tokens_per_expert, tokens_per_expert, group=ep_group)
    tokens_per_expert_2d = torch.stack(all_tokens_per_expert)  # [ep_size, num_experts]

    # Global tokens per expert (sum across ranks)
    global_tokens_per_expert = tokens_per_expert_2d.sum(dim=0)

    return (permuted_tokens, input_splits, output_splits, permuted_probs,
            restore_indices, tokens_per_expert, global_tokens_per_expert, tokens_per_expert_2d)

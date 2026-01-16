"""
MoE Baseline Implementation

This module implements the baseline MoE layer with standard AllToAll
communication. The backward pass uses the scheduler for dW overlap.

Key features:
- Standard Dispatch/Combine AllToAll pattern
- dW tasks registered for overlap during backward
- Compatible with scheduler-based backward optimization
"""

import torch
import torch.nn.functional as F
from typing import List

from fluid.core import _all_to_all, _compute_activation_derivative, _compute_activation_grad
from fluid.core.scheduler import get_backward_scheduler
from .router import compute_routing


class _MoEBaselineFunction(torch.autograd.Function):
    """
    Baseline MoE autograd function.

    Forward:
        1. Dispatch AllToAll: Send tokens to expert-owning ranks
        2. Expert computation: FC1 -> Activation -> FC2
        3. Combine AllToAll: Return results to original ranks

    Backward:
        1. Combine AllToAll (with dW overlap)
        2. Expert backward: Compute dX for FC1/FC2
        3. Register dW tasks for later execution
        4. Dispatch AllToAll (with dW overlap)
    """

    @staticmethod
    def forward(ctx, permuted_tokens, input_splits, output_splits,
                weight1, weight2, ep_group, activation_func,
                num_local_experts, tokens_per_expert_2d, layer_id=0,
                orig_weight1_2d=None, orig_weight2_2d=None):
        my_rank = ep_group.rank()
        ep_size = ep_group.size()
        device = permuted_tokens.device
        dtype = permuted_tokens.dtype
        hidden_size = permuted_tokens.shape[-1]
        ffn_hidden = weight1.shape[-1]

        input_splits_list = input_splits.tolist() if torch.is_tensor(input_splits) else list(input_splits)
        output_splits_list = output_splits.tolist() if torch.is_tensor(output_splits) else list(output_splits)

        # Dispatch AllToAll
        recv_tokens = _all_to_all(
            permuted_tokens.contiguous(),
            output_split_sizes=output_splits_list,
            input_split_sizes=input_splits_list,
            group=ep_group
        )

        total_recv = recv_tokens.shape[0]
        start_expert = my_rank * num_local_experts

        # Calculate tokens per expert
        tokens_per_expert_list = []
        for local_exp_idx in range(num_local_experts):
            global_exp_idx = start_expert + local_exp_idx
            total = sum(tokens_per_expert_2d[r, global_exp_idx].item() for r in range(ep_size))
            tokens_per_expert_list.append(total)

        # Build source-major -> expert-major reorder indices
        src_major_offsets = {}
        offset = 0
        for source_rank in range(ep_size):
            for local_exp_idx in range(num_local_experts):
                global_exp_idx = start_expert + local_exp_idx
                n_tokens = tokens_per_expert_2d[source_rank, global_exp_idx].item()
                src_major_offsets[(source_rank, local_exp_idx)] = (offset, n_tokens)
                offset += n_tokens

        # Build reorder indices
        reorder_indices = []
        split_sizes_rank_major = []
        for source_rank in range(ep_size):
            for local_exp_idx in range(num_local_experts):
                start_off, n_tokens = src_major_offsets[(source_rank, local_exp_idx)]
                split_sizes_rank_major.append(n_tokens)

        for local_exp_idx in range(num_local_experts):
            for source_rank in range(ep_size):
                start_off, n_tokens = src_major_offsets[(source_rank, local_exp_idx)]
                for i in range(n_tokens):
                    reorder_indices.append(start_off + i)

        if len(reorder_indices) > 0:
            reorder_indices_tensor = torch.tensor(reorder_indices, dtype=torch.long, device=device)
            recv_tokens_expert_major = recv_tokens[reorder_indices_tensor]
        else:
            reorder_indices_tensor = torch.tensor([], dtype=torch.long, device=device)
            recv_tokens_expert_major = recv_tokens

        # Expert computation with FC1 saved for backward
        all_fc1_list = []
        fc2_out_expert_major = torch.zeros(total_recv, hidden_size, dtype=dtype, device=device)

        offset = 0
        for local_exp_idx in range(num_local_experts):
            n_tokens = tokens_per_expert_list[local_exp_idx]
            if n_tokens > 0:
                exp_tokens = recv_tokens_expert_major[offset:offset + n_tokens]
                fc1_out = torch.matmul(exp_tokens, weight1[local_exp_idx])
                all_fc1_list.append(fc1_out)
                act_out = activation_func(fc1_out)
                fc2_out_expert_major[offset:offset + n_tokens] = torch.matmul(act_out, weight2[local_exp_idx])
                offset += n_tokens

        all_fc1 = torch.cat(all_fc1_list, dim=0) if all_fc1_list else torch.empty(0, ffn_hidden, dtype=dtype, device=device)

        # Reorder: expert-major -> source-major
        if len(reorder_indices_tensor) > 0:
            inverse_indices = torch.zeros_like(reorder_indices_tensor)
            inverse_indices[reorder_indices_tensor] = torch.arange(len(reorder_indices_tensor), device=device)
            fc2_out = fc2_out_expert_major[inverse_indices]
        else:
            inverse_indices = torch.tensor([], dtype=torch.long, device=device)
            fc2_out = fc2_out_expert_major

        # Combine AllToAll
        combined_output = _all_to_all(
            fc2_out.contiguous(),
            output_split_sizes=input_splits_list,
            input_split_sizes=output_splits_list,
            group=ep_group
        )

        # Save for backward
        ctx.save_for_backward(recv_tokens_expert_major, weight1.detach(), weight2.detach(), all_fc1,
                              reorder_indices_tensor, inverse_indices)
        # Use original 2D weights for gradient if provided, otherwise use 3D weights
        ctx._orig_weight1 = orig_weight1_2d if orig_weight1_2d is not None else weight1
        ctx._orig_weight2 = orig_weight2_2d if orig_weight2_2d is not None else weight2
        ctx._use_2d_layout = orig_weight1_2d is not None
        ctx.ep_group = ep_group
        ctx.activation_func = activation_func
        ctx.input_splits_list = input_splits_list
        ctx.output_splits_list = output_splits_list
        ctx.num_local_experts = num_local_experts
        ctx.ffn_hidden = ffn_hidden
        ctx.tokens_per_expert_list = tokens_per_expert_list
        ctx.split_sizes_rank_major = split_sizes_rank_major
        ctx.layer_id = layer_id

        return combined_output

    @staticmethod
    def backward(ctx, grad_output):
        recv_tokens_expert_major, weight1, weight2, all_fc1, \
            reorder_indices_tensor, inverse_indices = ctx.saved_tensors

        ep_group = ctx.ep_group
        activation_func = ctx.activation_func
        input_splits_list = ctx.input_splits_list
        output_splits_list = ctx.output_splits_list
        num_local_experts = ctx.num_local_experts
        ffn_hidden = ctx.ffn_hidden
        tokens_per_expert_list = ctx.tokens_per_expert_list
        layer_id = ctx.layer_id

        device = grad_output.device
        hidden_size = grad_output.shape[-1]
        total_recv = recv_tokens_expert_major.shape[0]

        # Compute activation derivative from saved fc1
        act_output = activation_func(all_fc1)
        act_deriv = _compute_activation_derivative(all_fc1, activation_func, gated_linear_unit=False)

        # Combine backward AllToAll with dW overlap
        scheduler = get_backward_scheduler()
        if scheduler.is_enabled():
            comm_stream = scheduler.comm_stream
            default_stream = scheduler.default_stream
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_combined = _all_to_all(
                    grad_output.contiguous(),
                    output_split_sizes=output_splits_list,
                    input_split_sizes=input_splits_list,
                    group=ep_group
                )
                event = torch.cuda.Event()
                event.record(comm_stream)
                scheduler.set_alltoall_end_event(event)
            scheduler.on_alltoall_start(comm_type=f"moe_combine_L{layer_id}")
            default_stream.wait_stream(comm_stream)
        else:
            grad_combined = _all_to_all(
                grad_output.contiguous(),
                output_split_sizes=output_splits_list,
                input_split_sizes=input_splits_list,
                group=ep_group
            )

        # Reorder grad: source-major -> expert-major
        if len(reorder_indices_tensor) > 0:
            grad_all_fc2 = grad_combined[reorder_indices_tensor]
        else:
            grad_all_fc2 = grad_combined

        # Compute grad_tokens and grad_fc1
        grad_all_tokens = torch.zeros(total_recv, hidden_size, dtype=grad_output.dtype, device=device)
        grad_all_fc1 = torch.zeros(total_recv, ffn_hidden, dtype=grad_output.dtype, device=device)

        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = tokens_per_expert_list[exp_idx]
            if n_tok > 0:
                grad_exp_act = torch.matmul(grad_all_fc2[start:start+n_tok], weight2[exp_idx].t())
                grad_exp_fc1 = _compute_activation_grad(
                    grad_exp_act, act_deriv[start:start+n_tok], gated_linear_unit=False
                )
                grad_all_tokens[start:start+n_tok] = torch.matmul(grad_exp_fc1, weight1[exp_idx].t())
                grad_all_fc1[start:start+n_tok] = grad_exp_fc1
                start += n_tok

        # Register dW tasks
        num_local_experts_saved = num_local_experts
        tokens_per_expert_saved = tokens_per_expert_list
        ffn_hidden_saved = ffn_hidden
        hidden_size_saved = hidden_size
        grad_all_fc2_saved = grad_all_fc2.detach()
        grad_all_fc1_saved = grad_all_fc1.detach()
        act_output_saved = act_output.detach()
        recv_tokens_saved = recv_tokens_expert_major.detach()
        orig_weight1 = ctx._orig_weight1
        orig_weight2 = ctx._orig_weight2
        use_2d_layout = ctx._use_2d_layout

        def compute_dw_weight2():
            # Compute gradients in 3D layout first
            grad_w2_3d = torch.zeros_like(weight2)
            start = 0
            for exp_idx in range(num_local_experts_saved):
                n_tok = tokens_per_expert_saved[exp_idx]
                if n_tok > 0:
                    grad_w2_3d[exp_idx] = torch.matmul(
                        act_output_saved[start:start+n_tok].t(),
                        grad_all_fc2_saved[start:start+n_tok]
                    )
                    start += n_tok
            if use_2d_layout:
                # Convert 3D [E, ffn, hidden] -> 2D [ffn * E, hidden]
                return grad_w2_3d.permute(1, 2, 0).reshape(ffn_hidden_saved * num_local_experts_saved, hidden_size_saved)
            return grad_w2_3d

        def compute_dw_weight1():
            # Compute gradients in 3D layout first
            grad_w1_3d = torch.zeros_like(weight1)
            start = 0
            for exp_idx in range(num_local_experts_saved):
                n_tok = tokens_per_expert_saved[exp_idx]
                if n_tok > 0:
                    grad_w1_3d[exp_idx] = torch.matmul(
                        recv_tokens_saved[start:start+n_tok].t(),
                        grad_all_fc1_saved[start:start+n_tok]
                    )
                    start += n_tok
            if use_2d_layout:
                # Convert 3D [E, hidden, ffn] -> 2D [hidden, ffn * E]
                return grad_w1_3d.permute(1, 2, 0).reshape(hidden_size_saved, ffn_hidden_saved * num_local_experts_saved)
            return grad_w1_3d

        scheduler.register_dw_task(
            layer_name=f"moe_weight2_L{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight2,
            priority=100,
            weight_param=orig_weight2,
        )
        scheduler.register_dw_task(
            layer_name=f"moe_weight1_L{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight1,
            priority=99,
            weight_param=orig_weight1,
        )

        # Dispatch backward AllToAll with dW overlap
        if len(inverse_indices) > 0:
            grad_dispatched = grad_all_tokens[inverse_indices].contiguous()
        else:
            grad_dispatched = grad_all_tokens.contiguous()

        if scheduler.is_enabled():
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_tokens = _all_to_all(
                    grad_dispatched,
                    output_split_sizes=input_splits_list,
                    input_split_sizes=output_splits_list,
                    group=ep_group
                )
                event = torch.cuda.Event()
                event.record(comm_stream)
                scheduler.set_alltoall_end_event(event)
            scheduler.on_alltoall_start(comm_type=f"moe_dispatch_L{layer_id}")
            default_stream.wait_stream(comm_stream)
        else:
            grad_tokens = _all_to_all(
                grad_dispatched,
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group
            )

        # Return gradients: (permuted_tokens, input_splits, output_splits, weight1, weight2,
        #                    ep_group, activation_func, num_local_experts, tokens_per_expert_2d,
        #                    layer_id, orig_weight1_2d, orig_weight2_2d)
        return (grad_tokens, None, None, None, None, None, None, None, None, None, None, None)


class MoEBaseline:
    """
    Baseline MoE layer with standard AllToAll.

    Uses scheduler-based dW overlap in backward pass.
    """

    def __init__(self, config, ep_group, device, dtype, layer_id=0):
        self.config = config
        self.ep_group = ep_group
        self.device = device
        self.dtype = dtype
        self.ep_size = ep_group.size()
        self.my_rank = ep_group.rank()
        self.layer_id = layer_id

        self.hidden_size = config['hidden_size']
        self.ffn_hidden_size = config['ffn_hidden_size']
        self.num_experts = config['num_experts']
        self.top_k = config['top_k']
        self.experts_per_rank = self.num_experts // self.ep_size

    def init_weights(self, requires_grad=True):
        """
        Initialize weights using Overlap-compatible layout.

        Weight layout (same as Overlap/Megatron GroupedMLP):
            weight1: [hidden, ffn * num_local_experts] - 2D layout
            weight2: [ffn * num_local_experts, hidden] - 2D layout

        Usage: view to 3D then permute for per-expert computation
            w1 = weight1.view(hidden, ffn, num_experts).permute(2, 0, 1)  # [E, H, F]
            w2 = weight2.view(ffn, hidden, num_experts).permute(2, 0, 1)  # [E, F, H]
        """
        self.router_weight = torch.randn(
            self.hidden_size, self.num_experts,
            dtype=torch.float32, device=self.device,
        ).requires_grad_(requires_grad)
        # Overlap-compatible 2D layout
        # NOTE: Create tensor first, then multiply by scale, then set requires_grad
        # to ensure the weight is a leaf tensor (non-leaf tensors cause graph retention issues)
        self.weight1 = (torch.randn(
            self.hidden_size, self.ffn_hidden_size * self.experts_per_rank,
            dtype=self.dtype, device=self.device,
        ) * 0.02).requires_grad_(requires_grad)
        self.weight2 = (torch.randn(
            self.ffn_hidden_size * self.experts_per_rank, self.hidden_size,
            dtype=self.dtype, device=self.device,
        ) * 0.02).requires_grad_(requires_grad)

    def forward(self, tokens, do_backward=False):
        """
        Forward pass.

        Args:
            tokens: [num_tokens, hidden_size]
            do_backward: Whether to run backward immediately

        Returns:
            output: [num_tokens, hidden_size]
        """
        # Routing
        # Note: router_weight needs to be passed with requires_grad for _RouterFunction to
        # register dW task. But we detach expert weights to avoid graph retention.
        permuted_tokens, input_splits, output_splits, probs, restore_indices, \
            local_tokens_per_expert, global_tokens_per_expert, tokens_per_expert_2d = \
            compute_routing(tokens, self.router_weight, self.num_experts, self.top_k,
                          self.ep_group, self.layer_id)

        # Convert 2D weight layout to 3D for per-expert computation
        # weight1: [hidden, ffn * E] -> [E, hidden, ffn]
        # weight2: [ffn * E, hidden] -> [E, ffn, hidden]
        # Detach weights to avoid retaining computation graph across iterations
        # (we compute dW manually in backward, so no need for autograd to track weight)
        w1_3d = self.weight1.detach().view(self.hidden_size, self.ffn_hidden_size, self.experts_per_rank).permute(2, 0, 1).contiguous()
        w2_3d = self.weight2.detach().view(self.ffn_hidden_size, self.hidden_size, self.experts_per_rank).permute(2, 0, 1).contiguous()

        # Expert computation
        combined = _MoEBaselineFunction.apply(
            permuted_tokens, input_splits, output_splits,
            w1_3d, w2_3d, self.ep_group, F.gelu,
            self.experts_per_rank, tokens_per_expert_2d, self.layer_id,
            self.weight1, self.weight2  # Pass original 2D weights for gradient
        )

        # Apply routing probabilities
        combined = combined * probs.unsqueeze(-1).to(combined.dtype)

        # Restore original order and reduce top-k
        restored = combined[restore_indices]
        num_tokens = tokens.shape[0]
        output = restored.view(num_tokens, self.top_k, -1).sum(dim=1)

        if do_backward:
            loss = output.sum()
            loss.backward()

        return output

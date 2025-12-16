# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn.functional as F

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from torch.nn.parameter import Parameter
from typing import Optional


# This is the new autograd function that implements the parallel backward pass.
class _FluidExpertComputation(torch.autograd.Function):
    """
    Custom autograd function for MoE expert computation that parallelizes
    the computation of dW and dX in the backward pass.
    """

    @staticmethod
    def forward(
        ctx,
        permuted_local_hidden_states,
        tokens_per_expert,
        permuted_probs,
        weight1,
        weight2,
        num_local_experts,
        hidden_size,
        ffn_hidden_size,
        gated_linear_unit,
        activation_func,
        moe_apply_probs_on_input,
        moe_router_topk,
    ):
        # Save tensors and configs for backward
        ctx.save_for_backward(
            permuted_local_hidden_states, tokens_per_expert, permuted_probs, weight1, weight2
        )
        ctx.num_local_experts = num_local_experts
        ctx.hidden_size = hidden_size
        ctx.ffn_hidden_size = ffn_hidden_size
        ctx.gated_linear_unit = gated_linear_unit
        ctx.activation_func = activation_func
        ctx.moe_apply_probs_on_input = moe_apply_probs_on_input

        # --- Re-implement the forward pass from GroupedMLP ---
        if moe_apply_probs_on_input:
            assert moe_router_topk == 1, "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() == 0:
            # Make sure params of experts still have gradients even given zero tokens.
            w1 = weight1.view(hidden_size, -1)
            w2 = weight2.view(-1, hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = activation_func(h, permuted_probs.unsqueeze(-1))
            fc2_output = torch.matmul(h, w2)
            ctx.fc1_output = h
            ctx.intermediate_parallel = h  # Simplified for zero-element case
            return fc2_output

        # Reshape the weights for the grouped GEMMs.
        w1 = weight1.view(num_local_experts, hidden_size, -1)
        w2 = weight2.view(num_local_experts, -1, hidden_size)

        # grouped_gemm requires tokens_per_expert to be on CPU
        tokens_per_expert_cpu = tokens_per_expert.cpu() if tokens_per_expert.is_cuda else tokens_per_expert

        # For now, use manual per-expert computation instead of grouped_gemm
        # TODO: Optimize with proper grouped_gemm usage
        fc1_outputs = []
        fc2_outputs = []
        start_idx = 0
        for expert_idx in range(num_local_experts):
            num_tokens = tokens_per_expert_cpu[expert_idx].item()
            if num_tokens == 0:
                continue

            # Slice tokens for this expert
            end_idx = start_idx + num_tokens
            expert_input = permuted_local_hidden_states[start_idx:end_idx]  # [num_tokens, hidden]
            expert_probs = permuted_probs[start_idx:end_idx]  # [num_tokens, 1]

            # FC1: [num_tokens, hidden] x [hidden, intermediate] = [num_tokens, intermediate]
            fc1_out = torch.matmul(expert_input, w1[expert_idx])

            # Activation with probs
            # expert_probs is already [num_tokens, 1], no need to unsqueeze again
            intermediate = activation_func(fc1_out, expert_probs)


            # FC2: [num_tokens, intermediate] x [intermediate, hidden] = [num_tokens, hidden]
            fc2_out = torch.matmul(intermediate, w2[expert_idx])

            fc1_outputs.append(fc1_out)
            fc2_outputs.append(intermediate)  # For backward

            if expert_idx == 0:
                fc1_output = fc1_out
                intermediate_parallel = intermediate
                fc2_output = fc2_out
            else:
                fc1_output = torch.cat([fc1_output, fc1_out], dim=0)
                intermediate_parallel = torch.cat([intermediate_parallel, intermediate], dim=0)
                fc2_output = torch.cat([fc2_output, fc2_out], dim=0)

            start_idx = end_idx

        # Save intermediate tensors for backward
        ctx.fc1_output = fc1_output
        ctx.intermediate_parallel = intermediate_parallel

        return fc2_output

    @staticmethod
    def backward(ctx, grad_fc2_output):
        # Retrieve saved tensors
        (
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
            weight1,
            weight2,
        ) = ctx.saved_tensors
        fc1_output = ctx.fc1_output
        intermediate_parallel = ctx.intermediate_parallel

        # Retrieve configs
        num_local_experts = ctx.num_local_experts
        hidden_size = ctx.hidden_size
        activation_func = ctx.activation_func
        moe_apply_probs_on_input = ctx.moe_apply_probs_on_input

        # Get Fluid scheduler
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        # Reshape weights
        w1 = weight1.view(num_local_experts, hidden_size, -1)
        w2 = weight2.view(num_local_experts, -1, hidden_size)

        tokens_per_expert_cpu = tokens_per_expert.cpu() if tokens_per_expert.is_cuda else tokens_per_expert

        # === CRITICAL PATH: Compute dX immediately ===
        grad_permuted_local_hidden_states = torch.zeros_like(permuted_local_hidden_states)

        start_idx = 0
        for expert_idx in range(num_local_experts):
            num_tokens = tokens_per_expert_cpu[expert_idx].item()
            if num_tokens == 0:
                continue

            end_idx = start_idx + num_tokens

            # Grad from FC2: grad_fc2_output @ w2.T
            grad_intermediate = torch.matmul(
                grad_fc2_output[start_idx:end_idx],
                w2[expert_idx].t()
            )

            # Grad through activation
            grad_fc1 = grad_intermediate

            # Grad from FC1: grad_fc1 @ w1.T (critical path)
            grad_input = torch.matmul(grad_fc1, w1[expert_idx].t())
            grad_permuted_local_hidden_states[start_idx:end_idx] = grad_input

            start_idx = end_idx

        # === LAZY REGISTRATION: Register dW computation ===
        # Detach tensors to avoid holding computation graph
        grad_fc2_output_saved = grad_fc2_output.detach()
        intermediate_parallel_saved = intermediate_parallel.detach()
        permuted_local_hidden_states_saved = permuted_local_hidden_states.detach()
        tokens_per_expert_saved = tokens_per_expert.detach()

        # Define dW computation function for all experts
        def compute_dw_weight2():
            """Compute grad_weight2 for all experts"""
            grad_w2_all = torch.zeros_like(weight2)
            w2_view = grad_w2_all.view(num_local_experts, -1, hidden_size)

            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert_saved[exp_idx].item()
                if n_tok == 0:
                    continue
                end = start + n_tok

                # grad_w2 = intermediate.T @ grad_output
                grad_w2 = torch.matmul(
                    intermediate_parallel_saved[start:end].t(),
                    grad_fc2_output_saved[start:end]
                )
                w2_view[exp_idx] = grad_w2
                start = end

            return grad_w2_all

        def compute_dw_weight1():
            """Compute grad_weight1 for all experts"""
            grad_w1_all = torch.zeros_like(weight1)
            w1_view = grad_w1_all.view(num_local_experts, hidden_size, -1)
            w2_view = weight2.view(num_local_experts, -1, hidden_size)

            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert_saved[exp_idx].item()
                if n_tok == 0:
                    continue
                end = start + n_tok

                # Recompute grad_fc1 (needed for w1 gradient)
                grad_intermediate = torch.matmul(
                    grad_fc2_output_saved[start:end],
                    w2_view[exp_idx].t()
                )
                grad_fc1 = grad_intermediate

                # grad_w1 = input.T @ grad_fc1
                grad_w1 = torch.matmul(
                    permuted_local_hidden_states_saved[start:end].t(),
                    grad_fc1
                )
                w1_view[exp_idx] = grad_w1
                start = end

            return grad_w1_all

        # Register dW tasks to scheduler
        # Higher priority for weight2 (closer to output)
        scheduler.register_dw_task(
            layer_name="moe_expert_weight2",
            layer_id=0,  # Could be parameterized if needed
            compute_fn=compute_dw_weight2,
            priority=100,
            weight_param=weight2,
        )

        scheduler.register_dw_task(
            layer_name="moe_expert_weight1",
            layer_id=0,
            compute_fn=compute_dw_weight1,
            priority=99,
            weight_param=weight1,
        )

        # Return None for weight gradients (scheduler will compute them)
        grad_weight1 = None
        grad_weight2 = None

        return (
            grad_permuted_local_hidden_states,  # 1
            None,  # 2: tokens_per_expert
            None,  # 3: permuted_probs
            grad_weight1,  # 4: None (scheduler computes)
            grad_weight2,  # 5: None (scheduler computes)
            None,  # 6: num_local_experts
            None,  # 7: hidden_size
            None,  # 8: ffn_hidden_size
            None,  # 9: gated_linear_unit
            None,  # 10: activation_func
            None,  # 11: moe_apply_probs_on_input
            None,  # 12: moe_router_topk
        )


class FluidGroupedMLP(MegatronModule):
    """
    A version of GroupedMLP that uses a custom autograd function
    to enable a parallel backward pass.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert not config.add_bias_linear, f"bias not supported in FluidGroupedMLP (got {config.add_bias_linear})"

        self.expert_parallel = config.expert_model_parallel_size > 1

        # Activation function logic from GroupedMLP
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        def activation_func_with_probs(x, probs):
            dtype = x.dtype
            res = self.activation_func(x) * probs
            return res.to(dtype)

        self.activation_func_with_probs = activation_func_with_probs

        # Weight initialization from GroupedMLP
        tp_size = pg_collection.expt_tp.size()

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        init_device = 'cpu' if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight1 = Parameter(
            torch.empty(
                self.config.hidden_size,
                fc1_output_size_per_partition,
                device=init_device,
                dtype=config.params_dtype,
            )
        )
        self.weight2 = Parameter(
            torch.empty(
                fc2_input_size_per_partition,
                self.config.hidden_size,
                device=init_device,
                dtype=config.params_dtype,
            )
        )

        if config.perform_initialization:
            if config.use_cpu_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    1,
                    config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    0,
                    config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                )
            else:
                _initialize_affine_weight_gpu(
                    self.weight1, config.init_method, partition_dim=1, is_expert=True
                )
                _initialize_affine_weight_gpu(
                    self.weight2, config.output_layer_init_method, partition_dim=0, is_expert=True
                )

        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        output = _FluidExpertComputation.apply(
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
            self.weight1,
            self.weight2,
            self.num_local_experts,
            self.config.hidden_size,
            self.config.moe_ffn_hidden_size,
            self.config.gated_linear_unit,
            self.activation_func_with_probs,
            self.config.moe_apply_probs_on_input,
            self.config.moe_router_topk,
        )
        # FluidGroupedMLP doesn't use bias, so return None for mlp_bias
        return output, None


# ============================================================
# FluidRouter - Router with dW overlap support
# ============================================================

class _FluidRouterFunc(torch.autograd.Function):
    """
    Fluid Router with lazy dW registration for overlap

    Router: Linear(hidden_size, num_experts, bias=False)
    """

    @staticmethod
    def forward(ctx, input, weight, layer_name, layer_id):
        """
        Forward: input @ weight.T

        Args:
            input: [num_tokens, hidden_size]
            weight: [num_experts, hidden_size]
        Returns:
            logits: [num_tokens, num_experts]
        """
        ctx.save_for_backward(input, weight)
        ctx.layer_name = layer_name
        ctx.layer_id = layer_id

        # Router forward: input @ weight.T
        # [num_tokens, hidden_size] @ [hidden_size, num_experts] = [num_tokens, num_experts]
        logits = torch.matmul(input, weight.t())

        return logits

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: Compute dX immediately, register dW for overlap

        Args:
            grad_output: [num_tokens, num_experts]
        Returns:
            grad_input: [num_tokens, hidden_size]
        """
        input, weight = ctx.saved_tensors

        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        # === CRITICAL PATH: Compute dX immediately ===
        # grad_input = grad_output @ weight
        # [num_tokens, num_experts] @ [num_experts, hidden_size] = [num_tokens, hidden_size]
        grad_input = torch.matmul(grad_output, weight)

        # === LAZY REGISTRATION: Register dW ===
        grad_output_saved = grad_output.detach()
        input_saved = input.detach()

        def compute_dw():
            # grad_weight = grad_output.T @ input
            # [num_experts, num_tokens] @ [num_tokens, hidden_size] = [num_experts, hidden_size]
            grad_weight = torch.matmul(grad_output_saved.t(), input_saved)
            return grad_weight

        scheduler.register_dw_task(
            layer_name=f"{ctx.layer_name}_router_weight",
            layer_id=ctx.layer_id,
            compute_fn=compute_dw,
            priority=0,
            weight_param=weight,
        )

        return grad_input, None, None, None


class FluidRouter(MegatronModule):
    """
    Fluid Router with dW overlap support

    Replaces Megatron's TopKRouter for computation-communication overlap.
    """

    def __init__(self, config: TransformerConfig, pg_collection=None, layer_number=None):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number

        # Router weight: [num_experts, hidden_size]
        self.weight = Parameter(
            torch.empty(
                config.num_moe_experts,
                config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )

        # Initialize weight
        if config.perform_initialization:
            _initialize_affine_weight_cpu(
                self.weight,
                config.num_moe_experts,
                config.hidden_size,
                config.num_moe_experts,
                partition_dim=0,
                init_method=config.init_method,
                params_dtype=config.params_dtype,
            )

    def forward(self, hidden_states: torch.Tensor):
        """
        Router forward with TopK selection

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            probs: [num_tokens, num_experts] - routing probabilities for all experts
            routing_map: [num_tokens, num_experts] - binary mask indicating selected top-k experts
        """
        # Use Fluid autograd function
        logits = _FluidRouterFunc.apply(
            hidden_states,
            self.weight,
            f"layer_{self.layer_number}" if self.layer_number is not None else "router",
            self.layer_number if self.layer_number is not None else 0,
        )

        # Apply softmax to get probabilities for all experts
        # probs: [num_tokens, num_experts]
        probs = F.softmax(logits, dim=-1)

        # TopK selection
        topk = self.config.moe_router_topk
        _, indices = torch.topk(probs, k=topk, dim=-1)

        # Create routing_map: [num_tokens, num_experts] binary mask
        num_tokens = hidden_states.shape[0]
        num_experts = self.config.num_moe_experts
        routing_map = torch.zeros(
            num_tokens, num_experts,
            dtype=torch.int32,
            device=hidden_states.device
        )

        # Set selected experts to 1
        routing_map.scatter_(1, indices, 1)

        return probs, routing_map


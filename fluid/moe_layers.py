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

# Import custom Fluid GroupGEMM kernels
import os
_USE_LOOP_FALLBACK = os.environ.get('USE_LOOP_FALLBACK', '0') == '1'

if _USE_LOOP_FALLBACK:
    FLUID_KERNELS_AVAILABLE = False
    print("[FluidMoE] Using loop-based matmul fallback (USE_LOOP_FALLBACK=1)")
else:
    try:
        from fluid.ops import fluid_kernels
        FLUID_KERNELS_AVAILABLE = True
        print("[FluidMoE] Using custom Fluid GroupGEMM kernels")
    except ImportError:
        FLUID_KERNELS_AVAILABLE = False
        print("[FluidMoE] Fluid kernels not available, using loop-based matmul fallback")


# Activation gradient computation helpers
import math
_GELU_CONST = math.sqrt(2.0 / math.pi)  # sqrt(2/π) ≈ 0.7978845608

def _gelu_grad_analytical(x):
    """
    Analytical GELU gradient (faster than torch.autograd.grad).

    GELU(x) = 0.5 * x * (1 + tanh(k))
    where k = sqrt(2/π) * (x + 0.044715 * x^3)

    GELU'(x) = 0.5 * (1 + tanh(k)) + 0.5 * x * sech²(k) * sqrt(2/π) * (1 + 3*0.044715*x²)
    """
    k = _GELU_CONST * (x + 0.044715 * x * x * x)
    tanh_k = torch.tanh(k)
    sech2_k = 1.0 - tanh_k * tanh_k  # sech²(k) = 1 - tanh²(k)
    dk_dx = _GELU_CONST * (1.0 + 3.0 * 0.044715 * x * x)
    return 0.5 * (1.0 + tanh_k) + 0.5 * x * sech2_k * dk_dx


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
        activation_func_type,  # 'gelu' or 'silu'
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
        ctx.activation_func_type = activation_func_type

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

        if FLUID_KERNELS_AVAILABLE:
            # Use custom Fluid GroupGEMM kernels
            tokens_per_expert_int = tokens_per_expert.to(torch.int32)
            fc1_output = fluid_kernels.grouped_gemm(
                permuted_local_hidden_states.half(), w1.half(),
                tokens_per_expert_int, trans_a=False, trans_b=False
            ).to(permuted_local_hidden_states.dtype)
            intermediate_parallel = activation_func(fc1_output, permuted_probs)
            fc2_output = fluid_kernels.grouped_gemm(
                intermediate_parallel.half(), w2.half(),
                tokens_per_expert_int, trans_a=False, trans_b=False
            ).to(permuted_local_hidden_states.dtype)
        else:
            # Loop fallback
            total_tokens = permuted_local_hidden_states.shape[0]
            ffn_size = w1.shape[2]
            fc1_output = torch.zeros(total_tokens, ffn_size, dtype=permuted_local_hidden_states.dtype, device=permuted_local_hidden_states.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                if n_tok > 0:
                    fc1_output[start:start+n_tok] = torch.matmul(permuted_local_hidden_states[start:start+n_tok], w1[exp_idx])
                    start += n_tok
            intermediate_parallel = activation_func(fc1_output, permuted_probs)
            fc2_output = torch.zeros(total_tokens, hidden_size, dtype=permuted_local_hidden_states.dtype, device=permuted_local_hidden_states.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                if n_tok > 0:
                    fc2_output[start:start+n_tok] = torch.matmul(intermediate_parallel[start:start+n_tok], w2[exp_idx])
                    start += n_tok

        # Pre-compute activation derivative in forward to speed up backward critical path
        # For non-GLU: save gelu'(fc1_output)
        # For GLU: save (gelu'(x_1), gelu(x_1), x_2) for grad_fc1 computation
        if gated_linear_unit:
            x_1, x_2 = torch.chunk(fc1_output, 2, dim=-1)
            if activation_func_type == 'silu':
                sig = torch.sigmoid(x_1)
                act_deriv = sig * (1 + x_1 * (1 - sig))
                act_val = x_1 * sig
            else:
                act_deriv = _gelu_grad_analytical(x_1)
                act_val = F.gelu(x_1)
            ctx.act_deriv = act_deriv  # activation'(x_1)
            ctx.act_val = act_val      # activation(x_1)
            ctx.x_2 = x_2              # x_2 for grad_x_1 = grad * act_deriv * x_2
        else:
            if activation_func_type == 'silu':
                sig = torch.sigmoid(fc1_output)
                ctx.act_deriv = sig * (1 + fc1_output * (1 - sig))
            else:
                ctx.act_deriv = _gelu_grad_analytical(fc1_output)

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
        gated_linear_unit = ctx.gated_linear_unit
        activation_func_type = ctx.activation_func_type

        # Get Fluid scheduler
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        # Reshape weights
        w1 = weight1.view(num_local_experts, hidden_size, -1)
        w2 = weight2.view(num_local_experts, -1, hidden_size)
        ffn_hidden_size = ctx.ffn_hidden_size

        # Retrieve pre-computed activation derivatives from forward
        act_deriv = ctx.act_deriv

        # Ensure probs has correct shape
        probs = permuted_probs.view(-1, 1) if permuted_probs.dim() == 1 else permuted_probs
        if probs.dim() == 1:
            probs = probs.unsqueeze(-1)

        # === CRITICAL PATH: Compute dX immediately ===
        if FLUID_KERNELS_AVAILABLE:
            tokens_per_expert_int = tokens_per_expert.to(torch.int32)
            # grad_intermediate = grad_fc2_output @ w2.T
            grad_intermediate = fluid_kernels.grouped_gemm(
                grad_fc2_output.half(), w2.half(),
                tokens_per_expert_int, trans_a=False, trans_b=True
            ).to(grad_fc2_output.dtype)

            # Compute grad_fc1 using pre-computed activation derivatives (fast!)
            if gated_linear_unit:
                act_val = ctx.act_val
                x_2 = ctx.x_2
                grad_x_1 = grad_intermediate * act_deriv * x_2 * probs
                grad_x_2 = grad_intermediate * act_val * probs
                grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
            else:
                grad_fc1 = grad_intermediate * act_deriv * probs

            # grad_input = grad_fc1 @ w1.T
            grad_permuted_local_hidden_states = fluid_kernels.grouped_gemm(
                grad_fc1.half(), w1.half(),
                tokens_per_expert_int, trans_a=False, trans_b=True
            ).to(grad_fc2_output.dtype)
        else:
            # Loop fallback
            total_tokens = grad_fc2_output.shape[0]
            intermediate_dim = intermediate_parallel.shape[-1]
            grad_intermediate = torch.zeros(total_tokens, intermediate_dim, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                if n_tok > 0:
                    grad_intermediate[start:start+n_tok] = torch.matmul(grad_fc2_output[start:start+n_tok], w2[exp_idx].t())
                    start += n_tok

            # Compute grad_fc1 using pre-computed activation derivatives (fast!)
            if gated_linear_unit:
                act_val = ctx.act_val
                x_2 = ctx.x_2
                grad_x_1 = grad_intermediate * act_deriv * x_2 * probs
                grad_x_2 = grad_intermediate * act_val * probs
                grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
            else:
                grad_fc1 = grad_intermediate * act_deriv * probs

            grad_permuted_local_hidden_states = torch.zeros(total_tokens, hidden_size, dtype=grad_fc2_output.dtype, device=grad_fc2_output.device)
            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx].item()
                if n_tok > 0:
                    grad_permuted_local_hidden_states[start:start+n_tok] = torch.matmul(grad_fc1[start:start+n_tok], w1[exp_idx].t())
                    start += n_tok

        # === LAZY REGISTRATION: Register dW computation ===
        # Detach tensors to avoid holding computation graph
        grad_fc2_output_saved = grad_fc2_output.detach()
        intermediate_parallel_saved = intermediate_parallel.detach()
        permuted_local_hidden_states_saved = permuted_local_hidden_states.detach()
        tokens_per_expert_saved = tokens_per_expert.detach()
        fc1_output_saved = fc1_output.detach()
        permuted_probs_saved = permuted_probs.detach()
        # Cache grad_fc1 directly - no need to recompute in compute_dw_weight1
        grad_fc1_saved = grad_fc1.detach()

        # Define dW computation function for all experts
        def compute_dw_weight2():
            """Compute grad_weight2 for all experts"""
            if FLUID_KERNELS_AVAILABLE:
                # Use custom Fluid GroupGEMM dW kernel
                # grad_w2 = intermediate.T @ grad_output
                # A = intermediate [total_tokens, ffn_hidden_size]
                # B = grad_fc2_output [total_tokens, hidden_size]
                tokens_per_expert_int = tokens_per_expert_saved.to(torch.int32)
                grad_w2_all = fluid_kernels.grouped_gemm_dw(
                    intermediate_parallel_saved.half(),
                    grad_fc2_output_saved.half(),
                    tokens_per_expert_int,
                    ffn_hidden_size,  # M: rows of dW (input dimension)
                    hidden_size       # N: cols of dW (output dimension)
                ).to(weight2.dtype)
                return grad_w2_all.view_as(weight2)
            else:
                # Loop fallback
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
            """Compute grad_weight1 for all experts using cached grad_fc1"""
            # Get actual ffn dimension from saved grad_fc1
            actual_ffn_dim = grad_fc1_saved.shape[-1]

            if FLUID_KERNELS_AVAILABLE:
                # Use custom Fluid GroupGEMM for dW
                # grad_w1 = input.T @ grad_fc1 (grad_fc1 already computed in dX path)
                tokens_per_expert_int = tokens_per_expert_saved.to(torch.int32)
                grad_w1_all = fluid_kernels.grouped_gemm_dw(
                    permuted_local_hidden_states_saved.half(),
                    grad_fc1_saved.half(),
                    tokens_per_expert_int,
                    hidden_size,      # M: rows of dW (input dimension)
                    actual_ffn_dim    # N: cols of dW (output dimension)
                ).to(weight1.dtype)
                return grad_w1_all.view_as(weight1)
            else:
                # Loop fallback
                grad_w1_all = torch.zeros_like(weight1)
                w1_view = grad_w1_all.view(num_local_experts, hidden_size, -1)

                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_per_expert_saved[exp_idx].item()
                    if n_tok == 0:
                        continue
                    end = start + n_tok

                    # grad_w1 = input.T @ grad_fc1 (use cached grad_fc1)
                    grad_w1 = torch.matmul(
                        permuted_local_hidden_states_saved[start:end].t(),
                        grad_fc1_saved[start:end]
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
            None,  # 13: activation_func_type
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
        # Determine activation function type for gradient computation
        if self.config.activation_func == F.silu:
            self.activation_func_type = 'silu'
        else:
            self.activation_func_type = 'gelu'  # default to gelu

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
            self.activation_func_type,  # Pass activation type for gradient computation
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


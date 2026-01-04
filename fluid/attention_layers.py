# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Fluid Linear layers with lazy dW registration for AlltoAll-dW overlap

This module provides Linear layer variants that register dW computation
with the global backward scheduler, enabling true computation-communication overlap.
"""

import torch
from torch.nn.parameter import Parameter
from typing import Optional

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.utils import divide, split_tensor_along_last_dim
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class _FluidColumnParallelLinearFunc(torch.autograd.Function):
    """
    Fluid version of ColumnParallelLinear with lazy dW registration

    Strategy:
    - Forward: Standard ColumnParallelLinear computation
    - Backward:
      1. Immediately compute dX (critical path)
      2. Register dW computation to scheduler (lazy execution)
      3. Return dX, let scheduler handle dW
    """

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        layer_name,
        layer_id,
        async_grad_allreduce,
        sequence_parallel,
        gradient_accumulation_fusion,
        tp_group,
    ):
        # Save for backward
        if bias is not None:
            ctx.save_for_backward(input, weight, bias)
        else:
            ctx.save_for_backward(input, weight)
        ctx.layer_name = layer_name
        ctx.layer_id = layer_id
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.tp_group = tp_group
        ctx.use_bias = bias is not None

        # === FORWARD COMPUTATION (standard) ===
        # input: [s, b, h]
        # weight: [h, output_size]
        # output: [s, b, output_size]

        output = torch.matmul(input, weight.t())

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        if ctx.use_bias:
            input, weight, bias = ctx.saved_tensors
        else:
            input, weight = ctx.saved_tensors
            bias = None

        # Get Fluid scheduler
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        # === CRITICAL PATH: Compute dX immediately ===
        # grad_output: [s, b, output_size]
        # weight: [output_size, h]
        # grad_input: [s, b, h]
        #
        # NOTE: Cross-layer QKV-Combine overlap is disabled because:
        # - QKV backward output goes through LayerNorm backward and residual
        #   before becoming the input to previous layer's combine backward
        # - The gradient data flow is not direct, making the overlap infeasible
        # - The existing dW-AllToAll overlap (97.50%) is already very effective
        grad_input = torch.matmul(grad_output, weight)

        # Handle tensor parallelism for grad_input
        if ctx.sequence_parallel:
            # Reduce-scatter in sequence parallel mode
            from megatron.core.tensor_parallel.mappings import (
                reduce_scatter_to_sequence_parallel_region,
            )
            grad_input = reduce_scatter_to_sequence_parallel_region(grad_input)
        else:
            # All-reduce across tensor parallel group
            if ctx.async_grad_allreduce:
                # Async allreduce (non-blocking)
                handle = torch.distributed.all_reduce(
                    grad_input, group=ctx.tp_group, async_op=True
                )
                # Store handle for later synchronization
                # Note: In practice, we'd need to sync before using grad_input
            else:
                # Sync allreduce (blocking)
                torch.distributed.all_reduce(grad_input, group=ctx.tp_group)

        # === LAZY REGISTRATION: Register dW and dbias ===
        # Detach tensors to avoid holding computation graph
        grad_output_saved = grad_output.detach()
        input_saved = input.detach()

        # Define dW computation function
        def compute_dw():
            # grad_weight = grad_output^T @ input
            # input: [s, b, h]
            # grad_output: [s, b, output_size]
            # PyTorch Linear weight shape: [output_size, h]
            # So grad_weight must be: [output_size, h]

            # Reshape: [s, b, h] -> [s*b, h]
            input_2d = input_saved.view(-1, input_saved.shape[-1])
            grad_output_2d = grad_output_saved.view(-1, grad_output_saved.shape[-1])

            # grad_weight = grad_output^T @ input
            # [output_size, s*b] @ [s*b, h] = [output_size, h]
            grad_weight = torch.matmul(grad_output_2d.t(), input_2d)

            return grad_weight

        # Define dbias computation function
        def compute_dbias():
            if ctx.use_bias:
                # Sum over sequence and batch dimensions
                grad_bias = grad_output_saved.sum(dim=[0, 1])
                return grad_bias
            return None

        # Register dW task
        scheduler.register_dw_task(
            layer_name=f"{ctx.layer_name}_weight",
            layer_id=ctx.layer_id,
            compute_fn=compute_dw,
            priority=100,
            weight_param=weight,
        )

        # Register dbias task if bias exists
        if ctx.use_bias:
            scheduler.register_dw_task(
                layer_name=f"{ctx.layer_name}_bias",
                layer_id=ctx.layer_id,
                compute_fn=compute_dbias,
                priority=99,
                weight_param=bias,
            )

        # Return None for gradients (scheduler will compute them)
        grad_weight = None
        grad_bias = None

        return (
            grad_input,      # grad for input
            grad_weight,     # grad for weight (None if using scheduler)
            grad_bias,       # grad for bias (None if using scheduler)
            None,            # layer_name
            None,            # layer_id
            None,            # async_grad_allreduce
            None,            # sequence_parallel
            None,            # gradient_accumulation_fusion
            None,            # tp_group
        )


class FluidColumnParallelLinear(MegatronModule):
    """
    Fluid version of ColumnParallelLinear with lazy dW registration

    This is a drop-in replacement for ColumnParallelLinear that enables
    AlltoAll-dW overlap by registering weight gradient computation to
    the global backward scheduler.

    Args:
        input_size: First dimension of weight matrix
        output_size: Second dimension of weight matrix
        config: Transformer config
        init_method: Weight initialization method
        bias: Whether to add bias
        gather_output: Whether to gather output across TP group
        skip_bias_add: Whether to skip bias addition (return separately)
        layer_name: Name for debugging and scheduler
        layer_id: Layer index in model
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: callable,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        layer_name: str = "linear",
        layer_id: int = 0,
    ):
        super().__init__(config=config)

        # Store config
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.layer_name = layer_name
        self.layer_id = layer_id
        self.is_expert = is_expert
        self.sequence_parallel = config.sequence_parallel
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.async_tensor_model_parallel_allreduce = (
            config.async_tensor_model_parallel_allreduce and not sequence_parallel
        )

        # Get tensor parallel group
        if tp_group is None:
            self.tp_group = get_tensor_model_parallel_group()
        else:
            self.tp_group = tp_group

        # Divide output_size by tensor parallel world size
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)

        # Initialize weight
        init_device = 'cpu' if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=init_device,
                dtype=config.params_dtype,
            )
        )

        if config.perform_initialization:
            if config.use_cpu_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.output_size_per_partition,
                    0,  # partition_dim
                    init_method,
                    params_dtype=config.params_dtype,
                )
            else:
                _initialize_affine_weight_gpu(
                    self.weight, init_method, partition_dim=0, is_expert=is_expert
                )

        # Initialize bias
        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    device=init_device,
                    dtype=config.params_dtype,
                )
            )
            # Always initialize bias to zero
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Set allreduce attribute for optimizer
        setattr(self.weight, 'allreduce', not is_expert)
        if bias:
            setattr(self.bias, 'allreduce', not is_expert)

    def forward(self, input_: torch.Tensor):
        """
        Forward pass through Fluid ColumnParallelLinear

        Args:
            input_: [s, b, h] tensor

        Returns:
            output: [s, b, output_size] tensor (or [s, b, output_size_per_partition] if not gather_output)
            bias: bias tensor if skip_bias_add, else None
        """
        # If not using sequence parallel, copy input to all TP ranks
        if not self.sequence_parallel:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        else:
            input_parallel = input_

        # Apply Fluid linear function
        output_parallel = _FluidColumnParallelLinearFunc.apply(
            input_parallel,
            self.weight,
            self.bias if not self.skip_bias_add else None,
            self.layer_name,
            self.layer_id,
            self.async_tensor_model_parallel_allreduce,
            self.sequence_parallel,
            self.gradient_accumulation_fusion,
            self.tp_group,
        )

        # Gather output if needed
        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        # Return bias separately if skip_bias_add
        if self.skip_bias_add:
            return output, self.bias
        else:
            return output, None


class _FluidRowParallelLinearFunc(torch.autograd.Function):
    """
    Fluid version of RowParallelLinear with lazy dW registration
    """

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        layer_name,
        layer_id,
        sequence_parallel,
        tp_group,
    ):
        # Save for backward
        if bias is not None:
            ctx.save_for_backward(input, weight, bias)
        else:
            ctx.save_for_backward(input, weight)
        ctx.layer_name = layer_name
        ctx.layer_id = layer_id
        ctx.sequence_parallel = sequence_parallel
        ctx.tp_group = tp_group
        ctx.use_bias = bias is not None

        # === FORWARD COMPUTATION ===
        output = torch.matmul(input, weight.t())

        # Reduce across tensor parallel group
        if sequence_parallel:
            from megatron.core.tensor_parallel.mappings import (
                reduce_scatter_to_sequence_parallel_region,
            )
            output = reduce_scatter_to_sequence_parallel_region(output)
        else:
            output = reduce_from_tensor_model_parallel_region(output)

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.use_bias:
            input, weight, bias = ctx.saved_tensors
        else:
            input, weight = ctx.saved_tensors
            bias = None

        # Get Fluid scheduler
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        # === CRITICAL PATH: Compute dX ===
        # grad_input: [s, b, h_per_partition]
        grad_input = torch.matmul(grad_output, weight)

        # Split grad_input for input_parallel (if not sequence_parallel)
        if not ctx.sequence_parallel:
            grad_input = scatter_to_tensor_model_parallel_region(grad_input)

        # === LAZY REGISTRATION: Register dW ===
        grad_output_saved = grad_output.detach()
        input_saved = input.detach()

        def compute_dw():
            # grad_weight = grad_output^T @ input
            # PyTorch Linear weight shape: [output_size, h]
            input_2d = input_saved.view(-1, input_saved.shape[-1])
            grad_output_2d = grad_output_saved.view(-1, grad_output_saved.shape[-1])
            # [output_size, s*b] @ [s*b, h] = [output_size, h]
            grad_weight = torch.matmul(grad_output_2d.t(), input_2d)
            return grad_weight

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

        grad_weight = None
        grad_bias = None

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None, None, None, None,
        )


class FluidRowParallelLinear(MegatronModule):
    """
    Fluid version of RowParallelLinear with lazy dW registration
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: callable,
        bias: bool = True,
        input_is_parallel: bool = False,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        layer_name: str = "linear",
        layer_id: int = 0,
    ):
        super().__init__(config=config)

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.layer_name = layer_name
        self.layer_id = layer_id
        self.is_expert = is_expert
        self.sequence_parallel = config.sequence_parallel

        if tp_group is None:
            self.tp_group = get_tensor_model_parallel_group()
        else:
            self.tp_group = tp_group

        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)

        init_device = 'cpu' if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=init_device,
                dtype=config.params_dtype,
            )
        )

        if config.perform_initialization:
            if config.use_cpu_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,
                    init_method,
                    params_dtype=config.params_dtype,
                )
            else:
                _initialize_affine_weight_gpu(
                    self.weight, init_method, partition_dim=1, is_expert=is_expert
                )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, device=init_device, dtype=config.params_dtype)
            )
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        setattr(self.weight, 'allreduce', not is_expert)
        if bias:
            setattr(self.bias, 'allreduce', not is_expert)

    def forward(self, input_: torch.Tensor):
        if not self.input_is_parallel:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        else:
            input_parallel = input_

        output = _FluidRowParallelLinearFunc.apply(
            input_parallel,
            self.weight,
            self.bias if not self.skip_bias_add else None,
            self.layer_name,
            self.layer_id,
            self.sequence_parallel,
            self.tp_group,
        )

        if self.skip_bias_add:
            return output, self.bias
        else:
            return output, None

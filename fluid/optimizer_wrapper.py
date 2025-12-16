"""
FluidMoE Optimizer Component

This module provides a custom optimizer wrapper that integrates with
the FluidMoE scheduler to ensure all dW tasks are completed before
parameter updates.

Design Philosophy:
- Like FluidSelfAttention and FluidMoELayer, this is a pluggable component
- Users can easily swap between Fluid optimizer and standard optimizer
- Provides a clean API: get_fluid_optimizer()
"""

from fluid.scheduler import get_backward_scheduler


class FluidOptimizerWrapper:
    """
    Wrapper around Megatron optimizer that calls scheduler.finish_batch()
    before optimizer.step() to ensure all dW tasks are completed.

    Usage:
        optimizer = get_megatron_optimizer(...)
        optimizer = FluidOptimizerWrapper(optimizer)
    """

    def __init__(self, optimizer):
        """
        Wrap a Megatron optimizer

        Args:
            optimizer: The original Megatron optimizer instance
        """
        self.optimizer = optimizer
        self.scheduler = get_backward_scheduler()

    def step(self, *args, **kwargs):
        """
        Execute optimizer step with automatic dW completion.

        This ensures all pending dW tasks are completed before
        updating parameters.
        """
        # Complete all remaining dW tasks before optimizer step
        if self.scheduler.is_enabled():
            self.scheduler.finish_batch()

        # Call original optimizer step
        return self.optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """Forward zero_grad to original optimizer"""
        return self.optimizer.zero_grad(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Forward state_dict to original optimizer"""
        return self.optimizer.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Forward load_state_dict to original optimizer"""
        return self.optimizer.load_state_dict(*args, **kwargs)

    def __getattr__(self, name):
        """Forward all other attributes to original optimizer"""
        return getattr(self.optimizer, name)

    def __repr__(self):
        return f"FluidOptimizerWrapper({self.optimizer})"


def get_fluid_optimizer(config, model, **optimizer_kwargs):
    """
    Get FluidMoE optimizer (wrapper around Megatron optimizer).

    This is the primary API for creating a Fluid-compatible optimizer,
    similar to get_fluid_custom_layers() for model layers.

    Args:
        config: Megatron OptimizerConfig or args
        model: The model to optimize
        **optimizer_kwargs: Additional arguments passed to get_megatron_optimizer

    Returns:
        FluidOptimizerWrapper: Wrapped optimizer with automatic dW completion

    Example:
        from fluid import get_fluid_optimizer
        from megatron.core.optimizer import get_megatron_optimizer

        # Standard Megatron way
        optimizer = get_megatron_optimizer(config, model)

        # Fluid way (with automatic dW completion)
        optimizer = get_fluid_optimizer(config, model)
    """
    from megatron.core.optimizer import get_megatron_optimizer

    # Create base Megatron optimizer
    base_optimizer = get_megatron_optimizer(config, model, **optimizer_kwargs)

    # Wrap with Fluid optimizer
    fluid_optimizer = FluidOptimizerWrapper(base_optimizer)

    print("[FluidMoE] Created FluidOptimizerWrapper (automatic dW completion enabled)")

    return fluid_optimizer


def wrap_optimizer(optimizer):
    """
    Wrap an existing optimizer with FluidOptimizerWrapper.

    Useful when you already have an optimizer instance and want to
    add Fluid's automatic dW completion.

    Args:
        optimizer: Existing optimizer instance

    Returns:
        FluidOptimizerWrapper: Wrapped optimizer

    Example:
        optimizer = get_megatron_optimizer(...)
        optimizer = wrap_optimizer(optimizer)
    """
    if isinstance(optimizer, FluidOptimizerWrapper):
        print("[FluidMoE] Optimizer already wrapped, returning as-is")
        return optimizer

    wrapped = FluidOptimizerWrapper(optimizer)
    print("[FluidMoE] Wrapped existing optimizer with FluidOptimizerWrapper")
    return wrapped

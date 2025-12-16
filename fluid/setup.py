"""
FluidMoE Setup Functions

Custom setup functions that replace Megatron's setup with Fluid components.
No monkey patching - fully custom implementation.
"""

from fluid import wrap_optimizer


def setup_model_and_optimizer(
    model_provider_func,
    model_type,
    checkpointing_context=None,
):
    """
    Custom setup_model_and_optimizer for FluidMoE.

    This is a complete custom implementation that:
    1. Calls Megatron's functions to create model and optimizer
    2. Wraps optimizer with FluidOptimizerWrapper
    3. Returns the wrapped components

    No monkey patching - just calls existing functions and wraps the result.
    """
    # Import Megatron functions
    from megatron.training import get_args
    from megatron.training.training import (
        get_model,
        unwrap_model,
        get_megatron_optimizer_config,
        get_optimizer_param_scheduler,
    )
    from megatron.core.optimizer import get_megatron_optimizer

    args = get_args()

    # 1. Create model (standard Megatron way)
    model = get_model(model_provider_func, model_type)

    # 2. Create optimizer (standard Megatron way)
    config, config_overrides = get_megatron_optimizer_config(args)

    optimizer = get_megatron_optimizer(
        config,
        model,
        config_overrides=config_overrides,
        use_gloo_process_groups=getattr(args, 'enable_gloo_process_groups', False),
        dump_param_to_param_group_map=getattr(args, 'dump_param_to_param_group_map', False),
    )

    # 3. Wrap optimizer with Fluid component
    optimizer = wrap_optimizer(optimizer)

    # 4. Create learning rate scheduler
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    return model, optimizer, opt_param_scheduler

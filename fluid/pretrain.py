"""
FluidMoE Pretrain - Generic Training Entry Point

This module provides a generic pretrain function for FluidMoE that works with
any model type. It handles the integration of Fluid components (custom layers
and optimizer wrapper) with Megatron's training infrastructure.

Key Features:
- Automatic optimizer wrapping with FluidOptimizerWrapper
- Works with any model architecture (GPT, BERT, T5, etc.)
- Fully custom setup function - no monkey patching at all
- Compatible with all Megatron features

Usage:
    from fluid.pretrain import pretrain
    from fluid import get_fluid_custom_layers

    def model_provider(...):
        layer_spec = get_fluid_custom_layers()
        model = YourModel(..., transformer_layer_spec=layer_spec)
        return model

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step_func=forward_step_func,
    )
"""

from megatron.training import training as megatron_training
from megatron.training import (
    initialize_megatron,
    get_args,
    get_timers,
    print_rank_0,
)
from megatron.training.initialize import set_jit_fusion_options
from fluid.setup import setup_model_and_optimizer as fluid_setup_model_and_optimizer


def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults=None,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
    store=None,
    inprocess_call_wrapper=None,
):
    """
    FluidMoE pretrain function - Generic training entry point.

    Fully custom implementation that uses Fluid's setup_model_and_optimizer
    instead of Megatron's. No monkey patching - completely custom flow.

    Args:
        Same as megatron.training.pretrain()

    Example:
        from fluid.pretrain import pretrain
        from megatron.core.enums import ModelType

        pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step_func=forward_step_func,
        )
    """
    if args_defaults is None:
        args_defaults = {}

    # 1. Initialize Megatron (standard)
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        store=store,
    )

    args = get_args()

    # Set default attributes if not set
    if not hasattr(args, 'iteration'):
        args.iteration = 0
    if not hasattr(args, 'num_floating_point_operations_so_far'):
        args.num_floating_point_operations_so_far = 0

    # 2. Set JIT fusion options (standard)
    set_jit_fusion_options()

    print_rank_0('> FluidMoE: Building model and optimizer with Fluid components ...')

    # 3. Setup model and optimizer (Fluid custom version)
    model, optimizer, opt_param_scheduler = fluid_setup_model_and_optimizer(
        model_provider,
        model_type,
    )

    print_rank_0('> FluidMoE: Model and optimizer built successfully!')

    # 4. Get model config (for train function)
    from megatron.training.training import get_model_config
    config = get_model_config(model[0])

    # 5. Get training datasets
    train_data_iterator, valid_data_iterator, test_data_iterator = None, None, None
    if train_valid_test_dataset_provider is not None:
        from megatron.training.training import build_train_valid_test_data_iterators
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        )

    # 6. Call Megatron's train function (standard)
    megatron_training.train(
        forward_step_func=forward_step_func,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        train_data_iterator=train_data_iterator,
        valid_data_iterator=valid_data_iterator,
        process_non_loss_data_func=process_non_loss_data_func,
        config=config,
        checkpointing_context={},  # Empty dict is fine
        non_loss_data_func=non_loss_data_func,
    )

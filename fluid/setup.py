"""
FluidMoE Setup Functions

Drop-in replacement for Megatron's setup_model_and_optimizer.

Gradient management is split between two systems:
  - Decoder params (MoE weights, attention, layernorm):
      Managed by FluidMoE's BackwardScheduler (dW deferred, AR on comm_stream)
  - Model-level params (embedding, output_layer, final_layernorm):
      Managed by a grad buffer with backward hooks (like Megatron DDP),
      AR done synchronously in finish_grad_sync()
"""

import torch
import torch.distributed as dist

from fluid.core import get_backward_scheduler


# ============================================================================
# FluidDDP: hybrid DDP for FluidMoE
# ============================================================================

class FluidDDP(torch.nn.Module):
    """Hybrid DDP: scheduler manages decoder params, grad buffer manages model-level params.

    Decoder params (MoE, attention, LN inside TransformerLayerFunction):
      - dW deferred to AlltoAll wait windows by scheduler
      - AR submitted on comm_stream after AlltoAll
      - finish_batch() completes remaining tasks

    Model-level params (embedding, output_layer, final_layernorm):
      - dW computed by autograd during backward (standard)
      - Gradients accumulated into contiguous grad buffer via hooks
      - AR done in finish_grad_sync() (pre-scale + SUM = AVG)
    """

    def __init__(self, config, module, scheduler, *,
                 dp_cp_group=None, expert_dp_group=None):
        """
        Args:
            config: Megatron TransformerConfig (kept for API compat; unused
                internally). Pass ``None`` for block-test usage.
            module: model to wrap. Either a Megatron-wrapped GPTModel
                (with ``decoder.layers``) or a standalone
                ``fluid.layer.TransformerModel`` (with ``.layers``).
            scheduler: FluidMoE BackwardScheduler instance.
            dp_cp_group: explicit data+context parallel process group. If
                ``None``, falls back to
                ``parallel_state.get_data_parallel_group(with_context_parallel=True)``.
                Pass explicitly when running outside Megatron's parallel_state
                (e.g. block-level benchmarks that build groups manually).
            expert_dp_group: explicit expert-data parallel group. ``None``
                falls back to ``parallel_state.get_data_modulo_expert_parallel_group()``.

        AR tuning values (gap budgets, bandwidth) are read from ``FLUIDMOE_*``
        env vars (set by callers like ``run_training.sh`` or block-bench
        scripts before constructing this wrapper).
        """
        super().__init__()
        self.module = module  # Megatron Float16Module(GPTModel) or standalone TransformerModel
        self.config = config
        from megatron.core.distributed import DistributedDataParallelConfig
        self.ddp_config = DistributedDataParallelConfig()
        self._scheduler = scheduler
        self._hooks = []
        self._explicit_dp_cp_group = dp_cp_group
        self._explicit_expert_dp_group = expert_dp_group
        self._setup_grad_buffer()
        self._configure_scheduler()

    def _get_raw_model(self):
        """Navigate through Float16Module to get the underlying model."""
        raw = self.module
        if hasattr(raw, 'module'):
            raw = raw.module
        return raw

    def _collect_fluid_layers(self):
        """Return list of inner FluidMoE TransformerLayer instances.

        Two supported layouts (block test vs production training):
          1. Megatron GPTModel with FluidTransformerLayer adapter:
             ``module.decoder.layers[i].layer`` is the FluidMoE TransformerLayer.
          2. Standalone ``fluid.layer.TransformerModel`` (block-test path):
             ``module.layers[i]`` is the FluidMoE TransformerLayer directly.
        A layer is recognized by having ``_get_qkv_weight`` (method unique to it).
        """
        raw = self._get_raw_model()
        if hasattr(raw, 'decoder') and hasattr(raw.decoder, 'layers'):
            layers_iter = raw.decoder.layers
        elif hasattr(raw, 'layers'):
            layers_iter = raw.layers
        else:
            return []
        fluid_layers = []
        for layer in layers_iter:
            if hasattr(layer, '_get_qkv_weight'):
                fluid_layers.append(layer)
            elif hasattr(layer, 'layer') and hasattr(layer.layer, '_get_qkv_weight'):
                fluid_layers.append(layer.layer)
        return fluid_layers

    def _setup_grad_buffer(self):
        """Allocate contiguous grad buffer for model-level params and register hooks.

        When the model has no params outside FluidMoE TransformerLayers (block-test
        case), buffer/hooks are skipped and ``self._grad_buffer`` is ``None``.
        """
        # Params managed by the scheduler: everything inside FluidMoE TransformerLayers.
        scheduler_param_ids = set()
        for fluid_layer in self._collect_fluid_layers():
            for p in fluid_layer.parameters():
                scheduler_param_ids.add(id(p))

        # Model-level params: everything NOT owned by a FluidMoE TransformerLayer
        # (embedding, final_layernorm, output_layer, etc.)
        self._buffer_params = [
            p for p in self.module.parameters() if id(p) not in scheduler_param_ids
        ]

        if not self._buffer_params:
            self._grad_buffer = None
            self._dp_cp_group = None
            return

        # DP+CP group for allreduce: explicit takes precedence over parallel_state.
        if self._explicit_dp_cp_group is not None:
            dp_cp_group = self._explicit_dp_cp_group
        else:
            from megatron.core import parallel_state as ps
            dp_cp_group = ps.get_data_parallel_group(with_context_parallel=True)
        dp_cp_size = dist.get_world_size(dp_cp_group)
        self._dp_cp_group = dp_cp_group if dp_cp_size > 1 else None
        self._dp_cp_size = dp_cp_size

        # Allocate contiguous grad buffer in param dtype so communication buffer
        # and main_grad follow the official Megatron DDP behavior when
        # grad_reduce_in_fp32=False.
        self._buffer_dtype = self._buffer_params[0].dtype
        total_numel = sum(p.numel() for p in self._buffer_params)
        self._grad_buffer = torch.zeros(total_numel, dtype=self._buffer_dtype,
                                        device=torch.cuda.current_device())

        # Map each param to a slice of the buffer and set main_grad
        offset = 0
        self._param_offsets = {}
        for p in self._buffer_params:
            numel = p.numel()
            self._param_offsets[p] = (offset, numel)
            # main_grad: optimizer reads from here (contiguous buffer in grad dtype)
            p.main_grad = self._grad_buffer[offset:offset + numel].view(p.shape)
            offset += numel

        # Register backward hooks to accumulate grads into buffer
        for p in self._buffer_params:
            hook = p.register_post_accumulate_grad_hook(self._grad_hook)
            self._hooks.append(hook)

    def _grad_hook(self, param):
        """Copy param.grad into the contiguous grad buffer."""
        if param.grad is not None:
            off, numel = self._param_offsets[param]
            self._grad_buffer[off:off + numel].add_(param.grad.view(-1).to(self._buffer_dtype))
            param.grad = None  # Free the separate grad tensor

    def _configure_scheduler(self):
        """Configure scheduler's allreduce groups and AR buffers for decoder params."""
        # Resolve groups: explicit kwargs take precedence over parallel_state.
        if self._explicit_dp_cp_group is not None:
            dp_cp_group = self._explicit_dp_cp_group
            dp_cp_size = dist.get_world_size(dp_cp_group)
        else:
            from megatron.core import parallel_state as ps
            dp_cp_group = ps.get_data_parallel_group(with_context_parallel=True)
            dp_cp_size = dist.get_world_size(dp_cp_group)

        if self._explicit_expert_dp_group is not None:
            expert_dp_group = self._explicit_expert_dp_group
            expert_dp_size = (dist.get_world_size(expert_dp_group)
                              if expert_dp_group is not None else 1)
        else:
            from megatron.core import parallel_state as ps
            try:
                expert_dp_group = ps.get_data_modulo_expert_parallel_group()
                expert_dp_size = dist.get_world_size(expert_dp_group) if expert_dp_group else 1
            except Exception:
                expert_dp_group = None
                expert_dp_size = 1

        # AR bandwidth and gap budgets from env (callers set FLUIDMOE_* vars).
        import os as _os
        _env_float = lambda key, default: float(_os.environ.get(key, str(default)))
        gap_budgets = {}
        for region, env_key in [('moe_combine', 'FLUIDMOE_GAP_MOE_COMBINE'),
                                ('moe_dispatch', 'FLUIDMOE_GAP_MOE_DISPATCH'),
                                ('attn_proj', 'FLUIDMOE_GAP_ATTN_PROJ'),
                                ('attn_qkv', 'FLUIDMOE_GAP_ATTN_QKV')]:
            g = _env_float(env_key, 0.0)
            if g > 0:
                gap_budgets[region] = g

        self._scheduler.configure_allreduce(
            enabled=True,
            shared_dp_group=dp_cp_group if dp_cp_size > 1 else None,
            expert_dp_group=expert_dp_group if expert_dp_size > 1 else None,
            shared_ar_bw=_env_float('FLUIDMOE_SHARED_AR_BW', 0.0),
            expert_ar_bw=_env_float('FLUIDMOE_EXPERT_AR_BW', 0.0),
            gap_budgets=gap_budgets if gap_budgets else None,
        )

        # Setup AR buffers for FluidMoE TransformerLayer params.
        fluid_layers = self._collect_fluid_layers()
        if fluid_layers:
            shared_params = []
            expert_params = []
            for layer in reversed(fluid_layers):
                expert_params.extend([layer.moe_w2, layer.moe_w1])
                shared_params.extend([
                    layer.router_weight,
                    layer.ln2_weight, layer.ln2_bias,
                    layer._get_proj_weight(),
                    layer._get_qkv_weight(),
                    layer.ln1_weight, layer.ln1_bias,
                ])
            self._scheduler.setup_ar_buffer(shared_params)
            if expert_params:
                self._scheduler.setup_expert_ar_buffer(expert_params)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def zero_grad_buffer(self):
        """Clear all gradients. Called at start of each iteration."""
        # Clear decoder params' grad and AR buffer write flags
        for p in self.module.parameters():
            p.grad = None
            if hasattr(p, '_ar_buf_written'):
                p._ar_buf_written = False
        # Zero the contiguous grad buffer for model-level params
        if self._grad_buffer is not None:
            self._grad_buffer.zero_()

    def finish_grad_sync(self, force_all_reduce: bool = False):
        """Complete all gradient computation and synchronization.

        1. scheduler.finish_batch(): decoder params dW + AR
        2. Grad buffer allreduce: model-level params AR
        """
        # Decoder params: scheduler completes remaining dW tasks + AR
        self._scheduler.finish_batch()

        # Model-level params: allreduce the grad buffer
        if self._dp_cp_group is not None and self._grad_buffer is not None:
            # Pre-scale by 1/dp_cp_size then SUM = AVG (matching Megatron DDP)
            self._grad_buffer.div_(self._dp_cp_size)
            dist.all_reduce(self._grad_buffer, group=self._dp_cp_group)

    def scale_gradients(self, scaling_factor):
        """Scale all gradients by scaling_factor (per-token loss normalization).

        Operates on contiguous flat buffers (3 kernel launches) instead of
        per-parameter iteration (500+ launches for many-expert models).
        """
        # Decoder shared params (router, ln, qkv, proj) — flat AR buffer
        if self._scheduler._shared_ar.fp32 is not None:
            self._scheduler._shared_ar.fp32.mul_(scaling_factor)
        if self._scheduler._shared_ar.bf16 is not None:
            self._scheduler._shared_ar.bf16.mul_(scaling_factor)
        # Decoder expert params (moe_w1, moe_w2) — flat AR buffer
        if self._scheduler._expert_ar.fp32 is not None:
            self._scheduler._expert_ar.fp32.mul_(scaling_factor)
        if self._scheduler._expert_ar.bf16 is not None:
            self._scheduler._expert_ar.bf16.mul_(scaling_factor)
        # Model-level params (embedding, output, final_ln) — grad buffer
        if self._grad_buffer is not None:
            self._grad_buffer.mul_(scaling_factor)

    def broadcast_params(self):
        """Broadcast parameters from rank 0."""
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.module.load_state_dict(state_dict, *args, **kwargs)

    def sharded_state_dict(self, *args, **kwargs):
        if hasattr(self.module, 'sharded_state_dict'):
            return self.module.sharded_state_dict(*args, **kwargs)
        return self.state_dict()


# ============================================================================
# FluidOptimizerWrapper
# ============================================================================

class FluidOptimizerWrapper:
    """Wraps Megatron's optimizer. Calls clear_iteration() after each step."""

    def __init__(self, optimizer, scheduler):
        self._optimizer = optimizer
        self._scheduler = scheduler

    def step(self, *args, **kwargs):
        result = self._optimizer.step(*args, **kwargs)
        self._scheduler.clear_iteration()
        return result

    def zero_grad(self, *args, **kwargs):
        return self._optimizer.zero_grad(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._optimizer, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._optimizer, name, value)

    @property
    def scheduler(self):
        return self._scheduler


# ============================================================================
# setup_model_and_optimizer: drop-in replacement for Megatron's version
# ============================================================================

def setup_model_and_optimizer(
    model_provider_func,
    model_type,
    checkpointing_context=None,
):
    """
    Replaces Megatron's setup_model_and_optimizer.

    Uses FluidDDP which splits gradient management:
      - Decoder params: scheduler (dW overlap + AR on comm_stream)
      - Model-level params: grad buffer + hooks (AR in finish_grad_sync)
    """
    from megatron.training import get_args
    from megatron.training.training import (
        get_model_config,
        get_megatron_optimizer_config,
        get_optimizer_param_scheduler,
    )
    from megatron.core import tensor_parallel
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.transformer.module import Float16Module
    from megatron.training.checkpointing import load_checkpoint

    args = get_args()

    # 1. Create raw model
    model = model_provider_func(pre_process=True, post_process=True)
    model.cuda(torch.cuda.current_device())

    # Match Megatron's standard build path so optimizer / grad-norm utilities
    # see the default TP metadata on every parameter.
    for param in model.parameters():
        tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # 2. Float16Module wrapping
    config = get_model_config(model)
    if args.fp16 or args.bf16:
        model = Float16Module(config, model)

    # 3. FluidDDP wrapping
    scheduler = get_backward_scheduler()
    scheduler.enable()
    model = FluidDDP(config, model, scheduler)
    model = [model]

    # 4. Create Megatron optimizer
    opt_config, config_overrides = get_megatron_optimizer_config(args)
    try:
        from megatron.training import get_timers
        opt_config.timers = get_timers()
    except (ImportError, AttributeError):
        pass

    optimizer = get_megatron_optimizer(
        opt_config,
        model,
        config_overrides=config_overrides,
        use_gloo_process_groups=getattr(args, 'enable_gloo_process_groups', False),
        dump_param_to_param_group_map=getattr(args, 'dump_param_to_param_group_map', False),
    )

    # 5. LR scheduler
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    # 6. Load checkpoint
    if (
        getattr(args, 'load', None) is not None
        or getattr(args, 'pretrained_checkpoint', None) is not None
    ):
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler,
            checkpointing_context=checkpointing_context,
        )
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    # 7. Wrap optimizer
    optimizer = FluidOptimizerWrapper(optimizer, scheduler)

    return model, optimizer, opt_param_scheduler

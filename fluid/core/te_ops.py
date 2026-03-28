"""
TransformerEngine functional wrappers for FluidMoE.

Provides TE-accelerated versions of LayerNorm, activation functions,
and high-level TE modules (DotProductAttention, Linear).
Falls back to PyTorch native ops if TE is not available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

try:
    from transformer_engine.pytorch.cpp_extensions import (
        layernorm_fwd as _te_layernorm_fwd,
        layernorm_bwd as _te_layernorm_bwd,
        gelu as _te_gelu,
        dgelu as _te_dgelu,
        silu as _te_silu,
        dsilu as _te_dsilu,
    )
    from transformer_engine_torch import DType as _te_DType

    _DTYPE_TO_TE = {
        torch.float32: _te_DType.kFloat32,
        torch.float16: _te_DType.kFloat16,
        torch.bfloat16: _te_DType.kBFloat16,
    }
    _HAS_TE = True
except ImportError:
    _HAS_TE = False

# TE high-level modules (DotProductAttention, Linear)
try:
    import transformer_engine.pytorch as _te_pytorch
    _HAS_TE_MODULES = True
except ImportError:
    _te_pytorch = None
    _HAS_TE_MODULES = False


def te_layernorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused LayerNorm forward using TE kernel.

    Args:
        input:  [..., hidden_size]
        weight: [hidden_size]
        bias:   [hidden_size]
        eps:    epsilon

    Returns:
        Normalized output, same shape as input.
    """
    if _HAS_TE and input.is_cuda:
        orig_shape = input.shape
        inp_2d = input.reshape(-1, orig_shape[-1])
        te_dtype = _DTYPE_TO_TE[input.dtype]
        out, _mu, _rsigma = _te_layernorm_fwd(
            inp_2d, weight, bias, eps,
            None,   # ln_out (pre-allocated output, None = allocate)
            None,   # quantizer
            te_dtype,
            0,      # sm_margin
            False,  # zero_centered_gamma
        )
        return out.reshape(orig_shape)
    return F.layer_norm(input, (input.shape[-1],), weight, bias, eps)


def te_layernorm_fwd_with_stats(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    """Fused LayerNorm forward that also returns mean and rsigma for backward.

    Returns:
        (output, mu, rsigma) where mu=[N], rsigma=[N], N=prod(shape[:-1]).
    """
    if _HAS_TE and input.is_cuda:
        orig_shape = input.shape
        inp_2d = input.reshape(-1, orig_shape[-1])
        te_dtype = _DTYPE_TO_TE[input.dtype]
        out, mu, rsigma = _te_layernorm_fwd(
            inp_2d, weight, bias, eps,
            None, None, te_dtype, 0, False,
        )
        return out.reshape(orig_shape), mu, rsigma
    # Fallback
    out = F.layer_norm(input, (input.shape[-1],), weight, bias, eps)
    inp_2d = input.reshape(-1, input.shape[-1]).float()
    mu = inp_2d.mean(dim=-1)
    rsigma = torch.rsqrt(inp_2d.var(dim=-1, unbiased=False) + eps)
    return out, mu, rsigma


def te_layernorm_bwd(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    mu: torch.Tensor,
    rsigma: torch.Tensor,
    weight: torch.Tensor,
):
    """Fused LayerNorm backward using TE kernel.

    Args:
        grad_output: [..., hidden_size] upstream gradient
        input:       [..., hidden_size] original forward input
        mu:          [N] mean from forward (N = prod(shape[:-1]))
        rsigma:      [N] reciprocal std from forward
        weight:      [hidden_size] LN weight

    Returns:
        (dx, dw, db) — dx same shape as input, dw/db shape [hidden_size].
    """
    if _HAS_TE and grad_output.is_cuda:
        orig_shape = grad_output.shape
        dy_2d = grad_output.reshape(-1, orig_shape[-1])
        inp_2d = input.reshape(-1, orig_shape[-1])
        dx, dw, db = _te_layernorm_bwd(
            dy_2d, inp_2d, mu, rsigma, weight,
            0,      # sm_margin
            False,  # zero_centered_gamma
        )
        return dx.reshape(orig_shape), dw, db
    # Fallback: PyTorch autograd
    with torch.enable_grad():
        x = input.detach().float().requires_grad_(True)
        w = weight.detach().float().requires_grad_(True)
        b = torch.zeros_like(w).requires_grad_(True)
        out = F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)
        dx, dw, db = torch.autograd.grad(out, (x, w, b), grad_output.float())
    return dx.to(input.dtype), dw.to(weight.dtype), db.to(weight.dtype)


def te_gelu(input: torch.Tensor) -> torch.Tensor:
    """GeLU activation using TE kernel."""
    if _HAS_TE and input.is_cuda:
        return _te_gelu(input, None)
    return F.gelu(input)


def te_silu(input: torch.Tensor) -> torch.Tensor:
    """SiLU activation using TE kernel."""
    if _HAS_TE and input.is_cuda:
        return _te_silu(input, None)
    return F.silu(input)


def te_dgelu(grad: torch.Tensor, fwd_input: torch.Tensor) -> torch.Tensor:
    """GeLU backward using TE kernel.

    Args:
        grad:      gradient from upstream
        fwd_input: original input to GeLU (before activation)

    Returns:
        Gradient w.r.t. input.
    """
    if _HAS_TE and grad.is_cuda:
        return _te_dgelu(grad, fwd_input, None)
    # Fallback: manual exact GELU gradient
    import math
    _SQRT_2 = math.sqrt(2.0)
    _INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(fwd_input / _SQRT_2))
    pdf = torch.exp(-0.5 * fwd_input * fwd_input) * _INV_SQRT_2PI
    return grad * (cdf + fwd_input * pdf)


def te_dsilu(grad: torch.Tensor, fwd_input: torch.Tensor) -> torch.Tensor:
    """SiLU backward using TE kernel.

    Args:
        grad:      gradient from upstream
        fwd_input: original input to SiLU (before activation)

    Returns:
        Gradient w.r.t. input.
    """
    if _HAS_TE and grad.is_cuda:
        return _te_dsilu(grad, fwd_input, None)
    # Fallback: manual SiLU gradient
    sig = torch.sigmoid(fwd_input)
    return grad * sig * (1 + fwd_input * (1 - sig))


def create_te_dpa(
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
    layer_number: Optional[int] = None,
) -> Optional[nn.Module]:
    """Create TE DotProductAttention module.

    Args:
        num_heads:    number of Q heads (local, after CP split)
        head_dim:     dimension per head
        num_kv_heads: number of KV heads (local); None = same as num_heads
        layer_number: optional layer index

    Returns:
        te.DotProductAttention instance, or None if TE unavailable.
    """
    if not _HAS_TE_MODULES:
        return None
    kwargs = dict(
        num_attention_heads=num_heads,
        kv_channels=head_dim,
        attention_dropout=0.0,
        qkv_format="sbhd",
        attn_mask_type="causal",
    )
    if num_kv_heads is not None and num_kv_heads != num_heads:
        kwargs["num_gqa_groups"] = num_kv_heads
    if layer_number is not None:
        kwargs["layer_number"] = layer_number
    return _te_pytorch.DotProductAttention(**kwargs)


def create_te_linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
    params_dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    init_method: Optional[Callable] = None,
    get_rng_state_tracker: Optional[Callable] = None,
    parallel_mode: Optional[str] = None,
) -> Optional[nn.Module]:
    """Create TE Linear module.

    Args:
        in_features:  input dimension
        out_features: output dimension
        bias:         include bias (default False)
        params_dtype: parameter dtype
        device:       parameter device
        init_method:  weight init callable
        get_rng_state_tracker: Megatron RNG tracker callable for deterministic init
        parallel_mode: TE parallel mode ("column", "row", or None).
                       Must match Megatron's TELinear to ensure identical init path.

    Returns:
        te.Linear instance, or None if TE unavailable.
    """
    if not _HAS_TE_MODULES:
        return None
    kwargs = dict(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        tp_size=1,   # Match Megatron TELinear (TP=1)
        tp_group=None,
    )
    if parallel_mode is not None:
        kwargs["parallel_mode"] = parallel_mode
    if params_dtype is not None:
        kwargs["params_dtype"] = params_dtype
    if device is not None:
        kwargs["device"] = str(device) if isinstance(device, torch.device) else device
    if init_method is not None:
        kwargs["init_method"] = init_method
    if get_rng_state_tracker is not None:
        kwargs["get_rng_state_tracker"] = get_rng_state_tracker
    return _te_pytorch.Linear(**kwargs)


__all__ = [
    'te_layernorm_fwd', 'te_layernorm_fwd_with_stats', 'te_layernorm_bwd',
    'te_gelu', 'te_silu',
    'te_dgelu', 'te_dsilu',
    '_HAS_TE', '_HAS_TE_MODULES',
    'create_te_dpa', 'create_te_linear',
]

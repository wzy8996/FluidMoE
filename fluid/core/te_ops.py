"""
TransformerEngine functional wrappers for FluidMoE.

Provides TE-accelerated versions of LayerNorm and activation functions.
Falls back to PyTorch native ops if TE is not available.
"""

import torch
import torch.nn.functional as F

try:
    from transformer_engine.pytorch.cpp_extensions import (
        layernorm_fwd as _te_layernorm_fwd,
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


__all__ = [
    'te_layernorm_fwd',
    'te_gelu', 'te_silu',
    'te_dgelu', 'te_dsilu',
    '_HAS_TE',
]

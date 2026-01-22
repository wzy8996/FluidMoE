"""
Common Utility Functions for FluidMoE

Contains:
- Activation function derivatives (GELU, SiLU)
- Chunk size optimization
- Other common utilities
"""

import os
import math
import torch
import torch.nn.functional as F


# Constants for GELU computation
_SQRT_2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _gelu_grad_exact(x: torch.Tensor) -> torch.Tensor:
    """
    Exact GELU gradient using error function.

    GELU(x) = x * Phi(x) where Phi is CDF of standard normal
    GELU'(x) = Phi(x) + x * phi(x) where phi is PDF of standard normal

    Args:
        x: Input tensor

    Returns:
        Gradient of GELU at x
    """
    cdf = 0.5 * (1.0 + torch.erf(x / _SQRT_2))
    pdf = torch.exp(-0.5 * x * x) * _INV_SQRT_2PI
    return cdf + x * pdf


def _compute_activation_derivative(fc1_output, activation_func, gated_linear_unit=False):
    """
    Pre-compute activation function derivative for backward pass.

    This allows reusing the derivative computation across multiple operations
    in the backward pass, avoiding redundant computation.

    Args:
        fc1_output: Output of FC1 layer [tokens, ffn_hidden]
        activation_func: Activation function (F.gelu or F.silu)
        gated_linear_unit: Whether using GLU (SwiGLU/GeGLU)

    Returns:
        If gated_linear_unit:
            (act_deriv, act_val, x_2) tuple
        Else:
            act_deriv tensor
    """
    fc1_detached = fc1_output.detach()

    if gated_linear_unit:
        x_1, x_2 = torch.chunk(fc1_detached, 2, dim=-1)
        if activation_func == F.silu or (hasattr(activation_func, '__name__') and 'silu' in activation_func.__name__.lower()):
            sig = torch.sigmoid(x_1)
            act_deriv = sig * (1 + x_1 * (1 - sig))
            act_val = x_1 * sig
        else:
            act_deriv = _gelu_grad_exact(x_1)
            act_val = F.gelu(x_1)
        return act_deriv, act_val, x_2
    else:
        if activation_func == F.silu or (hasattr(activation_func, '__name__') and 'silu' in activation_func.__name__.lower()):
            sig = torch.sigmoid(fc1_detached)
            act_deriv = sig * (1 + fc1_detached * (1 - sig))
        else:
            act_deriv = _gelu_grad_exact(fc1_detached)
        return act_deriv


def _compute_activation_grad(grad_act, act_deriv, act_val=None, x_2=None, gated_linear_unit=False):
    """
    Compute gradient through activation using pre-computed derivative.

    Args:
        grad_act: Gradient from FC2 [tokens, ffn_hidden or ffn_hidden/2]
        act_deriv: Pre-computed activation derivative
        act_val: Pre-computed activation value (for GLU)
        x_2: Second half of FC1 output (for GLU)
        gated_linear_unit: Whether using GLU

    Returns:
        grad_fc1: Gradient w.r.t. FC1 output [tokens, ffn_hidden]
    """
    if gated_linear_unit:
        grad_x_1 = grad_act * x_2 * act_deriv
        grad_x_2 = grad_act * act_val
        grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
    else:
        grad_fc1 = grad_act * act_deriv
    return grad_fc1


def get_optimal_num_chunks(
    hidden_size: int,
    total_tokens: int,
    ffn_hidden: int,
    estimated_a2a_time_us: float = 100.0,
) -> int:
    """
    Estimate optimal number of chunks for dX + AllToAll overlap.

    The goal is to balance chunk compute time with AllToAll time:
    - If chunk is too small: AllToAll launch overhead dominates
    - If chunk is too large: Less overlap opportunity

    Args:
        hidden_size: Model hidden dimension
        total_tokens: Total number of tokens
        ffn_hidden: FFN hidden dimension
        estimated_a2a_time_us: Estimated AllToAll time in microseconds

    Returns:
        Recommended number of chunks (1, 2, 4, or 8)
    """
    # Use environment variable if set
    default_chunks = int(os.environ.get('FLUID_DX_CHUNKS', '4'))

    # Ensure hidden_size is divisible by num_chunks
    for num_chunks in [default_chunks, 4, 2, 1]:
        if hidden_size % num_chunks == 0:
            return num_chunks

    return 1


__all__ = [
    '_gelu_grad_exact',
    '_compute_activation_derivative',
    '_compute_activation_grad',
    'get_optimal_num_chunks',
]

"""
NVTX (NVIDIA Tools Extension) utilities for profiling FluidMoE.

Usage:
    1. Add NVTX markers to code:
       from fluid.core.nvtx import nvtx_range, nvtx_mark

       with nvtx_range("forward_dispatch"):
           # code

       nvtx_mark("checkpoint")

    2. Profile with nsys:
       nsys profile -o profile --trace=cuda,nvtx torchrun --nproc_per_node=2 your_script.py

    3. View in Nsight Systems:
       nsys-ui profile.nsys-rep

Color coding:
    - Green: Forward pass operations
    - Blue: Backward pass operations
    - Yellow: Communication (P2P, AllToAll)
    - Red: dW computation
    - Purple: Scheduler/sync operations
"""

import torch
from contextlib import contextmanager
from typing import Optional
import os

# Check if NVTX profiling is enabled
NVTX_ENABLED = os.environ.get('FLUID_NVTX', '0') == '1'

# Try to import NVTX
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    nvtx = None
    NVTX_AVAILABLE = False


# Color constants for NVTX ranges (ARGB format)
class Colors:
    GREEN = 0xFF00FF00      # Forward
    DARK_GREEN = 0xFF008800
    BLUE = 0xFF0000FF       # Backward
    DARK_BLUE = 0xFF000088
    YELLOW = 0xFFFFFF00     # Communication
    ORANGE = 0xFFFF8800
    RED = 0xFFFF0000        # dW tasks
    PURPLE = 0xFF8800FF     # Scheduler
    CYAN = 0xFF00FFFF       # Compute
    GRAY = 0xFF888888


@contextmanager
def nvtx_range(name: str, color: Optional[int] = None):
    """
    Context manager for NVTX range annotation.

    Args:
        name: Name of the range (displayed in Nsight)
        color: Optional ARGB color (use Colors class)

    Example:
        with nvtx_range("dispatch_p2p", Colors.YELLOW):
            # P2P communication code
    """
    if NVTX_ENABLED and NVTX_AVAILABLE:
        nvtx.range_push(name)
        try:
            yield
        finally:
            nvtx.range_pop()
    else:
        yield


def nvtx_mark(name: str):
    """
    Insert an instant NVTX marker.

    Args:
        name: Name of the marker

    Example:
        nvtx_mark("alltoall_start")
    """
    if NVTX_ENABLED and NVTX_AVAILABLE:
        nvtx.mark(name)


def nvtx_range_push(name: str):
    """Push an NVTX range (must be paired with nvtx_range_pop)."""
    if NVTX_ENABLED and NVTX_AVAILABLE:
        nvtx.range_push(name)


def nvtx_range_pop():
    """Pop an NVTX range."""
    if NVTX_ENABLED and NVTX_AVAILABLE:
        nvtx.range_pop()


# Convenience decorators
def nvtx_annotate(name: str, color: Optional[int] = None):
    """
    Decorator to annotate a function with NVTX range.

    Example:
        @nvtx_annotate("router_forward", Colors.GREEN)
        def router_forward(...):
            ...
    """
    def decorator(func):
        if not (NVTX_ENABLED and NVTX_AVAILABLE):
            return func

        def wrapper(*args, **kwargs):
            nvtx.range_push(name)
            try:
                return func(*args, **kwargs)
            finally:
                nvtx.range_pop()
        return wrapper
    return decorator


__all__ = [
    'nvtx_range',
    'nvtx_mark',
    'nvtx_range_push',
    'nvtx_range_pop',
    'nvtx_annotate',
    'Colors',
    'NVTX_ENABLED',
    'NVTX_AVAILABLE',
]

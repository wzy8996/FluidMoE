"""
NVTX (NVIDIA Tools Extension) utilities for profiling FluidMoE.

Usage:
    1. Add NVTX markers to code:
       from fluid.core.nvtx import nvtx_range_push, nvtx_range_pop

       nvtx_range_push("forward_dispatch")
       # code
       nvtx_range_pop()

    2. Profile with nsys:
       nsys profile -o profile --trace=cuda,nvtx torchrun --nproc_per_node=2 your_script.py

    3. View in Nsight Systems:
       nsys-ui profile.nsys-rep
"""

import os
from contextlib import contextmanager

# Check if NVTX profiling is enabled
NVTX_ENABLED = os.environ.get('FLUID_NVTX', '0') == '1'

# Try to import NVTX
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    nvtx = None
    NVTX_AVAILABLE = False


@contextmanager
def nvtx_range(name: str):
    """Context manager for NVTX range annotation."""
    if NVTX_ENABLED and NVTX_AVAILABLE:
        nvtx.range_push(name)
        try:
            yield
        finally:
            nvtx.range_pop()
    else:
        yield


def nvtx_range_push(name: str):
    """Push an NVTX range (must be paired with nvtx_range_pop)."""
    if NVTX_ENABLED and NVTX_AVAILABLE:
        nvtx.range_push(name)


def nvtx_range_pop():
    """Pop an NVTX range."""
    if NVTX_ENABLED and NVTX_AVAILABLE:
        nvtx.range_pop()


__all__ = [
    'nvtx_range',
    'nvtx_range_push',
    'nvtx_range_pop',
    'NVTX_ENABLED',
    'NVTX_AVAILABLE',
]

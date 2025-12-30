# Copyright (c) 2024, FluidMoE Team. All rights reserved.

"""
FluidMoE CUDA Kernels

This module provides optimized CUDA kernels for MoE computation:
- grouped_gemm: Grouped GEMM for expert computation
- grouped_gemm_dw: Grouped GEMM for weight gradients
- grouped_gemm_single_chunk: Single chunk GEMM for pipelining
- grouped_gemm_dx_pipelined: Pipelined dX with CUDA events
"""

import os
import importlib.util

# Load the compiled extension
_so_path = os.path.join(os.path.dirname(__file__), "fluid_kernels.so")

if os.path.exists(_so_path):
    spec = importlib.util.spec_from_file_location("fluid_kernels", _so_path)
    fluid_kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fluid_kernels)
else:
    raise ImportError(
        f"fluid_kernels.so not found at {_so_path}. "
        "Please build it with: cd fluid/kernels && bash build.sh"
    )

__all__ = ["fluid_kernels"]

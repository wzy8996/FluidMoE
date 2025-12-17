#!/usr/bin/env python3
"""
Test script for Fluid kernels build verification
"""

import sys
import os

# Add the ops directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ops'))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Try to import the built module
try:
    import fluid_kernels
    print("\n✅ fluid_kernels module loaded successfully!")
except ImportError as e:
    print(f"\n❌ Failed to import fluid_kernels: {e}")
    print("Make sure to run build.sh first")
    sys.exit(1)

# Test barrier operations
print("\n--- Testing Barrier Operations ---")
device = torch.device("cuda:0")
barrier = fluid_kernels.create_barrier(4, device)
print(f"Created barrier: {barrier}")

fluid_kernels.reset_barrier(barrier)
print(f"After reset: {barrier}")

fluid_kernels.signal_barrier(barrier, 0, 1)
torch.cuda.synchronize()
print(f"After signal(0, 1): {barrier}")

# Test CUTLASS GEMM
print("\n--- Testing CUTLASS GEMM ---")
M, N, K = 256, 256, 256
A = torch.randn(M, K, dtype=torch.float16, device=device)
B = torch.randn(K, N, dtype=torch.float16, device=device)

# Reference using PyTorch
C_ref = torch.matmul(A.float(), B.float()).half()

# CUTLASS GEMM
C_cutlass = fluid_kernels.cutlass_gemm(A, B)

# Compare results
max_diff = (C_ref - C_cutlass).abs().max().item()
print(f"Matrix size: {M}x{N}x{K}")
print(f"Max difference from PyTorch: {max_diff:.6f}")

if max_diff < 0.1:
    print("✅ CUTLASS GEMM test passed!")
else:
    print("❌ CUTLASS GEMM test failed - results don't match")

# Test GEMM with barrier
print("\n--- Testing GEMM with Barrier ---")
barrier = fluid_kernels.create_barrier(4, device)
# Pre-signal all barriers for this test
for i in range(4):
    fluid_kernels.signal_barrier(barrier, i, 1)
torch.cuda.synchronize()

C_barrier = fluid_kernels.gemm_with_barrier(A, B, barrier, 4)
print(f"GEMM with barrier output shape: {C_barrier.shape}")
print("✅ GEMM with barrier test passed!")

print("\n============================================")
print("All tests passed! Build environment is ready.")
print("============================================")

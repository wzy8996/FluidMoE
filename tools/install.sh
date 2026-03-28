#!/bin/bash
# FluidMoE Installation Script
#
# Prerequisites: PyTorch with CUDA already installed.
# Usage: bash install.sh
set -euo pipefail

python -c "import torch; print('PyTorch '+ torch.__version__ +', CUDA '+ str(torch.version.cuda))" 2>/dev/null || {
    echo "ERROR: PyTorch not found. Install PyTorch with CUDA first."
    exit 1
}

# CUDA extension build flags
export NVTE_FRAMEWORK=pytorch
export APEX_CPP_EXT=1
export APEX_CUDA_EXT=1

echo "[1/7] Build tools ..."
pip install packaging ninja pybind11 six psutil regex pyyaml

echo "[2/7] Triton ..."
pip install triton==3.3.1

echo "[3/7] TransformerEngine ..."
pip install --no-build-isolation transformer-engine[pytorch]==2.10.0

echo "[4/7] grouped_gemm ..."
pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm.git@172fada89fa7364fe5d026b3a0dfab58b591ffdd

echo "[5/7] Megatron-Core ..."
pip install megatron-core==0.16.0

echo "[6/7] Apex ..."
pip install --no-build-isolation git+https://github.com/NVIDIA/apex.git@a7b872ed8514295fae16a6df894b5ffab298a008

echo "[7/7] DeepSpeed, flash-attn, Tutel ..."
pip install deepspeed==0.18.8
pip install --no-build-isolation flash-attn==2.8.3
pip install --no-build-isolation tutel@https://github.com/microsoft/Tutel/archive/refs/tags/v0.4.1.tar.gz

echo "Done."

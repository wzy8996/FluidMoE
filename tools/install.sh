#!/bin/bash
# FluidMoE Installation Script
#
# Prerequisites: PyTorch with CUDA already installed in current conda env.
# Usage: conda activate <your_env> && bash install.sh
set -euo pipefail

python -c "import torch; print('PyTorch '+ torch.__version__ +', CUDA '+ str(torch.version.cuda))" 2>/dev/null || {
    echo "ERROR: PyTorch not found. Install PyTorch with CUDA first."
    exit 1
}

# CUDA extension build flags
export NVTE_FRAMEWORK=pytorch
export APEX_CPP_EXT=1
export APEX_CUDA_EXT=1

echo "[1/8] Build tools ..."
python -m pip install packaging ninja pybind11 six psutil regex pyyaml

echo "[2/8] Triton ..."
python -m pip install triton==3.3.1

echo "[3/8] TransformerEngine ..."
python -m pip install --no-build-isolation transformer-engine[pytorch]==2.10.0

echo "[4/8] grouped_gemm ..."
python -m pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm.git@172fada89fa7364fe5d026b3a0dfab58b591ffdd

echo "[5/8] Megatron-Core ..."
python -m pip install megatron-core==0.16.0

echo "[6/8] Apex ..."
python -m pip install --no-build-isolation git+https://github.com/NVIDIA/apex.git@a7b872ed8514295fae16a6df894b5ffab298a008

echo "[7/8] DeepSpeed, flash-attn, Tutel ..."
python -m pip install deepspeed==0.18.8
python -m pip install --no-build-isolation flash-attn==2.8.3
python -m pip install --no-build-isolation tutel@https://github.com/microsoft/Tutel/archive/refs/tags/v0.4.1.tar.gz

echo "[8/8] NVSHMEM (for P2P overlap on NVLink) ..."
NVSHMEM_DIR="${HOME}/nvshmem"
if [ -f "${NVSHMEM_DIR}/lib/libnvshmem_host.so" ]; then
    echo "  NVSHMEM already installed at ${NVSHMEM_DIR}, skipping."
else
    echo "  Downloading NVSHMEM 3.6.5 (CUDA 12) ..."
    NVSHMEM_URL="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.6.5_cuda12-archive.tar.xz"
    TMP_FILE="$(mktemp /tmp/nvshmem-XXXXXX.tar.xz)"
    trap 'rm -f "${TMP_FILE}"' EXIT
    wget -q "${NVSHMEM_URL}" -O "${TMP_FILE}" || {
        echo "  WARNING: NVSHMEM download failed. P2P will use NCCL fallback."
        echo "  You can install NVSHMEM manually later to ~/nvshmem/"
        echo "Done (NVSHMEM skipped)."
        exit 0
    }
    mkdir -p "${NVSHMEM_DIR}"
    tar xf "${TMP_FILE}" -C "${NVSHMEM_DIR}" --strip-components=1
    echo "  Installed to ${NVSHMEM_DIR}"
fi
echo "  NVSHMEM auto-detection: FluidMoE will find ~/nvshmem automatically."

echo "Done."

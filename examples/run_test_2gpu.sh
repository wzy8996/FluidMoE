#!/bin/bash
# Quick test script for 2 GPUs
set -e

# Activate conda environment
source /home/zju/miniconda3/bin/activate megatron

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/home/zju/wzy/Fluid:/home/zju/wzy/Megatron-LM:$PYTHONPATH

# ========================================
# Profiling Configuration
# ========================================
# Set ENABLE_NSYS=1 to enable nsys profiling
# Usage: ENABLE_NSYS=1 bash run_test_2gpu.sh
ENABLE_NSYS=${ENABLE_NSYS:-0}
NSYS_OUTPUT=${NSYS_OUTPUT:-"fluid_profile"}
NSYS_DELAY=${NSYS_DELAY:-5}
NSYS_DURATION=${NSYS_DURATION:-20}
LOG_OUTPUT=${LOG_OUTPUT:-"training_output.log"}

echo "=========================================="
echo "FluidMoE Quick Test (2 GPUs)"
echo "=========================================="
echo "Config:"
echo "  - GPUs: 2"
echo "  - Layers: 2"
echo "  - Hidden: 512"
echo "  - Experts: 4"
echo "  - SP: 2, EP: 2"
if [ "$ENABLE_NSYS" = "1" ]; then
    echo "  - Profiling: ENABLED (nsys)"
    echo "  - Profile output: ${NSYS_OUTPUT}.nsys-rep"
    echo "  - Log output: ${LOG_OUTPUT}"
    echo "  - Delay: ${NSYS_DELAY}s, Duration: ${NSYS_DURATION}s"
else
    echo "  - Profiling: DISABLED"
    echo "  - Log output: ${LOG_OUTPUT}"
fi
echo "=========================================="
echo ""

# Build nsys command prefix if profiling is enabled
NSYS_CMD=""
if [ "$ENABLE_NSYS" = "1" ]; then
    NSYS_CMD="nsys profile \
        -o ${NSYS_OUTPUT} \
        --stats=true \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --delay=${NSYS_DELAY} \
        --duration=${NSYS_DURATION} \
        --export=sqlite"

    echo "[Profiling] Starting nsys profiler..."
    echo "[Profiling] Will start profiling after ${NSYS_DELAY}s delay"
    echo "[Profiling] Command output will be saved to: ${LOG_OUTPUT}"
    echo ""
fi

# Simple 2 GPU test with SP=2, EP=2
# Use tee to save output to log file while still displaying on screen
$NSYS_CMD /home/zju/miniconda3/envs/megatron/bin/torchrun \
    --nproc_per_node 2 \
    --nnodes 1 \
    pretrain_gpt_moe.py \
    --tensor-model-parallel-size 1 \
    --context-parallel-size 2 \
    --expert-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --num-layers 2 \
    --hidden-size 512 \
    --ffn-hidden-size 2048 \
    --num-attention-heads 8 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --num-experts 4 \
    --moe-router-topk 2 \
    --moe-grouped-gemm \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --lr 1e-4 \
    --train-iters 10 \
    --bf16 \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 50257 \
    --save /tmp/fluid-test \
    --save-interval 1000 \
    --log-interval 1 \
    --no-save-optim \
    --no-save-rng \
    --no-load-optim \
    --no-load-rng 2>&1 | tee "${LOG_OUTPUT}"

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="

# Show profiling results if enabled
if [ "$ENABLE_NSYS" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Profiling Results"
    echo "=========================================="
    echo "Profile report: ${NSYS_OUTPUT}.nsys-rep"
    echo "Training log: ${LOG_OUTPUT}"
    echo ""
    echo "View results with:"
    echo "  1. Command-line stats:"
    echo "     nsys stats ${NSYS_OUTPUT}.nsys-rep"
    echo ""
    echo "  2. GPU kernel summary:"
    echo "     nsys stats --report cuda_gpu_kern_sum ${NSYS_OUTPUT}.nsys-rep"
    echo ""
    echo "  3. CUDA API summary:"
    echo "     nsys stats --report cuda_api_sum ${NSYS_OUTPUT}.nsys-rep"
    echo ""
    echo "  4. GUI viewer (if available):"
    echo "     nsys-ui ${NSYS_OUTPUT}.nsys-rep"
    echo ""
    echo "  5. Export to CSV:"
    echo "     nsys stats --report cuda_gpu_kern_sum ${NSYS_OUTPUT}.nsys-rep --format csv -o kernels.csv"
    echo "=========================================="
else
    echo ""
    echo "Training log saved to: ${LOG_OUTPUT}"
fi

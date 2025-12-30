#!/bin/bash
# Quick test script for 2 GPUs with configurable model sizes
set -e

# Activate conda environment
source /home/zju/miniconda3/bin/activate megatron

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/home/zju/wzy/FluidMoE:/home/zju/wzy/Megatron-LM:$PYTHONPATH

# ========================================
# Model Size Configuration
# ========================================
# Usage: bash run_test_2gpu.sh [small|medium|large|xlarge]
# Default: small
MODEL_SIZE=${1:-small}

case $MODEL_SIZE in
    small)
        # Small: quick test, ~150ms/iter
        HIDDEN=512
        FFN=2048
        SEQ=1024
        HEADS=8
        LAYERS=4
        ;;
    medium)
        # Medium: ~300ms/iter
        HIDDEN=1024
        FFN=4096
        SEQ=2048
        HEADS=8
        LAYERS=4
        ;;
    large)
        # Large: dX+A2A pipeline should help
        HIDDEN=1024
        FFN=4096
        SEQ=4096
        HEADS=8
        LAYERS=4
        ;;
    xlarge)
        # XLarge: dX+A2A pipeline should definitely help
        # Reduced FFN size to fit in memory with recompute
        HIDDEN=1536
        FFN=6144
        SEQ=4096
        HEADS=12
        LAYERS=2
        ;;
    comm_heavy)
        # Comm-heavy: Large FFN with longer seq for better overlap
        HIDDEN=1024
        FFN=24576
        SEQ=2048
        HEADS=8
        LAYERS=4
        ;;
    moe_heavy)
        # MoE-heavy: Maximize MoE comm relative to Attention
        # - Small hidden (reduces Attention compute: O(seq² × hidden))
        # - Long seq (increases MoE comm: O(seq × hidden))
        # - Large FFN (increases MoE compute)
        # - Few layers (reduces total Attention)
        # MoE AllToAll size per layer: seq × hidden × 2 (bf16)
        #   = 4096 × 512 × 2 = 4MB per AllToAll
        # Attention vs MoE ratio: seq²×hidden vs seq×hidden×FFN
        #   = 4096² × 512 vs 4096 × 512 × 8192
        #   = 8.6B vs 17.2B (MoE compute > Attention!)
        HIDDEN=512
        FFN=8192
        SEQ=4096
        HEADS=8
        LAYERS=2
        ;;
    moe_heavy2)
        # MoE-heavy v2: Even more tokens, smaller model
        # - Tiny hidden (minimize Attention compute)
        # - Longer seq (more tokens to communicate)
        # - Moderate FFN
        # MoE AllToAll size: 6144 × 384 × 2 = 4.7MB
        HIDDEN=384
        FFN=4096
        SEQ=6144
        HEADS=6
        LAYERS=2
        ;;
    many_experts)
        # Many experts: Test with larger number of experts
        # - Moderate hidden size
        # - Small FFN per expert (since we have many)
        # - Medium seq length for reasonable AllToAll size
        # Use with NUM_EXPERTS=8 or 16
        HIDDEN=768
        FFN=3072
        SEQ=2048
        HEADS=12
        LAYERS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Usage: bash run_test_2gpu.sh [small|medium|large|xlarge]"
        exit 1
        ;;
esac

# Fixed parallelism: CP=2, EP=2
CP=2
EP=2

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
echo "FluidMoE Test (2 GPUs) - ${MODEL_SIZE^^}"
echo "=========================================="
echo "Config:"
echo "  - GPUs: 2"
echo "  - Model size: $MODEL_SIZE"
echo "  - Layers: $LAYERS"
echo "  - Hidden: $HIDDEN"
echo "  - FFN: $FFN"
echo "  - Experts: 2 (num_local_experts=1)"
echo "  - Seq: $SEQ"
echo "  - CP: $CP, EP: $EP"
echo "  - FLUID_DX_NUM_CHUNKS: ${FLUID_DX_NUM_CHUNKS:-1}"
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

# Run training
$NSYS_CMD /home/zju/miniconda3/envs/megatron/bin/torchrun \
    --nproc_per_node 2 \
    --nnodes 1 \
    pretrain_gpt_moe.py \
    --tensor-model-parallel-size 1 \
    --context-parallel-size $CP \
    --expert-model-parallel-size $EP \
    --pipeline-model-parallel-size 1 \
    --num-layers $LAYERS \
    --hidden-size $HIDDEN \
    --ffn-hidden-size $FFN \
    --num-attention-heads $HEADS \
    --seq-length $SEQ \
    --max-position-embeddings $SEQ \
    --num-experts ${NUM_EXPERTS:-2} \
    --moe-router-topk ${MOE_TOPK:-2} \
    --moe-grouped-gemm \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --lr 1e-4 \
    --train-iters 10 \
    --bf16 \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --recompute-activations \
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

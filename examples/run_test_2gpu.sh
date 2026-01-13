#!/bin/bash
# FluidMoE Test Script for 2 GPUs
# Tests baseline vs overlap mode with full forward+backward training
set -e

# Activate conda environment
source /home/zju/miniconda3/bin/activate megatron

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/home/zju/wzy/FluidMoE:/home/zju/wzy/Megatron-LM:$PYTHONPATH

# ========================================
# Configuration
# ========================================
# MODE: baseline | overlap | compare (run both and compare)
MODE=${MODE:-compare}

# Number of iterations
ITERS=${ITERS:-10}

# Model size
MODEL_SIZE=${1:-medium}

case $MODEL_SIZE in
    small)
        HIDDEN=512; FFN=2048; SEQ=1024; HEADS=8; LAYERS=4
        ;;
    medium)
        HIDDEN=1024; FFN=4096; SEQ=2048; HEADS=8; LAYERS=4
        ;;
    large)
        HIDDEN=2048; FFN=8192; SEQ=4096; HEADS=16; LAYERS=4
        ;;
    xlarge)
        HIDDEN=4096; FFN=16384; SEQ=4096; HEADS=32; LAYERS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Valid: small | medium | large | xlarge"
        exit 1
        ;;
esac

CP=2
EP=2

# ========================================
# Helper Functions
# ========================================

run_test() {
    local mode_name=$1
    local overlap_flag=$2

    echo ""
    echo "=========================================="
    echo "Running: $mode_name (forward + backward)"
    echo "=========================================="

    export FLUID_FORWARD_OVERLAP=$overlap_flag

    /home/zju/miniconda3/envs/megatron/bin/torchrun \
        --nproc_per_node 2 \
        --nnodes 1 \
        examples/pretrain_gpt_moe.py \
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
        --num-experts 2 \
        --moe-router-topk 2 \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type allgather \
        --micro-batch-size 1 \
        --global-batch-size 2 \
        --lr 1e-4 \
        --train-iters $ITERS \
        --bf16 \
        --disable-bias-linear \
        --use-distributed-optimizer \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50257 \
        --log-interval 1 \
        --timing-log-level 1 2>&1 | tee /tmp/fluid_${mode_name}.log

    # Extract average iteration time (get last few iterations and average)
    local times=$(grep "elapsed time per iteration" /tmp/fluid_${mode_name}.log | tail -5 | grep -oP 'iteration \(ms\): \K[\d.]+')
    local sum=0
    local count=0
    for t in $times; do
        sum=$(echo "$sum + $t" | bc)
        count=$((count + 1))
    done
    local avg_time=$(echo "scale=2; $sum / $count" | bc)
    echo "${mode_name}: $avg_time ms/iter" >> /tmp/fluid_results.txt
    echo "  -> $mode_name avg time: $avg_time ms/iter (from last $count iterations)"
}

print_comparison() {
    echo ""
    echo "=========================================="
    echo "Comparison Results"
    echo "=========================================="

    local base_time=$(grep "^baseline:" /tmp/fluid_results.txt | grep -oP '[\d.]+')
    local over_time=$(grep "^overlap:" /tmp/fluid_results.txt | grep -oP '[\d.]+')

    printf "%-15s %15s\n" "Mode" "Time (ms/iter)"
    printf "%-15s %15s\n" "----" "--------------"
    printf "%-15s %13.2f ms\n" "Baseline" "$base_time"
    printf "%-15s %13.2f ms\n" "Overlap" "$over_time"

    if [ -n "$base_time" ] && [ -n "$over_time" ]; then
        local speedup=$(echo "scale=2; $base_time / $over_time" | bc)
        printf "%-15s %14.2fx\n" "Speedup" "$speedup"
    fi
    echo "=========================================="
}

# ========================================
# Main
# ========================================

echo "=========================================="
echo "FluidMoE Benchmark - ${MODEL_SIZE^^}"
echo "=========================================="
echo "Config:"
echo "  - Mode: $MODE"
echo "  - Iterations: $ITERS"
echo "  - Hidden: $HIDDEN, FFN: $FFN"
echo "  - Seq: $SEQ, Heads: $HEADS"
echo "  - Layers: $LAYERS"
echo "  - CP: $CP, EP: $EP"
echo "=========================================="

# Clear results file
> /tmp/fluid_results.txt

case $MODE in
    baseline)
        run_test "baseline" "0"
        ;;
    overlap)
        run_test "overlap" "1"
        ;;
    compare)
        run_test "baseline" "0"
        run_test "overlap" "1"
        print_comparison
        ;;
esac

echo ""
echo "Done! Logs saved to /tmp/fluid_*.log"

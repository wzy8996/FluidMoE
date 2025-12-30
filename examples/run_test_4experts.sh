#!/bin/bash
# Test script for 2 GPUs with 4 experts (num_local_experts=2)
set -e

# Activate conda environment
source /home/zju/miniconda3/bin/activate megatron

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/home/zju/wzy/FluidMoE:/home/zju/wzy/Megatron-LM:$PYTHONPATH

# ========================================
# Model Size Configuration
# ========================================
# Usage: bash run_test_4experts.sh [small|medium|large|xlarge]
# Default: small
MODEL_SIZE=${1:-small}

case $MODEL_SIZE in
    small)
        HIDDEN=512
        FFN=2048
        SEQ=1024
        HEADS=8
        LAYERS=4
        ;;
    medium)
        HIDDEN=1024
        FFN=4096
        SEQ=2048
        HEADS=8
        LAYERS=4
        ;;
    large)
        HIDDEN=1024
        FFN=4096
        SEQ=4096
        HEADS=8
        LAYERS=4
        ;;
    xlarge)
        HIDDEN=2048
        FFN=8192
        SEQ=4096
        HEADS=16
        LAYERS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Usage: bash run_test_4experts.sh [small|medium|large|xlarge]"
        exit 1
        ;;
esac

# Fixed parallelism: CP=2, EP=2
CP=2
EP=2
NUM_EXPERTS=4  # 4 experts with EP=2 means num_local_experts=2

echo "=========================================="
echo "FluidMoE Test (2 GPUs) - 4 Experts - ${MODEL_SIZE^^}"
echo "=========================================="
echo "Config:"
echo "  - GPUs: 2"
echo "  - Model size: $MODEL_SIZE"
echo "  - Experts: $NUM_EXPERTS (num_local_experts=2)"
echo "  - Layers: $LAYERS"
echo "  - Hidden: $HIDDEN"
echo "  - FFN: $FFN"
echo "  - Seq: $SEQ"
echo "  - CP: $CP, EP: $EP"
echo "  - FLUID_DX_NUM_CHUNKS: ${FLUID_DX_NUM_CHUNKS:-1}"
echo "=========================================="
echo ""

# Run training
/home/zju/miniconda3/envs/megatron/bin/torchrun \
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
    --num-experts $NUM_EXPERTS \
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
    --no-load-rng 2>&1 | tee training_output_4experts.log

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="

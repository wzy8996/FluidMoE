#!/usr/bin/env bash
# Training: Megatron vs FluidMoE via Megatron's pretrain()
#
# Both use identical Megatron infrastructure (distributed optimizer, mixed precision,
# gradient clipping, lr scheduling). Only the MoE scheduling differs.
#
# Model configs are read from tools/model_configs.py.
# Experiment configs (parallelism, chunks, AR) are read from tools/experiment_configs.py.
#
# Usage:
#   bash examples/run_training.sh                          # default: mixtral_8x7b
#   MODEL=mixtral_8x7b bash examples/run_training.sh      # different model
#   TRAIN_ITERS=500 bash examples/run_training.sh          # more iterations
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

MODEL="${MODEL:-mixtral_8x7b}"
TRAIN_ITERS="${TRAIN_ITERS:-300}"
LR_WARMUP="${LR_WARMUP:- 30}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SEED="${SEED:-105}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.0}"
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
DATASET_SOURCE="${DATASET_SOURCE:-mock}"

PROJ_ROOT="$(pwd)"

# Read model config from tools/model_configs.py
read -r HIDDEN FFN NUM_HEADS NUM_KV_HEADS NUM_EXPERTS TOP_K NUM_LAYERS SEQ_LEN BATCH_SIZE CAP_FACTOR <<< $(
"${PYTHON_BIN}" -c "
import sys; sys.path.insert(0, '${PROJ_ROOT}')
from tools.model_configs import get_model_config
c = get_model_config('${MODEL}')
print(c['hidden_size'], c['ffn_hidden'], c['num_heads'], c['num_kv_heads'],
      c['num_experts'], c['top_k'], c['num_layers'], c['seq_len'],
      c['batch_size'], c['capacity_factor'])
")

# Read experiment config from tools/experiment_configs.py
read -r DP_SIZE CP_SIZE EP_SIZE \
     MOE_COMBINE_CHUNKS MOE_DISPATCH_CHUNKS ATTN_PROJ_CHUNKS ATTN_QKV_CHUNKS \
     SHARED_AR_BW EXPERT_AR_BW \
     GAP_MOE_COMBINE GAP_MOE_DISPATCH GAP_ATTN_PROJ GAP_ATTN_QKV <<< $(
"${PYTHON_BIN}" -c "
import sys; sys.path.insert(0, '${PROJ_ROOT}')
from tools.experiment_configs import get_block_benchmark_defaults
d = get_block_benchmark_defaults()
g = d.get('gap_budgets', {})
print(d['dp_size'], d['cp_size'], d['ep_size'],
      d.get('moe_combine_chunks', 4), d.get('moe_dispatch_chunks', 4),
      d.get('attn_proj_chunks', 2), d.get('attn_qkv_chunks', 4),
      d.get('shared_ar_bw', 0.0), d.get('expert_ar_bw', 0.0),
      g.get('moe_combine', 0.0), g.get('moe_dispatch', 0.0),
      g.get('attn_proj', 0.0), g.get('attn_qkv', 0.0))
")

NPROC=$((DP_SIZE * CP_SIZE))

# Use seq_len from model_configs.py as-is (paper-scale runs need full
# context). Set SEQ_LEN_OVERRIDE to override on memory-constrained machines.
SEQ_LEN="${SEQ_LEN_OVERRIDE:-${SEQ_LEN}}"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${BATCH_SIZE}}"
# Megatron requires global_batch_size to be divisible by micro_batch_size * dp_size.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * DP_SIZE))}"

case "${DATASET_SOURCE}" in
    mock)
        DATASET_ARGS=(
            --dataset-source mock
            --mock-data
            --tokenizer-type NullTokenizer
            --vocab-size 50257
        )
        ;;
    wikitext)
        DATASET_ARGS=(
            --dataset-source wikitext
            --tokenizer-type NullTokenizer
            --vocab-size 50257
        )
        ;;
    *)
        echo "Unsupported DATASET_SOURCE=${DATASET_SOURCE}. Use mock or wikitext." >&2
        exit 1
        ;;
esac

if [[ "${DATASET_SOURCE}" == "wikitext" ]]; then
    CACHE_DIR="${PROJ_ROOT}/examples/.data_cache"
    MISSING_SPLITS=()
    for SPLIT in train validation test; do
        if [[ ! -f "${CACHE_DIR}/wikitext103_${SPLIT}_gpt2.pt" ]]; then
            MISSING_SPLITS+=("${SPLIT}")
        fi
    done
    if [[ ${#MISSING_SPLITS[@]} -gt 0 ]]; then
        echo "WikiText cache missing for: ${MISSING_SPLITS[*]}"
        echo "  First run may spend time downloading/tokenizing before the first iteration appears."
        echo "  Current config uses training-only mode (eval-iters=0), so only train cache is required."
    fi
fi

COMMON_ARGS=(
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN}"
    --ffn-hidden-size "${FFN}"
    --num-attention-heads "${NUM_HEADS}"
    --group-query-attention
    --num-query-groups "${NUM_KV_HEADS}"
    --num-experts "${NUM_EXPERTS}"
    --moe-router-topk "${TOP_K}"
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --lr 3e-5
    --min-lr 1e-5
    --lr-warmup-iters "$(( LR_WARMUP < TRAIN_ITERS ? LR_WARMUP : TRAIN_ITERS / 2 ))"
    --lr-decay-iters "${TRAIN_ITERS}"
    --lr-decay-style cosine
    --context-parallel-size "${CP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --bf16
    --hidden-dropout "${HIDDEN_DROPOUT}"
    --attention-dropout "${ATTENTION_DROPOUT}"
    --disable-bias-linear
    --log-interval 1
    --save-interval 10000
    --seed "${SEED}"
    --eval-iters 0
    --eval-interval 10000
    "${DATASET_ARGS[@]}"
)

# FluidMoE-specific: chunk and AR parameters from experiment_configs
export FLUIDMOE_MOE_COMBINE_CHUNKS="${MOE_COMBINE_CHUNKS}"
export FLUIDMOE_MOE_DISPATCH_CHUNKS="${MOE_DISPATCH_CHUNKS}"
export FLUIDMOE_ATTN_PROJ_CHUNKS="${ATTN_PROJ_CHUNKS}"
export FLUIDMOE_ATTN_QKV_CHUNKS="${ATTN_QKV_CHUNKS}"
export FLUIDMOE_SHARED_AR_BW="${SHARED_AR_BW}"
export FLUIDMOE_EXPERT_AR_BW="${EXPERT_AR_BW}"
export FLUIDMOE_GAP_MOE_COMBINE="${GAP_MOE_COMBINE}"
export FLUIDMOE_GAP_MOE_DISPATCH="${GAP_MOE_DISPATCH}"
export FLUIDMOE_GAP_ATTN_PROJ="${GAP_ATTN_PROJ}"
export FLUIDMOE_GAP_ATTN_QKV="${GAP_ATTN_QKV}"

echo "============================================================"
echo "Training: Megatron vs FluidMoE (via Megatron pretrain())"
echo "  model=${MODEL}  GPUs=${NPROC} (dp=${DP_SIZE} cp=${CP_SIZE} ep=${EP_SIZE})"
echo "  hidden=${HIDDEN}  ffn=${FFN}  experts=${NUM_EXPERTS}  top_k=${TOP_K}"
echo "  layers=${NUM_LAYERS}  seq=${SEQ_LEN}  batch=${MICRO_BATCH_SIZE}"
echo "  data=${DATASET_SOURCE}"
echo "  dropout: hidden=${HIDDEN_DROPOUT}  attention=${ATTENTION_DROPOUT}"
echo "  chunks: R1=${MOE_COMBINE_CHUNKS} R2=${MOE_DISPATCH_CHUNKS} R3=${ATTN_PROJ_CHUNKS} R4=${ATTN_QKV_CHUNKS}"
echo "  train_iters=${TRAIN_ITERS}  lr_warmup=${LR_WARMUP}  seed=${SEED}"
echo "============================================================"
echo

TMPDIR_RUN=$(mktemp -d)
MEG_LOG="${TMPDIR_RUN}/megatron.log"
FLUID_LOG="${TMPDIR_RUN}/fluidmoe.log"
DS_LOG="${TMPDIR_RUN}/deepspeed.log"

# Return 0 on success, non-zero on failure. Caller decides whether to continue.
# (Previously this called `exit` on failure, which made one bad framework abort
# the whole 3-system comparison.)
run_training_job() {
    local label="$1"
    local log_file="$2"
    shift 2

    set +e
    "$@" 2>&1 | tee "${log_file}" | grep --line-buffered "iteration.*lm loss"
    local pipe_status=("${PIPESTATUS[@]}")
    set -e

    if [[ ${pipe_status[0]} -ne 0 ]]; then
        echo
        echo "${label} failed (exit=${pipe_status[0]}). Last 80 log lines:"
        tail -n 80 "${log_file}"
        return "${pipe_status[0]}"
    fi

    if [[ ${pipe_status[2]} -ne 0 ]]; then
        echo
        echo "${label} produced no iteration logs. Last 80 log lines:"
        tail -n 80 "${log_file}"
        return 1
    fi
    return 0
}

# `set -e` would make a non-zero return from `run_training_job` abort the whole
# script; disable around the per-framework block so one failure doesn't kill
# the others. Track exit codes for the summary.
set +e
DS_RC=0; MEG_RC=0; FLUID_RC=0

echo "[1/3] DeepSpeed-MoE Baseline ..."
run_training_job \
    "DeepSpeed-MoE Baseline" \
    "${DS_LOG}" \
    ${TORCHRUN_BIN} --nproc_per_node="${NPROC}" examples/pretrain_deepspeed.py "${COMMON_ARGS[@]}"
DS_RC=$?
echo

echo "[2/3] Megatron Baseline ..."
run_training_job \
    "Megatron Baseline" \
    "${MEG_LOG}" \
    ${TORCHRUN_BIN} --nproc_per_node="${NPROC}" examples/pretrain_megatron.py "${COMMON_ARGS[@]}"
MEG_RC=$?
echo

echo "[3/3] FluidMoE ..."
run_training_job \
    "FluidMoE" \
    "${FLUID_LOG}" \
    ${TORCHRUN_BIN} --nproc_per_node="${NPROC}" examples/pretrain_fluidmoe.py "${COMMON_ARGS[@]}"
FLUID_RC=$?
echo

set -e

# Per-framework status line so `cat run.log | tail` shows what worked.
echo "Run status:  DeepSpeed=$([[ $DS_RC -eq 0 ]] && echo OK || echo FAIL)  " \
              "Megatron=$([[ $MEG_RC -eq 0 ]] && echo OK || echo FAIL)  " \
              "FluidMoE=$([[ $FLUID_RC -eq 0 ]] && echo OK || echo FAIL)"
echo

# ── Summary: compute stable step time (skip first LR_WARMUP steps) ──
echo "============================================================"
echo "Summary"
echo "============================================================"
"${PYTHON_BIN}" -c "
import os, re, sys

def parse_log(path, warmup):
    times, losses = [], []
    if not os.path.exists(path):
        return times, losses
    pat = re.compile(r'iteration +(\d+)/.*elapsed time per iteration \(ms\): ([\d.]+).*lm loss: ([\d.E+\-]+).*grad norm: ([\d.]+)')
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                step, t, loss, gn = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
                if step > warmup:
                    times.append(t)
                losses.append((step, loss, gn))
    return times, losses

warmup = ${LR_WARMUP}
results = {
    'DeepSpeed': parse_log('${DS_LOG}', warmup),
    'Megatron':  parse_log('${MEG_LOG}', warmup),
    'FluidMoE':  parse_log('${FLUID_LOG}', warmup),
}

avgs = {}
for name, (times, losses) in results.items():
    if times:
        avg = sum(times) / len(times)
        avgs[name] = avg
        last_loss = losses[-1][1] if losses else float('nan')
        last_gn = losses[-1][2] if losses else float('nan')
        print(f'  {name:10s} avg step time: {avg:7.1f} ms  ({len(times)} steps)  '
              f'final loss={last_loss:.4f}  gn={last_gn:.3f}')
    else:
        print(f'  {name:10s} (no parsed iterations)')

print()
# All meaningful pairwise speedups for whichever frameworks succeeded.
if 'FluidMoE' in avgs and 'Megatron' in avgs:
    print(f'  Speedup FluidMoE / Megatron : {avgs[\"Megatron\"] / avgs[\"FluidMoE\"]:.2f}x')
if 'FluidMoE' in avgs and 'DeepSpeed' in avgs:
    print(f'  Speedup FluidMoE / DeepSpeed: {avgs[\"DeepSpeed\"] / avgs[\"FluidMoE\"]:.2f}x')
if 'DeepSpeed' in avgs and 'Megatron' in avgs:
    print(f'  Speedup DeepSpeed / Megatron: {avgs[\"Megatron\"] / avgs[\"DeepSpeed\"]:.2f}x')
if not avgs:
    print('  ERROR: no framework produced parseable iterations', file=sys.stderr)
"
echo "============================================================"
rm -rf "${TMPDIR_RUN}"

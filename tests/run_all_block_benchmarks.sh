#!/usr/bin/env bash
set -euo pipefail


TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${TESTS_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

MODEL="${MODEL:-mixtral_8x7b}"

read_default() {
  local key="$1"
  "${PYTHON_BIN}" - "${ROOT_DIR}" "$key" <<'PY'
import sys
import importlib.util
from pathlib import Path

root_dir = Path(sys.argv[1])
key = sys.argv[2]
cfg_path = root_dir / "tools" / "experiment_configs.py"
spec = importlib.util.spec_from_file_location("fluidmoe_experiment_configs", cfg_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

defaults = module.get_block_benchmark_defaults()
print(defaults[key])
PY
}

# Multi-node settings (all from experiment_configs, only NODE_RANK needs env)
NNODES="${NNODES:-$(read_default nnodes)}"
MASTER_ADDR="${MASTER_ADDR:-$(read_default master_addr)}"
MASTER_PORT="${MASTER_PORT:-$(read_default master_port)}"
NODE_RANK="${NODE_RANK:-0}"

DP_SIZE="${DP_SIZE:-$(read_default dp_size)}"
CP_SIZE="${CP_SIZE:-$(read_default cp_size)}"
EP_SIZE="${EP_SIZE:-$(read_default ep_size)}"
WARMUP="${WARMUP:-$(read_default warmup)}"
ITERS="${ITERS:-$(read_default iters)}"

# Per-node GPU count: total GPUs = dp * cp, split across nodes
TOTAL_GPUS=$(( DP_SIZE * CP_SIZE ))
NPROC_PER_NODE=$(( TOTAL_GPUS / NNODES ))
if (( NPROC_PER_NODE * NNODES != TOTAL_GPUS )); then
  echo "ERROR: dp*cp=$TOTAL_GPUS is not divisible by nnodes=$NNODES" >&2
  exit 1
fi

echo "============================================================"
echo "Run All Block Benchmarks"
echo "  model=${MODEL}"
if (( NNODES > 1 )); then
  echo "  nnodes=${NNODES}, node_rank=${NODE_RANK}"
  echo "  master=${MASTER_ADDR}:${MASTER_PORT}"
fi
echo "  nproc_per_node=${NPROC_PER_NODE} (total_gpus=${TOTAL_GPUS})"
echo "  dp=${DP_SIZE} cp=${CP_SIZE} ep=${EP_SIZE}"
echo "============================================================"
echo

# Build torchrun args for single-node or multi-node
TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE}")
if (( NNODES > 1 )); then
  TORCHRUN_ARGS+=(
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )
fi

COMMON_ARGS=(
  tests/run_block_benchmark.py
  --model "${MODEL}"
  --dp-size "${DP_SIZE}"
  --cp-size "${CP_SIZE}"
  --ep-size "${EP_SIZE}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
)

DEEPSPEED_LOG="$(mktemp)"
MEGATRON_LOG="$(mktemp)"
FLUIDMOE_LOG="$(mktemp)"
OVERLAP_LOG="$(mktemp)"
trap 'rm -f "${DEEPSPEED_LOG}" "${MEGATRON_LOG}" "${FLUIDMOE_LOG}" "${OVERLAP_LOG}"' EXIT

echo "[1/4] DeepSpeed Baseline"
"${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" "${COMMON_ARGS[@]}" --impl deepspeed | tee "${DEEPSPEED_LOG}"
echo

echo "[2/4] Megatron Baseline"
"${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" "${COMMON_ARGS[@]}" --impl megatron | tee "${MEGATRON_LOG}"
echo

echo "[3/4] FluidMoE"
"${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" "${COMMON_ARGS[@]}" --impl fluidmoe | tee "${FLUIDMOE_LOG}"
echo

echo "[4/4] Overlap Analysis"
"${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" \
  tests/overlap_ratio_analyzer.py --model "${MODEL}" \
  --dp-size "${DP_SIZE}" --cp-size "${CP_SIZE}" --ep-size "${EP_SIZE}" \
  --warmup "${WARMUP}" --iters "${ITERS}" | tee "${OVERLAP_LOG}"
echo

"${PYTHON_BIN}" - "${DEEPSPEED_LOG}" "${MEGATRON_LOG}" "${FLUIDMOE_LOG}" "${OVERLAP_LOG}" <<'PY'
import re
import sys

deepspeed_log, megatron_log, fluidmoe_log, overlap_log = sys.argv[1:5]

with open(deepspeed_log, "r", encoding="utf-8") as f:
    deepspeed_text = f.read()
with open(megatron_log, "r", encoding="utf-8") as f:
    megatron_text = f.read()
with open(fluidmoe_log, "r", encoding="utf-8") as f:
    fluidmoe_text = f.read()
with open(overlap_log, "r", encoding="utf-8") as f:
    overlap_text = f.read().strip()

baseline_re = r"RESULT impl=\S+ forward_ms=([0-9.]+) iter_ms=([0-9.]+)"
deepspeed_match = re.search(baseline_re, deepspeed_text)
megatron_match = re.search(baseline_re, megatron_text)
fluidmoe_match = re.search(
    r"RESULT impl=fluidmoe(?:-\w+)? forward_ms=([0-9.]+) "
    r"f_iter_ms=([0-9.]+) fb_iter_ms=([0-9.]+) full_iter_ms=([0-9.]+)",
    fluidmoe_text,
)

failed = []
if deepspeed_match is None: failed.append("deepspeed")
if megatron_match is None: failed.append("megatron")
if fluidmoe_match is None: failed.append("fluidmoe")
if failed:
    raise SystemExit(f"Failed to parse RESULT lines from: {', '.join(failed)}")

ds_fwd, ds_iter = map(float, deepspeed_match.groups())
meg_fwd, meg_iter = map(float, megatron_match.groups())
fluid_fwd, fluid_f, fluid_fb, fluid_full = map(float, fluidmoe_match.groups())

a2a_overlap = None
ar_overlap = None
for line in overlap_text.splitlines():
    line = line.strip()
    m = re.search(r'a2a_overlap=([0-9.]+)', line)
    if m:
        a2a_overlap = float(m.group(1))
    m = re.search(r'ar_overlap=([0-9.]+)', line)
    if m:
        ar_overlap = float(m.group(1))

print("============================================================")
print("Benchmark Results")
print("------------------------------------------------------------")
print(f"  DeepSpeed Baseline:  forward={ds_fwd:.2f}ms  iter={ds_iter:.2f}ms")
print(f"  Megatron Baseline:   forward={meg_fwd:.2f}ms  iter={meg_iter:.2f}ms")
print(f"  FluidMoE:            forward={fluid_fwd:.2f}ms  F={fluid_f:.2f}ms  FB={fluid_fb:.2f}ms  full={fluid_full:.2f}ms")
if a2a_overlap is not None:
    print(f"  A2A overlap ratio:   {a2a_overlap:.1%}")
if ar_overlap is not None:
    print(f"  AR overlap ratio:    {ar_overlap:.1%}")
print("------------------------------------------------------------")
print("Speedup (vs Megatron Baseline)")
print(f"  forward speedup    = {meg_fwd / fluid_fwd:.3f}x")
print(f"  F iter speedup     = {meg_iter / fluid_f:.3f}x  (no overlap)")
print(f"  FB iter speedup    = {meg_iter / fluid_fb:.3f}x  (+dW || A2A)")
print(f"  full iter speedup  = {meg_iter / fluid_full:.3f}x  (+inline AR)")
print("------------------------------------------------------------")
print("Speedup (vs DeepSpeed Baseline)")
print(f"  forward speedup    = {ds_fwd / fluid_fwd:.3f}x")
print(f"  F iter speedup     = {ds_iter / fluid_f:.3f}x")
print(f"  FB iter speedup    = {ds_iter / fluid_fb:.3f}x")
print(f"  full iter speedup  = {ds_iter / fluid_full:.3f}x")
print("============================================================")
PY

#!/usr/bin/env bash
# Note: -e is intentionally OFF — individual baselines may OOM or fail and
# we want the rest to still run. Each job's exit status is captured.
set -uo pipefail


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

# Per-job runner that records exit status and never aborts the whole script.
# Caller passes label, log path, and the command. Returns 0 on success, 1 on
# failure; the script logs which jobs failed but continues to the next one.
run_one() {
  local label="$1"; local log="$2"; shift 2
  echo "${label}"
  "$@" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  if (( rc != 0 )); then
    echo "  [WARN] ${label} FAILED (exit=${rc}) — continuing with next baseline."
    echo "  Last 20 log lines:"
    tail -n 20 "${log}" | sed 's/^/    /'
  fi
  echo
  return $rc
}

run_one "[1/4] DeepSpeed Baseline" "${DEEPSPEED_LOG}" \
  "${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" "${COMMON_ARGS[@]}" --impl deepspeed
DS_RC=$?

run_one "[2/4] Megatron Baseline" "${MEGATRON_LOG}" \
  "${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" "${COMMON_ARGS[@]}" --impl megatron
MEG_RC=$?

run_one "[3/4] FluidMoE (full mode: scheduler + inline AR)" "${FLUIDMOE_LOG}" \
  "${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" "${COMMON_ARGS[@]}" --impl fluidmoe-full
FLU_RC=$?

run_one "[4/4] Overlap Analysis" "${OVERLAP_LOG}" \
  "${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" \
  tests/overlap_ratio_analyzer.py --model "${MODEL}" \
  --dp-size "${DP_SIZE}" --cp-size "${CP_SIZE}" --ep-size "${EP_SIZE}" \
  --warmup "${WARMUP}" --iters "${ITERS}"
OVL_RC=$?

"${PYTHON_BIN}" - "${DEEPSPEED_LOG}" "${MEGATRON_LOG}" "${FLUIDMOE_LOG}" "${OVERLAP_LOG}" <<'PY'
import re
import sys

deepspeed_log, megatron_log, fluidmoe_log, overlap_log = sys.argv[1:5]

baseline_re = r"RESULT impl=\S+ forward_ms=([0-9.]+) iter_ms=([0-9.]+)"

def parse_baseline(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    m = re.search(baseline_re, text)
    if m is None:
        return None
    return tuple(map(float, m.groups()))

ds = parse_baseline(deepspeed_log)
meg = parse_baseline(megatron_log)
flu = parse_baseline(fluidmoe_log)

a2a_overlap = ar_overlap = None
try:
    overlap_text = open(overlap_log, "r", encoding="utf-8").read()
    for line in overlap_text.splitlines():
        m = re.search(r'a2a_overlap=([0-9.]+)', line)
        if m: a2a_overlap = float(m.group(1))
        m = re.search(r'ar_overlap=([0-9.]+)', line)
        if m: ar_overlap = float(m.group(1))
except OSError:
    pass

print("============================================================")
print("Benchmark Results")
print("------------------------------------------------------------")
def fmt(name, res):
    if res is None:
        print(f"  {name:<22} [FAILED — see log]")
    else:
        fwd, iter_ms = res
        print(f"  {name:<22} forward={fwd:.2f}ms  iter={iter_ms:.2f}ms")
fmt("DeepSpeed Baseline:", ds)
fmt("Megatron Baseline:",  meg)
fmt("FluidMoE (full):",    flu)
if a2a_overlap is not None:
    print(f"  A2A overlap ratio:   {a2a_overlap:.1%}")
if ar_overlap is not None:
    print(f"  AR overlap ratio:    {ar_overlap:.1%}")
print("------------------------------------------------------------")
if flu is not None:
    fluid_fwd, fluid_iter = flu
    if meg is not None:
        meg_fwd, meg_iter = meg
        print("Speedup (vs Megatron Baseline)")
        print(f"  forward speedup    = {meg_fwd / fluid_fwd:.3f}x")
        print(f"  iter speedup       = {meg_iter / fluid_iter:.3f}x")
        print("------------------------------------------------------------")
    if ds is not None:
        ds_fwd, ds_iter = ds
        print("Speedup (vs DeepSpeed Baseline)")
        print(f"  forward speedup    = {ds_fwd / fluid_fwd:.3f}x")
        print(f"  iter speedup       = {ds_iter / fluid_iter:.3f}x")
        print("------------------------------------------------------------")
else:
    print("FluidMoE failed — speedup numbers omitted.")
    print("------------------------------------------------------------")
print()
print("Note: this script runs FluidMoE in 'full' mode (scheduler + inline AR).")
print("      For the F/FB/full ablation, run run_block_benchmark.py three times")
print("      with --impl=fluidmoe-{f,fb,full}.")
print("============================================================")
PY

# Final exit code: 0 only if ALL four jobs succeeded. Useful for CI.
if (( DS_RC != 0 || MEG_RC != 0 || FLU_RC != 0 || OVL_RC != 0 )); then
  echo
  echo "[!] One or more jobs failed: ds=${DS_RC} meg=${MEG_RC} flu=${FLU_RC} overlap=${OVL_RC}"
  exit 1
fi

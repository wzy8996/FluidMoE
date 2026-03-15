#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${TOOLS_DIR}/.." && pwd)"
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

DP_SIZE="${DP_SIZE:-$(read_default dp_size)}"
CP_SIZE="${CP_SIZE:-$(read_default cp_size)}"
EP_SIZE="${EP_SIZE:-$(read_default ep_size)}"
WARMUP="${WARMUP:-$(read_default warmup)}"
ITERS="${ITERS:-$(read_default iters)}"

NPROC_PER_NODE=$(( DP_SIZE * CP_SIZE ))

echo "============================================================"
echo "Run All Block Benchmarks"
echo "  model=${MODEL}"
echo "  nproc_per_node=${NPROC_PER_NODE} (dp=${DP_SIZE} * cp=${CP_SIZE})"
echo "  dp=${DP_SIZE} cp=${CP_SIZE} ep=${EP_SIZE}"
echo "============================================================"
echo

COMMON_ARGS=(
  --nproc_per_node="${NPROC_PER_NODE}"
  tools/run_block_benchmark.py
  --model "${MODEL}"
  --dp-size "${DP_SIZE}"
  --cp-size "${CP_SIZE}"
  --ep-size "${EP_SIZE}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
)

NATIVE_LOG="$(mktemp)"
MEGATRON_LOG="$(mktemp)"
FLUIDMOE_LOG="$(mktemp)"
OVERLAP_LOG="$(mktemp)"
trap 'rm -f "${NATIVE_LOG}" "${MEGATRON_LOG}" "${FLUIDMOE_LOG}" "${OVERLAP_LOG}"' EXIT

echo "[1/4] Native Baseline (bulk AllToAll)"
"${TORCHRUN_BIN}" "${COMMON_ARGS[@]}" --impl native | tee "${NATIVE_LOG}"
echo

echo "[2/4] Megatron Baseline"
"${TORCHRUN_BIN}" "${COMMON_ARGS[@]}" --impl megatron | tee "${MEGATRON_LOG}"
echo

echo "[3/4] FluidMoE"
"${TORCHRUN_BIN}" "${COMMON_ARGS[@]}" --impl fluidmoe | tee "${FLUIDMOE_LOG}"
echo

echo "[4/4] Overlap Analysis"
"${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" \
  tools/overlap_ratio_analyzer.py --model "${MODEL}" \
  --dp-size "${DP_SIZE}" --cp-size "${CP_SIZE}" --ep-size "${EP_SIZE}" \
  --warmup "${WARMUP}" --iters "${ITERS}" | tee "${OVERLAP_LOG}"
echo

"${PYTHON_BIN}" - "${NATIVE_LOG}" "${MEGATRON_LOG}" "${FLUIDMOE_LOG}" "${OVERLAP_LOG}" <<'PY'
import re
import sys

native_log, megatron_log, fluidmoe_log, overlap_log = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(native_log, "r", encoding="utf-8") as f:
    native_text = f.read()
with open(megatron_log, "r", encoding="utf-8") as f:
    megatron_text = f.read()
with open(fluidmoe_log, "r", encoding="utf-8") as f:
    fluidmoe_text = f.read()
with open(overlap_log, "r", encoding="utf-8") as f:
    overlap_text = f.read().strip()

baseline_re = r"RESULT impl=\S+ forward_ms=([0-9.]+) iter_ms=([0-9.]+)"
native_match = re.search(baseline_re, native_text)
megatron_match = re.search(baseline_re, megatron_text)
fluidmoe_match = re.search(
    r"RESULT impl=fluidmoe forward_ms=([0-9.]+) sync_iter_ms=([0-9.]+) interleaved_iter_ms=([0-9.]+)",
    fluidmoe_text,
)

if native_match is None or megatron_match is None or fluidmoe_match is None:
    raise SystemExit("Failed to parse RESULT lines from benchmark output.")

nat_fwd, nat_iter = map(float, native_match.groups())
meg_fwd, meg_iter = map(float, megatron_match.groups())
fluid_fwd, fluid_sync, fluid_inter = map(float, fluidmoe_match.groups())

# Parse overlap ratio (last line is the float value from rank 0)
overlap_ratio = None
for line in reversed(overlap_text.splitlines()):
    line = line.strip()
    try:
        overlap_ratio = float(line)
        break
    except ValueError:
        continue

print("============================================================")
print("Benchmark Results")
print("------------------------------------------------------------")
print(f"  Native Baseline:   forward={nat_fwd:.2f}ms  iter={nat_iter:.2f}ms")
print(f"  Megatron Baseline: forward={meg_fwd:.2f}ms  iter={meg_iter:.2f}ms")
print(f"  FluidMoE:          forward={fluid_fwd:.2f}ms  sync_iter={fluid_sync:.2f}ms  interleaved_iter={fluid_inter:.2f}ms")
if overlap_ratio is not None:
    print(f"  A2A overlap ratio: {overlap_ratio:.1%}")
print("------------------------------------------------------------")
print("Speedup (vs Native Baseline)")
print(f"  forward speedup      = {nat_fwd / fluid_fwd:.3f}x")
print(f"  sync iter speedup    = {nat_iter / fluid_sync:.3f}x")
print(f"  interleaved speedup  = {nat_iter / fluid_inter:.3f}x")
print("------------------------------------------------------------")
print("Speedup (vs Megatron Baseline)")
print(f"  forward speedup      = {meg_fwd / fluid_fwd:.3f}x")
print(f"  sync iter speedup    = {meg_iter / fluid_sync:.3f}x")
print(f"  interleaved speedup  = {meg_iter / fluid_inter:.3f}x")
print("============================================================")
PY

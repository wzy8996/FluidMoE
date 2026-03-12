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

MEGATRON_LOG="$(mktemp)"
FLUIDMOE_LOG="$(mktemp)"
OVERLAP_LOG="$(mktemp)"
trap 'rm -f "${MEGATRON_LOG}" "${FLUIDMOE_LOG}" "${OVERLAP_LOG}"' EXIT

echo "[1/3] Megatron"
"${TORCHRUN_BIN}" "${COMMON_ARGS[@]}" --impl megatron | tee "${MEGATRON_LOG}"
echo

echo "[2/3] FluidMoE"
"${TORCHRUN_BIN}" "${COMMON_ARGS[@]}" --impl fluidmoe | tee "${FLUIDMOE_LOG}"
echo

echo "[3/3] Overlap Analysis"
"${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" \
  tools/overlap_ratio_analyzer.py --model "${MODEL}" \
  --dp-size "${DP_SIZE}" --cp-size "${CP_SIZE}" --ep-size "${EP_SIZE}" \
  --warmup "${WARMUP}" --iters "${ITERS}" | tee "${OVERLAP_LOG}"
echo

"${PYTHON_BIN}" - "${MEGATRON_LOG}" "${FLUIDMOE_LOG}" "${OVERLAP_LOG}" <<'PY'
import re
import sys

megatron_log, fluidmoe_log, overlap_log = sys.argv[1], sys.argv[2], sys.argv[3]

with open(megatron_log, "r", encoding="utf-8") as f:
    megatron_text = f.read()
with open(fluidmoe_log, "r", encoding="utf-8") as f:
    fluidmoe_text = f.read()
with open(overlap_log, "r", encoding="utf-8") as f:
    overlap_text = f.read().strip()

megatron_match = re.search(r"RESULT impl=megatron forward_ms=([0-9.]+) iter_ms=([0-9.]+)", megatron_text)
fluidmoe_match = re.search(
    r"RESULT impl=fluidmoe forward_ms=([0-9.]+) sync_iter_ms=([0-9.]+) interleaved_iter_ms=([0-9.]+)",
    fluidmoe_text,
)

if megatron_match is None or fluidmoe_match is None:
    raise SystemExit("Failed to parse RESULT lines from benchmark output.")

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
print("Speedup Summary")
print(f"  forward speedup      = {meg_fwd / fluid_fwd:.3f}x")
print(f"  sync iter speedup    = {meg_iter / fluid_sync:.3f}x")
print(f"  interleaved speedup  = {meg_iter / fluid_inter:.3f}x")
if overlap_ratio is not None:
    print(f"  A2A overlap ratio    = {overlap_ratio:.1%}")
print("============================================================")
PY

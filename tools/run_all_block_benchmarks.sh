#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${TOOLS_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

MODEL="${MODEL:-mixtral_8x7b}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DP_SIZE="${DP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-2}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"

echo "============================================================"
echo "Run All Block Benchmarks"
echo "  model=${MODEL}"
echo "  nproc_per_node=${NPROC_PER_NODE}"
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
trap 'rm -f "${MEGATRON_LOG}" "${FLUIDMOE_LOG}"' EXIT

echo "[1/2] Megatron"
"${TORCHRUN_BIN}" "${COMMON_ARGS[@]}" --impl megatron | tee "${MEGATRON_LOG}"
echo

echo "[2/2] FluidMoE"
"${TORCHRUN_BIN}" "${COMMON_ARGS[@]}" --impl fluidmoe | tee "${FLUIDMOE_LOG}"
echo

"${PYTHON_BIN}" - "${MEGATRON_LOG}" "${FLUIDMOE_LOG}" <<'PY'
import re
import sys

megatron_log, fluidmoe_log = sys.argv[1], sys.argv[2]

with open(megatron_log, "r", encoding="utf-8") as f:
    megatron_text = f.read()
with open(fluidmoe_log, "r", encoding="utf-8") as f:
    fluidmoe_text = f.read()

megatron_match = re.search(r"RESULT impl=megatron forward_ms=([0-9.]+) iter_ms=([0-9.]+)", megatron_text)
fluidmoe_match = re.search(
    r"RESULT impl=fluidmoe forward_ms=([0-9.]+) sync_iter_ms=([0-9.]+) interleaved_iter_ms=([0-9.]+)",
    fluidmoe_text,
)

if megatron_match is None or fluidmoe_match is None:
    raise SystemExit("Failed to parse RESULT lines from benchmark output.")

meg_fwd, meg_iter = map(float, megatron_match.groups())
fluid_fwd, fluid_sync, fluid_inter = map(float, fluidmoe_match.groups())

print("============================================================")
print("Speedup Summary")
print(f"  forward speedup      = {meg_fwd / fluid_fwd:.3f}x")
print(f"  sync iter speedup    = {meg_iter / fluid_sync:.3f}x")
print(f"  interleaved speedup  = {meg_iter / fluid_inter:.3f}x")
print("============================================================")
PY

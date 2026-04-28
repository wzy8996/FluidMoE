#!/usr/bin/env bash
# Per-model torchrun wrapper for tests/comm_ratio_table.py.
#
# Why: a hang or NCCL timeout in one model would otherwise kill the rest of
# the multi-model run. Splitting into one torchrun process per model makes
# each run independent — one failure does not block the others.
#
# Usage:
#   bash tests/run_comm_ratio_table.sh \
#     --dp-size 2 --cp-size 4 --ep-size 4 --seq-len 32768 \
#     --warmup 10 --iters 30
#
# Outputs are concatenated into ``comm_ratio_${TIMESTAMP}.md`` for easy
# copy-paste into the paper.

set -uo pipefail

NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
MODELS=(dbrx_base deepseek_v3_mha_proxy glm4_5_air_mha_proxy qwen3_30b_a3b)

# Forward all CLI args (--dp-size / --cp-size / --ep-size / --seq-len /
# --warmup / --iters) verbatim to the python script.
FORWARD_ARGS=("$@")

TS=$(date +%Y%m%d_%H%M%S)
OUT="comm_ratio_${TS}.md"
echo "# comm_ratio_table — $(date)" > "${OUT}"
echo "# Forward args: ${FORWARD_ARGS[*]}" >> "${OUT}"
echo "" >> "${OUT}"
echo "| Model | Fwd A2A exp. | Bwd A2A exp. | DP sync tail | Σ exposed | Iter ms |" >> "${OUT}"
echo "|---|---:|---:|---:|---:|---:|" >> "${OUT}"

for MODEL in "${MODELS[@]}"; do
  echo
  echo "================================================"
  echo "[$(date +%H:%M:%S)] Running ${MODEL}..."
  echo "================================================"

  LOG="comm_ratio_${TS}_${MODEL}.log"
  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    tests/comm_ratio_table.py --models "${MODEL}" \
    "${FORWARD_ARGS[@]}" > "${LOG}" 2>&1
  RC=$?

  if (( RC == 0 )); then
    # Extract the markdown row (line starting with "| <model_name>").
    ROW=$(grep -m1 "^| ${MODEL} " "${LOG}" || true)
    if [[ -n "${ROW}" ]]; then
      echo "${ROW}" >> "${OUT}"
      echo "  -> ${ROW}"
    else
      echo "| ${MODEL} | — | — | — | — | RAN BUT NO ROW (see ${LOG}) |" >> "${OUT}"
      echo "  -> exit 0 but no markdown row; check ${LOG}"
    fi
  else
    echo "| ${MODEL} | — | — | — | — | FAILED (exit=${RC}, see ${LOG}) |" >> "${OUT}"
    echo "  -> FAILED (exit=${RC}); see ${LOG}"
  fi
done

echo
echo "================================================"
echo "Done. Combined output: ${OUT}"
echo "================================================"
cat "${OUT}"

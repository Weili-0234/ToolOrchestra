#!/usr/bin/env bash
set -euo pipefail

# Run a small HLE serving sweep on 5090 using `run_hle_serving_one_w130.sh`.
#
# Usage:
#   export CKPT_DIR=...
#   export INDEX_DIR=...
#   bash scripts/5090/hle_serving/run_hle_serving_sweep.sh <scheduler> [rep_list]
#
# scheduler: baseline | continuum | trnew
# rep_list:  comma-separated (default: "1")
#
# C_LIST is fixed per current experiment decision:
#   [48, 72, 96, 120, 144, 24]

SCHEDULER="${1:?scheduler required (baseline|continuum|trnew)}"
REP_LIST="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IFS=',' read -r -a REPS <<< "${REP_LIST}"

C_LIST=(48 72 96 120 144 24)

echo "[sweep] scheduler=${SCHEDULER} reps=${REP_LIST} C_LIST=${C_LIST[*]}"

for rep in "${REPS[@]}"; do
  for c in "${C_LIST[@]}"; do
    echo "[sweep] starting scheduler=${SCHEDULER} C=${c} rep=${rep}"
    bash "${SCRIPT_DIR}/run_hle_serving_one_w130.sh" "${SCHEDULER}" "${c}" "${rep}"
  done
done


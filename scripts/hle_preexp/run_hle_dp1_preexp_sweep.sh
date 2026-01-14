#!/usr/bin/env bash
# Convenience wrapper to run the full (C × rep) sweep for ONE scheduler.
#
# Order is fixed to match the experiment plan:
#   scheduler -> concurrency -> rep
#
# Usage:
#   bash scripts/hle_preexp/run_hle_dp1_preexp_sweep.sh <scheduler>
#
# Override concurrencies:
#   C_LIST="48 72 96" bash scripts/hle_preexp/run_hle_dp1_preexp_sweep.sh baseline

set -euo pipefail

SCHEDULER="${1:?scheduler required (baseline|continuum|thunderreact)}"
C_LIST="${C_LIST:-48 72 96 120 144 168 192 216}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for c in ${C_LIST}; do
  for rep in 1 2 3; do
    echo "=== sweep scheduler=${SCHEDULER} C=${c} rep=${rep} ==="
    bash "${SCRIPT_DIR}/run_hle_dp1_preexp_one_w130.sh" "${SCHEDULER}" "${c}" "${rep}"
  done
done


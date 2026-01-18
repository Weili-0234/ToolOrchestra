#!/usr/bin/env bash
set -euo pipefail

# Run TAU2Bench inference sweep on a 4×5090 machine with 4 parallel workers (GPU0-3).
#
# Queue format: one task per line
#   <method> <C> <rep>
# where method ∈ {tr,baseline,continuum}.
#
# Workers atomically pop the first line under a shared flock, run one setting
# (restarting vLLM/TR to clear cache), then repeat until the queue is empty.
#
# Usage (on 5090):
#   export CKPT_DIR=...
#   bash scripts/5090/tau2_bench/run_tau2bench_parallel_4gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMPARE_REPO_DIR="$(cd "${REPO_DIR}/.." && pwd)"

CKPT_DIR="${CKPT_DIR:?ERROR: CKPT_DIR not set}"

# Sweep config
C_LIST_STR="${C_LIST_STR:-32,64,96,128}"
REP_LIST="${REP_LIST:-1,2,3}"
DOMAINS="${DOMAINS:-retail telecom airline}"

# Report paths (one per method)
REPORT_TR_MD="${REPORT_TR_MD:-${COMPARE_REPO_DIR}/tau2-serving-5090/tau2bench_tr_10_130.md}"
REPORT_BASELINE_MD="${REPORT_BASELINE_MD:-${COMPARE_REPO_DIR}/tau2-serving-5090/tau2bench_baseline_10_130.md}"
REPORT_CONTINUUM_MD="${REPORT_CONTINUUM_MD:-${COMPARE_REPO_DIR}/tau2-serving-5090/tau2bench_continuum_10_130.md}"

START_WATCHERS="${START_WATCHERS:-1}"

# Port bases (per worker)
ORCH_PORT_BASE="${ORCH_PORT_BASE:-1900}"
TR_ROUTER_PORT_BASE="${TR_ROUTER_PORT_BASE:-28000}"
TR_BACKEND_PORT_BASE="${TR_BACKEND_PORT_BASE:-28100}"

TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="${REPO_DIR}/outputs/tau2_serving_5090_parallel_${TS}"
mkdir -p "${SWEEP_DIR}/logs"

TASKS_FILE="${SWEEP_DIR}/task_queue.txt"
LOCK_FILE="${SWEEP_DIR}/task_queue.lock"

WATCH_PIDS=()

cleanup() {
  set +e
  for pid in "${WATCH_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT

IFS=',' read -r -a C_LIST <<< "${C_LIST_STR}"
IFS=',' read -r -a REP_ARR <<< "${REP_LIST}"

# Build DONE set from existing report state files (optional).
declare -A DONE=()
for st in "${REPORT_TR_MD}.state.json" "${REPORT_BASELINE_MD}.state.json" "${REPORT_CONTINUUM_MD}.state.json"; do
  if [[ ! -f "${st}" ]]; then
    continue
  fi
  python - <<'PY' "${st}" 2>/dev/null || true
import json, re, sys
p=sys.argv[1]
obj=json.load(open(p, "r", encoding="utf-8"))
for out_dir in obj.get("processed_out_dirs", []):
    m=re.search(r"tau2_serving_5090_(?P<m>baseline|continuum|tr)_c(?P<c>\d+)_rep(?P<r>\d+)_", str(out_dir))
    if not m:
        continue
    print(m.group("m"), m.group("c"), m.group("r"))
PY
done | while read -r m c r; do
  [[ -z "${m}" || -z "${c}" || -z "${r}" ]] && continue
  DONE["${m}:${c}:${r}"]=1
done

rm -f "${TASKS_FILE}" "${LOCK_FILE}" "${TASKS_FILE}.tmp"

# Generate task queue in priority order:
#   rep asc, C asc, method priority tr > baseline > continuum
for rep in "${REP_ARR[@]}"; do
  for c in "${C_LIST[@]}"; do
    for m in tr baseline continuum; do
      if [[ -n "${DONE["${m}:${c}:${rep}"]+x}" ]]; then
        continue
      fi
      echo "${m} ${c} ${rep}" >> "${TASKS_FILE}"
    done
  done
done

echo "[sweep] sweep_dir=${SWEEP_DIR}"
echo "[sweep] task_queue=${TASKS_FILE} ($(wc -l < "${TASKS_FILE}") lines)"

SINCE_UNIX="${SINCE_UNIX:-$(date +%s)}"
if [[ "${START_WATCHERS}" == "1" ]]; then
  echo "[watch] starting watchers..."
  python "${SCRIPT_DIR}/watch_tau2bench_metrics.py" \
    --method tr \
    --report-md "${REPORT_TR_MD}" \
    --interval-sec 60 \
    --since-unix "${SINCE_UNIX}" \
    > "${SWEEP_DIR}/logs/watch_tr.log" 2>&1 &
  WATCH_PIDS+=("$!")
  python "${SCRIPT_DIR}/watch_tau2bench_metrics.py" \
    --method baseline \
    --report-md "${REPORT_BASELINE_MD}" \
    --interval-sec 60 \
    --since-unix "${SINCE_UNIX}" \
    > "${SWEEP_DIR}/logs/watch_baseline.log" 2>&1 &
  WATCH_PIDS+=("$!")
  python "${SCRIPT_DIR}/watch_tau2bench_metrics.py" \
    --method continuum \
    --report-md "${REPORT_CONTINUUM_MD}" \
    --interval-sec 60 \
    --since-unix "${SINCE_UNIX}" \
    > "${SWEEP_DIR}/logs/watch_continuum.log" 2>&1 &
  WATCH_PIDS+=("$!")
else
  echo "[watch] disabled (START_WATCHERS=0)"
fi

run_worker() {
  local worker_id="${1:?worker_id}"
  local gpu="${2:?gpu}"
  local orch_port="${3:?orch_port}"
  local tr_router_port="${4:?tr_router_port}"
  local tr_backend_port="${5:?tr_backend_port}"
  local first_m="${6:-}"
  local first_c="${7:-}"
  local first_r="${8:-}"

  echo "[worker${worker_id}] GPU=${gpu} ports orch=${orch_port} tr_router=${tr_router_port} tr_backend=${tr_backend_port}"

  run_one() {
    local m="${1:?method}"
    local c="${2:?c}"
    local r="${3:?rep}"
    echo "[worker${worker_id}] start method=${m} C=${c} rep=${r}"
    ORCH_GPU="${gpu}" \
      ORCH_PORT="${orch_port}" \
      TR_ROUTER_PORT="${tr_router_port}" \
      TR_BACKEND_PORT="${tr_backend_port}" \
      DOMAINS="${DOMAINS}" \
      bash "${SCRIPT_DIR}/run_tau2bench_one_w130.sh" "${m}" "${c}" "${r}"
    echo "[worker${worker_id}] done method=${m} C=${c} rep=${r}"
  }

  if [[ -n "${first_m}" && -n "${first_c}" && -n "${first_r}" ]]; then
    run_one "${first_m}" "${first_c}" "${first_r}"
  fi

  while true; do
    line="$(
      (
        flock -x 200
        if [[ ! -s "${TASKS_FILE}" ]]; then
          exit 1
        fi
        IFS= read -r first_line < "${TASKS_FILE}"
        tail -n +2 "${TASKS_FILE}" > "${TASKS_FILE}.tmp" && mv "${TASKS_FILE}.tmp" "${TASKS_FILE}"
        printf '%s\n' "${first_line}"
      ) 200>"${LOCK_FILE}"
    )" || break

    local m=""
    local c=""
    local r=""
    read -r m c r _rest <<< "${line}"
    [[ -z "${m}" || -z "${c}" || -z "${r}" ]] && continue
    run_one "${m}" "${c}" "${r}"
  done
}

# Deterministic first wave (keeps all 4 GPUs busy and follows priority):
# GPU0: tr 32 1
# GPU1: baseline 32 1
# GPU2: continuum 32 1
# GPU3: tr 64 1
run_worker 0 0 "$((ORCH_PORT_BASE + 0))" "$((TR_ROUTER_PORT_BASE + 0))" "$((TR_BACKEND_PORT_BASE + 0))" tr 32 1 &
P0=$!
run_worker 1 1 "$((ORCH_PORT_BASE + 1))" "$((TR_ROUTER_PORT_BASE + 1))" "$((TR_BACKEND_PORT_BASE + 1))" baseline 32 1 &
P1=$!
run_worker 2 2 "$((ORCH_PORT_BASE + 2))" "$((TR_ROUTER_PORT_BASE + 2))" "$((TR_BACKEND_PORT_BASE + 2))" continuum 32 1 &
P2=$!
run_worker 3 3 "$((ORCH_PORT_BASE + 3))" "$((TR_ROUTER_PORT_BASE + 3))" "$((TR_BACKEND_PORT_BASE + 3))" tr 64 1 &
P3=$!

wait "${P0}" "${P1}" "${P2}" "${P3}" || true
echo "[sweep] DONE"

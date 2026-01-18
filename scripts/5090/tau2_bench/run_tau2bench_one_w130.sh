#!/usr/bin/env bash
# Run ONE TAU2Bench setting on a 5090 worker GPU with fixed 10–130min window metrics.
#
# Per-setting policy:
# - Restart vLLM (and TR router if applicable) to clear KV/prefix cache.
# - Hard timeout (default 145m) to keep sweep progressing.
#
# Usage (on 5090):
#   export CKPT_DIR=/path/to/Nemotron-Orchestrator-8B
#   bash scripts/5090/tau2_bench/run_tau2bench_one_w130.sh <method> <C> <rep>
#
# method:
#   - baseline   (standard vLLM, no router)
#   - continuum  (vllm-continuum, no router; --scheduling-policy continuum)
#   - tr         (ThunderReact-new router + 1 backend)

set -euo pipefail

METHOD="${1:?method required (baseline|continuum|tr)}"
CONCURRENCY="${2:?concurrency required}"
REP_IDX="${3:-1}"

WINDOW_START_SEC="${WINDOW_START_SEC:-600}"
WINDOW_END_SEC="${WINDOW_END_SEC:-7800}"
EVAL_TIMEOUT_MIN="${EVAL_TIMEOUT_MIN:-145}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-2}"

ORCH_GPU="${ORCH_GPU:-0}"
ORCH_PORT="${ORCH_PORT:-1900}"
TR_ROUTER_PORT="${TR_ROUTER_PORT:-28000}"
TR_BACKEND_PORT="${TR_BACKEND_PORT:-28100}"

VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.95}"

BASELINE_CONDA_ENV="${BASELINE_CONDA_ENV:-vllm1}"
TR_CONDA_ENV="${TR_CONDA_ENV:-vllm1}"
CONTINUUM_CONDA_ENV="${CONTINUUM_CONDA_ENV:-vllm-continuum}"

CKPT_DIR="${CKPT_DIR:?ERROR: CKPT_DIR not set}"
DOMAINS="${DOMAINS:-retail telecom airline}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMPARE_REPO_DIR="$(cd "${REPO_DIR}/.." && pwd)"
TAU2_DIR="${REPO_DIR}/evaluation/tau2-bench"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_DIR}/outputs/tau2_serving_5090_${METHOD}_c${CONCURRENCY}_rep${REP_IDX}_${TS}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

USAGE_JSONL="${OUT_DIR}/orchestrator_usage.jsonl"
METRICS_CSV="${OUT_DIR}/prefix_cache_timeseries.csv"
GPU_UTIL_CSV="${OUT_DIR}/gpu_sm_util_timeseries.csv"
MODEL_CONFIG_PATH="${OUT_DIR}/model_config_tau2_5090.json"
SUMMARY_JSON="${OUT_DIR}/window_summary.json"
STEPS_JSON="${OUT_DIR}/steps_summary.json"
COMBINED_JSON="${OUT_DIR}/combined_summary.json"

ORCH_LOG="${LOG_DIR}/orchestrator.log"
TR_ROUTER_LOG="${LOG_DIR}/tr_router.log"
EVAL_STDOUT_LOG="${LOG_DIR}/eval_stdout.log"
SAMPLER_LOG="${LOG_DIR}/metrics_sampler.log"
GPU_SAMPLER_LOG="${LOG_DIR}/gpu_sm_sampler.log"

ORCH_PID=""
TR_PID=""
SAMPLER_PID=""
GPU_SAMPLER_PID=""

source ~/miniconda3/etc/profile.d/conda.sh

ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null || true
echo "[ulimit] nofile soft=$(ulimit -Sn) hard=$(ulimit -Hn)"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

port_in_use() {
  local port="${1:?port}"
  lsof -tiTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
}

kill_port_listeners() {
  local port="${1:?port}"
  local pids=""
  pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "[cleanup] killing listeners on port ${port}: ${pids}"
  kill ${pids} >/dev/null 2>&1 || true
  sleep 0.2
  kill -9 ${pids} >/dev/null 2>&1 || true
}

wait_port_free() {
  local port="${1:?port}"
  local timeout_sec="${2:-20}"
  local t0
  t0="$(date +%s)"
  while port_in_use "${port}"; do
    if (( $(date +%s) - t0 > timeout_sec )); then
      return 1
    fi
    sleep 0.2
  done
  return 0
}

cleanup() {
  set +e
  echo "[cleanup] stopping background processes..."

  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" >/dev/null 2>&1; then
    kill "${SAMPLER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GPU_SAMPLER_PID}" ]] && kill -0 "${GPU_SAMPLER_PID}" >/dev/null 2>&1; then
    kill "${GPU_SAMPLER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${TR_PID}" ]] && kill -0 "${TR_PID}" >/dev/null 2>&1; then
    kill "${TR_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${ORCH_PID}" ]] && kill -0 "${ORCH_PID}" >/dev/null 2>&1; then
    kill "${ORCH_PID}" >/dev/null 2>&1 || true
  fi

  # Kill ports last (best-effort).
  kill_port_listeners "${ORCH_PORT}" || true
  kill_port_listeners "${TR_ROUTER_PORT}" || true
  kill_port_listeners "${TR_BACKEND_PORT}" || true
}
trap cleanup EXIT

echo "=========================================="
echo "=== TAU2Bench (5090) (${METHOD}) ==="
echo "=========================================="
echo "C=${CONCURRENCY} rep=${REP_IDX}"
echo "window offsets (sec): ${WINDOW_START_SEC}..${WINDOW_END_SEC}"
echo "timeout: ${EVAL_TIMEOUT_MIN}m"
echo "GPU: ${ORCH_GPU}"
echo "ports: orch=${ORCH_PORT} tr_router=${TR_ROUTER_PORT} tr_backend=${TR_BACKEND_PORT}"
echo "out: ${OUT_DIR}"

# Clean ports for this worker.
kill_port_listeners "${ORCH_PORT}"
kill_port_listeners "${TR_ROUTER_PORT}"
kill_port_listeners "${TR_BACKEND_PORT}"
wait_port_free "${ORCH_PORT}" || { echo "ERROR: port ${ORCH_PORT} still in use after cleanup" >&2; exit 1; }
wait_port_free "${TR_ROUTER_PORT}" || { echo "ERROR: port ${TR_ROUTER_PORT} still in use after cleanup" >&2; exit 1; }
wait_port_free "${TR_BACKEND_PORT}" || { echo "ERROR: port ${TR_BACKEND_PORT} still in use after cleanup" >&2; exit 1; }

log "phase=preflight start"
echo "[preflight] checking expert tunnels..."
for p in 1910 1911 1912 1913 1904 1905 1906 1920; do
  if ! curl -sf --max-time 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
    echo "ERROR: expert tunnel not ready on 127.0.0.1:${p} (/health failed)" >&2
    exit 1
  fi
done
echo "[preflight] expert tunnels OK."
log "phase=preflight done"

echo "[model_config] writing ${MODEL_CONFIG_PATH}"
cat > "${MODEL_CONFIG_PATH}" <<EOF
{
  "_comment": "Auto-generated for TAU2Bench 5090 inference: orchestrator local, experts via localhost SSH tunnels.",
  "vllm_model_config_path": "${MODEL_CONFIG_PATH}",
  "${CKPT_DIR}": [{"ip_addr": "127.0.0.1", "port": "${ORCH_PORT}"}],

  "openai/gpt-oss-20b": [
    {"ip_addr": "127.0.0.1", "port": "1910"},
    {"ip_addr": "127.0.0.1", "port": "1911"},
    {"ip_addr": "127.0.0.1", "port": "1912"},
    {"ip_addr": "127.0.0.1", "port": "1913"}
  ],
  "Qwen/Qwen3-32B-FP8": [
    {"ip_addr": "127.0.0.1", "port": "1904"},
    {"ip_addr": "127.0.0.1", "port": "1905"},
    {"ip_addr": "127.0.0.1", "port": "1906"}
  ],
  "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": [
    {"ip_addr": "127.0.0.1", "port": "1920"}
  ],

  "oss_expert_mapping": {
    "expert-1": "openai/gpt-oss-20b",
    "expert-2": "Qwen/Qwen3-32B-FP8",
    "expert-3": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
  }
}
EOF

start_baseline() {
  conda activate "${BASELINE_CONDA_ENV}"
  export CUDA_VISIBLE_DEVICES="${ORCH_GPU}"
  vllm serve "${CKPT_DIR}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-force-include-usage \
    --host 0.0.0.0 \
    --port "${ORCH_PORT}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    > "${ORCH_LOG}" 2>&1 &
  ORCH_PID=$!
}

start_continuum() {
  conda activate "${CONTINUUM_CONDA_ENV}"
  export CUDA_VISIBLE_DEVICES="${ORCH_GPU}"
  export RUN_OUTPUT_DIR="${OUT_DIR}"
  vllm serve "${CKPT_DIR}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-force-include-usage \
    --host 0.0.0.0 \
    --port "${ORCH_PORT}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --scheduling-policy continuum \
    > "${ORCH_LOG}" 2>&1 &
  ORCH_PID=$!
}

start_tr() {
  conda activate "${TR_CONDA_ENV}"
  export CUDA_VISIBLE_DEVICES="${ORCH_GPU}"
  vllm serve "${CKPT_DIR}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-force-include-usage \
    --host 0.0.0.0 \
    --port "${TR_BACKEND_PORT}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    > "${ORCH_LOG}" 2>&1 &
  ORCH_PID=$!

  echo "[tr] waiting backend /health..."
  for _ in $(seq 1 600); do
    if ! kill -0 "${ORCH_PID}" >/dev/null 2>&1; then
      echo "ERROR: TR backend exited early. See ${ORCH_LOG}" >&2
      exit 1
    fi
    if curl -sf --max-time 2 "http://127.0.0.1:${TR_BACKEND_PORT}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  echo "[tr] backend ready."

  # Start ThunderReact-new router (sticky routing by program_id).
  # ThunderReact package lives at compare-repo root (ThunderReact -> ThunderReact-new symlink).
  export PYTHONPATH="${COMPARE_REPO_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
  (
    cd "${COMPARE_REPO_DIR}"
    python -m ThunderReact \
      --host 0.0.0.0 \
      --port "${TR_ROUTER_PORT}" \
      --router tr \
      --backends "http://127.0.0.1:${TR_BACKEND_PORT}" \
      --profile \
      --profile-dir "${OUT_DIR}/tr_profiles" \
      --metrics \
      --metrics-interval 5 \
      > "${TR_ROUTER_LOG}" 2>&1
  ) &
  TR_PID=$!

  echo "[tr] waiting router /health..."
  for _ in $(seq 1 600); do
    if ! kill -0 "${TR_PID}" >/dev/null 2>&1; then
      echo "ERROR: TR router exited early. See ${TR_ROUTER_LOG}" >&2
      exit 1
    fi
    if curl -sf --max-time 2 "http://127.0.0.1:${TR_ROUTER_PORT}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  echo "[tr] router ready."

  # Update orchestrator endpoint in model_config to point to TR router.
  python - <<PY
import json
from pathlib import Path
p=Path("${MODEL_CONFIG_PATH}")
cfg=json.loads(p.read_text())
cfg["${CKPT_DIR}"]=[{"ip_addr":"127.0.0.1","port":"${TR_ROUTER_PORT}"}]
p.write_text(json.dumps(cfg,indent=2))
print("[model_config] updated orchestrator endpoint -> TR router")
PY
}

log "phase=orchestrator start method=${METHOD}"
case "${METHOD}" in
  baseline) start_baseline ;;
  continuum) start_continuum ;;
  tr) start_tr ;;
  *) echo "ERROR: unknown method=${METHOD}" >&2; exit 2 ;;
esac

# Wait orchestrator endpoint ready (router for TR, vLLM for baseline/continuum).
READY_PORT="${ORCH_PORT}"
READY_HEALTH_PATH="/health"
if [[ "${METHOD}" == "tr" ]]; then
  READY_PORT="${TR_ROUTER_PORT}"
  READY_HEALTH_PATH="/health"
fi

echo "[orchestrator] waiting ready on port ${READY_PORT}..."
for _ in $(seq 1 600); do
  if [[ "${METHOD}" != "tr" ]]; then
    if ! kill -0 "${ORCH_PID}" >/dev/null 2>&1; then
      echo "ERROR: orchestrator exited early. See ${ORCH_LOG}" >&2
      exit 1
    fi
  fi
  if curl -sf --max-time 2 "http://127.0.0.1:${READY_PORT}${READY_HEALTH_PATH}" >/dev/null 2>&1 && \
     curl -sf --max-time 2 "http://127.0.0.1:${READY_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
echo "[orchestrator] ready."
log "phase=orchestrator ready"

# Start samplers (metrics port: backend for TR, vLLM port otherwise).
METRICS_PORT="${ORCH_PORT}"
if [[ "${METHOD}" == "tr" ]]; then
  METRICS_PORT="${TR_BACKEND_PORT}"
fi

echo "[sampler] starting /metrics sampler (interval=${SAMPLE_INTERVAL_SEC}s) from port ${METRICS_PORT}..."
bash "${REPO_DIR}/scripts/hle_preexp/kv_prefix_cache_hit_sampler.sh" \
  "http://127.0.0.1:${METRICS_PORT}/metrics" \
  "${METRICS_CSV}" \
  "${SAMPLE_INTERVAL_SEC}" \
  > "${SAMPLER_LOG}" 2>&1 &
SAMPLER_PID=$!

echo "[sampler] starting GPU SM sampler (interval=${SAMPLE_INTERVAL_SEC}s) gpu_index=${ORCH_GPU}..."
(source ~/miniconda3/etc/profile.d/conda.sh && conda activate "${BASELINE_CONDA_ENV}" && \
  python "${REPO_DIR}/scripts/preexp/gpu_sm_util_sampler.py" \
    --out-csv "${GPU_UTIL_CSV}" \
    --gpu-index "${ORCH_GPU}" \
    --interval-sec "${SAMPLE_INTERVAL_SEC}" \
    > "${GPU_SAMPLER_LOG}" 2>&1) &
GPU_SAMPLER_PID=$!

# Run tau2-bench evaluation (driver).
export REPO_PATH="${REPO_DIR}"
export TOOL_ORCH_USAGE_LOG_PATH="${USAGE_JSONL}"
export PYTHONUNBUFFERED=1

if [[ "${METHOD}" == "tr" ]]; then
  export TAU2_TRNEW=1
  export ROUTER_URL="http://127.0.0.1:${TR_ROUTER_PORT}"
else
  export TAU2_TRNEW=0
  unset ROUTER_URL >/dev/null 2>&1 || true
fi

log "phase=eval start"
cd "${TAU2_DIR}"
set +e
timeout --signal=TERM --kill-after=30s "${EVAL_TIMEOUT_MIN}m" bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate '${BASELINE_CONDA_ENV}' && \
  exec python run_oss.py \
    --agent-model '${CKPT_DIR}' \
    --skip-server-start \
    --schedule-mode global \
    --model-config-path '${MODEL_CONFIG_PATH}' \
    --domains ${DOMAINS} \
    --max-concurrency '${CONCURRENCY}' \
    --num-trials 1 \
    --num-repeats 4 \
    --max-steps 50 \
    --user-llm 'openai/gpt-oss-20b' \
    --log-level PROFILE \
    --output-dir '${OUT_DIR}/outputs' \
    --log-dir '${LOG_DIR}' \
  " 2>&1 | tee -a "${EVAL_STDOUT_LOG}"
EVAL_RC=${PIPESTATUS[0]}
set -e
echo "[eval] exit_code=${EVAL_RC} (expected 124 if timed out)"
log "phase=eval done rc=${EVAL_RC}"

echo "[summarize] computing window stats (${WINDOW_START_SEC}..${WINDOW_END_SEC})..."
python "${REPO_DIR}/scripts/preexp/summarize_dp1_prefix_cache_window.py" \
  --metrics-csv "${METRICS_CSV}" \
  --usage-jsonl "${USAGE_JSONL}" \
  --eval-log "${LOG_DIR}/eval_global.log" \
  --start-offset-sec "${WINDOW_START_SEC}" \
  --end-offset-sec "${WINDOW_END_SEC}" \
  --windows "${WINDOW_START_SEC}:${WINDOW_END_SEC}" \
  --out-json "${SUMMARY_JSON}" \
  >/dev/null 2>&1 || true

python "${REPO_DIR}/scripts/preexp/summarize_steps_per_sec_window.py" \
  --usage-jsonl "${USAGE_JSONL}" \
  --eval-log "${LOG_DIR}/eval_global.log" \
  --start-offset-sec "${WINDOW_START_SEC}" \
  --end-offset-sec "${WINDOW_END_SEC}" \
  > "${STEPS_JSON}" 2>/dev/null || true

SUMMARY_JSON="${SUMMARY_JSON}" STEPS_JSON="${STEPS_JSON}" METRICS_CSV="${METRICS_CSV}" USAGE_JSONL="${USAGE_JSONL}" GPU_UTIL_CSV="${GPU_UTIL_CSV}" \
  python - <<'PY' > "${COMBINED_JSON}"
import json, os
from pathlib import Path

summary = {}
steps = {}
sp = Path(os.environ["SUMMARY_JSON"])
tp = Path(os.environ["STEPS_JSON"])
if sp.exists():
    try:
        summary = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        summary = {}
if tp.exists():
    try:
        steps = json.loads(tp.read_text(encoding="utf-8"))
    except Exception:
        steps = {}

out = {
    "ok": bool(summary.get("ok")),
    "t0_first_request_unix": summary.get("t0_first_request_unix"),
    "t0_first_request_iso": summary.get("t0_first_request_iso"),
    "windows": summary.get("windows"),
    "steps": steps,
    "paths": {
        "metrics_csv": os.environ.get("METRICS_CSV"),
        "usage_jsonl": os.environ.get("USAGE_JSONL"),
        "gpu_util_csv": os.environ.get("GPU_UTIL_CSV"),
    },
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY

echo "[done] OUT_DIR=${OUT_DIR}"

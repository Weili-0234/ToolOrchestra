#!/usr/bin/env bash
# Run ONE HLE DP=1 scheduler pre-experiment setting on a single GPU node (10–130min window).
#
# Window definition:
#   t0 = first successful orchestrator response (from orchestrator_usage.jsonl)
#   window = [t0 + 10min, t0 + 130min]  (600s .. 7800s)
#
# Usage (run on compute node inside a long-lived salloc allocation):
#   export CKPT_DIR=/path/to/Nemotron-Orchestrator-8B
#   bash scripts/hle_preexp/run_hle_dp1_preexp_one_w130.sh <scheduler> <max_concurrency> <rep_idx>
#
# scheduler:
#   - baseline      (vllm-router, session routing by job_id)
#   - continuum     (vllm-router + vLLM --scheduling-policy continuum)
#   - thunderreact  (ThunderReact multinode_router.py + single backend)
#
# Outputs:
#   - Output dir: <repo>/outputs/hle_preexp_dp1_<sched>_c<conc>_t4_r16_rep<rep>_<ts>/
#   - Logs dir:   $HOME/logs/hle_preexp_dp1_<sched>_c<conc>_t4_r16_rep<rep>_<ts>/

set -euo pipefail

SCHEDULER="${1:?scheduler required (baseline|continuum|thunderreact)}"
CONCURRENCY="${2:?max_concurrency required (e.g., 48/72/96/120/144/168/192/216)}"
REP_IDX="${3:?rep_idx required (1..3)}"

ROUTER_PORT="${ROUTER_PORT:-8000}"
BACKEND_PORT="${BACKEND_PORT:-8100}"

ORCH_GPU_ID="${ORCH_GPU_ID:-auto}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.95}"

NUM_TRIALS="${NUM_TRIALS:-4}"
NUM_REPEATS="${NUM_REPEATS:-16}"
EVAL_TIMEOUT_MIN="${EVAL_TIMEOUT_MIN:-145}"

WINDOW_START_SEC="${WINDOW_START_SEC:-600}"
WINDOW_END_SEC="${WINDOW_END_SEC:-7800}"

SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT_DIR="$(cd "${REPO_DIR}/.." && pwd)"
EVAL_DIR="${REPO_DIR}/evaluation"

CKPT_DIR="${CKPT_DIR:?ERROR: CKPT_DIR not set}"

case "${SCHEDULER}" in
  baseline) PREEXP_CONDA_ENV_DEFAULT="vllm-router" ;;
  continuum) PREEXP_CONDA_ENV_DEFAULT="vllm-router-continuum" ;;
  thunderreact) PREEXP_CONDA_ENV_DEFAULT="vllm1" ;;
  *)
    echo "ERROR: unknown scheduler '${SCHEDULER}' (expected baseline|continuum|thunderreact)" >&2
    exit 2
    ;;
esac

source ~/.bashrc 2>/dev/null || true
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${PREEXP_CONDA_ENV:-${PREEXP_CONDA_ENV_DEFAULT}}" >/dev/null 2>&1 || true
fi

ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null || true
echo "[ulimit] nofile soft=$(ulimit -Sn) hard=$(ulimit -Hn)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python 2>/dev/null || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${HOME}/miniconda3/envs/${PREEXP_CONDA_ENV:-${PREEXP_CONDA_ENV_DEFAULT}}/bin/python"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python executable not found (need conda env ${PREEXP_CONDA_ENV:-${PREEXP_CONDA_ENV_DEFAULT}})." >&2
  exit 127
fi
echo "[env] python_bin=${PYTHON_BIN}"

VLLM_BIN="${VLLM_BIN:-}"
if [[ -z "${VLLM_BIN}" ]]; then
  VLLM_BIN="$(command -v vllm 2>/dev/null || true)"
fi
if [[ -z "${VLLM_BIN}" ]]; then
  VLLM_BIN="${HOME}/miniconda3/envs/${PREEXP_CONDA_ENV:-${PREEXP_CONDA_ENV_DEFAULT}}/bin/vllm"
fi
if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "ERROR: vllm executable not found (need conda env ${PREEXP_CONDA_ENV:-${PREEXP_CONDA_ENV_DEFAULT}})." >&2
  exit 127
fi
echo "[env] vllm_bin=${VLLM_BIN}"

VLLM_ROUTER_BIN="${VLLM_ROUTER_BIN:-}"
if [[ -z "${VLLM_ROUTER_BIN}" ]]; then
  VLLM_ROUTER_BIN="$(command -v vllm-router 2>/dev/null || true)"
fi

# Fix for potential unbound variables in setup_envs.sh when running under bash -u.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
export TOGETHER_API_KEY="${TOGETHER_API_KEY:-}"
export NEBIUS_API_KEY="${NEBIUS_API_KEY:-}"
export TAVILY_KEY="${TAVILY_KEY:-}"
source "${REPO_DIR}/setup_envs.sh" > /dev/null 2>&1 || true

detect_running_job_out() {
  local job_name="$1"
  local fallback_glob="$2"
  if command -v squeue >/dev/null 2>&1; then
    local jid
    jid="$(squeue --me -h -n "${job_name}" -t RUNNING -o '%i' 2>/dev/null | head -n 1 || true)"
    if [[ -n "${jid}" ]]; then
      local p="${ROOT_DIR}/slurm/${job_name}/${job_name}_${jid}.out"
      if [[ -f "${p}" ]]; then echo "${p}"; return 0; fi
    fi
  fi
  ls -t ${fallback_glob} 2>/dev/null | head -1 || true
}

extract_ipv4_from_file() {
  local f="$1"
  grep -m1 -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' "${f}" 2>/dev/null || echo ""
}

# Expert/retrieval endpoints (prefer explicit env vars; otherwise autodetect from our sbatch logs).
HLE_EXPERT_REASONER_IP="${HLE_EXPERT_REASONER_IP:-}"
HLE_EXPERT_ANSWER_IP="${HLE_EXPERT_ANSWER_IP:-}"
HLE_EXPERT_SEARCH_IP="${HLE_EXPERT_SEARCH_IP:-}"
HLE_RETRIEVAL_IP="${HLE_RETRIEVAL_IP:-}"

HLE_EXPERT_REASONER_PORT_BASE="${HLE_EXPERT_REASONER_PORT_BASE:-1810}"
HLE_EXPERT_ANSWER_PORT_BASE="${HLE_EXPERT_ANSWER_PORT_BASE:-1820}"
HLE_EXPERT_SEARCH_PORT_BASE="${HLE_EXPERT_SEARCH_PORT_BASE:-1840}"
HLE_RETRIEVAL_PORT="${HLE_RETRIEVAL_PORT:-8765}"

if [[ -z "${HLE_EXPERT_REASONER_IP}" ]]; then
  OUT="$(detect_running_job_out "hle_expert_reasoner" "${ROOT_DIR}/slurm/hle_expert_logs/reasoner/hle_expert_reasoner_*.out")"
  HLE_EXPERT_REASONER_IP="$(extract_ipv4_from_file "${OUT}")"
fi
if [[ -z "${HLE_EXPERT_ANSWER_IP}" ]]; then
  OUT="$(detect_running_job_out "hle_expert_answer" "${ROOT_DIR}/slurm/hle_expert_logs/answer/hle_expert_answer_*.out")"
  HLE_EXPERT_ANSWER_IP="$(extract_ipv4_from_file "${OUT}")"
fi
if [[ -z "${HLE_EXPERT_SEARCH_IP}" ]]; then
  OUT="$(detect_running_job_out "hle_expert_search" "${ROOT_DIR}/slurm/hle_expert_logs/search/hle_expert_search_*.out")"
  HLE_EXPERT_SEARCH_IP="$(extract_ipv4_from_file "${OUT}")"
fi
if [[ -z "${HLE_RETRIEVAL_IP}" ]]; then
  OUT="$(detect_running_job_out "hle_retrieval" "${ROOT_DIR}/slurm/hle_retrieval_logs/hle_retrieval_*.out")"
  HLE_RETRIEVAL_IP="$(extract_ipv4_from_file "${OUT}")"
fi

if [[ -z "${HLE_EXPERT_REASONER_IP}" || -z "${HLE_EXPERT_ANSWER_IP}" || -z "${HLE_EXPERT_SEARCH_IP}" || -z "${HLE_RETRIEVAL_IP}" ]]; then
  echo "ERROR: Missing expert/retrieval IPs." >&2
  echo "Set env vars: HLE_EXPERT_REASONER_IP, HLE_EXPERT_ANSWER_IP, HLE_EXPERT_SEARCH_IP, HLE_RETRIEVAL_IP" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${ROOT_DIR}/outputs/hle_preexp_dp1_${SCHEDULER}_c${CONCURRENCY}_t${NUM_TRIALS}_r${NUM_REPEATS}_rep${REP_IDX}_${TIMESTAMP}"
LOG_DIR="${HOME}/logs/hle_preexp_dp1_${SCHEDULER}_c${CONCURRENCY}_t${NUM_TRIALS}_r${NUM_REPEATS}_rep${REP_IDX}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

METRICS_CSV="${OUTPUT_DIR}/prefix_cache_timeseries.csv"
USAGE_JSONL="${OUTPUT_DIR}/orchestrator_usage.jsonl"
GPU_UTIL_CSV="${OUTPUT_DIR}/gpu_sm_util_timeseries.csv"
MODEL_CONFIG_PATH="${OUTPUT_DIR}/model_config_hle_dp1.json"
SUMMARY_JSON="${OUTPUT_DIR}/window_summary.json"
STEPS_JSON="${OUTPUT_DIR}/steps_summary.json"
COMBINED_JSON="${OUTPUT_DIR}/combined_summary.json"

ORCH_BACKEND_LOG="${LOG_DIR}/orchestrator_backend.log"
ORCH_ROUTER_LOG="${LOG_DIR}/orchestrator_router.log"

echo "=========================================="
echo "=== HLE PreExp: DP=1 (${SCHEDULER}) (10–130min) ==="
echo "=========================================="
echo "conda_env: ${PREEXP_CONDA_ENV:-${PREEXP_CONDA_ENV_DEFAULT}}"
echo "max_concurrency: ${CONCURRENCY}"
echo "rep_idx: ${REP_IDX}"
echo "num_trials: ${NUM_TRIALS}"
echo "num_repeats: ${NUM_REPEATS}"
echo "eval timeout: ${EVAL_TIMEOUT_MIN}m"
echo "window offsets (sec): ${WINDOW_START_SEC}..${WINDOW_END_SEC}"
echo "router: http://127.0.0.1:${ROUTER_PORT}"
echo "backend: http://127.0.0.1:${BACKEND_PORT}"
echo "experts:"
echo "  reasoner_ip=${HLE_EXPERT_REASONER_IP} ports=${HLE_EXPERT_REASONER_PORT_BASE}..$((HLE_EXPERT_REASONER_PORT_BASE+3))"
echo "  answer_ip=${HLE_EXPERT_ANSWER_IP} ports=${HLE_EXPERT_ANSWER_PORT_BASE}..$((HLE_EXPERT_ANSWER_PORT_BASE+3))"
echo "  search_ip=${HLE_EXPERT_SEARCH_IP} ports=${HLE_EXPERT_SEARCH_PORT_BASE}..$((HLE_EXPERT_SEARCH_PORT_BASE+3))"
echo "  retrieval_ip=${HLE_RETRIEVAL_IP} port=${HLE_RETRIEVAL_PORT}"
echo "output: ${OUTPUT_DIR}"
echo "logs: ${LOG_DIR}"

echo "[cleanup] Killing any previous processes on ports ${ROUTER_PORT}/${BACKEND_PORT} (best-effort)..."
fuser -k "${ROUTER_PORT}/tcp" >/dev/null 2>&1 || true
fuser -k "${BACKEND_PORT}/tcp" >/dev/null 2>&1 || true
pkill -f "vllm serve ${CKPT_DIR}" >/dev/null 2>&1 || true
pkill -f "vllm-router" >/dev/null 2>&1 || true
pkill -f "multinode_router.py" >/dev/null 2>&1 || true

pick_gpu_id() {
  local best_id=""
  local best_free="-1"
  while IFS=',' read -r idx used total; do
    idx="$(echo "${idx}" | tr -d ' ')"
    used="$(echo "${used}" | tr -d ' ')"
    total="$(echo "${total}" | tr -d ' ')"
    [[ "${idx}" =~ ^[0-9]+$ ]] || continue
    [[ "${used}" =~ ^[0-9]+$ ]] || continue
    [[ "${total}" =~ ^[0-9]+$ ]] || continue
    free=$((total - used))
    if (( free > best_free )); then
      best_free="${free}"
      best_id="${idx}"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
  echo "${best_id:-0}"
}

if [[ "${ORCH_GPU_ID}" == "auto" ]]; then
  ORCH_GPU_ID="$(pick_gpu_id)"
  echo "[gpu] ORCH_GPU_ID=auto -> selected GPU ${ORCH_GPU_ID}"
fi
if ! [[ "${ORCH_GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: ORCH_GPU_ID must be a GPU index or 'auto' (got: ${ORCH_GPU_ID})" >&2
  exit 1
fi

# persistent cache for torch compile (separate across schedulers)
CACHE_BASE_DIR="${HOME}/haokang/cache/vllm_compile/hle_dp1/${SCHEDULER}"
mkdir -p "${CACHE_BASE_DIR}"
HOSTNAME="$(hostname)"
INSTANCE_CACHE_DIR="${CACHE_BASE_DIR}/${HOSTNAME}/orchestrator_gpu${ORCH_GPU_ID}"
mkdir -p "${INSTANCE_CACHE_DIR}/inductor" "${INSTANCE_CACHE_DIR}/vllm"

cleanup() {
  echo "[cleanup] stopping sampler/eval/orchestrator..."
  jobs -p | xargs -r kill >/dev/null 2>&1 || true
  fuser -k "${ROUTER_PORT}/tcp" >/dev/null 2>&1 || true
  fuser -k "${BACKEND_PORT}/tcp" >/dev/null 2>&1 || true
  pkill -f "vllm serve ${CKPT_DIR}" >/dev/null 2>&1 || true
  pkill -f "vllm-router" >/dev/null 2>&1 || true
  pkill -f "multinode_router.py" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[orchestrator] Starting backend (${SCHEDULER})..."
export CUDA_VISIBLE_DEVICES="${ORCH_GPU_ID}"
BACKEND_CMD=(
  "${VLLM_BIN}" serve "${CKPT_DIR}"
  --host 0.0.0.0
  --port "${BACKEND_PORT}"
  --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --enable-force-include-usage
  --enable-auto-tool-choice
  --tool-call-parser hermes
)
if [[ "${SCHEDULER}" == "continuum" ]]; then
  BACKEND_CMD+=(--scheduling-policy continuum)
  export RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_DIR}/continuum_exp}"
  mkdir -p "${RUN_OUTPUT_DIR}"
  echo "[continuum] RUN_OUTPUT_DIR=${RUN_OUTPUT_DIR}"
fi

echo "[orchestrator-backend] ${BACKEND_CMD[*]}"
TORCHINDUCTOR_CACHE_DIR="${INSTANCE_CACHE_DIR}/inductor" \
VLLM_CACHE_ROOT="${INSTANCE_CACHE_DIR}/vllm" \
  setsid "${BACKEND_CMD[@]}" 2>&1 | tee "${ORCH_BACKEND_LOG}" &
BACKEND_PID=$!

echo "[orchestrator-backend] Waiting for /health..."
for _ in $(seq 1 900); do
  if ! kill -0 "${BACKEND_PID}" >/dev/null 2>&1; then
    echo "ERROR: backend vLLM exited early. See ${ORCH_BACKEND_LOG}" >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "[orchestrator] Starting router (${SCHEDULER})..."
if [[ "${SCHEDULER}" == "thunderreact" ]]; then
  export VLLM_BACKENDS="http://127.0.0.1:${BACKEND_PORT}"
  export ROUTER_PORT
  export ROUTER_HOST="0.0.0.0"
  export ROUTER_LOG_LEVEL="${ROUTER_LOG_LEVEL:-info}"
  export ROUTER_URL="http://127.0.0.1:${ROUTER_PORT}"
  setsid "${PYTHON_BIN}" "${REPO_DIR}/multinode_router.py" 2>&1 | tee "${ORCH_ROUTER_LOG}" &
  ROUTER_PID=$!
else
  if [[ -z "${VLLM_ROUTER_BIN}" || ! -x "${VLLM_ROUTER_BIN}" ]]; then
    echo "ERROR: vllm-router binary not found (need conda env ${PREEXP_CONDA_ENV:-${PREEXP_CONDA_ENV_DEFAULT}})." >&2
    exit 127
  fi
  unset ROUTER_URL || true
  setsid "${VLLM_ROUTER_BIN}" --host 0.0.0.0 --port "${ROUTER_PORT}" \
    --service-discovery static \
    --static-backends "http://127.0.0.1:${BACKEND_PORT}" \
    --static-models "${CKPT_DIR}" \
    --routing-logic session \
    --session-key "job_id" \
    --engine-stats-interval "${SAMPLE_INTERVAL_SEC}" \
    --log-stats \
    2>&1 | tee "${ORCH_ROUTER_LOG}" &
  ROUTER_PID=$!
fi

echo "[orchestrator-router] Waiting for router readiness..."
for _ in $(seq 1 900); do
  if ! kill -0 "${ROUTER_PID}" >/dev/null 2>&1; then
    echo "ERROR: router exited early. See ${ORCH_ROUTER_LOG}" >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${ROUTER_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "[precheck] Validating cached_tokens is present..."
ORCH_BASE_URL="http://127.0.0.1:${ROUTER_PORT}/v1" "${PYTHON_BIN}" - <<'PY'
import os, time
from openai import OpenAI

base_url = os.environ["ORCH_BASE_URL"]
client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=30.0)
models = client.models.list()
model_id = models.data[0].id if models.data else None
if not model_id:
    raise SystemExit("ERROR: no model id from /v1/models")

job_id = "hle_precheck:job"
pt_any = False
cached_any = False
last_usage = None
for i in range(6):
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role":"user","content":"ping"}],
        max_tokens=1,
        temperature=0,
        extra_body={"job_id": job_id, "is_last_step": False},
    )
    u = getattr(resp, "usage", None)
    last_usage = u
    pt = getattr(u, "prompt_tokens", None)
    ptd = getattr(u, "prompt_tokens_details", None)
    ct = None
    if ptd is not None:
        ct = getattr(ptd, "cached_tokens", None) or getattr(ptd, "cached_prompt_tokens", None)
    if pt is not None:
        pt_any = True
    if ct is not None and float(ct) > 0:
        cached_any = True
    print(f"[precheck] attempt={i+1} prompt_tokens={pt} cached_tokens={ct}")
    time.sleep(0.2)

if not pt_any:
    raise SystemExit(f"ERROR: missing prompt_tokens in response usage. last_usage={last_usage}")
if not cached_any:
    raise SystemExit(
        "ERROR: could not observe any cached_tokens>0 across repeated identical requests. "
        "Per-request cache hit metric would be unusable."
    )
print(f"[precheck] ok model_id={model_id} (observed cached_tokens>0)")
PY

echo "[metrics] Starting /metrics sampler (${SAMPLE_INTERVAL_SEC}s)..."
bash "${SCRIPT_DIR}/kv_prefix_cache_hit_sampler.sh" "http://127.0.0.1:${BACKEND_PORT}/metrics" "${METRICS_CSV}" "${SAMPLE_INTERVAL_SEC}" \
  > "${LOG_DIR}/metrics_sampler.log" 2>&1 &

echo "[gpu] Starting GPU SM util sampler (${SAMPLE_INTERVAL_SEC}s)..."
"${PYTHON_BIN}" "${REPO_DIR}/scripts/preexp/gpu_sm_util_sampler.py" \
  --out-csv "${GPU_UTIL_CSV}" \
  --gpu-index "${ORCH_GPU_ID}" \
  --interval-sec "${SAMPLE_INTERVAL_SEC}" \
  > "${LOG_DIR}/gpu_util_sampler.log" 2>&1 &

echo "[eval] Generating HLE model config..."
CKPT_DIR="${CKPT_DIR}" ROUTER_PORT="${ROUTER_PORT}" \
  HLE_EXPERT_REASONER_IP="${HLE_EXPERT_REASONER_IP}" HLE_EXPERT_REASONER_PORT_BASE="${HLE_EXPERT_REASONER_PORT_BASE}" \
  HLE_EXPERT_ANSWER_IP="${HLE_EXPERT_ANSWER_IP}" HLE_EXPERT_ANSWER_PORT_BASE="${HLE_EXPERT_ANSWER_PORT_BASE}" \
  HLE_EXPERT_SEARCH_IP="${HLE_EXPERT_SEARCH_IP}" HLE_EXPERT_SEARCH_PORT_BASE="${HLE_EXPERT_SEARCH_PORT_BASE}" \
  HLE_RETRIEVAL_IP="${HLE_RETRIEVAL_IP}" HLE_RETRIEVAL_PORT="${HLE_RETRIEVAL_PORT}" \
  MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH}" \
  "${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path

ckpt = os.environ["CKPT_DIR"]
router_port = int(os.environ["ROUTER_PORT"])

def ports(base: int, n: int = 4):
    return [base + i for i in range(n)]

cfg = {
    "_comment": "Auto-generated model config for HLE rollout preexp (DP=1 orchestrator behind router)",
    "vllm_model_config_path": os.environ["MODEL_CONFIG_PATH"],
    ckpt: [{"ip_addr": "127.0.0.1", "port": router_port}],
    "Qwen/Qwen2.5-Coder-14B-Instruct": [
        {"ip_addr": os.environ["HLE_EXPERT_REASONER_IP"], "port": p} for p in ports(int(os.environ["HLE_EXPERT_REASONER_PORT_BASE"]))
    ],
    "Qwen/Qwen3-32B-FP8": [
        {"ip_addr": os.environ["HLE_EXPERT_ANSWER_IP"], "port": p} for p in ports(int(os.environ["HLE_EXPERT_ANSWER_PORT_BASE"]))
    ],
    "openai/gpt-oss-20b": [
        {"ip_addr": os.environ["HLE_EXPERT_SEARCH_IP"], "port": p} for p in ports(int(os.environ["HLE_EXPERT_SEARCH_PORT_BASE"]))
    ],
    "retrieval": [{"ip_addr": os.environ["HLE_RETRIEVAL_IP"], "port": int(os.environ["HLE_RETRIEVAL_PORT"])}],
}

out = Path(os.environ["MODEL_CONFIG_PATH"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[model_config] wrote {out}")
PY

export TOOL_ORCH_USAGE_LOG_PATH="${USAGE_JSONL}"
export HLE_LOG_STREAM=1

HLE_EXAMPLE_PATH="${HLE_EXAMPLE_PATH:-${EVAL_DIR}/hle.jsonl}"
HLE_MAX_ROUNDS="${HLE_MAX_ROUNDS:-50}"
HLE_ORCH_MAX_TOKENS="${HLE_ORCH_MAX_TOKENS:-12000}"

echo "[eval] Starting eval_hle_rollout_oss.py (timeout ${EVAL_TIMEOUT_MIN}m)..."
cd "${EVAL_DIR}"
set +e
timeout "${EVAL_TIMEOUT_MIN}m" "${PYTHON_BIN}" eval_hle_rollout_oss.py \
  --model_name "${CKPT_DIR}" \
  --output_dir "${OUTPUT_DIR}/outputs" \
  --model_config "${MODEL_CONFIG_PATH}" \
  --example_path "${HLE_EXAMPLE_PATH}" \
  --concurrency "${CONCURRENCY}" \
  --num_trials "${NUM_TRIALS}" \
  --num_repeats "${NUM_REPEATS}" \
  --max_rounds "${HLE_MAX_ROUNDS}" \
  --orch_max_tokens "${HLE_ORCH_MAX_TOKENS}" \
  --log_level PROFILE \
  --log_file "${LOG_DIR}/hle_profile.log" \
  2>&1 | tee "${LOG_DIR}/eval_driver.log"
EVAL_RC=${PIPESTATUS[0]}
set -e
echo "[eval] exit_code=${EVAL_RC} (expected 124 if timed out)"

echo "[summarize] Computing window stats (${WINDOW_START_SEC}..${WINDOW_END_SEC})..."
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_hle_prefix_cache_window.py" \
  --metrics-csv "${METRICS_CSV}" \
  --usage-jsonl "${USAGE_JSONL}" \
  --eval-log "${LOG_DIR}/eval_driver.log" \
  --start-offset-sec "${WINDOW_START_SEC}" \
  --end-offset-sec "${WINDOW_END_SEC}" \
  --windows "${WINDOW_START_SEC}:${WINDOW_END_SEC}" \
  --out-json "${SUMMARY_JSON}" \
  >/dev/null 2>&1
SUM_RC=$?

"${PYTHON_BIN}" "${REPO_DIR}/scripts/preexp/summarize_steps_per_sec_window.py" \
  --usage-jsonl "${USAGE_JSONL}" \
  --eval-log "${LOG_DIR}/eval_driver.log" \
  --start-offset-sec "${WINDOW_START_SEC}" \
  --end-offset-sec "${WINDOW_END_SEC}" \
  > "${STEPS_JSON}" 2>/dev/null
STEPS_RC=$?

SUMMARY_JSON="${SUMMARY_JSON}" STEPS_JSON="${STEPS_JSON}" METRICS_CSV="${METRICS_CSV}" USAGE_JSONL="${USAGE_JSONL}" GPU_UTIL_CSV="${GPU_UTIL_CSV}" CONCURRENCY="${CONCURRENCY}" REP_IDX="${REP_IDX}" SCHEDULER="${SCHEDULER}" \
  "${PYTHON_BIN}" - <<'PY' > "${COMBINED_JSON}"
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
    "scheduler": os.environ.get("SCHEDULER"),
    "concurrency": int(os.environ["CONCURRENCY"]),
    "rep_idx": int(os.environ["REP_IDX"]),
    "t0_first_request_unix": summary.get("t0_first_request_unix"),
    "t0_first_request_iso": summary.get("t0_first_request_iso"),
    "window": {},
    "raw": {
        "summary_json": str(sp),
        "steps_json": str(tp),
        "metrics_csv": os.environ.get("METRICS_CSV"),
        "usage_jsonl": os.environ.get("USAGE_JSONL"),
    },
}

win = (summary.get("windows") or {}).get("600-7800") or {}
metrics = (win.get("metrics") or {}) if isinstance(win, dict) else {}
usage = (win.get("response_usage") or {}) if isinstance(win, dict) else {}
trials = (win.get("trials") or {}) if isinstance(win, dict) else {}
pre = (metrics.get("preemptions") or {}) if isinstance(metrics, dict) else {}

steps_block = (steps.get("steps") or {}) if isinstance(steps, dict) else {}
steps_per_sec = steps_block.get("steps_per_sec")

trials_per_sec = trials.get("trials_per_sec")
trials_per_min = (trials_per_sec * 60.0) if isinstance(trials_per_sec, (int, float)) else None

out["window"] = {
    "start_offset_sec": 600,
    "end_offset_sec": 7800,
    "window_start_iso": win.get("window_start_iso"),
    "window_end_iso": win.get("window_end_iso"),
    "duration_sec": 7200,
    "server_prefix_hit_ratio": metrics.get("hit_ratio"),
    "server_kv_cache_usage_mean_perc": metrics.get("kv_cache_usage_mean_perc"),
    "server_latency_means_seconds": metrics.get("latency_means_seconds"),
    "server_decode_over_e2e_ratio": metrics.get("decode_over_e2e_ratio"),
    "server_preemptions_sum_delta": pre.get("sum_delta_preemptions"),
    "server_preemptions_delta_mean_per_interval": pre.get("delta_preemptions_mean_per_interval"),
    "server_preemptions_per_sec": pre.get("preemptions_per_sec"),
    "request_cached_tokens_ratio_avg": usage.get("cached_tokens_ratio_request_avg"),
    "usage_responses": usage.get("responses"),
    "trials_completed": trials.get("trials_completed_in_window"),
    "trials_per_sec": trials_per_sec,
    "trials_per_min": trials_per_min,
    "steps_completed": steps_block.get("step_complete_in_window"),
    "steps_per_sec": steps_per_sec,
}

# GPU SM util mean over the same window (best-effort).
gpu_csv = Path(os.environ.get("GPU_UTIL_CSV", ""))
gpu_sm_sum = 0.0
gpu_sm_n = 0
gpu_idx = None
if gpu_csv.exists():
    try:
        import csv as _csv

        t0 = summary.get("t0_first_request_unix")
        if isinstance(t0, (int, float)):
            w_start = float(t0) + 600.0
            w_end = float(t0) + 7800.0
            with gpu_csv.open("r", encoding="utf-8") as f:
                r = _csv.DictReader(f)
                for row in r:
                    try:
                        ts = float(row.get("ts_unix", "nan"))
                        sm = float(row.get("sm_util", "nan"))
                    except Exception:
                        continue
                    if ts != ts or sm != sm:
                        continue
                    if ts < w_start or ts > w_end:
                        continue
                    gpu_sm_sum += sm
                    gpu_sm_n += 1
                    if gpu_idx is None:
                        try:
                            gpu_idx = int(float(row.get("gpu_index", "nan")))
                        except Exception:
                            gpu_idx = None
    except Exception:
        pass
out["window"]["gpu_index"] = gpu_idx
out["window"]["gpu_sm_util_mean"] = (gpu_sm_sum / gpu_sm_n) if gpu_sm_n > 0 else None
out["window"]["gpu_sm_util_samples"] = gpu_sm_n
out["raw"]["gpu_util_csv"] = str(gpu_csv)

print(json.dumps(out, indent=2, ensure_ascii=False))
PY
COMBINED_RC=$?
set -e

REPORT_MD="${ROOT_DIR}/hle-bench-rollout/preexp_hle_dp1_report_10_130.md"
mkdir -p "$(dirname "${REPORT_MD}")"
if [[ ! -f "${REPORT_MD}" ]]; then
  cat > "${REPORT_MD}" <<'MD'
# PreExp: HLE DP=1 Scheduler Comparison (10–130min window)

- Orchestrator vLLM restarted per run (clears prefix/KV cache)
- `t0` = first successful orchestrator response (from `orchestrator_usage.jsonl`)
- Window = `t0+10min .. t0+130min` (2 hours)
- Throughput:
  - `trials/min`: count of `[HLE_TRIAL_COMPLETE]` markers / 120
  - `steps/sec`: count of `[PROFILE] ... type=step_complete ...` / 7200
- Server metrics: backend `/metrics` sampled every 2s (hit ratio is Δhits/Δqueries in tokens)
- Preemption severity: mean Δ(`vllm:num_preemptions*`) per 2s sample
- Request metric: `cached_tokens/prompt_tokens` request-avg over successful responses in window

## Baseline (vllm-router)

| C | rep | t0_iso | window_start_iso | window_end_iso | server_hit_ratio | request_hit_avg | kv_usage_mean_perc | e2e_s | decode_s | decode_over_e2e | preemptions_delta_mean_2s | trials/min | steps/sec | usage_responses | gpu_sm_util_mean | output_dir | log_dir | combined_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Continuum (vllm-router-continuum)

| C | rep | t0_iso | window_start_iso | window_end_iso | server_hit_ratio | request_hit_avg | kv_usage_mean_perc | e2e_s | decode_s | decode_over_e2e | preemptions_delta_mean_2s | trials/min | steps/sec | usage_responses | gpu_sm_util_mean | output_dir | log_dir | combined_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## ThunderReact (multinode_router)

| C | rep | t0_iso | window_start_iso | window_end_iso | server_hit_ratio | request_hit_avg | kv_usage_mean_perc | e2e_s | decode_s | decode_over_e2e | preemptions_delta_mean_2s | trials/min | steps/sec | usage_responses | gpu_sm_util_mean | output_dir | log_dir | combined_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Failures
MD
fi

append_failure() {
  local reason="$1"
  {
    echo "- $(date -Is) scheduler=${SCHEDULER} C=${CONCURRENCY} rep=${REP_IDX} reason=${reason} output_dir=${OUTPUT_DIR} log_dir=${LOG_DIR}"
  } >> "${REPORT_MD}"
}

if [[ "${SUM_RC}" != "0" || "${STEPS_RC}" != "0" || "${COMBINED_RC}" != "0" || ! -s "${COMBINED_JSON}" ]]; then
  append_failure "summarize_failed(sum_rc=${SUM_RC},steps_rc=${STEPS_RC},combined_rc=${COMBINED_RC},eval_rc=${EVAL_RC})"
  echo "[summarize] ERROR: summarize failed; recorded failure in ${REPORT_MD}" >&2
  exit 1
fi

SCHEDULER="${SCHEDULER}" CONCURRENCY="${CONCURRENCY}" REP_IDX="${REP_IDX}" OUTPUT_DIR="${OUTPUT_DIR}" LOG_DIR="${LOG_DIR}" COMBINED_JSON="${COMBINED_JSON}" REPORT_MD="${REPORT_MD}" \
  "${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path

row_path = Path(os.environ["COMBINED_JSON"])
row = json.loads(row_path.read_text(encoding="utf-8")) if row_path.exists() else {}
w = row.get("window") or {}
lat = w.get("server_latency_means_seconds") or {}

scheduler = (os.environ.get("SCHEDULER") or "").strip().lower()
report_path = Path(os.environ["REPORT_MD"])

def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return f"{x:.6f}"
    return str(x)

def find_section(lines: list[str], section_header: str) -> int:
    for i, line in enumerate(lines):
        if line.strip() == section_header:
            return i
    return -1

def find_table_after(lines: list[str], start_idx: int) -> int:
    for i in range(start_idx, len(lines)):
        if lines[i].startswith("| C | rep | t0_iso |"):
            return i
    return -1

section = None
if scheduler == "baseline":
    section = "## Baseline (vllm-router)"
elif scheduler == "continuum":
    section = "## Continuum (vllm-router-continuum)"
elif scheduler == "thunderreact":
    section = "## ThunderReact (multinode_router)"
else:
    raise SystemExit(f"unknown scheduler section: {scheduler}")

lines = report_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
sec_idx = find_section(lines, section)
if sec_idx < 0:
    raise SystemExit(f"missing section {section}")
hdr_idx = find_table_after(lines, sec_idx)
if hdr_idx < 0:
    raise SystemExit("missing table header")

# Find insertion point (after contiguous '|' rows)
sep_idx = hdr_idx + 1
first_row = sep_idx + 1
end_idx = first_row
while end_idx < len(lines) and lines[end_idx].startswith("|"):
    end_idx += 1

prefix = f"| {float(os.environ['CONCURRENCY']):.6f} | {float(os.environ['REP_IDX']):.6f} |"
for i in range(first_row, end_idx):
    if lines[i].startswith(prefix):
        raise SystemExit(0)

cells = [
    fmt(row.get("concurrency")),
    fmt(row.get("rep_idx")),
    fmt(row.get("t0_first_request_iso")),
    fmt(w.get("window_start_iso")),
    fmt(w.get("window_end_iso")),
    fmt(w.get("server_prefix_hit_ratio")),
    fmt(w.get("request_cached_tokens_ratio_avg")),
    fmt(w.get("server_kv_cache_usage_mean_perc")),
    fmt(lat.get("e2e_request_latency_seconds")),
    fmt(lat.get("request_decode_time_seconds")),
    fmt(w.get("server_decode_over_e2e_ratio")),
    fmt(w.get("server_preemptions_delta_mean_per_interval")),
    fmt(w.get("trials_per_min")),
    fmt(w.get("steps_per_sec")),
    fmt(w.get("usage_responses")),
    fmt(w.get("gpu_sm_util_mean")),
    str(Path(os.environ["OUTPUT_DIR"])),
    str(Path(os.environ["LOG_DIR"])),
    str(row_path),
]
row_line = "| " + " | ".join(cells) + " |\n"
lines.insert(end_idx, row_line)
report_path.write_text("".join(lines), encoding="utf-8")
PY

echo "[summarize] combined_json=${COMBINED_JSON}"
echo "[summarize] report_md=${REPORT_MD}"
echo "[done] HLE preexp run finished."


#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash evaluation/launch_hle_inference.sh --method <baseline|continuum|thunderagent> --concurrency <C> \
    --ckpt <CKPT_DIR> --index-dir <INDEX_DIR> [options]

Required:
  --method         baseline | continuum | thunderagent
  --concurrency    HLE concurrency (eval batch size)
  --ckpt           Orchestrator-8B checkpoint path
  --index-dir      Directory containing eval.index + eval.jsonl

Common options:
  --example-path   Path to hle.jsonl (default: evaluation/hle.jsonl)
  --output-dir     Output directory (default: evaluation/outputs/hle_local)
  --log-dir        Log directory (default: evaluation/logs/hle_local)
  --model-config   Model config JSON path (default: evaluation/model_configs/hle_local_router.json)
  --model-name     Served model name (default: --ckpt)
  --model-type     Orchestrator model type (default: Qwen/Qwen3-8B)
  --orchestrator-gpu  GPU for vLLM (default: 0)
  --retrieval-gpu     GPU for retriever (default: 1)
  --backend-port      vLLM backend port (default: 8100)
  --router-port       ThunderAgent router port (default: 8000)
  --retrieval-port    Retriever port (default: 1401)
  --vllm-env          Conda env for baseline/thunderagent (default: vllm1)
  --cont-env          Conda env for continuum (default: vllm-continuum)
  --router-env        Conda env for ThunderAgent router (default: vllm1)
  --retriever-env     Conda env for retriever (default: retriever-clean)
  --conda-sh          Path to conda.sh (default: /root/miniconda3/etc/profile.d/conda.sh)
  --max-rounds        Max rounds per task (default: 50)
  --log-level         HLE log level (default: INFO)
USAGE
}

METHOD=""
CONCURRENCY=""
CKPT_DIR=""
INDEX_DIR=""
EXAMPLE_PATH=""
OUTPUT_DIR=""
LOG_DIR=""
MODEL_CONFIG=""
MODEL_NAME=""
MODEL_TYPE="Qwen/Qwen3-8B"
ORCH_GPU=0
RET_GPU=1
BACKEND_PORT=8100
ROUTER_PORT=8000
RETRIEVAL_PORT=1401
VLLM_ENV="vllm1"
CONT_ENV="vllm-continuum"
ROUTER_ENV="vllm1"
RETRIEVER_ENV="retriever-clean"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
MAX_ROUNDS=50
LOG_LEVEL="INFO"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --ckpt) CKPT_DIR="$2"; shift 2 ;;
    --index-dir) INDEX_DIR="$2"; shift 2 ;;
    --example-path) EXAMPLE_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --model-type) MODEL_TYPE="$2"; shift 2 ;;
    --orchestrator-gpu) ORCH_GPU="$2"; shift 2 ;;
    --retrieval-gpu) RET_GPU="$2"; shift 2 ;;
    --backend-port) BACKEND_PORT="$2"; shift 2 ;;
    --router-port) ROUTER_PORT="$2"; shift 2 ;;
    --retrieval-port) RETRIEVAL_PORT="$2"; shift 2 ;;
    --vllm-env) VLLM_ENV="$2"; shift 2 ;;
    --cont-env) CONT_ENV="$2"; shift 2 ;;
    --router-env) ROUTER_ENV="$2"; shift 2 ;;
    --retriever-env) RETRIEVER_ENV="$2"; shift 2 ;;
    --conda-sh) CONDA_SH="$2"; shift 2 ;;
    --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$METHOD" || -z "$CONCURRENCY" || -z "$CKPT_DIR" || -z "$INDEX_DIR" ]]; then
  usage
  exit 1
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda.sh not found at $CONDA_SH (set --conda-sh)"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/evaluation"

if [[ -z "$EXAMPLE_PATH" ]]; then
  EXAMPLE_PATH="${EVAL_DIR}/hle.jsonl"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${EVAL_DIR}/outputs/hle_local"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${EVAL_DIR}/logs/hle_local"
fi
if [[ -z "$MODEL_CONFIG" ]]; then
  MODEL_CONFIG="${EVAL_DIR}/model_configs/hle_local_router.json"
fi
if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME="${CKPT_DIR}"
fi

case "$METHOD" in
  baseline)
    ROUTER_MODE="default"
    VLLM_ENV_SELECTED="$VLLM_ENV"
    VLLM_ARGS=()
    ;;
  continuum)
    ROUTER_MODE="default"
    VLLM_ENV_SELECTED="$CONT_ENV"
    VLLM_ARGS=(--scheduling-policy continuum)
    ;;
  thunderagent)
    ROUTER_MODE="tr"
    VLLM_ENV_SELECTED="$VLLM_ENV"
    VLLM_ARGS=()
    ;;
  *)
    echo "ERROR: invalid --method '$METHOD'"
    usage
    exit 1
    ;;
esac

mkdir -p "$(dirname "$MODEL_CONFIG")" "$OUTPUT_DIR" "$LOG_DIR"

MODEL_NAME_ENV="${MODEL_NAME}" \
MODEL_CONFIG_ENV="${MODEL_CONFIG}" \
RETRIEVAL_PORT_ENV="${RETRIEVAL_PORT}" \
ROUTER_PORT_ENV="${ROUTER_PORT}" \
python - <<'PY'
import json
import os

model_name = os.environ["MODEL_NAME_ENV"]
model_config_path = os.environ["MODEL_CONFIG_ENV"]
retrieval_port = os.environ["RETRIEVAL_PORT_ENV"]
router_port = os.environ["ROUTER_PORT_ENV"]

cfg = {
    "retrieval": [{"ip_addr": "127.0.0.1", "port": str(retrieval_port)}],
    "Qwen/Qwen2.5-Math-72B-Instruct": [],
    "Qwen/Qwen3-32B": [],
    "Qwen/Qwen2.5-Math-7B-Instruct": [],
    "meta-llama/Llama-3.3-70B-Instruct": [],
    model_name: [{"ip_addr": "127.0.0.1", "port": str(router_port)}],
    "Qwen/Qwen2.5-Coder-32B-Instruct": [],
    "vllm_model_config_path": model_config_path,
}
with open(model_config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"Wrote model_config: {model_config_path}")
PY

cleanup() {
  echo "Stopping background processes..."
  [[ -n "${RETRIEVER_PID:-}" ]] && kill "${RETRIEVER_PID}" >/dev/null 2>&1 || true
  [[ -n "${VLLM_PID:-}" ]] && kill "${VLLM_PID}" >/dev/null 2>&1 || true
  [[ -n "${ROUTER_PID:-}" ]] && kill "${ROUTER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting retriever on GPU ${RET_GPU} (env=${RETRIEVER_ENV})..."
(
  source "$CONDA_SH"
  conda activate "$RETRIEVER_ENV"
  export INDEX_DIR="${INDEX_DIR}"
  CUDA_VISIBLE_DEVICES="${RET_GPU}" \
    python "${EVAL_DIR}/retrieval_hle.py" \
      --port "${RETRIEVAL_PORT}" \
      --new_cache_dir "${EVAL_DIR}/cache/hle" \
      --example_id_file "${EVAL_DIR}/examples.json"
) >"${LOG_DIR}/retrieval.log" 2>&1 &
RETRIEVER_PID=$!

echo "Starting vLLM backend on GPU ${ORCH_GPU} (env=${VLLM_ENV_SELECTED})..."
(
  source "$CONDA_SH"
  conda activate "$VLLM_ENV_SELECTED"
  CUDA_VISIBLE_DEVICES="${ORCH_GPU}" \
    vllm serve "${CKPT_DIR}" \
      --port "${BACKEND_PORT}" \
      --served-model-name "${MODEL_NAME}" \
      --enable-auto-tool-choice \
      --tool-call-parser hermes \
      "${VLLM_ARGS[@]}"
) >"${LOG_DIR}/vllm_backend.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM backend health (up to 30 min)..."
VLLM_READY=0
for _ in $(seq 1 1800); do
  if curl -sf "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    VLLM_READY=1
    break
  fi
  sleep 1
done
if [[ "$VLLM_READY" -ne 1 ]]; then
  echo "ERROR: vLLM backend did not become ready within 30 minutes."
  exit 1
fi

echo "Starting ThunderAgent router (mode=${ROUTER_MODE}, env=${ROUTER_ENV})..."
(
  source "$CONDA_SH"
  conda activate "$ROUTER_ENV"
  export PYTHONPATH="/workspace/ThunderAgent:${PYTHONPATH:-}"
  python -m ThunderAgent \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT}" \
    --backends "http://127.0.0.1:${BACKEND_PORT}" \
    --router "${ROUTER_MODE}" \
    --metrics
) >"${LOG_DIR}/router.log" 2>&1 &
ROUTER_PID=$!

echo "Waiting for router health..."
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null 2>&1; then
    echo "Router is healthy."
    break
  fi
  sleep 2
done
sleep 1

export ROUTER_URL="http://127.0.0.1:${ROUTER_PORT}"
export HLE_LOG_LEVEL="${LOG_LEVEL}"

echo "Starting eval (concurrency=${CONCURRENCY})..."
(
  source "$CONDA_SH"
  conda activate "$VLLM_ENV_SELECTED"
  cd "${EVAL_DIR}"
  python "${EVAL_DIR}/eval_hle_local.py" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_config "${MODEL_CONFIG}" \
    --max_rounds "${MAX_ROUNDS}" \
    --model_type "${MODEL_TYPE}" \
    --example_path "${EXAMPLE_PATH}" \
    --concurrency "${CONCURRENCY}" \
    --log_level "${LOG_LEVEL}"
)

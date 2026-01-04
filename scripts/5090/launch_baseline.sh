#!/bin/bash
set -euo pipefail

# Baseline (Standard vLLM) run on 5090 box.
#
# Usage:
#   export CKPT_DIR=/path/to/Nemotron-Orchestrator-8B
#   bash scripts/5090/launch_baseline.sh 32

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_ORCH_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_DIR="${CKPT_DIR:?CKPT_DIR env var must be set}"
CONCURRENCY="${1:-32}"
DOMAINS="${DOMAINS:-retail telecom airline}"
LOG_LEVEL="${LOG_LEVEL:-PROFILE}"
#
# Nemotron-Orchestrator-8B checkpoint includes a built-in `chat_template` in tokenizer_config.json,
# so we default to NOT overriding --chat-template here.
# If you want to override, export CHAT_TEMPLATE_PATH=/path/to/template.jinja.
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-}"
#
# NVIDIA tau2-bench default for Orchestrator uses hermes; keep this as the default.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
ORCH_PORT="${ORCH_PORT:-1900}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TOOL_ORCH_DIR}/outputs/baseline_c${CONCURRENCY}_${TS}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Start vLLM
CHAT_TEMPLATE_ARGS=()
if [[ -n "${CHAT_TEMPLATE_PATH}" ]]; then
  CHAT_TEMPLATE_ARGS=(--chat-template "${CHAT_TEMPLATE_PATH}")
fi

CUDA_VISIBLE_DEVICES=0 vllm serve "${CKPT_DIR}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  "${CHAT_TEMPLATE_ARGS[@]}" \
  --port "${ORCH_PORT}" \
  --gpu-memory-utilization 0.95 \
  > "${LOG_DIR}/vllm_baseline_${TS}.log" 2>&1 &
VLLM_PID=$!

sleep 120
echo "vLLM ready (PID: ${VLLM_PID})"

# Create model config for run_local.py (--skip-server-start requires it)
MODEL_CONFIG_PATH="${OUTPUT_DIR}/model_config_local.json"
cat > "${MODEL_CONFIG_PATH}" <<EOF
{
  "${CKPT_DIR}": [{"ip_addr": "127.0.0.1", "port": "${ORCH_PORT}"}],
  "vllm_model_config_path": "${MODEL_CONFIG_PATH}"
}
EOF

# Run evaluation
cd "${TOOL_ORCH_DIR}/evaluation/tau2-bench"
python run_local.py \
  --agent-model "${CKPT_DIR}" \
  --skip-server-start \
  --model-config-path "${MODEL_CONFIG_PATH}" \
  --domains ${DOMAINS} \
  --max-concurrency "${CONCURRENCY}" \
  --log-level "${LOG_LEVEL}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

kill "${VLLM_PID}" 2>/dev/null || true



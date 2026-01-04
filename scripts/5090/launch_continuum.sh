#!/bin/bash
set -euo pipefail

# Continuum scheduling experiment run on 5090 box (vllm-continuum env).
#
# Usage:
#   export CKPT_DIR=/path/to/Nemotron-Orchestrator-8B
#   bash scripts/5090/launch_continuum.sh 32
#
# Remote expert mode (Together H100 experts via localhost SSH tunnel):
#   1) In another tmux pane: bash scripts/5090/tunnel_experts_via_head.sh
#   2) Run this script. It will generate a model_config that points experts to 127.0.0.1 ports.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm-continuum

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_ORCH_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_DIR="${CKPT_DIR:?CKPT_DIR env var must be set}"
CONCURRENCY="${1:-32}"
DOMAINS="${DOMAINS:-retail telecom airline}"
LOG_LEVEL="${LOG_LEVEL:-PROFILE}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TOOL_ORCH_DIR}/outputs/continuum_c${CONCURRENCY}_${TS}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

#
# Nemotron-Orchestrator-8B checkpoint includes a built-in `chat_template` in tokenizer_config.json,
# so we default to NOT overriding --chat-template here.
# If you want to override, export CHAT_TEMPLATE_PATH=/path/to/template.jinja.
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-}"
#
# NVIDIA tau2-bench default for Orchestrator uses hermes; keep this as the default.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
ORCH_PORT="${ORCH_PORT:-1900}"

# Continuum output directory (scheduler timestamps, etc.)
export RUN_OUTPUT_DIR="${OUTPUT_DIR}"

# Start Continuum-enabled vLLM
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
  --scheduling-policy continuum \
  > "${LOG_DIR}/vllm_continuum_${TS}.log" 2>&1 &
VLLM_PID=$!

sleep 120
echo "Continuum vLLM ready (PID: ${VLLM_PID})"

# Start KV cache sampler (sample orchestrator backend /metrics)
KV_CSV="${OUTPUT_DIR}/kv_cache_timeseries.csv"
bash "${TOOL_ORCH_DIR}/scripts/rollout/collect_kv_cache_timeseries.sh" "127.0.0.1" "${KV_CSV}" 5 "${ORCH_PORT}" > "${LOG_DIR}/kv_sampler_${TS}.log" 2>&1 &
KV_PID=$!
echo "KV sampler started (PID: ${KV_PID}) -> ${KV_CSV}"

# Create model config for run_oss.py --skip-server-start
# - Orchestrator endpoint is local vLLM-continuum on ORCH_PORT.
# - Expert endpoints are local forwarded ports created by scripts/5090/tunnel_experts_via_head.sh.
MODEL_CONFIG_PATH="${OUTPUT_DIR}/model_config_5090_continuum.json"
cat > "${MODEL_CONFIG_PATH}" <<EOF
{
  "${CKPT_DIR}": [{"ip_addr": "127.0.0.1", "port": "${ORCH_PORT}"}],

  "openai/gpt-oss-20b": [
    {"ip_addr": "127.0.0.1", "port": "1910"},
    {"ip_addr": "127.0.0.1", "port": "1911"},
    {"ip_addr": "127.0.0.1", "port": "1912"},
    {"ip_addr": "127.0.0.1", "port": "1913"}
  ],
  "Qwen/Qwen3-32B-FP8": [
    {"ip_addr": "127.0.0.1", "port": "1904"},
    {"ip_addr": "127.0.0.1", "port": "1905"}
  ],
  "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": [
    {"ip_addr": "127.0.0.1", "port": "1920"},
    {"ip_addr": "127.0.0.1", "port": "1921"}
  ],

  "oss_expert_mapping": {
    "expert-1": "openai/gpt-oss-20b",
    "expert-2": "Qwen/Qwen3-32B-FP8",
    "expert-3": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
  },

  "vllm_model_config_path": "${MODEL_CONFIG_PATH}"
}
EOF

# Run evaluation
cd "${TOOL_ORCH_DIR}/evaluation/tau2-bench"
python run_oss.py \
  --agent-model "${CKPT_DIR}" \
  --skip-server-start \
  --schedule-mode global \
  --model-config-path "${MODEL_CONFIG_PATH}" \
  --domains ${DOMAINS} \
  --max-concurrency "${CONCURRENCY}" \
  --log-level "${LOG_LEVEL}" \
  --output-dir "${OUTPUT_DIR}/outputs" \
  --log-dir "${LOG_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/driver.log"

# Copy scheduler timestamps if present
cp "${RUN_OUTPUT_DIR}/scheduler_timestamps" "${OUTPUT_DIR}/" 2>/dev/null || true

kill "${KV_PID}" "${VLLM_PID}" 2>/dev/null || true



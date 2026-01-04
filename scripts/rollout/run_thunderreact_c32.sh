#!/bin/bash
# Run ThunderReact experiment with concurrency=32
# Usage: run_thunderreact_c32.sh [expert_1_ip] [expert_2_ip] [expert_3_ip] [orch_ip]
#
# Prerequisites:
# 1. Expert LLMs deployed (sbatch slurm/experts/deploy_expert_*.sbatch)
# 2. Orchestrator deployed (sbatch slurm/orchestrator/deploy_thunderreact.sbatch)
# 3. IPs extracted from SLURM job outputs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TAU2_DIR="${REPO_DIR}/evaluation/tau2-bench"

detect_running_job_out() {
    # Prefer the currently RUNNING job's .out file (via squeue) to avoid picking stale logs.
    # Usage: detect_running_job_out <slurm_job_name> <fallback_glob>
    local job_name="$1"
    local fallback_glob="$2"

    if command -v squeue >/dev/null 2>&1; then
        local jid
        jid="$(squeue --me -h -n "${job_name}" -o '%i' 2>/dev/null | head -n 1 || true)"
        if [[ -n "${jid}" ]]; then
            # Our expert jobs write .out into slurm/expert_logs/expert-*/ with name %x_%j.out
            local p1="${REPO_DIR}/slurm/expert_logs/expert-1/${job_name}_${jid}.out"
            local p2="${REPO_DIR}/slurm/expert_logs/expert-2/${job_name}_${jid}.out"
            local p3="${REPO_DIR}/slurm/expert_logs/expert-3/${job_name}_${jid}.out"
            local p4="${REPO_DIR}/${job_name}_${jid}.out"
            if [[ -f "${p1}" ]]; then echo "${p1}"; return 0; fi
            if [[ -f "${p2}" ]]; then echo "${p2}"; return 0; fi
            if [[ -f "${p3}" ]]; then echo "${p3}"; return 0; fi
            if [[ -f "${p4}" ]]; then echo "${p4}"; return 0; fi
        fi
    fi

    # Fallback: latest matching file
    ls -t ${fallback_glob} 2>/dev/null | head -1 || true
}

extract_first_ipv4() {
    # Extract the first IPv4 address from the latest matching SLURM output file.
    # This is robust to banners like "=====" that appear before the IP.
    local pattern="$1"
    local latest
    latest="$(ls -t ${pattern} 2>/dev/null | head -1 || true)"
    if [ -z "${latest}" ]; then
        echo ""
        return 0
    fi
    grep -m1 -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' "${latest}" 2>/dev/null || echo ""
}

# Get IPs from args or try to extract from SLURM outputs (latest file, first IPv4 match).
# Expert jobs now write their Slurm .out/.err into slurm/expert_logs/expert-{1,2,3}/ for cleanliness.
EXPERT_1_OUT="$(detect_running_job_out "rollout_expert_1" "${REPO_DIR}/slurm/expert_logs/expert-1/rollout_expert_1_*.out")"
EXPERT_2_OUT="$(detect_running_job_out "rollout_expert_2" "${REPO_DIR}/slurm/expert_logs/expert-2/rollout_expert_2_*.out")"
EXPERT_3_OUT="$(detect_running_job_out "rollout_expert_3" "${REPO_DIR}/slurm/expert_logs/expert-3/rollout_expert_3_*.out")"

EXPERT_1_IP="${1:-$(grep -m1 -Eo '([0-9]{1,3}\\.){3}[0-9]{1,3}' "${EXPERT_1_OUT}" 2>/dev/null || echo "")}"
EXPERT_2_IP="${2:-$(grep -m1 -Eo '([0-9]{1,3}\\.){3}[0-9]{1,3}' "${EXPERT_2_OUT}" 2>/dev/null || echo "")}"
EXPERT_3_IP="${3:-$(grep -m1 -Eo '([0-9]{1,3}\\.){3}[0-9]{1,3}' "${EXPERT_3_OUT}" 2>/dev/null || echo "")}"
ORCH_IP="${4:-$(extract_first_ipv4 "rollout_orch_thunderreact_*.out")}"

if [[ -z "${EXPERT_1_IP}" || -z "${EXPERT_2_IP}" || -z "${EXPERT_3_IP}" || -z "${ORCH_IP}" ]]; then
    echo "ERROR: Missing IPs. Provide them as arguments or ensure SLURM output files exist."
    echo ""
    echo "Usage: $0 <expert_1_ip> <expert_2_ip> <expert_3_ip> <orch_ip>"
    echo ""
    echo "Or extract from SLURM outputs:"
    echo "  EXPERT_1_IP=\$(grep -m1 -Eo '([0-9]{1,3}\\.){3}[0-9]{1,3}' \$(ls -t ${REPO_DIR}/slurm/expert_logs/expert-1/rollout_expert_1_*.out | head -1))"
    echo "  EXPERT_2_IP=\$(grep -m1 -Eo '([0-9]{1,3}\\.){3}[0-9]{1,3}' \$(ls -t ${REPO_DIR}/slurm/expert_logs/expert-2/rollout_expert_2_*.out | head -1))"
    echo "  EXPERT_3_IP=\$(grep -m1 -Eo '([0-9]{1,3}\\.){3}[0-9]{1,3}' \$(ls -t ${REPO_DIR}/slurm/expert_logs/expert-3/rollout_expert_3_*.out | head -1))"
    echo "  ORCH_IP=\$(grep -m1 -Eo '([0-9]{1,3}\\.){3}[0-9]{1,3}' \$(ls -t rollout_orch_thunderreact_*.out | head -1))"
    exit 1
fi

CKPT_DIR="${CKPT_DIR:?ERROR: CKPT_DIR not set}"
CONCURRENCY=32
SCHEDULER="thunderreact"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${REPO_DIR}/outputs/rollout_${SCHEDULER}_c${CONCURRENCY}_${TIMESTAMP}"
LOG_DIR="${HOME}/logs/rollout_${SCHEDULER}_c${CONCURRENCY}_${TIMESTAMP}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "=========================================="
echo "=== ThunderReact Rollout Experiment ==="
echo "=========================================="
echo "Concurrency: ${CONCURRENCY}"
echo "Orchestrator: ${ORCH_IP}"
echo "Expert-1: ${EXPERT_1_IP}"
echo "Expert-2: ${EXPERT_2_IP}"
echo "Expert-3: ${EXPERT_3_IP}"
echo "Output: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"
echo ""

# Step 1: Health check
echo "=== Step 1: Health Check ==="
"${SCRIPT_DIR}/health_check.sh" "${ORCH_IP}" "${EXPERT_1_IP}" "${EXPERT_2_IP}" "${EXPERT_3_IP}" "${SCHEDULER}" || {
    echo "ERROR: Health check failed. Please verify all services are running."
    exit 1
}

# Step 2: Generate model config
echo ""
echo "=== Step 2: Generate Model Config ==="
cd "${TAU2_DIR}"
python "${REPO_DIR}/scripts/gen_model_config.py" \
    "${ORCH_IP}" "${EXPERT_1_IP}" "${EXPERT_2_IP}" "${EXPERT_3_IP}" \
    --scheduler "${SCHEDULER}" \
    --output "model_config_rollout.json"

# Set ROUTER_URL for ThunderReact
export ROUTER_URL="http://${ORCH_IP}:8000"
echo "ROUTER_URL=${ROUTER_URL}"

# Step 3: Smoke test (fast sanity check, keeps logs for debugging)
echo ""
echo "=== Step 3: Smoke Test (mock domain) ==="
SMOKE_NUM_TASKS="${SMOKE_NUM_TASKS:-2}"
SMOKE_MAX_CONCURRENCY="${SMOKE_MAX_CONCURRENCY:-4}"
SMOKE_OUTPUT_DIR="${OUTPUT_DIR}/smoke"
SMOKE_LOG_DIR="${LOG_DIR}/smoke"
mkdir -p "${SMOKE_OUTPUT_DIR}" "${SMOKE_LOG_DIR}"

cd "${TAU2_DIR}"
python run_oss.py \
    --agent-model "${CKPT_DIR}" \
    # -mini
    --user-llm gpt-5 \
    --skip-server-start \
    --schedule-mode global \
    --model-config-path model_config_rollout.json \
    --domains mock \
    --num-tasks "${SMOKE_NUM_TASKS}" \
    --max-concurrency "${SMOKE_MAX_CONCURRENCY}" \
    --use_model_tool \
    --log-level DEBUG \
    --output-dir "${SMOKE_OUTPUT_DIR}/results" \
    --log-dir "${SMOKE_LOG_DIR}" \
    2>&1 | tee "${SMOKE_LOG_DIR}/smoke_driver.log"

# Step 4: Full run (PROFILE logs + time-series KV-cache usage)
echo ""
echo "=== Step 4: Full Run (PROFILE) ==="
DOMAINS=(${DOMAINS:-retail telecom airline})
FULL_OUTPUT_DIR="${OUTPUT_DIR}/full"
FULL_LOG_DIR="${LOG_DIR}/full"
mkdir -p "${FULL_OUTPUT_DIR}/metrics" "${FULL_LOG_DIR}"

KV_SAMPLE_INTERVAL_SEC="${KV_SAMPLE_INTERVAL_SEC:-5}"
KV_CSV="${FULL_OUTPUT_DIR}/metrics/kv_cache_usage.csv"

echo "Domains: ${DOMAINS[*]}"
echo "Max concurrency: ${CONCURRENCY}"
echo "KV-cache sample interval (sec): ${KV_SAMPLE_INTERVAL_SEC}"
echo ""

# Start KV-cache sampler in background (best-effort)
"${SCRIPT_DIR}/collect_kv_cache_timeseries.sh" "${ORCH_IP}" "${KV_CSV}" "${KV_SAMPLE_INTERVAL_SEC}" &
KV_SAMPLER_PID=$!
trap 'kill "${KV_SAMPLER_PID}" 2>/dev/null || true' EXIT

FULL_START_TS="$(date +%s)"
cd "${TAU2_DIR}"
python run_oss.py \
    --agent-model "${CKPT_DIR}" \
    --user-llm gpt-5 \
    --skip-server-start \
    --schedule-mode global \
    --model-config-path model_config_rollout.json \
    --domains "${DOMAINS[@]}" \
    --max-concurrency "${CONCURRENCY}" \
    --use_model_tool \
    --log-level PROFILE \
    --output-dir "${FULL_OUTPUT_DIR}/results" \
    --log-dir "${FULL_LOG_DIR}" \
    2>&1 | tee "${FULL_LOG_DIR}/full_driver.log"
FULL_END_TS="$(date +%s)"

# Stop KV sampler and keep the CSV
kill "${KV_SAMPLER_PID}" 2>/dev/null || true
wait "${KV_SAMPLER_PID}" 2>/dev/null || true

# Step 5: Collect metrics (PROFILE stats + router state + summary)
echo ""
echo "=== Step 5: Collecting Metrics ==="
export FULL_START_TS FULL_END_TS
"${SCRIPT_DIR}/collect_metrics.sh" "${FULL_OUTPUT_DIR}" "${FULL_LOG_DIR}" "${SCHEDULER}" "${ROUTER_URL}" || true

echo ""
echo "=========================================="
echo "=== Experiment Complete ==="
echo "=========================================="
echo "Results: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"
echo "Full results: ${OUTPUT_DIR}/full/results"
echo "Full metrics: ${OUTPUT_DIR}/full/metrics"
echo "Full logs: ${LOG_DIR}/full"
echo "KV cache CSV: ${OUTPUT_DIR}/full/metrics/kv_cache_usage.csv"
echo "Summary JSON: ${OUTPUT_DIR}/full/metrics/rollout_summary.json"
echo "Model config: ${TAU2_DIR}/model_config_rollout.json"

#!/bin/bash
# Collect metrics from rollout experiment
# Usage: collect_metrics.sh <output_dir> <log_dir> <scheduler> [router_url]

set -euo pipefail

OUTPUT_DIR="${1:?OUTPUT_DIR required}"
LOG_DIR="${2:?LOG_DIR required}"
SCHEDULER="${3:-baseline}"
ROUTER_URL="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=== Collecting metrics for ${SCHEDULER} ==="
echo "Output: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"

mkdir -p "${OUTPUT_DIR}/metrics"

# Parse PROFILE logs for latency statistics.
# New global scheduler mode writes ${LOG_DIR}/tau2_global.log; legacy mode writes tau2_<domain>.log.
if [[ -f "${LOG_DIR}/tau2_global.log" ]]; then
    echo "Parsing PROFILE logs for global..."
    python "${REPO_DIR}/scripts/analyze_profile.py" "${LOG_DIR}/tau2_global.log" \
        > "${OUTPUT_DIR}/metrics/latency_stats_global.json" 2>/dev/null || true
else
    for domain in mock retail telecom airline; do
        if [[ -f "${LOG_DIR}/tau2_${domain}.log" ]]; then
            echo "Parsing PROFILE logs for ${domain}..."
            python "${REPO_DIR}/scripts/analyze_profile.py" "${LOG_DIR}/tau2_${domain}.log" \
                > "${OUTPUT_DIR}/metrics/latency_stats_${domain}.json" 2>/dev/null || true
        fi
    done
fi

# Summarize throughput: tasks/min + steps/sec (step_complete PROFILE events)
if [[ -d "${LOG_DIR}" ]]; then
    echo "Summarizing rollout metrics..."
    python "${REPO_DIR}/scripts/rollout/summarize_rollout_metrics.py" \
        --log-dir "${LOG_DIR}" \
        --domains retail telecom airline \
        --out-json "${OUTPUT_DIR}/metrics/rollout_summary.json" \
        > "${OUTPUT_DIR}/metrics/rollout_summary_print.json" 2>/dev/null || true
fi

# ThunderReact-specific: router state
if [[ "${SCHEDULER}" == "thunderreact" ]]; then
    if [[ -n "${ROUTER_URL}" ]]; then
        echo "Collecting ThunderReact router metrics..."
        curl -sf "${ROUTER_URL}/backends" > "${OUTPUT_DIR}/metrics/router_backends.json" 2>/dev/null || true
        curl -sf "${ROUTER_URL}/programs" > "${OUTPUT_DIR}/metrics/router_programs.json" 2>/dev/null || true
    else
        echo "Warning: ROUTER_URL not set, skipping router metrics"
    fi
fi

# Continuum-specific: scheduler timestamps
if [[ "${SCHEDULER}" == "continuum" ]]; then
    echo "Collecting Continuum scheduler timestamps..."
    find "${LOG_DIR}" -name "scheduler_timestamps*" -exec cp {} "${OUTPUT_DIR}/metrics/" \; 2>/dev/null || true
fi

# Copy driver logs if present (helpful for quick debugging without going into ~/logs)
for f in full_driver.log smoke_driver.log; do
    if [[ -f "${LOG_DIR}/${f}" ]]; then
        cp "${LOG_DIR}/${f}" "${OUTPUT_DIR}/metrics/${f}" 2>/dev/null || true
    fi
done

echo "=== Metrics collection complete ==="
echo "Results saved to: ${OUTPUT_DIR}/metrics/"

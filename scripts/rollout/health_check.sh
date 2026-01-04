#!/bin/bash
# Health check script for rollout experiments
# Usage: health_check.sh <orch_ip> <expert_1_ip> <expert_2_ip> <expert_3_ip> [scheduler]

set -euo pipefail

ORCH_IP="${1:-}"
EXPERT_1_IP="${2:-}"
EXPERT_2_IP="${3:-}"
EXPERT_3_IP="${4:-}"
SCHEDULER="${5:-baseline}"

check_endpoint() {
    local ip=$1
    local port=$2
    local name=$3
    local timeout=${4:-5}

    if curl -sf --connect-timeout "${timeout}" "http://${ip}:${port}/health" > /dev/null 2>&1; then
        echo "[OK] ${name} at ${ip}:${port}"
        return 0
    else
        echo "[FAIL] ${name} at ${ip}:${port}"
        return 1
    fi
}

FAILED=0

if [[ -n "${ORCH_IP}" ]]; then
    echo "=== Checking Orchestrator (${SCHEDULER}) ==="
    if [[ "${SCHEDULER}" == "thunderreact" ]]; then
        check_endpoint "${ORCH_IP}" 8000 "ThunderReact Router" || FAILED=1
        # Also check backends
        echo "--- Checking ThunderReact backends ---"
        for port in 8100 8101 8102 8103 8104 8105 8106 8107; do
            check_endpoint "${ORCH_IP}" "${port}" "Backend" || true  # Don't fail on backend checks
        done
    else
        for port in 1900 1901 1902 1903 1904 1905 1906 1907; do
            check_endpoint "${ORCH_IP}" "${port}" "Orchestrator" || FAILED=1
        done
    fi
fi

if [[ -n "${EXPERT_1_IP}" ]]; then
    echo ""
    echo "=== Checking Expert-1 (gpt-oss-20b) at ${EXPERT_1_IP} ==="
    for port in 1910 1911 1912 1913; do
        check_endpoint "${EXPERT_1_IP}" "${port}" "Expert-1" || FAILED=1
    done
fi

if [[ -n "${EXPERT_2_IP}" ]]; then
    echo ""
    echo "=== Checking Expert-2 (Qwen3-32B) at ${EXPERT_2_IP} ==="
    for port in 1904 1905; do
        check_endpoint "${EXPERT_2_IP}" "${port}" "Expert-2" || FAILED=1
    done
fi

if [[ -n "${EXPERT_3_IP}" ]]; then
    echo ""
    echo "=== Checking Expert-3 (Qwen3-Next-80B) at ${EXPERT_3_IP} ==="
    for port in 1920 1921; do
        check_endpoint "${EXPERT_3_IP}" "${port}" "Expert-3" || FAILED=1
    done
fi

echo ""
if [[ ${FAILED} -eq 0 ]]; then
    echo "=== All health checks passed ==="
else
    echo "=== Some health checks failed ==="
fi

exit ${FAILED}

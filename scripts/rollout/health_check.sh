#!/bin/bash
# Health check script for rollout experiments
# Usage: health_check.sh <orch_ip> <expert_1_ip> <expert_2_ip> <expert_3_ip> [scheduler]

set -euo pipefail

ORCH_IP="${1:-}"
EXPERT_1_IP="${2:-}"
EXPERT_2_IP="${3:-}"
EXPERT_3_IP="${4:-}"
SCHEDULER="${5:-baseline}"

# NOTE: Orchestrator backends (ports 8100+) are typically bound to localhost on the orchestrator node
# (router talks to http://localhost:810x). When this script runs from a different node, probing
# ${ORCH_IP}:810x will often fail even if the system is healthy. Therefore backend checks are
# optional by default.
CHECK_ORCH_BACKENDS="${CHECK_ORCH_BACKENDS:-0}"
REQUIRE_ORCH_BACKENDS="${REQUIRE_ORCH_BACKENDS:-0}"

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
    else
        check_endpoint "${ORCH_IP}" 8000 "vllm-router" || FAILED=1
    fi

    if [[ "${CHECK_ORCH_BACKENDS}" -eq 1 ]]; then
        # Best-effort only; do not fail unless REQUIRE_ORCH_BACKENDS=1.
        echo "--- Checking Orchestrator backends (best effort) ---"
        ANY_BACKEND_OK=0
        for port in 8100 8101 8102 8103 8104 8105 8106 8107; do
            if check_endpoint "${ORCH_IP}" "${port}" "Backend" 2; then
                ANY_BACKEND_OK=1
            fi
        done
        if [[ "${ANY_BACKEND_OK}" -eq 0 ]]; then
            if [[ "${REQUIRE_ORCH_BACKENDS}" -eq 1 ]]; then
                echo "[FAIL] No orchestrator backends responded on ports 8100-8107 (REQUIRE_ORCH_BACKENDS=1)"
                FAILED=1
            else
                echo "[WARN] No orchestrator backends responded on ports 8100-8107 (likely bound to localhost); continuing"
            fi
        fi
    else
        echo "--- Skipping orchestrator backend checks (CHECK_ORCH_BACKENDS=0) ---"
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
    if [[ -n "${TOGETHER_API_KEY:-}" ]]; then
        echo "[SKIP] TOGETHER_API_KEY set; expert-3 may be routed via Together (no local /health)."
    else
        for port in 1920 1921; do
            check_endpoint "${EXPERT_3_IP}" "${port}" "Expert-3" || FAILED=1
        done
    fi
fi

echo ""
if [[ ${FAILED} -eq 0 ]]; then
    echo "=== All health checks passed ==="
else
    echo "=== Some health checks failed ==="
fi

exit ${FAILED}

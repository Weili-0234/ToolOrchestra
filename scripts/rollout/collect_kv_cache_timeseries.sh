#!/bin/bash
# Collect vLLM KV-cache usage time series from Orchestrator backends.
# Designed for ThunderReact rollout where backends are on ports 8100-8107.
#
# Usage:
#   collect_kv_cache_timeseries.sh <orch_ip> <out_csv> [interval_sec] [ports_csv]
#
# Example:
#   collect_kv_cache_timeseries.sh 172.27.27.153 /tmp/kv_cache.csv 5 "8100,8101,8102,8103,8104,8105,8106,8107"

set -euo pipefail

ORCH_IP="${1:?ORCH_IP required}"
OUT_CSV="${2:?OUT_CSV required}"
INTERVAL_SEC="${3:-5}" # 5 seconds by default
PORTS_CSV="${4:-8100,8101,8102,8103,8104,8105,8106,8107}"

mkdir -p "$(dirname "${OUT_CSV}")"

if [ ! -f "${OUT_CSV}" ]; then
  echo "ts_iso,ts_unix,port,metric,value" > "${OUT_CSV}"
fi

ports=()
IFS=',' read -r -a ports <<< "${PORTS_CSV}"

echo "[kv_cache] Sampling vLLM metrics from ${ORCH_IP} ports=${PORTS_CSV} interval_sec=${INTERVAL_SEC}"
echo "[kv_cache] Writing: ${OUT_CSV}"

trap 'echo "[kv_cache] Stopping sampler"; exit 0' INT TERM

while true; do
  ts_unix="$(date +%s)"
  ts_iso="$(date -Is)"
  for port in "${ports[@]}"; do
    # Keep timeouts tight so a single bad port doesn't block sampling.
    metrics="$(curl -sf --connect-timeout 2 --max-time 4 "http://${ORCH_IP}:${port}/metrics" 2>/dev/null || true)"
    if [ -z "${metrics}" ]; then
      echo "${ts_iso},${ts_unix},${port},vllm:kv_cache_usage_perc,NaN" >> "${OUT_CSV}"
      continue
    fi

    # Prometheus format lines look like:
    #   vllm:kv_cache_usage_perc{...} 12.34
    # Some versions use gpu_cache_usage_perc; capture both.
    echo "${metrics}" | awk -v ts_iso="${ts_iso}" -v ts_unix="${ts_unix}" -v port="${port}" '
      $0 ~ /^vllm:(kv_cache_usage_perc|gpu_cache_usage_perc)(\{| )/ {
        # split by whitespace: metric{labels} value
        key=$1
        val=$2
        if (val == "") { val="NaN" }
        gsub(/,/, ";", key) # keep CSV stable if labels contain commas
        print ts_iso "," ts_unix "," port "," key "," val
      }
    ' >> "${OUT_CSV}" || true
  done
  sleep "${INTERVAL_SEC}"
done



#!/usr/bin/env bash
# Sample vLLM prefix-cache token hit-rate + KV usage + preemptions from /metrics.
#
# This is a HLE-specific variant:
# - default interval: 2s
# - includes vLLM preemption counters (delta per interval)
#
# Output CSV columns:
#   ts_iso,ts_unix,
#   hits_total,queries_total,delta_hits,delta_queries,delta_miss_tokens,hit_ratio,
#   preemptions_total,delta_preemptions,
#   kv_cache_usage_perc,num_requests_running,num_requests_waiting,
#   <metric>_sum,<metric>_count,delta_<metric>_sum,delta_<metric>_count,<metric>_mean
#
# Hist metrics (if present):
#   - time_to_first_token_seconds
#   - inter_token_latency_seconds
#   - e2e_request_latency_seconds
#   - request_prefill_time_seconds
#   - request_decode_time_seconds
#
# Usage:
#   bash kv_prefix_cache_hit_sampler.sh http://127.0.0.1:8100/metrics /tmp/prefix_cache.csv [interval_sec]

set -euo pipefail

METRICS_URL="${1:?metrics_url required}"
OUT_CSV="${2:?out_csv required}"
INTERVAL_SEC="${3:-2}"

mkdir -p "$(dirname "${OUT_CSV}")"

KV_USAGE_METRIC="vllm:kv_cache_usage_perc"
NUM_RUNNING_METRIC="vllm:num_requests_running"
NUM_WAITING_METRIC="vllm:num_requests_waiting"

HIST_BASES=(
  "time_to_first_token_seconds"
  "inter_token_latency_seconds"
  "e2e_request_latency_seconds"
  "request_prefill_time_seconds"
  "request_decode_time_seconds"
)

append_hist_header() {
  local base
  for base in "${HIST_BASES[@]}"; do
    echo -n ",${base}_sum,${base}_count,delta_${base}_sum,delta_${base}_count,${base}_mean"
  done
}

{
  echo -n "ts_iso,ts_unix"
  echo -n ",hits_total,queries_total,delta_hits,delta_queries,delta_miss_tokens,hit_ratio"
  echo -n ",preemptions_total,delta_preemptions"
  echo -n ",kv_cache_usage_perc,num_requests_running,num_requests_waiting"
  append_hist_header
  echo
} > "${OUT_CSV}"

extract_counter() {
  local metric_name="$1"
  local text="$2"
  echo "${text}" | awk -v m="${metric_name}" '
    $1 ~ ("^"m"\\{") || $1 == m { val=$NF }
    END { if (val=="") val=""; print val }
  '
}

detect_metric_name() {
  local kind="$1" # hits|queries|preemptions
  local text="$2"
  local cand=""
  if [[ "${kind}" == "hits" ]]; then
    cand="$(echo "${text}" | grep -Eo '^[^#[:space:]]+' | grep -Ei 'prefix_cache.*(hit|hits).*total' | head -n 1 || true)"
    [[ -z "${cand}" ]] && cand="$(echo "${text}" | grep -Eo '^[^#[:space:]]+' | grep -Ei 'prefix.*cache.*(hit|hits).*token' | head -n 1 || true)"
    [[ -z "${cand}" ]] && cand="$(echo "${text}" | grep -Eo '^[^#[:space:]]+' | grep -Ei 'prefix_cache_hits_total' | head -n 1 || true)"
  elif [[ "${kind}" == "queries" ]]; then
    cand="$(echo "${text}" | grep -Eo '^[^#[:space:]]+' | grep -Ei 'prefix_cache.*(quer|queries).*total' | head -n 1 || true)"
    [[ -z "${cand}" ]] && cand="$(echo "${text}" | grep -Eo '^[^#[:space:]]+' | grep -Ei 'prefix.*cache.*(quer|queries).*token' | head -n 1 || true)"
    [[ -z "${cand}" ]] && cand="$(echo "${text}" | grep -Eo '^[^#[:space:]]+' | grep -Ei 'prefix_cache_queries_total' | head -n 1 || true)"
  else
    cand="$(echo "${text}" | grep -Eo '^[^#[:space:]]+' | grep -Ei 'num_preemptions.*total' | head -n 1 || true)"
    [[ -z "${cand}" ]] && cand="$(echo "${text}" | grep -Eo '^[^#[:space:]]+' | grep -Ei 'num_preemptions' | head -n 1 || true)"
  fi
  cand="${cand%%\{*}"
  echo "${cand}"
}

calc_delta_clamp() {
  local cur="$1"
  local prev="$2"
  awk -v c="${cur}" -v p="${prev}" 'BEGIN{
    if (c=="nan" || p=="") { print "nan"; exit }
    d=c-p; if (d<0) d=0;
    printf "%.10f", d
  }'
}

calc_ratio() {
  local num="$1"
  local den="$2"
  awk -v n="${num}" -v d="${den}" 'BEGIN{
    if (n=="nan" || d=="nan" || d<=0) { print "nan"; exit }
    printf "%.10f", n/d
  }'
}

hits_metric=""
queries_metric=""
preempt_metric=""

prev_hits=""
prev_queries=""
prev_preempt=""

while true; do
  ts_unix="$(date +%s)"
  ts_iso="$(date -Is)"

  metrics="$(curl -fsS --max-time 2 "${METRICS_URL}" 2>/dev/null || true)"

  [[ -z "${hits_metric}" ]] && hits_metric="$(detect_metric_name "hits" "${metrics}")"
  [[ -z "${queries_metric}" ]] && queries_metric="$(detect_metric_name "queries" "${metrics}")"
  [[ -z "${preempt_metric}" ]] && preempt_metric="$(detect_metric_name "preemptions" "${metrics}")"

  if [[ -n "${hits_metric}" && -n "${queries_metric}" && -z "${prev_hits}" ]]; then
    echo "[sampler] hits_metric=${hits_metric} queries_metric=${queries_metric} preempt_metric=${preempt_metric}" >&2
  fi

  hits="nan"
  queries="nan"
  preemptions="nan"

  if [[ -n "${hits_metric}" ]]; then
    v="$(extract_counter "${hits_metric}" "${metrics}")"
    [[ -n "${v}" ]] && hits="${v}"
  fi
  if [[ -n "${queries_metric}" ]]; then
    v="$(extract_counter "${queries_metric}" "${metrics}")"
    [[ -n "${v}" ]] && queries="${v}"
  fi
  if [[ -n "${preempt_metric}" ]]; then
    v="$(extract_counter "${preempt_metric}" "${metrics}")"
    [[ -n "${v}" ]] && preemptions="${v}"
  fi

  kv_usage="$(extract_counter "${KV_USAGE_METRIC}" "${metrics}")"
  num_running="$(extract_counter "${NUM_RUNNING_METRIC}" "${metrics}")"
  num_waiting="$(extract_counter "${NUM_WAITING_METRIC}" "${metrics}")"
  [[ -z "${kv_usage}" ]] && kv_usage="nan"
  [[ -z "${num_running}" ]] && num_running="nan"
  [[ -z "${num_waiting}" ]] && num_waiting="nan"

  delta_hits="nan"
  delta_queries="nan"
  delta_miss="nan"
  hit_ratio="nan"

  delta_preemptions="nan"

  if [[ "${hits}" != "nan" && "${queries}" != "nan" && -n "${prev_hits}" && -n "${prev_queries}" ]]; then
    delta_hits="$(calc_delta_clamp "${hits}" "${prev_hits}")"
    delta_queries="$(calc_delta_clamp "${queries}" "${prev_queries}")"
    delta_miss="$(awk -v dh="${delta_hits}" -v dq="${delta_queries}" 'BEGIN{
      if (dh=="nan" || dq=="nan") { print "nan"; exit }
      m=dq-dh; if (m<0) m=0;
      printf "%.10f", m
    }')"
    hit_ratio="$(calc_ratio "${delta_hits}" "${delta_queries}")"
  fi

  if [[ "${preemptions}" != "nan" && -n "${prev_preempt}" ]]; then
    delta_preemptions="$(calc_delta_clamp "${preemptions}" "${prev_preempt}")"
  fi

  hist_cells=""
  for base in "${HIST_BASES[@]}"; do
    sum_metric="vllm:${base}_sum"
    cnt_metric="vllm:${base}_count"
    cur_sum="$(extract_counter "${sum_metric}" "${metrics}")"
    cur_cnt="$(extract_counter "${cnt_metric}" "${metrics}")"
    [[ -z "${cur_sum}" ]] && cur_sum="nan"
    [[ -z "${cur_cnt}" ]] && cur_cnt="nan"

    prev_sum_var="prev_${base}_sum"
    prev_cnt_var="prev_${base}_cnt"
    prev_sum="${!prev_sum_var-}"
    prev_cnt="${!prev_cnt_var-}"

    dsum="nan"
    dcnt="nan"
    mean="nan"
    if [[ "${cur_sum}" != "nan" && "${cur_cnt}" != "nan" && -n "${prev_sum}" && -n "${prev_cnt}" ]]; then
      dsum="$(calc_delta_clamp "${cur_sum}" "${prev_sum}")"
      dcnt="$(calc_delta_clamp "${cur_cnt}" "${prev_cnt}")"
      mean="$(calc_ratio "${dsum}" "${dcnt}")"
    fi
    hist_cells="${hist_cells},${cur_sum},${cur_cnt},${dsum},${dcnt},${mean}"
    printf -v "${prev_sum_var}" "%s" "${cur_sum}"
    printf -v "${prev_cnt_var}" "%s" "${cur_cnt}"
  done

  echo "${ts_iso},${ts_unix},${hits},${queries},${delta_hits},${delta_queries},${delta_miss},${hit_ratio},${preemptions},${delta_preemptions},${kv_usage},${num_running},${num_waiting}${hist_cells}" >> "${OUT_CSV}"

  prev_hits="${hits}"
  prev_queries="${queries}"
  prev_preempt="${preemptions}"

  sleep "${INTERVAL_SEC}"
done


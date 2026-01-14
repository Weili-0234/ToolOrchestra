#!/usr/bin/env bash
# Wait until the current Continuum restart-per-C sweep finishes C=168, then:
#   1) stop the sweep job (so it doesn't proceed to the next C)
#   2) move retrieval service onto research-secure-02 (1 GPU)
#   3) move the watcher onto research-secure-02 (CPU-only)
#   4) resume remaining C settings in a new sweep job, appending into the same TSV
#
# This script is designed to run on the head node (no tmux dependency).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/junxiong/haokang/compare-ThunderReact-ToolOrchestra}"
cd "${ROOT_DIR}"

TARGET_NODE="${TARGET_NODE:-research-secure-02}"

SWEEP_JOB_ID="${SWEEP_JOB_ID:-13969}"
SWEEP_LOG="${SWEEP_LOG:-${ROOT_DIR}/slurm/hle_eval_logs/hle_continuum_restart_per_c_${SWEEP_JOB_ID}.out}"

WAIT_FOR_C="${WAIT_FOR_C:-168}"
RESUME_C_LIST="${RESUME_C_LIST:-192 216 240 264}"

SUMMARY_TSV="${SUMMARY_TSV:-${ROOT_DIR}/outputs/hle_continuum_restart_per_c_20260110_224315.tsv}"
REPORT_MD="${REPORT_MD:-${ROOT_DIR}/hle-bench-rollout/hle_preexp_dp1_continuum_restart_per_c_report_10_130.md}"

RETRIEVAL_JOB_NAME="${RETRIEVAL_JOB_NAME:-hle_retrieval}"
WATCH_JOB_NAME="${WATCH_JOB_NAME:-hle_continuum_sweep_watch}"

log() {
  echo "[$(date -Is)] $*"
}

log "ROOT_DIR=${ROOT_DIR}"
log "TARGET_NODE=${TARGET_NODE}"
log "SWEEP_JOB_ID=${SWEEP_JOB_ID}"
log "SWEEP_LOG=${SWEEP_LOG}"
log "WAIT_FOR_C=${WAIT_FOR_C}"
log "RESUME_C_LIST=${RESUME_C_LIST}"
log "SUMMARY_TSV=${SUMMARY_TSV}"
log "REPORT_MD=${REPORT_MD}"

if [[ ! -f "${SWEEP_LOG}" ]]; then
  log "ERROR: sweep log not found: ${SWEEP_LOG}"
  exit 1
fi
if [[ ! -f "${SUMMARY_TSV}" ]]; then
  log "ERROR: summary TSV not found: ${SUMMARY_TSV}"
  exit 1
fi

marker="\\[sweep\\] C=${WAIT_FOR_C}: done\\."
log "Waiting for marker in sweep log: ${marker}"
while ! rg -n "${marker}" "${SWEEP_LOG}" >/dev/null 2>&1; do
  sleep 15
done
log "Detected C=${WAIT_FOR_C} done. Proceeding with migration+resume."

# Stop the current sweep job so it doesn't proceed to the next C with the old RET_IP.
if squeue -j "${SWEEP_JOB_ID}" -h >/dev/null 2>&1; then
  log "Canceling sweep job ${SWEEP_JOB_ID} (after C=${WAIT_FOR_C} done)."
  scancel "${SWEEP_JOB_ID}" || true
else
  log "Sweep job ${SWEEP_JOB_ID} not in queue; continuing anyway."
fi

# Move retrieval onto TARGET_NODE:
# - cancel any existing retrieval job (by name)
old_ret_jid="$(squeue --me -h -n "${RETRIEVAL_JOB_NAME}" -t RUNNING -o '%i' 2>/dev/null | head -n 1 || true)"
if [[ -n "${old_ret_jid}" ]]; then
  log "Canceling existing retrieval job ${old_ret_jid} (${RETRIEVAL_JOB_NAME})."
  scancel "${old_ret_jid}" || true
else
  log "No running retrieval job named ${RETRIEVAL_JOB_NAME}."
fi

log "Submitting retrieval on ${TARGET_NODE}..."
ret_submit_out="$(sbatch --nodelist="${TARGET_NODE}" oss-ToolOrchestra/slurm/hle/deploy_hle_retrieval_hle.sbatch)"
log "${ret_submit_out}"
ret_jid="$(echo "${ret_submit_out}" | awk '{print $NF}')"
if [[ -z "${ret_jid}" ]]; then
  log "ERROR: failed to parse retrieval job id from: ${ret_submit_out}"
  exit 1
fi

ret_dir="${ROOT_DIR}/slurm/hle_retrieval_logs/${RETRIEVAL_JOB_NAME}_${ret_jid}"
ret_ip_file="${ret_dir}/node_ip.txt"
log "Waiting for retrieval node ip file: ${ret_ip_file}"
for _ in $(seq 1 180); do
  if [[ -s "${ret_ip_file}" ]]; then
    break
  fi
  sleep 2
done
if [[ ! -s "${ret_ip_file}" ]]; then
  log "ERROR: retrieval node ip file not found/non-empty after wait: ${ret_ip_file}"
  exit 1
fi
ret_ip="$(cat "${ret_ip_file}" | head -n 1 | tr -d '[:space:]')"
log "Retrieval is on ${ret_ip} (job ${ret_jid}). Waiting for /health..."
for _ in $(seq 1 180); do
  if curl -sf --max-time 2 "http://${ret_ip}:8765/health" >/dev/null 2>&1; then
    log "Retrieval /health OK."
    break
  fi
  sleep 2
done
if ! curl -sf --max-time 2 "http://${ret_ip}:8765/health" >/dev/null 2>&1; then
  log "ERROR: retrieval never became healthy: http://${ret_ip}:8765/health"
  exit 1
fi

# Move watcher onto TARGET_NODE:
old_watch_jid="$(squeue --me -h -n "${WATCH_JOB_NAME}" -t RUNNING -o '%i' 2>/dev/null | head -n 1 || true)"
if [[ -n "${old_watch_jid}" ]]; then
  log "Canceling existing watcher job ${old_watch_jid} (${WATCH_JOB_NAME})."
  scancel "${old_watch_jid}" || true
else
  log "No running watcher job named ${WATCH_JOB_NAME}."
fi

log "Submitting watcher on ${TARGET_NODE}..."
watch_submit_out="$(sbatch --nodelist="${TARGET_NODE}" --export=ALL,TSV_PATH="${SUMMARY_TSV}",REPORT_MD="${REPORT_MD}",INTERVAL_SEC=30 oss-ToolOrchestra/slurm/hle/monitor_hle_continuum_restart_per_c.sbatch)"
log "${watch_submit_out}"

# Resume remaining C settings in a new sweep job (append to same TSV).
log "Submitting continuation sweep on ${TARGET_NODE}: C_LIST=${RESUME_C_LIST}"
resume_submit_out="$(sbatch --nodelist="${TARGET_NODE}" --exclude=research-secure-19 --export=ALL,C_LIST="${RESUME_C_LIST}",SWEEP_SUMMARY="${SUMMARY_TSV}" oss-ToolOrchestra/slurm/hle/run_hle_continuum_restart_per_c.sbatch)"
log "${resume_submit_out}"

log "Done. Retrieval moved to ${TARGET_NODE}. Queue resumed with C_LIST=${RESUME_C_LIST}."

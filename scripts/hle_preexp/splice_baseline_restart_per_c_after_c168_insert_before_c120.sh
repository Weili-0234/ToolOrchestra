#!/usr/bin/env bash
# Wait until the current baseline restart-per-C sweep finishes a given C, then:
#   1) submit a continuation baseline sweep job (dependency: afterany:<old_job>)
#   2) cancel the current sweep job so it doesn't proceed to the next C with the old queue
#
# This is designed to run on the head node (no tmux dependency).
#
# Typical use for this repo's current run:
#   - wait for C=168 to finish
#   - resume with: 144, 288, 312, 336, 120, 96, 72, 48
#     (i.e., insert 288/312/336 before C=120).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/junxiong/haokang/compare-ThunderReact-ToolOrchestra}"
cd "${ROOT_DIR}"

TARGET_NODE="${TARGET_NODE:-research-secure-02}"

SWEEP_JOB_ID="${SWEEP_JOB_ID:?ERROR: set SWEEP_JOB_ID (current baseline sweep job id)}"
SWEEP_LOG="${SWEEP_LOG:-${ROOT_DIR}/slurm/hle_eval_logs/hle_baseline_restart_per_c_${SWEEP_JOB_ID}.out}"

WAIT_FOR_C="${WAIT_FOR_C:-168}"
RESUME_C_LIST="${RESUME_C_LIST:-144 288 312 336 120 96 72 48}"

SUMMARY_TSV="${SUMMARY_TSV:-}"
if [[ -z "${SUMMARY_TSV}" ]]; then
  SUMMARY_TSV="$(ls -1t "${ROOT_DIR}"/outputs/hle_baseline_restart_per_c_*.tsv 2>/dev/null | head -n 1 || true)"
fi

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

if [[ ! -f "${SWEEP_LOG}" ]]; then
  log "ERROR: sweep log not found: ${SWEEP_LOG}"
  exit 1
fi
if [[ -z "${SUMMARY_TSV}" || ! -f "${SUMMARY_TSV}" ]]; then
  log "ERROR: summary TSV not found: ${SUMMARY_TSV}"
  exit 1
fi

log "Submitting continuation baseline sweep (dependency: afterany:${SWEEP_JOB_ID})..."
resume_submit_out="$(
  sbatch \
    --dependency="afterany:${SWEEP_JOB_ID}" \
    --nodelist="${TARGET_NODE}" \
    --exclude=research-secure-19 \
    --export=ALL,C_LIST="${RESUME_C_LIST}",SWEEP_SUMMARY="${SUMMARY_TSV}" \
    oss-ToolOrchestra/slurm/hle/run_hle_baseline_restart_per_c.sbatch
)"
log "${resume_submit_out}"
resume_jid="$(echo "${resume_submit_out}" | awk '{print $NF}')"
if [[ -z "${resume_jid}" ]]; then
  log "ERROR: failed to parse resume job id from: ${resume_submit_out}"
  exit 1
fi
log "Resume job id: ${resume_jid}"

marker="\\[sweep\\] C=${WAIT_FOR_C}: done\\."
log "Waiting for marker in sweep log: ${marker}"
while ! rg -n "${marker}" "${SWEEP_LOG}" >/dev/null 2>&1; do
  sleep 15
done
log "Detected C=${WAIT_FOR_C} done. Canceling sweep job ${SWEEP_JOB_ID} so resume can start."

if squeue -j "${SWEEP_JOB_ID}" -h >/dev/null 2>&1; then
  scancel "${SWEEP_JOB_ID}" || true
else
  log "Sweep job ${SWEEP_JOB_ID} not in queue; resume job will start when dependency is satisfied."
fi

log "Done. After sweep job ${SWEEP_JOB_ID} ends, resume job ${resume_jid} will run C_LIST='${RESUME_C_LIST}' appending to ${SUMMARY_TSV}."


#!/bin/bash
#SBATCH --partition batch
#SBATCH --time 24:00:00
#SBATCH --nodes 2
#SBATCH --gpus-per-node=4
#SBATCH --job-name tr_multibackend
#SBATCH --output=tr_multibackend_%j.out
#SBATCH --error=tr_multibackend_%j.err

set -euo pipefail

# Phase 6: ThunderReact multi-backend (later rollout in the experiment plan).
#
# This job starts:
# - 4x vLLM instances per node (1 GPU each), ports 8100-8103 on each node
# - 1x ThunderReact router (FastAPI) on the allocation's first node, port ${ROUTER_PORT:-8000}
#
# NOTE: For SLURM cluster usage, manage via tmux on head node as documented in:
#   oss-ToolOrchestra/.cursor/rules/work-on-slurm-cluster-via-tmux.mdc

source ~/.bashrc
conda activate vllm1

if [[ -z "${CKPT_DIR:-}" ]]; then
  echo "ERROR: CKPT_DIR env var must be set to the Orchestrator-8B checkpoint path."
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_ORCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${HOME}/logs/tr_multibackend_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

cleanup() {
  echo "[tr_multibackend] Cleaning up background job steps..."
  jobs -p | xargs -r kill || true
}
trap cleanup EXIT

NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
echo "[tr_multibackend] Nodes: ${NODES[*]}" | tee "${LOG_DIR}/nodes.txt"

# Launch vLLM backends (one per GPU, per node)
BACKENDS=()
for node in "${NODES[@]}"; do
  for i in 0 1 2 3; do
    port=$((8100 + i))
    BACKENDS+=("http://${node}:${port}")
    echo "[tr_multibackend] Starting vLLM on ${node} port ${port}" | tee -a "${LOG_DIR}/launch.log"
    srun --nodes=1 --ntasks=1 --nodelist="${node}" --exclusive \
      bash -lc "vllm serve \"${CKPT_DIR}\" --enable-auto-tool-choice --tool-call-parser hermes --port ${port} --gpu-memory-utilization 0.95" \
      2>&1 | tee "${LOG_DIR}/vllm_${node}_port${port}.log" &
    sleep 2
  done
done

export VLLM_BACKENDS="$(IFS=, ; echo "${BACKENDS[*]}")"
export ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
export ROUTER_PORT="${ROUTER_PORT:-8000}"

echo "[tr_multibackend] VLLM_BACKENDS=${VLLM_BACKENDS}" | tee "${LOG_DIR}/backends.txt"
echo "[tr_multibackend] Starting router on ${ROUTER_HOST}:${ROUTER_PORT}" | tee -a "${LOG_DIR}/launch.log"

cd "${TOOL_ORCH_DIR}"
python multinode_router.py 2>&1 | tee "${LOG_DIR}/router.log"



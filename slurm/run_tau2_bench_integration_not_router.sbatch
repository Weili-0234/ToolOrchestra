#!/bin/bash
#SBATCH --partition batch
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name tau2_integ_no_router
#SBATCH --output=/home/%u/logs/tau2_integ_no_router_%j.out
#SBATCH --error=/home/%u/logs/tau2_integ_no_router_%j.err

set -euo pipefail

# Activate conda env (CUDA 12.9-compatible torch lives in vllm1 on this cluster)
source /home/junxiong/miniconda3/etc/profile.d/conda.sh
conda activate vllm1

# Locate repo root by walking up from SLURM_SUBMIT_DIR until setup_envs.sh is found.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
TOOL_ORCH_DIR=""
for candidate in \
  "${SUBMIT_DIR}" \
  "$(cd "${SUBMIT_DIR}/.." && pwd)" \
  "$(cd "${SUBMIT_DIR}/../.." && pwd)" \
  "$(cd "${SUBMIT_DIR}/../../.." && pwd)"; do
  if [[ -f "${candidate}/setup_envs.sh" ]]; then
    TOOL_ORCH_DIR="${candidate}"
    break
  fi
  if [[ -f "${candidate}/oss-ToolOrchestra/setup_envs.sh" ]]; then
    TOOL_ORCH_DIR="${candidate}/oss-ToolOrchestra"
    break
  fi
done
if [[ -z "${TOOL_ORCH_DIR}" ]]; then
  echo "ERROR: could not locate setup_envs.sh from SLURM_SUBMIT_DIR=${SUBMIT_DIR}"
  exit 2
fi

# setup_envs.sh references $LD_LIBRARY_PATH; with `set -u` enabled, ensure it's defined.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Set REPO_PATH / HF_HOME / keys
source "${TOOL_ORCH_DIR}/setup_envs.sh"

LOG_DIR="${HOME}/logs/tau2_integ_no_router_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

{
  echo "[tau2_integ_no_router] date: $(date)"
  echo "[tau2_integ_no_router] node: $(hostname)"
  echo "[tau2_integ_no_router] python: $(which python)"
  echo "[tau2_integ_no_router] REPO_PATH=${REPO_PATH:-}"
  python -c "import torch; print('[tau2_integ_no_router] torch:', torch.__version__)"
} | tee "${LOG_DIR}/env.txt"

cd "${REPO_PATH}/evaluation/tau2-bench"

echo "[tau2_integ_no_router] Running: pytest -m 'integration and not router'"
pytest -m "integration and not router" 2>&1 | tee "${LOG_DIR}/integration_not_router.log"

echo "[tau2_integ_no_router] Done."



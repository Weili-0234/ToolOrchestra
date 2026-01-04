#!/bin/bash
#SBATCH --partition batch
#SBATCH --time 00:30:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name tau2_pytest
#SBATCH --output=/home/%u/logs/tau2_pytest_%j.out
#SBATCH --error=/home/%u/logs/tau2_pytest_%j.err

set -euo pipefail

# Activate conda env (CUDA 12.9-compatible torch lives in vllm1 on this cluster)
source /home/junxiong/miniconda3/etc/profile.d/conda.sh
conda activate vllm1

# Locate repo root by walking up from SLURM_SUBMIT_DIR until setup_envs.sh is found.
# (SLURM may copy the script to /var/lib/slurm/... so BASH_SOURCE is not reliable.)
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
  # If submitting from the compare repo root, the active code lives under oss-ToolOrchestra/
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

# Set REPO_PATH / HF_HOME / keys (as requested)
source "${TOOL_ORCH_DIR}/setup_envs.sh"

LOG_DIR="${HOME}/logs/tau2_pytest_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

{
  echo "[tau2_pytest] date: $(date)"
  echo "[tau2_pytest] node: $(hostname)"
  echo "[tau2_pytest] python: $(which python)"
  echo "[tau2_pytest] REPO_PATH=${REPO_PATH:-}"
  python -c "import torch; print('[tau2_pytest] torch:', torch.__version__)"
} | tee "${LOG_DIR}/env.txt"

cd "${REPO_PATH}/evaluation/tau2-bench"

echo "[tau2_pytest] Running unit-only tests (-m 'not integration')"
pytest -m "not integration" 2>&1 | tee "${LOG_DIR}/unit.log"

echo "[tau2_pytest] Running router integration tests (-m 'integration and router')"
pytest -m "integration and router" 2>&1 | tee "${LOG_DIR}/router_integration.log"

echo "[tau2_pytest] Done."



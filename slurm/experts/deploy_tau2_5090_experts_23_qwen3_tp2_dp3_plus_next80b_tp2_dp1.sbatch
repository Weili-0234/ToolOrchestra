#!/bin/bash
#SBATCH --partition batch
#SBATCH --time 96:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --job-name rl_toolorch_expert23
#SBATCH --output=slurm/tau2_expert_logs/expert-23/%x_%j.out
#SBATCH --error=slurm/tau2_expert_logs/expert-23/%x_%j.err

set -euo pipefail

# TAU2Bench 5090 inference experiment (H100 cluster experts):
# - expert-2: Qwen/Qwen3-32B-FP8
#   - TP=2, DP=3 (3 instances, each uses 2 GPUs)
#   - Ports: 1904-1906
# - expert-3: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
#   - TP=2, DP=1 (1 instance uses 2 GPUs)
#   - Port: 1920
#
# Placement: expert-2 and expert-3 share a single 8×H100 node (exclusive).

source ~/.bashrc 2>/dev/null || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_ROOT="${SUBMIT_DIR}/slurm/tau2_expert_logs/expert-23"
LOG_DIR="${LOG_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

CACHE_BASE_DIR="${HOME}/haokang/cache/vllm_compile"
mkdir -p "${CACHE_BASE_DIR}"

NODE_IP=""
if command -v ip >/dev/null 2>&1; then
  NODE_IP="$(
    ip -4 -o addr show scope global 2>/dev/null \
      | awk '{print $4}' \
      | cut -d/ -f1 \
      | grep -E '^172\\.' \
      | head -n 1 \
      || true
  )"
fi
if [[ -z "${NODE_IP}" ]]; then
  NODE_IP="$(hostname -I | awk '{print $1}')"
fi
HOSTNAME="$(hostname)"
echo "${NODE_IP}" | tee "${LOG_DIR}/node_ip.txt"

echo "[tau2_5090_experts_23] Node IP: ${NODE_IP}"
echo "[tau2_5090_experts_23] Hostname: ${HOSTNAME}"
echo "[tau2_5090_experts_23] Log dir: ${LOG_DIR}"
echo "[tau2_5090_experts_23] Cache base dir: ${CACHE_BASE_DIR}"

check_gpus_clean() {
  local mem_threshold_mib="${GPU_CLEAN_MEM_THRESHOLD_MIB:-500}"
  echo "[tau2_5090_experts_23] Preflight GPU check (mem_threshold_mib=${mem_threshold_mib})"
  nvidia-smi || true
  local apps
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [ -n "${apps}" ]; then
    echo "[tau2_5090_experts_23] ERROR: Detected existing GPU compute processes (node not clean)."
    echo "${apps}"
    exit 1
  fi
  local mem_used
  mem_used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
  if echo "${mem_used}" | awk -F',' -v thr="${mem_threshold_mib}" '{gsub(/ /,"",$2); if ($2+0 > thr) bad=1} END{exit bad?0:1}'; then
    echo "[tau2_5090_experts_23] ERROR: Detected GPU memory usage above threshold (node not clean):"
    echo "${mem_used}"
    exit 1
  fi
  echo "[tau2_5090_experts_23] Preflight GPU check: OK."
}

check_gpus_clean

cleanup() {
  echo "[tau2_5090_experts_23] Cleaning up background processes..."
  jobs -p | xargs -r kill || true
}
trap cleanup EXIT

# expert-2 (Qwen3-32B-FP8): 3 instances, GPUs 0-5, ports 1904-1906
for i in 0 1 2; do
  gpu_start=$((i * 2))
  gpus="${gpu_start},$((gpu_start + 1))"
  port=$((1904 + i))

  INSTANCE_CACHE_DIR="${CACHE_BASE_DIR}/${HOSTNAME}/tau2_5090_expert2_instance_${i}"
  mkdir -p "${INSTANCE_CACHE_DIR}/inductor" "${INSTANCE_CACHE_DIR}/vllm"

  echo "[tau2_5090_experts_23] Starting Qwen/Qwen3-32B-FP8 instance ${i} on GPU ${gpus} port ${port}"
  CUDA_VISIBLE_DEVICES="${gpus}" \
    TORCHINDUCTOR_CACHE_DIR="${INSTANCE_CACHE_DIR}/inductor" \
    VLLM_CACHE_ROOT="${INSTANCE_CACHE_DIR}/vllm" \
    vllm serve Qwen/Qwen3-32B-FP8 \
      --tensor-parallel-size 2 \
      --host 0.0.0.0 \
      --port "${port}" \
      --gpu-memory-utilization 0.95 \
      > "${LOG_DIR}/expert2_instance_${i}.log" 2>&1 &
  sleep 60
done

# expert-3 (Qwen3-Next-80B-A3B-Instruct-FP8): 1 instance, GPUs 6-7, port 1920
INSTANCE_CACHE_DIR="${CACHE_BASE_DIR}/${HOSTNAME}/tau2_5090_expert3_instance_0"
mkdir -p "${INSTANCE_CACHE_DIR}/inductor" "${INSTANCE_CACHE_DIR}/vllm"
echo "[tau2_5090_experts_23] Starting Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 on GPU 6-7 port 1920"
CUDA_VISIBLE_DEVICES="6,7" \
  TORCHINDUCTOR_CACHE_DIR="${INSTANCE_CACHE_DIR}/inductor" \
  VLLM_CACHE_ROOT="${INSTANCE_CACHE_DIR}/vllm" \
  vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 1920 \
    --gpu-memory-utilization 0.88 \
    --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 4}' \
    --tokenizer-mode auto \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    > "${LOG_DIR}/expert3_instance_0.log" 2>&1 &

echo "[tau2_5090_experts_23] All instances started. Waiting..."
wait

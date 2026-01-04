#!/bin/bash
#SBATCH --partition batch
#SBATCH --time 48:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --job-name rollout_expert_1
#SBATCH --output=slurm/expert_logs/expert-1/%x_%j.out
#SBATCH --error=slurm/expert_logs/expert-1/%x_%j.err

set -euo pipefail

# Expert-1: openai/gpt-oss-20b with TP=2, DP=4 (4 instances, each using 2 GPUs)
# Ports: 1910-1913

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_ROOT="${SUBMIT_DIR}/slurm/expert_logs/expert-1"
LOG_DIR="${LOG_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

# persistent cache for torch compile
CACHE_BASE_DIR="${HOME}/haokang/cache/vllm_compile"
mkdir -p "${CACHE_BASE_DIR}"

NODE_IP="$(hostname -I | awk '{print $1}')"
HOSTNAME="$(hostname)"
echo "${NODE_IP}" | tee "${LOG_DIR}/node_ip.txt"

echo "[expert_1] Node IP: ${NODE_IP}"
echo "[expert_1] Hostname: ${HOSTNAME}"
echo "[expert_1] Log dir: ${LOG_DIR}"
echo "[expert_1] Cache base dir: ${CACHE_BASE_DIR}"

check_gpus_clean() {
    # Ensure the allocated node's GPUs are not occupied by other jobs/processes.
    # This prints to stdout/stderr so it is recorded in Slurm .out/.err.
    local mem_threshold_mib="${GPU_CLEAN_MEM_THRESHOLD_MIB:-500}"

    echo "[expert_1] Preflight GPU check (mem_threshold_mib=${mem_threshold_mib})"
    echo "[expert_1] nvidia-smi:"
    nvidia-smi || true

    # Any compute process is a strong signal the node isn't clean.
    local apps
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -n "${apps}" ]; then
        echo "[expert_1] ERROR: Detected existing GPU compute processes (node not clean)."
        echo "${apps}"
        exit 1
    fi

    # Fallback: check for unexpected memory usage even if compute-apps is empty.
    local mem_used
    mem_used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
    if echo "${mem_used}" | awk -F',' -v thr="${mem_threshold_mib}" '{gsub(/ /,"",$2); if ($2+0 > thr) bad=1} END{exit bad?0:1}'; then
        echo "[expert_1] ERROR: Detected GPU memory usage above threshold (node not clean):"
        echo "${mem_used}"
        exit 1
    fi

    echo "[expert_1] Preflight GPU check: OK (no existing compute processes, mem usage within threshold)."
}

check_gpus_clean

cleanup() {
    echo "[expert_1] Cleaning up background processes..."
    jobs -p | xargs -r kill || true
}
trap cleanup EXIT

# TP=2, DP=4: 4 instances, each using 2 GPUs
for i in 0 1 2 3; do
    gpu_start=$((i * 2))
    gpu_end=$((gpu_start + 1))
    port=$((1910 + i))
    
    # Set cache directories for each instance to avoid concurrent write corruption
    INSTANCE_CACHE_DIR="${CACHE_BASE_DIR}/${HOSTNAME}/expert_1_instance_${i}"
    mkdir -p "${INSTANCE_CACHE_DIR}/inductor" "${INSTANCE_CACHE_DIR}/vllm"

    echo "[expert_1] Starting openai/gpt-oss-20b instance ${i} on GPU ${gpu_start}-${gpu_end} port ${port}"
    CUDA_VISIBLE_DEVICES="${gpu_start},${gpu_end}" \
    TORCHINDUCTOR_CACHE_DIR="${INSTANCE_CACHE_DIR}/inductor" \
    VLLM_CACHE_ROOT="${INSTANCE_CACHE_DIR}/vllm" \
    vllm serve openai/gpt-oss-20b \
        --tensor-parallel-size 2 \
        --port "${port}" \
        --gpu-memory-utilization 0.95 \
        | tee "${LOG_DIR}/instance_${i}.log" 2>&1 &
    sleep 60
done

echo "[expert_1] All instances started. Waiting..."
wait

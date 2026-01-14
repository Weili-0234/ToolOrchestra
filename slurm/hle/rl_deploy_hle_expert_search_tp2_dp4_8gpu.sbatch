#!/bin/bash
#SBATCH --partition batch
#SBATCH --time 96:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --job-name rl_hle_expert_search
#SBATCH --output=/home/junxiong/haokang/compare-ThunderReact-ToolOrchestra/slurm/hle_expert_logs/search/%x_%j.out
#SBATCH --error=/home/junxiong/haokang/compare-ThunderReact-ToolOrchestra/slurm/hle_expert_logs/search/%x_%j.err

set -euo pipefail

# HLE Expert (Search): openai/gpt-oss-20b with TP=2, DP=4 (4 instances, each using 2 GPUs)
# Ports: 1840-1843

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm1

LOG_ROOT="/home/junxiong/haokang/compare-ThunderReact-ToolOrchestra/slurm/hle_expert_logs/search"
LOG_DIR="${LOG_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

# persistent cache for torch compile
CACHE_BASE_DIR="${HOME}/haokang/cache/vllm_compile"
mkdir -p "${CACHE_BASE_DIR}"

NODE_IP="$(hostname -I | awk '{print $1}')"
HOSTNAME="$(hostname)"
echo "${NODE_IP}" | tee "${LOG_DIR}/node_ip.txt"

echo "[rl_hle_expert_search] Node IP: ${NODE_IP}"
echo "[rl_hle_expert_search] Hostname: ${HOSTNAME}"
echo "[rl_hle_expert_search] Log dir: ${LOG_DIR}"
echo "[rl_hle_expert_search] Cache base dir: ${CACHE_BASE_DIR}"

check_gpus_clean() {
    local mem_threshold_mib="${GPU_CLEAN_MEM_THRESHOLD_MIB:-500}"
    echo "[rl_hle_expert_search] Preflight GPU check (mem_threshold_mib=${mem_threshold_mib})"
    nvidia-smi || true

    local apps
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -n "${apps}" ]; then
        echo "[rl_hle_expert_search] ERROR: Detected existing GPU compute processes (node not clean)."
        echo "${apps}"
        exit 1
    fi

    local mem_used
    mem_used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
    if echo "${mem_used}" | awk -F',' -v thr="${mem_threshold_mib}" '{gsub(/ /,"",$2); if ($2+0 > thr) bad=1} END{exit bad?0:1}'; then
        echo "[rl_hle_expert_search] ERROR: Detected GPU memory usage above threshold (node not clean):"
        echo "${mem_used}"
        exit 1
    fi

    echo "[rl_hle_expert_search] Preflight GPU check: OK."
}

check_ports_free() {
    local ports=("$@")
    python - <<'PY' "${ports[@]}"
import socket, sys
ports = [int(p) for p in sys.argv[1:]]
bad = []
for p in ports:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", p))
    except OSError:
        bad.append(p)
    finally:
        s.close()
if bad:
    print(f"[rl_hle_expert_search] ERROR: ports already in use: {bad}", file=sys.stderr)
    raise SystemExit(1)
print(f"[rl_hle_expert_search] Port preflight OK: {ports}")
PY
}

cleanup() {
    echo "[rl_hle_expert_search] Cleaning up background processes..."
    jobs -p | xargs -r kill || true
}
trap cleanup EXIT

check_gpus_clean
check_ports_free 1840 1841 1842 1843

MODEL="openai/gpt-oss-20b"
BASE_PORT=1840

# TP=2, DP=4: 4 instances, each using 2 GPUs
for i in 0 1 2 3; do
    gpu_start=$((i * 2))
    gpu_end=$((gpu_start + 1))
    port=$((BASE_PORT + i))

    INSTANCE_CACHE_DIR="${CACHE_BASE_DIR}/${HOSTNAME}/rl_hle_search_instance_${i}"
    mkdir -p "${INSTANCE_CACHE_DIR}/inductor" "${INSTANCE_CACHE_DIR}/vllm"

    echo "[rl_hle_expert_search] Starting ${MODEL} instance ${i} on GPU ${gpu_start}-${gpu_end} port ${port}"
    CUDA_VISIBLE_DEVICES="${gpu_start},${gpu_end}" \
    TORCHINDUCTOR_CACHE_DIR="${INSTANCE_CACHE_DIR}/inductor" \
    VLLM_CACHE_ROOT="${INSTANCE_CACHE_DIR}/vllm" \
    vllm serve "${MODEL}" \
        --tensor-parallel-size 2 \
        --port "${port}" \
        --gpu-memory-utilization 0.95 \
        | tee "${LOG_DIR}/instance_${i}.log" 2>&1 &
    sleep 60
done

echo "[rl_hle_expert_search] All instances started. Waiting..."
wait

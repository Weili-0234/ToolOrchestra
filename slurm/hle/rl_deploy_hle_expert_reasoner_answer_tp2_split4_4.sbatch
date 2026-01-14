#!/bin/bash
#SBATCH --partition batch
#SBATCH --time 96:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --job-name rl_hle_expert_reasoner_answer
#SBATCH --output=/home/junxiong/haokang/compare-ThunderReact-ToolOrchestra/slurm/hle_expert_logs/reasoner_answer/%x_%j.out
#SBATCH --error=/home/junxiong/haokang/compare-ThunderReact-ToolOrchestra/slurm/hle_expert_logs/reasoner_answer/%x_%j.err

set -euo pipefail

# HLE Experts (Reasoner+Answer) on one 8×H100 node:
# - Reasoner: Qwen/Qwen2.5-Coder-14B-Instruct with TP=2, 2 instances (4 GPUs total)
#   Ports: 1810-1811 (GPUs 0-1, 2-3)
# - Answer: Qwen/Qwen3-32B-FP8 with TP=2, 2 instances (4 GPUs total)
#   Ports: 1820-1821 (GPUs 4-5, 6-7)

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm1

LOG_ROOT="/home/junxiong/haokang/compare-ThunderReact-ToolOrchestra/slurm/hle_expert_logs/reasoner_answer"
LOG_DIR="${LOG_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

# persistent cache for torch compile
CACHE_BASE_DIR="${HOME}/haokang/cache/vllm_compile"
mkdir -p "${CACHE_BASE_DIR}"

NODE_IP="$(hostname -I | awk '{print $1}')"
HOSTNAME="$(hostname)"
echo "${NODE_IP}" | tee "${LOG_DIR}/node_ip.txt"

echo "[rl_hle_expert_reasoner_answer] Node IP: ${NODE_IP}"
echo "[rl_hle_expert_reasoner_answer] Hostname: ${HOSTNAME}"
echo "[rl_hle_expert_reasoner_answer] Log dir: ${LOG_DIR}"
echo "[rl_hle_expert_reasoner_answer] Cache base dir: ${CACHE_BASE_DIR}"

check_gpus_clean() {
    local mem_threshold_mib="${GPU_CLEAN_MEM_THRESHOLD_MIB:-500}"
    echo "[rl_hle_expert_reasoner_answer] Preflight GPU check (mem_threshold_mib=${mem_threshold_mib})"
    nvidia-smi || true

    local apps
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -n "${apps}" ]; then
        echo "[rl_hle_expert_reasoner_answer] ERROR: Detected existing GPU compute processes (node not clean)."
        echo "${apps}"
        exit 1
    fi

    local mem_used
    mem_used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
    if echo "${mem_used}" | awk -F',' -v thr="${mem_threshold_mib}" '{gsub(/ /,"",$2); if ($2+0 > thr) bad=1} END{exit bad?0:1}'; then
        echo "[rl_hle_expert_reasoner_answer] ERROR: Detected GPU memory usage above threshold (node not clean):"
        echo "${mem_used}"
        exit 1
    fi

    echo "[rl_hle_expert_reasoner_answer] Preflight GPU check: OK."
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
    print(f"[rl_hle_expert_reasoner_answer] ERROR: ports already in use: {bad}", file=sys.stderr)
    raise SystemExit(1)
print(f"[rl_hle_expert_reasoner_answer] Port preflight OK: {ports}")
PY
}

cleanup() {
    echo "[rl_hle_expert_reasoner_answer] Cleaning up background processes..."
    jobs -p | xargs -r kill || true
}
trap cleanup EXIT

check_gpus_clean
check_ports_free 1810 1811 1820 1821

start_instance() {
    local name="$1"
    local model="$2"
    local port="$3"
    local cuda_visible="$4"
    local cache_suffix="$5"

    local INSTANCE_CACHE_DIR="${CACHE_BASE_DIR}/${HOSTNAME}/${cache_suffix}"
    mkdir -p "${INSTANCE_CACHE_DIR}/inductor" "${INSTANCE_CACHE_DIR}/vllm"

    echo "[rl_hle_expert_reasoner_answer] Starting ${name} (${model}) CUDA_VISIBLE_DEVICES=${cuda_visible} port=${port}"
    CUDA_VISIBLE_DEVICES="${cuda_visible}" \
    TORCHINDUCTOR_CACHE_DIR="${INSTANCE_CACHE_DIR}/inductor" \
    VLLM_CACHE_ROOT="${INSTANCE_CACHE_DIR}/vllm" \
    vllm serve "${model}" \
        --tensor-parallel-size 2 \
        --port "${port}" \
        --gpu-memory-utilization 0.95 \
        | tee "${LOG_DIR}/${name}_port_${port}.log" 2>&1 &
}

# Reasoner (GPUs 0-3): 2 instances
start_instance "reasoner_0" "Qwen/Qwen2.5-Coder-14B-Instruct" 1810 "0,1" "rl_hle_reasoner_instance_0"
sleep 60
start_instance "reasoner_1" "Qwen/Qwen2.5-Coder-14B-Instruct" 1811 "2,3" "rl_hle_reasoner_instance_1"
sleep 60

# Answer (GPUs 4-7): 2 instances
start_instance "answer_0" "Qwen/Qwen3-32B-FP8" 1820 "4,5" "rl_hle_answer_instance_0"
sleep 60
start_instance "answer_1" "Qwen/Qwen3-32B-FP8" 1821 "6,7" "rl_hle_answer_instance_1"
sleep 60

echo "[rl_hle_expert_reasoner_answer] All instances started. Waiting..."
wait

#!/bin/bash
#SBATCH --partition batch
#SBATCH --time 48:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --exclude=research-secure-18
#SBATCH --job-name rollout_orch_thunderreact
#SBATCH --output=rollout_orch_thunderreact_%j.out
#SBATCH --error=rollout_orch_thunderreact_%j.err

set -euo pipefail

# Orchestrator-8B with ThunderReact Router
# 8 vLLM backends (one per GPU) on ports 8100-8107
# ThunderReact router on port 8000

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm1

# Fix for potential unbound variable in setup_envs.sh
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
# setup_envs.sh prints key prefixes like ${NEBIUS_API_KEY:0:10} under bash -u; ensure they are defined.
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
export TOGETHER_API_KEY="${TOGETHER_API_KEY:-}"
export NEBIUS_API_KEY="${NEBIUS_API_KEY:-}"
export TAVILY_KEY="${TAVILY_KEY:-}"
LOG_DIR="${HOME}/logs/rollout_orch_thunderreact_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"
source /home/junxiong/haokang/ToolOrchestra/setup_envs.sh > "${LOG_DIR}/setup_envs.log"
CKPT_DIR="${CKPT_DIR:?ERROR: CKPT_DIR not set}"
TOOL_ORCH_DIR="${TOOL_ORCH_DIR:-/home/junxiong/haokang/ToolOrchestra}"

# persistent cache for torch compile
CACHE_BASE_DIR="${HOME}/haokang/cache/vllm_compile"
mkdir -p "${CACHE_BASE_DIR}"

NODE_IP="$(hostname -I | awk '{print $1}')"
HOSTNAME="$(hostname)"
echo "${NODE_IP}" | tee "${LOG_DIR}/node_ip.txt"

echo "[thunderreact] Node IP: ${NODE_IP}"
echo "[thunderreact] Hostname: ${HOSTNAME}"
echo "[thunderreact] CKPT_DIR: ${CKPT_DIR}"
echo "[thunderreact] Log dir: ${LOG_DIR}"
echo "[thunderreact] Cache base dir: ${CACHE_BASE_DIR}"

check_gpus_clean() {
    # Ensure the allocated node's GPUs are not occupied by other jobs/processes.
    # This prints to stdout/stderr so it is recorded in Slurm .out/.err.
    local mem_threshold_mib="${GPU_CLEAN_MEM_THRESHOLD_MIB:-500}"
    local dirty_nodes_file="${DIRTY_NODES_FILE:-${HOME}/haokang/cache/slurm_dirty_nodes.txt}"

    handle_dirty_node() {
        local reason="${1:-unknown}"
        mkdir -p "$(dirname "${dirty_nodes_file}")"
        echo "${HOSTNAME}" >> "${dirty_nodes_file}" || true

        echo "[thunderreact] ERROR: Node marked dirty (${reason}). Hostname=${HOSTNAME}" >&2
        echo "[thunderreact] Dirty nodes file: ${dirty_nodes_file}" >&2

        exit 1
    }

    echo "[thunderreact] Preflight GPU check (mem_threshold_mib=${mem_threshold_mib})"
    echo "[thunderreact] Preflight dirty-nodes settings: DIRTY_NODES_FILE=${dirty_nodes_file}"
    echo "[thunderreact] nvidia-smi:"
    nvidia-smi || true

    # Any compute process is a strong signal the node isn't clean.
    local apps
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -n "${apps}" ]; then
        echo "[thunderreact] Detected existing GPU compute processes (node not clean):" >&2
        echo "${apps}" >&2
        handle_dirty_node "compute_processes_detected"
    fi

    # Fallback: check for unexpected memory usage even if compute-apps is empty.
    local mem_used
    mem_used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
    if echo "${mem_used}" | awk -F',' -v thr="${mem_threshold_mib}" '{gsub(/ /,"",$2); if ($2+0 > thr) bad=1} END{exit bad?0:1}'; then
        echo "[thunderreact] Detected GPU memory usage above threshold (node not clean):" >&2
        echo "${mem_used}" >&2
        handle_dirty_node "memory_usage_above_threshold"
    fi

    echo "[thunderreact] Preflight GPU check: OK (no existing compute processes, mem usage within threshold)."
}

check_gpus_clean

cleanup() {
    echo "[thunderreact] Cleaning up background processes..."
    jobs -p | xargs -r kill || true
}
trap cleanup EXIT

# Start 8 vLLM backends on ports 8100-8107
BACKENDS=""
for i in 0 1 2 3 4 5 6 7; do
    port=$((8100 + i))
    BACKENDS="${BACKENDS:+${BACKENDS},}http://localhost:${port}"
    
    # Set cache directories for each backend to avoid concurrent write corruption
    INSTANCE_CACHE_DIR="${CACHE_BASE_DIR}/${HOSTNAME}/orchestrator_gpu${i}"
    mkdir -p "${INSTANCE_CACHE_DIR}/inductor" "${INSTANCE_CACHE_DIR}/vllm"

    echo "[thunderreact] Starting vLLM backend ${i} on GPU ${i} port ${port}"
    CUDA_VISIBLE_DEVICES=${i} \
    TORCHINDUCTOR_CACHE_DIR="${INSTANCE_CACHE_DIR}/inductor" \
    VLLM_CACHE_ROOT="${INSTANCE_CACHE_DIR}/vllm" \
    vllm serve "${CKPT_DIR}" \
        --enable-auto-tool-choice \
        --tool-call-parser llama3_json \
        --chat-template "${TOOL_ORCH_DIR}/evaluation/tool_chat_template_llama3.1_json.jinja" \
        --port "${port}" \
        --gpu-memory-utilization 0.95 \
        | tee "${LOG_DIR}/backend_${i}.log" 2>&1 &
    sleep 30
done

# Wait for backends to be ready
echo "[thunderreact] Waiting 5 minutes for backends to be ready..."
sleep 300

# Verify backends are responding
echo "[thunderreact] Checking backend health..."
FAILED=0
for i in 0 1 2 3 4 5 6 7; do
    port=$((8100 + i))
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "[thunderreact] Backend ${i} (port ${port}): OK" | tee -a "${LOG_DIR}/backend_${i}.log"
    else
        echo "[thunderreact] Backend ${i} (port ${port}): FAILED" | tee -a "${LOG_DIR}/backend_${i}.log"
        FAILED=1
    fi
done

if [ "${FAILED}" -ne 0 ]; then
    echo "[thunderreact] One or more backends failed health check. Dumping last 80 lines of each backend log to help debugging..."
    for i in 0 1 2 3 4 5 6 7; do
        echo "==================== backend_${i}.log (tail -n 80) ===================="
        tail -n 80 "${LOG_DIR}/backend_${i}.log" || true
    done
    echo "[thunderreact] Exiting due to backend startup failure."
    exit 1
fi

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | tee "${LOG_DIR}/nvidia-smi.log"

echo "[thunderreact] Starting ThunderReact router..."
echo "[thunderreact] VLLM_BACKENDS=${BACKENDS}"

# Start ThunderReact router
export VLLM_BACKENDS="${BACKENDS}"
export ROUTER_HOST="0.0.0.0"
export ROUTER_PORT="8000"

cd "${TOOL_ORCH_DIR}"
python multinode_router.py 2>&1 | tee "${LOG_DIR}/router.log"

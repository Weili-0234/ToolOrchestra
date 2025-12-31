#!/bin/bash

#SBATCH --partition batch
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name oss_eval_node2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=oss_eval_node2.out
#SBATCH --error=oss_eval_node2.err

set -x

# Print node IP for service discovery
hostname -i

# Setup environment - must source conda.sh first for non-interactive shells
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm1

# Verify vllm is available
which vllm || echo "ERROR: vllm not found in PATH"

echo "HF_HOME: $HF_HOME"
echo "CKPT_DIR: $CKPT_DIR"
echo "Job: oss_eval_node2"
echo "Node: $(hostname)"

# Set cache directories
CACHE_BASE="${USER_PATH:-$HOME}/cache/vllm"
mkdir -p "$CACHE_BASE"

# ===========================================================================
# GPU Cleanup Check: Verify node is clean before launching vLLM servers
# ===========================================================================
# IMPORTANT: sbatch may allocate nodes that appear idle in SLURM but actually
# have running processes. This check ensures the node is clean before we start.
echo "=== GPU Status at Job Start ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv
echo "=== End GPU Status ==="

# Check for any existing GPU processes
echo "=== Checking for existing GPU processes ==="
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | wc -l)
if [ "$GPU_PROCESSES" -gt 0 ]; then
    echo "WARNING: Found $GPU_PROCESSES existing GPU processes!"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo ""
    echo "=== Attempting to kill stale processes ==="
    # Kill any GPU processes not owned by us (with timeout)
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
        if [ -n "$pid" ]; then
            PROC_USER=$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')
            if [ "$PROC_USER" = "$USER" ]; then
                echo "Killing our own stale process: PID=$pid"
                kill -9 $pid 2>/dev/null || true
            else
                echo "WARNING: Found process owned by $PROC_USER (PID=$pid) - cannot kill"
            fi
        fi
    done
    sleep 5
    # Re-check after cleanup
    GPU_PROCESSES_AFTER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_PROCESSES_AFTER" -gt 0 ]; then
        echo "ERROR: Still have $GPU_PROCESSES_AFTER GPU processes after cleanup. Node may be occupied."
        nvidia-smi
        # Continue anyway but log the warning - let the user decide
    else
        echo "GPU cleanup successful - all GPUs are now free"
    fi
else
    echo "All GPUs are clean - no existing processes found"
fi
echo "=== End GPU Cleanup Check ==="

# Expert: openai/gpt-oss-20b (instance 0)
export VLLM_CACHE_ROOT="$CACHE_BASE/oss_eval_node2_openai_gpt-oss-20b_0"
export CUDA_VISIBLE_DEVICES=0
vllm serve openai/gpt-oss-20b \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 1910 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    &
sleep 60

# Expert: openai/gpt-oss-20b (instance 1)
export VLLM_CACHE_ROOT="$CACHE_BASE/oss_eval_node2_openai_gpt-oss-20b_1"
export CUDA_VISIBLE_DEVICES=1
vllm serve openai/gpt-oss-20b \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 1911 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    &
sleep 60

# Expert: openai/gpt-oss-20b (instance 2)
export VLLM_CACHE_ROOT="$CACHE_BASE/oss_eval_node2_openai_gpt-oss-20b_2"
export CUDA_VISIBLE_DEVICES=2
vllm serve openai/gpt-oss-20b \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 1912 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    &
sleep 60

# Expert: openai/gpt-oss-20b (instance 3)
export VLLM_CACHE_ROOT="$CACHE_BASE/oss_eval_node2_openai_gpt-oss-20b_3"
export CUDA_VISIBLE_DEVICES=3
vllm serve openai/gpt-oss-20b \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 1913 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    &
sleep 60

# Keep job running
sleep 15000

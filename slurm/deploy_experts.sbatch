#!/bin/bash
#SBATCH --partition batch
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name expert_llms
#SBATCH --output=expert_llms_%j.out
#SBATCH --error=expert_llms_%j.err

set -euo pipefail

# Expert LLM deployment on an H100 node (single node, 8 GPUs).
# Run from a tmux-managed shell on the SLURM head node (see .cursor rules).

source ~/.bashrc
conda activate vllm1

LOG_DIR="${HOME}/logs/expert_llms_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

NODE_IP="$(hostname -I | awk '{print $1}')"
echo "NODE_IP=${NODE_IP}" | tee "${LOG_DIR}/expert_node_ip.txt" > /tmp/expert_node_ip.txt

echo "[deploy_experts] Node IP: ${NODE_IP}"
echo "[deploy_experts] Logs: ${LOG_DIR}"

cleanup() {
  echo "[deploy_experts] Caught exit; killing background servers..."
  jobs -p | xargs -r kill || true
}
trap cleanup EXIT

# Qwen3-32B-FP8 (expert-2): GPU 0-3
for i in 0 1 2 3; do
  port=$((1904 + i))
  echo "[deploy_experts] Starting Qwen/Qwen3-32B-FP8 on GPU ${i} port ${port}"
  CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen3-32B-FP8 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port "${port}" \
    --gpu-memory-utilization 0.95 \
    > "${LOG_DIR}/qwen3_32b_fp8_${port}.log" 2>&1 &
  sleep 30
done

# Qwen2.5-Coder-14B (expert-1): GPU 4-5
for i in 0 1; do
  gpu=$((4 + i))
  port=$((1910 + i))
  echo "[deploy_experts] Starting Qwen/Qwen2.5-Coder-14B-Instruct on GPU ${gpu} port ${port}"
  CUDA_VISIBLE_DEVICES=$gpu vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port "${port}" \
    --gpu-memory-utilization 0.95 \
    > "${LOG_DIR}/qwen2p5_coder_14b_${port}.log" 2>&1 &
  sleep 30
done

# gpt-oss-20b: GPU 6-7
for i in 0 1; do
  gpu=$((6 + i))
  port=$((1916 + i))
  echo "[deploy_experts] Starting openai/gpt-oss-20b on GPU ${gpu} port ${port}"
  CUDA_VISIBLE_DEVICES=$gpu vllm serve openai/gpt-oss-20b \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port "${port}" \
    --gpu-memory-utilization 0.95 \
    > "${LOG_DIR}/gpt_oss_20b_${port}.log" 2>&1 &
  sleep 30
done

echo "[deploy_experts] All expert LLMs started. Waiting..."
wait



# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToolOrchestra is a framework for training small orchestrator models that coordinate intelligent tools and specialized LLMs. The Orchestrator-8B model alternates between reasoning and tool calling in multi-turn interactions, using reinforcement learning (outcome, efficiency, and preference rewards) for joint optimization.

## Environment Setup

Three separate Conda environments are required:

```bash
# Training environment
conda create -n toolorchestra python=3.12 -y
conda activate toolorchestra
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
pip install -e training/rollout

# Retrieval environment (for HLE/FRAMES evaluation)
conda create -n retriever python=3.12 -y
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install transformers datasets pyserini psutil uvicorn fastapi tavily-python
conda install -c pytorch -c nvidia faiss-gpu
pip install flash-attn --no-build-isolation

# vLLM environment (for model serving)
conda create -n vllm1 python=3.12 -y
conda activate vllm1
pip install torch "transformers<4.54.0" vllm==0.9.2
pip install -e evaluation/tau2-bench
```

## Required Environment Variables

```bash
export INDEX_DIR='/path/to/index'           # HuggingFace index files
export CKPT_DIR='/path/to/checkpoint'       # Model checkpoints
export HF_HOME='/path/to/huggingface'
export REPO_PATH='/path/to/this_repo'
export TAVILY_KEY='...'                     # Tavily search API
export TOGETHER_API_KEY='...'               # Together AI API
export WANDB_API_KEY='...'
export OSS_KEY='...'                        # NVIDIA NGC key
export CLIENT_ID='...'
export CLIENT_SECRET='...'
```

## Common Commands

```bash
# Training (from training/ directory)
cd training
python resume_h100.py

# Evaluation (from evaluation/ directory)
cd evaluation
python run_hle.py      # HLE benchmark (requires vllm1 + retriever envs)
python run_frames.py   # FRAMES benchmark (requires vllm1 + retriever envs)

# τ²-Bench evaluation
cd evaluation/tau2-bench
python run.py          # Standard run
python run_oss.py      # OSS models run

# Data preparation
python prepare_sft_data.py
```

## Architecture

### Core Components

- **LLM_CALL.py**: Unified LLM interface supporting OpenAI/Azure, vLLM, Claude, Together AI, and Nebius endpoints. The `get_llm_response()` function handles routing based on model name and `model_type` parameter.

- **evaluation/**: Benchmark evaluation scripts
  - `eval_hle.py`, `eval_frames.py`: Main evaluation logic with tool calling
  - `run_hle.py`, `run_frames.py`: Orchestration scripts that spawn model servers
  - `tau2-bench/`: τ²-Bench environment simulation (airline, retail, telecom, bank, etc.)
  - `tools.json`: Tool definitions (search, answer, enhance_reasoning with model tiers)
  - `model_config_*.json`: vLLM server endpoint configurations

- **training/**: RL training infrastructure
  - `resume_h100.py`: SLURM job orchestrator for multi-node training (monitors job queue, auto-restarts)
  - `recipe/dapo/`: DAPO training recipe
  - `rollout/tau2/`: Environment rollout for RL with tau2 domain integrations
  - `lead_agent/`: LLM agent with tool support for training

- **data_synthesis/**: Task synthesis pipeline
  - `run.ipynb`: Main synthesis notebook
  - `prompts/`: Generation prompts for schemas, tools, tasks, data models

### Model Orchestration

The orchestrator selects from tiered models based on task complexity (defined in `MODEL_MAPPING` in eval scripts):
- **Tier 1** (answer-1/search-1/reasoner-1): GPT-5 class
- **Tier 2** (answer-2/search-2/reasoner-2): GPT-5-mini class
- **Tier 3** (answer-3/search-3/reasoner-3): OSS models (Qwen, Llama-70B)
- **Math** (answer-math-1/2): Qwen2.5-Math models

### Model Configuration

vLLM endpoints are configured via JSON files (e.g., `model_config_oss_new.json`):
```json
{
    "model_name": [
        {"ip_addr": "hostname", "port": 1800}
    ],
    "retrieval": [{"ip_addr": "hostname", "port": 8765}]
}
```

## Customization Points

- **LLM backends**: Modify `get_llm_response()` in `LLM_CALL.py`
- **Prompts**: Lines 455-458 in `eval_hle.py`, lines 506-509 in `eval_frames.py`
- **Tool config**: `tool_config` in line 27 of `eval_frames.py`/`eval_hle.py`
- **Parallel experiments**: `EXPERIMENT_NAME1/2/3` env vars or in `training/resume_h100.py`

## SLURM + tmux Workflow

On the SLURM cluster, manage long-running jobs through tmux sessions:

```bash
# Create tmux session and allocate nodes
tmux new -s <session_name> -d
tmux send-keys -t <session_name> 'salloc -N 1 -t 48:00:00 --gres=gpu:4 --exclude=research-secure-02,research-secure-03,research-secure-07,research-secure-09' Enter

# Wait for allocation, then run commands
sleep 15
tmux capture-pane -t <session_name> -p | tail -10  # Check status

# Run long tasks with logging (always background + tee)
LOG_DIR="$HOME/logs/exp_$(date +%Y%m%d_%H%M%S)"
tmux send-keys -t <session_name> "mkdir -p $LOG_DIR" Enter
tmux send-keys -t <session_name> "srun python train.py 2>&1 | tee ${LOG_DIR}/train.log &" Enter
```

Key rules:
- Always use `tee` to save logs (tmux only keeps last 1000 lines)
- Background long-running tasks with `&`
- Use home directory for logs (accessible from head node)
- Request ≥48h allocation time

Common operations:
```bash
tmux list-sessions                              # List sessions
tmux capture-pane -t <session> -p | tail -50   # View output
squeue --me                                     # Check jobs
scancel <job_id>                                # Cancel job
tail -f ~/logs/train.log                        # Watch logs from head node
```

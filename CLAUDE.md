# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToolOrchestra is a framework for training small orchestrator models that coordinate intelligent tools and specialized LLMs. The Orchestrator-8B model alternates between reasoning and tool calling in multi-turn interactions, using reinforcement learning (outcome, efficiency, and preference rewards) for joint optimization.

## Environment Setup

```bash
# Training environment (Conda)
conda create -n toolorchestra python=3.12 -y
conda activate toolorchestra
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
pip install -e training/rollout

# Retrieval environment (separate Conda env)
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
cd tau2-bench && python run.py  # τ²-Bench (requires vllm1 env)

# Data preparation
python prepare_sft_data.py
```

## Architecture

### Core Components

- **LLM_CALL.py**: Unified LLM interface supporting OpenAI/Azure, vLLM, Claude, and NVIDIA NIM endpoints. Modify `get_llm_response()` for custom LLM backends.

- **evaluation/**: Benchmark evaluation scripts
  - `eval_hle.py`, `eval_frames.py`: Main evaluation logic with tool calling
  - `run_hle.py`, `run_frames.py`: Orchestration scripts that spawn model servers
  - `tau2-bench/`: τ²-Bench environment simulation (airline, retail, telecom domains)
  - `tools.json`: Tool definitions (search, answer, enhance_reasoning with model tiers)

- **training/**: RL training infrastructure
  - `resume_h100.py`: SLURM job orchestrator for multi-node training
  - `verl/`: VERL-based training framework (protocol, workers, trainer)
  - `rollout/`: Environment rollout for RL with tau2 integration
  - `recipe/dapo/`: DAPO training recipe

- **data_synthesis/**: Task synthesis pipeline
  - `run.ipynb`: Main synthesis notebook
  - `prompts/`: Generation prompts for schemas, tools, tasks, data models

### Model Orchestration

The orchestrator selects from tiered models based on task complexity:
- **answer-1/search-1/reasoner-1**: High capability (GPT-5 tier)
- **answer-2/search-2/reasoner-2**: Medium capability
- **answer-3/search-3/reasoner-3**: Efficient models (Qwen, Llama)
- **answer-math-1/2**: Math-specialized models

Model mappings are defined in `prepare_sft_data.py:MODEL_MAPPING`.

## Customization Points

- **LLM backends**: Modify `get_llm_response()` in `LLM_CALL.py`
- **Prompts**: Lines 455-458 in `eval_hle.py`, lines 506-509 in `eval_frames.py`
- **Tool config**: `tool_config` in line 27 of `eval_frames.py`/`eval_hle.py`
- **Parallel experiments**: `EXPERIMENT_NAME1/2/3` in `training/resume_h100.py`

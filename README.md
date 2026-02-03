# Overview

This folder contains reproducible end-to-end guides for running HLE (Humanity's Last Exam) evaluations with **ThunderAgent + vLLM** using the ToolOrchestra multi-agent workflow.

![ThunderAgent overview](../docs/thunder/figures/thunder.jpg)

## ToolOrchestra + ThunderAgent Use Case

### Prerequisites
- Python 3.12
- [Conda](https://docs.conda.io/en/latest/) package manager
- 2× GPUs (tested on RTX 5090) with CUDA drivers
- API keys: OpenAI (`OPENAI_API_KEY`), Together AI (`TOGETHER_API_KEY`), Tavily (`TAVILY_KEY`)

### Intro

Run the HLE benchmark through ToolOrchestra's multi-agent workflow, where:
- **Orchestrator-8B** (local, served via vLLM) coordinates tool calls
- **Expert Models** (external APIs) handle specific tasks:
  - `search` → OpenAI `gpt-5-mini`
  - `enhance_reasoning` → OpenAI `gpt-5`
  - `answer` → Together AI `openai/gpt-oss-120b`

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HLE Evaluation                              │
├─────────────────────────────────────────────────────────────────────┤
│  eval_hle_local.py                                                  │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────┐                                               │
│  │ ThunderAgent     │◄──── Manages KV cache, scheduling             │
│  │ Router           │      (--router tr or --router default)        │
│  └────────┬─────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────┐         ┌─────────────────────────────────┐   │
│  │ vLLM Server      │         │ Expert APIs (External)          │   │
│  │ Orchestrator-8B  │         │ • gpt-5-mini (search)           │   │
│  │ (GPU 0)          │         │ • gpt-5 (enhance_reasoning)     │   │
│  └──────────────────┘         │ • gpt-oss-120b (answer)         │   │
│                               └─────────────────────────────────┘   │
│  ┌──────────────────┐                                               │
│  │ Retriever        │◄──── FAISS GPU index for document search      │
│  │ (GPU 1)          │                                               │
│  └──────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Environment Setup

We use three conda environments:

```bash
# vLLM environment (for Orchestrator-8B serving)
conda create -n vllm1 python=3.12 -y
conda activate vllm1
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.10.1 transformers hf_transfer matplotlib

# vLLM-Continuum environment (for Continuum scheduler comparison)
# Follow instructions in vllm-continuum repo

# Retriever environment (for FAISS GPU index)
conda create -n retriever-clean python=3.12 -y
conda activate retriever-clean
conda install -y -c conda-forge --force-reinstall "numpy<2" "scipy<2" "scikit-learn<2" numpy-base
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
conda install -y -c pytorch -c nvidia faiss-gpu
pip install packaging ninja psutil
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install transformers datasets uvicorn fastapi tavily-python hf_transfer
```

### 2. Verify Environments

```bash
# Verify retriever environment
conda activate retriever-clean
python -c 'import faiss; n=faiss.get_num_gpus(); print(f"Faiss GPUs: {n}"); assert n > 0'

# Verify vLLM environments
conda activate vllm1
python -c "from vllm import LLM; print('vllm1 OK')"
```

### 3. Download Required Data

```bash
# Clone the retrieval index
git clone https://huggingface.co/datasets/multi-train/index
export INDEX_DIR='/path/to/index'

# Clone the Orchestrator model
git clone https://huggingface.co/nvidia/Nemotron-Orchestrator-8B
export CKPT_DIR='/path/to/Nemotron-Orchestrator-8B'
```

### 4. Configure Environment Variables

```bash
# Copy the template and fill in your values
cp setup_envs.sh_example setup_envs.sh

# Edit setup_envs.sh with your paths and API keys:
#   CKPT_DIR, INDEX_DIR, OPENAI_API_KEY, TOGETHER_API_KEY, TAVILY_KEY

# Source the environment
source setup_envs.sh
```

## How to Run Experiments

### Quick Start (Single Command)

```bash
# Source environment variables first
source setup_envs.sh

# Run baseline with concurrency=64
bash evaluation/launch_hle_inference.sh \
    --method baseline \
    --concurrency 64 \
    --ckpt "$CKPT_DIR" \
    --index-dir "$INDEX_DIR"

# Run ThunderAgent with concurrency=64
bash evaluation/launch_hle_inference.sh \
    --method thunderagent \
    --concurrency 64 \
    --ckpt "$CKPT_DIR" \
    --index-dir "$INDEX_DIR"

# Run Continuum with concurrency=64
bash evaluation/launch_hle_inference.sh \
    --method continuum \
    --concurrency 64 \
    --ckpt "$CKPT_DIR" \
    --index-dir "$INDEX_DIR"
```

### Manual Step-by-Step

#### 1. Start Retriever (GPU 1)
```bash
conda activate retriever-clean
CUDA_VISIBLE_DEVICES=1 python evaluation/retrieval_hle.py \
    --port 1401 \
    --new_cache_dir evaluation/cache/hle \
    --example_id_file evaluation/examples.json
```

#### 2. Start vLLM Backend (GPU 0)
```bash
# For baseline/thunderagent:
conda activate vllm1
CUDA_VISIBLE_DEVICES=0 vllm serve "$CKPT_DIR" \
    --port 8100 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# For continuum:
conda activate vllm-continuum
CUDA_VISIBLE_DEVICES=0 vllm serve "$CKPT_DIR" \
    --port 8100 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --scheduling-policy continuum
```

#### 3. Start ThunderAgent Router
```bash
conda activate vllm1
export PYTHONPATH="/workspace/ThunderAgent:${PYTHONPATH:-}"

# For baseline/continuum (pure proxy mode):
python -m ThunderAgent \
    --backends http://localhost:8100 \
    --port 8000 \
    --router default \
    --metrics

# For thunderagent (capacity scheduling):
python -m ThunderAgent \
    --backends http://localhost:8100 \
    --port 8000 \
    --router tr \
    --metrics
```

#### 4. Run Evaluation
```bash
conda activate vllm1
export ROUTER_URL="http://127.0.0.1:8000"

cd evaluation
python eval_hle_local.py \
    --model_name "$CKPT_DIR" \
    --output_dir outputs/hle_local \
    --model_config model_configs/hle_local_router.json \
    --max_rounds 50 \
    --model_type "Qwen/Qwen3-8B" \
    --example_path hle.jsonl \
    --concurrency 64
```

### Launch Script Options

```
Usage:
  bash evaluation/launch_hle_inference.sh --method <METHOD> --concurrency <C> \
    --ckpt <CKPT_DIR> --index-dir <INDEX_DIR> [options]

Required:
  --method         baseline | continuum | thunderagent
  --concurrency    HLE concurrency (eval batch size)
  --ckpt           Orchestrator-8B checkpoint path
  --index-dir      Directory containing eval.index + eval.jsonl

Common options:
  --orchestrator-gpu  GPU for vLLM (default: 0)
  --retrieval-gpu     GPU for retriever (default: 1)
  --backend-port      vLLM backend port (default: 8100)
  --router-port       ThunderAgent router port (default: 8000)
  --max-rounds        Max rounds per task (default: 50)
  --log-level         HLE log level (default: INFO)
```

## What We Changed for ThunderAgent Integration

To integrate ThunderAgent with your own agent workflow, you need two key changes:

### 1. Program ID Injection

Location: [`evaluation/eval_hle_local.py`](evaluation/eval_hle_local.py) (`run_hle_trial()`).

**What**: Assign a unique `program_id` per HLE trial and pass it via `extra_body.program_id` on every orchestrator call.

**Why**: ThunderAgent uses this field to separate requests into per-program state for KV cache management.

```python
# Create a unique program_id per HLE trial
program_id = f"hle-{trial_id}"

# Pass program_id in extra_body for every orchestrator request
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    extra_body={"program_id": program_id}
)
```

### 2. Program Release Hook

Location: [`evaluation/eval_hle_local.py`](evaluation/eval_hle_local.py) (`finally:` block).

**What**: Send `POST /programs/release` to ThunderAgent with the same `program_id` after the trial finishes.

**Why**: Frees router-side bookkeeping (tokens / pause-resume state) so finished programs do not linger.

```python
# After trial completes, release the program_id
import requests

router_url = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000")
try:
    requests.post(
        f"{router_url}/programs/release",
        json={"program_id": program_id},
        timeout=5
    )
except Exception:
    pass  # Best-effort cleanup
```

## Methods Comparison

| Method | vLLM Env | Router Mode | Scheduler |
|--------|----------|-------------|-----------|
| `baseline` | vllm1 | `--router default` | Standard vLLM FCFS |
| `continuum` | vllm-continuum | `--router default` | Continuum TTL pinning |
| `thunderagent` | vllm1 | `--router tr` | ThunderAgent capacity scheduling |

## Expert Model Configuration

| Tool | Expert Model | Provider | API Key |
|------|--------------|----------|---------|
| `search` | gpt-5-mini | OpenAI | `OPENAI_API_KEY` |
| `enhance_reasoning` | gpt-5 | OpenAI | `OPENAI_API_KEY` |
| `answer` | openai/gpt-oss-120b | Together AI | `TOGETHER_API_KEY` |

## Repository Layout

```text
ToolOrchestra/
├── README.md                    # This file
├── setup_envs.sh_example        # Environment template (copy to setup_envs.sh)
├── LLM_CALL.py                  # Unified LLM interface (OpenAI, Together, vLLM)
├── evaluation/
│   ├── launch_hle_inference.sh  # Main launch script
│   ├── eval_hle_local.py        # HLE evaluation with local Orchestrator
│   ├── eval_hle_oss.py          # HLE evaluation with OSS experts
│   ├── retrieval_hle.py         # FAISS GPU retrieval service
│   ├── hle.jsonl                # HLE benchmark dataset
│   ├── tools.json               # Tool definitions
│   ├── model_configs/           # Generated model config files
│   ├── outputs/                 # Evaluation outputs
│   └── logs/                    # Service logs
├── scripts/
│   ├── 5090/                    # RTX 5090 launch scripts
│   ├── analysis/                # Result analysis scripts
│   └── hle_serving/             # HLE serving helpers
├── assets/                      # Figures and documentation assets
└── training/                    # Training code (optional)
```

## Troubleshooting

### Common Issues

1. **vLLM server not starting**: Check GPU memory and CUDA drivers. Orchestrator-8B requires ~16GB VRAM.

2. **Retriever FAISS errors**: Ensure `eval.index` and `eval.jsonl` exist in `INDEX_DIR`.

3. **API key errors**: Verify `OPENAI_API_KEY` and `TOGETHER_API_KEY` are set correctly.

4. **Module not found**: Ensure `PYTHONPATH` includes `/workspace/ThunderAgent`.

### Logs

Check logs in `evaluation/logs/hle_local/`:
- `vllm_backend.log` - vLLM server logs
- `router.log` - ThunderAgent router logs
- `retrieval.log` - Retriever service logs

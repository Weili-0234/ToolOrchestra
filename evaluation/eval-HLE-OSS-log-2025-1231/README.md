# HLE OSS Evaluation Log - 2025-12-31

## Overview

Evaluation of HLE benchmark using OSS expert models:
- **Orchestrator**: Nemotron-Orchestrator-8B (DP=4)
- **enhance_reasoning**: Qwen/Qwen2.5-Coder-14B-Instruct (DP=8)
- **answer**: Qwen/Qwen3-32B-FP8 (DP=8, non-thinking mode)
- **search**: openai/gpt-oss-20b (DP=4)
- **LLM-as-Judge**: gpt-5 (external API)

---

## 1. vLLM Server Deployment

### 1.1 Orchestrator (research-secure-05, ports 1800-1803)

```bash
# tmux session: oss_new_node1
# After SLURM allocation
for i in 0 1 2 3; do
    PORT=$((1800 + i))
    export CUDA_VISIBLE_DEVICES=$i
    export VLLM_CACHE_ROOT="/tmp/vllm_cache_orch_$i"

    vllm serve /home/junxiong/haokang/ckpt/nvidia/Nemotron-Orchestrator-8B/ \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --port $PORT \
        --gpu-memory-utilization 0.85 \
        2>&1 | tee ~/logs/oss_eval/orch_port${PORT}.log &
    sleep 30
done
```

### 1.2 Qwen2.5-Coder-14B (research-secure-11, ports 1810-1817, DP=8)

```bash
# tmux session: oss_new_node2
for i in 0 1 2 3 4 5 6 7; do
    PORT=$((1810 + i))
    export CUDA_VISIBLE_DEVICES=$i

    vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
        --port $PORT \
        --gpu-memory-utilization 0.90 \
        2>&1 | tee ~/logs/oss_eval/coder14b_port${PORT}.log &
    sleep 20
done
```

### 1.3 Qwen3-32B-FP8 (research-secure-12, ports 1820-1827, DP=8)

```bash
# tmux session: oss_new_node3
for i in 0 1 2 3 4 5 6 7; do
    PORT=$((1820 + i))
    export CUDA_VISIBLE_DEVICES=$i

    vllm serve Qwen/Qwen3-32B-FP8 \
        --port $PORT \
        --gpu-memory-utilization 0.90 \
        2>&1 | tee ~/logs/oss_eval/qwen3_32b_port${PORT}.log &
    sleep 20
done
```

### 1.4 GPT-OSS-20b (172.27.31.10, ports 1840-1843)

```bash
# On separate node (172.27.31.10)
for i in 0 1 2 3; do
    PORT=$((1840 + i))
    export CUDA_VISIBLE_DEVICES=$i

    vllm serve openai/gpt-oss-20b \
        --enable-auto-tool-choice \
        --tool-call-parser openai \
        --port $PORT \
        --gpu-memory-utilization 0.90 \
        2>&1 | tee ~/logs/oss_eval/gpt20b_hle_${PORT}.log &
    sleep 20
done
```

**Note**: Must use `--tool-call-parser openai` (NOT `hermes`) for gpt-oss models.

### 1.5 Retrieval Services (research-secure-13, ports 8765 & 8766)

```bash
# tmux session: oss_new_retrieval
conda activate retriever
cd /home/junxiong/haokang/ToolOrchestra/evaluation

# HLE retrieval (port 8765)
export CUDA_VISIBLE_DEVICES=0
python retrieval_hle.py --port 8765 2>&1 | tee ~/logs/oss_eval/retrieval_hle.log &

# Wiki retrieval (port 8766)
export CUDA_VISIBLE_DEVICES=1
python retrieval_wiki.py --port 8766 2>&1 | tee ~/logs/oss_eval/retrieval_wiki.log &
```

### tmux Sessions Summary

| Session | Node | Service |
|---------|------|---------|
| oss_new_node1 | research-secure-05 | Orchestrator-8B |
| oss_new_node2 | research-secure-11 | Qwen2.5-Coder-14B |
| oss_new_node3 | research-secure-12 | Qwen3-32B-FP8 |
| oss_new_retrieval | research-secure-13 | retrieval_hle + retrieval_wiki |
| (sbatch job) | 172.27.31.10 | gpt-oss-20b |

---

## 2. Code Changes Required

### 2.1 LLM_CALL.py - Disable Nebius Fallback

**Problem**: When `NEBIUS_API_KEY` is set, Qwen3-32B calls were routed to Nebius API instead of local vLLM.

**Fix** (line ~498):
```python
# Before
use_nebius = nebius_api_key and 'qwen3-32b' in model.lower()

# After - only use Nebius when model_config is not provided
use_nebius = nebius_api_key and 'qwen3-32b' in model.lower() and not model_config
```

### 2.2 eval_hle_oss.py - Qwen3 Non-Thinking Mode

Add `extra_body` for Qwen3-32B to disable thinking:
```python
response = get_llm_response(
    model=model_name,
    messages=prompt2,
    model_type="vllm",
    model_config=arguments["vllm_model_configs"][model_name],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ...
)
```

### 2.3 eval_hle_oss.py - Context Length

Qwen3-32B-FP8 has 32K context limit (not 80K):
```python
# For answer tool
max_context_length = 24000  # Leave room for output
```

---

## 3. Evaluation Runs

### Run 1 - Initial Attempt (Failed)

```bash
source ~/haokang/ToolOrchestra/setup_envs.sh && \
cd ~/haokang/ToolOrchestra/evaluation && \
conda activate vllm1 && \
python eval_hle_oss.py \
    --model_name "/home/junxiong/haokang/ckpt/nvidia/Nemotron-Orchestrator-8B/" \
    --output_dir outputs_oss_hle_full_new \
    --model_config model_config_oss_new.json \
    --example_path hle.jsonl \
    --concurrency 64 \
    --max_rounds 50 \
    2>&1 | tee ~/logs/oss_eval/hle_full_new_run1.log
```

**Result**: Failed with `[Errno 24] Too many open files` at ~1758/1900 tasks.

**Issue**: Calling Nebius API (external) instead of local vLLM for Qwen3-32B.

### Run 2 - With ulimit fix (Partial Success → Stuck)

```bash
ulimit -n 65536 && \
source ~/haokang/ToolOrchestra/setup_envs.sh && \
cd ~/haokang/ToolOrchestra/evaluation && \
conda activate vllm1 && \
nohup python eval_hle_oss.py \
    --model_name "/home/junxiong/haokang/ckpt/nvidia/Nemotron-Orchestrator-8B/" \
    --output_dir outputs_oss_hle_full_new \
    --model_config model_config_oss_new.json \
    --example_path hle.jsonl \
    --concurrency 64 \
    --max_rounds 50 \
    > ~/logs/oss_eval/hle_full_new_run2.log 2>&1 &
```

**Result**: Completed 1780/1900, then process died silently.

**Issue**: Using `nohup ... > log 2>&1 &` causes stdout to be fully-buffered. When process dies, buffered output is lost. Log stopped at 01:37, process was gone by 08:19 (~7 hours no progress).

**Root Cause Analysis**:
- Last 200 log lines: 85 requests started, 0 completed → process died mid-flight
- No error messages in log (buffered output lost)
- Orchestrator-8B had 151 requests >300s (max 407s), causing worker starvation

### Run 3 - Remaining Tasks

```bash
# First create file with remaining 120 tasks
python3 << 'EOF'
import json
from pathlib import Path

all_examples = []
with open("hle.jsonl") as f:
    for line in f:
        all_examples.append(json.loads(line))

all_ids = {e.get("_id") or e.get("id") for e in all_examples}
completed_ids = {f.stem for f in Path("outputs_oss_hle_full_new").glob("*.json")}
remaining_ids = all_ids - completed_ids

remaining_examples = [e for e in all_examples if (e.get("_id") or e.get("id")) in remaining_ids]
with open("hle_missing_120.jsonl", "w") as f:
    for e in remaining_examples:
        f.write(json.dumps(e) + "\n")
EOF

# Run with remaining tasks (in tmux foreground, NOT nohup)
ulimit -n 65536 && \
source ~/haokang/ToolOrchestra/setup_envs.sh && \
cd ~/haokang/ToolOrchestra/evaluation && \
conda activate vllm1 && \
python eval_hle_oss.py \
    --model_name "/home/junxiong/haokang/ckpt/nvidia/Nemotron-Orchestrator-8B/" \
    --output_dir outputs_oss_hle_full_new \
    --model_config model_config_oss_new.json \
    --example_path hle_missing_120.jsonl \
    --concurrency 32 \
    --max_rounds 50 \
    2>&1 | tee ~/logs/oss_eval/hle_full_new_run3.log
```

**Result**: Completed remaining tasks.

---

## 4. Latency Analysis

### Tool Latency Statistics (1900 tasks)

Run analysis:
```bash
python analyze_tool_latency.py outputs_oss_hle_full_new
```

#### End-to-End Latency (seconds)

| Tool | Count | Mean | Std | Min | Max | P50 | P90 | P99 | <5s% |
|------|-------|------|-----|-----|-----|-----|-----|-----|------|
| search | ~400 | 8.0s | 6.4s | 0.7s | 37s | 7.1s | 16.2s | 25.3s | 42% |
| answer | ~500 | 34.5s | 20.7s | 3.8s | 104s | 34.4s | 60.6s | 78.5s | 1.2% |
| enhance_reasoning | ~270 | 11.5s | 13.4s | 2.3s | 90s | 8.5s | 14.6s | 70s | 8.9% |

#### Expert LLM Call Latency (seconds)

| Tool | Model | Count | Mean | P50 | P90 | <5s% |
|------|-------|-------|------|-----|-----|------|
| search | gpt-oss-20b | ~400 | 2.5s | 1.6s | 4.9s | 90% |
| answer | Qwen3-32B-FP8 | ~500 | 32.7s | 31.5s | 59.4s | 2.4% |
| enhance_reasoning | Qwen2.5-Coder-14B | ~270 | 8.0s | 7.7s | 12.2s | 15% |

### Orchestrator Latency Analysis

| Metric | Value |
|--------|-------|
| Total requests | 5318 |
| Mean latency | 80.7s |
| Max latency | 407s |
| Requests >300s | 151 |

**Finding**: Orchestrator is the bottleneck. Long-running tasks (50 rounds × 80s = 67 min) cause worker starvation.

---

## 5. Issues Encountered & Fixes

### Issue 1: gpt-oss-20b 500 Error
- **Symptom**: `Hermes2ProToolParser.extract_tool_calls() got unexpected keyword argument 'token_ids'`
- **Fix**: Use `--tool-call-parser openai` instead of `hermes`

### Issue 2: Qwen3-32B Context Length Error
- **Symptom**: `maximum context length is 32768 tokens. However, your request has 80034 input tokens`
- **Fix**: Set `max_context_length = 24000` in eval_hle_oss.py

### Issue 3: Nebius API Fallback
- **Symptom**: `Error calling Nebius API: [Errno 24] Too many open files`
- **Root Cause**: `NEBIUS_API_KEY` in env triggered external API calls instead of local vLLM
- **Fix**: Modified LLM_CALL.py to skip Nebius when `model_config` is provided

### Issue 4: Process Died Silently
- **Symptom**: 1780/1900 completed, then no progress for 7 hours, process gone
- **Root Cause**: `nohup ... > log 2>&1 &` uses full buffering; buffered output lost on crash
- **Fix**: Run in tmux foreground with `2>&1 | tee log.log` (line-buffered)

### Issue 5: Retrieval Cold Start
- **Symptom**: Early retrieval requests 8s mean, later 0.2s P50
- **Root Cause**: FAISS index and embedding model need warmup
- **Impact**: Not a bug, expected behavior

---

## 6. Configuration Files

### model_config_oss_new.json

```json
{
    "_comment": "Configuration for OSS evaluation",
    "/home/junxiong/haokang/ckpt/nvidia/Nemotron-Orchestrator-8B/": [
        {"ip_addr": "research-secure-05", "port": 1800},
        {"ip_addr": "research-secure-05", "port": 1801},
        {"ip_addr": "research-secure-05", "port": 1802},
        {"ip_addr": "research-secure-05", "port": 1803}
    ],
    "openai/gpt-oss-20b": [
        {"ip_addr": "172.27.31.10", "port": 1840},
        {"ip_addr": "172.27.31.10", "port": 1841},
        {"ip_addr": "172.27.31.10", "port": 1842},
        {"ip_addr": "172.27.31.10", "port": 1843}
    ],
    "Qwen/Qwen2.5-Coder-14B-Instruct": [
        {"ip_addr": "research-secure-11", "port": 1810},
        ... (ports 1810-1817)
    ],
    "Qwen/Qwen3-32B-FP8": [
        {"ip_addr": "research-secure-12", "port": 1820},
        ... (ports 1820-1827)
    ],
    "retrieval": [{"ip_addr": "research-secure-13", "port": 8765}],
    "wiki_retrieval": [{"ip_addr": "research-secure-13", "port": 8766}]
}
```

---

## 7. Lessons Learned

1. **Never use `nohup ... > log &` for long-running jobs** - Use tmux foreground with `| tee log.log`
2. **Always verify model routing** - Check logs for correct vLLM endpoints, not external APIs
3. **Set `ulimit -n 65536`** before running high-concurrency evaluations
4. **Orchestrator is the bottleneck** - Consider reducing `max_rounds` or adding timeouts
5. **Cold start is expected** - First ~100 tasks will have higher latency for retrieval
6. **gpt-oss models need `--tool-call-parser openai`** - Not `hermes`

---

## 8. Useful Commands

### Check vLLM endpoints
```bash
for p in 1800 1801 1802 1803; do
    echo -n "Orch-8B :$p - "
    timeout 3 curl -s "http://research-secure-05:$p/v1/models" | grep -o '"id"' && echo "OK" || echo "FAIL"
done
```

### Analyze latency
```bash
python analyze_tool_latency.py outputs_oss_hle_full_new
python analyze_tool_latency.py outputs_oss_hle_full_new --plots --bins 20
```

### Count progress
```bash
ls outputs_oss_hle_full_new/*.json | wc -l
```

### Create missing tasks file
```bash
python3 -c "
import json
from pathlib import Path
all_ids = {json.loads(l).get('_id') or json.loads(l).get('id') for l in open('hle.jsonl')}
done_ids = {f.stem for f in Path('outputs_oss_hle_full_new').glob('*.json')}
missing = all_ids - done_ids
print(f'Missing: {len(missing)}')
"
```

---

## 9. FRAMES Evaluation (To be completed)

After HLE completes, run FRAMES:

```bash
ulimit -n 65536 && \
source ~/haokang/ToolOrchestra/setup_envs.sh && \
cd ~/haokang/ToolOrchestra/evaluation && \
conda activate vllm1 && \
python eval_frames_oss.py \
    --model_name "/home/junxiong/haokang/ckpt/nvidia/Nemotron-Orchestrator-8B/" \
    --output_dir outputs_oss_frames_full_new \
    --model_config model_config_oss_new.json \
    --example_file_path frames.jsonl \
    --concurrency 64 \
    --max_rounds 50 \
    2>&1 | tee ~/logs/oss_eval/frames_full_new_run1.log
```

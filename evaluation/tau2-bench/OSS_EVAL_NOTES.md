# Tau2-Bench OSS Evaluation Notes

**Date**: 2025-12-30
**Experiment**: Full tau2-bench evaluation with OSS expert models

## 1. Model Configuration

| Model | Role | Node | GPUs | Ports | Config |
|-------|------|------|------|-------|--------|
| Nemotron-Orchestrator-8B | Agent | Node1 | 0-3 | 1900-1903 | TP=1, DP=4 |
| Qwen/Qwen3-32B-FP8 | Expert-2 | Node1 | 4-7 | 1904-1907 | TP=1, DP=4, non-thinking |
| openai/gpt-oss-20b | Expert-1 | Node2 | 0-3 | 1910-1913 | TP=1, DP=4, no speculator |
| Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 | Expert-3 | Node3 | 0-7 | 1920-1923 | TP=2, DP=4, MTP 4 tokens |

## 2. Changes Made

### 2.1 run_oss.py

1. **SLURM time extended**: Changed from `04:00:00` to `24:00:00` for long-running evaluations

2. **GPU cleanup check added** to SLURM_HEADER:
   ```bash
   # Check for existing GPU processes before starting vLLM
   GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | wc -l)
   if [ "$GPU_PROCESSES" -gt 0 ]; then
       # Kill stale processes owned by current user
       ...
   fi
   ```

3. **Qwen3-32B-FP8 config fixed**: Removed `--chat-template-kwargs` (not supported in vLLM 0.12.0)

4. **Qwen3-Next-80B memory reduced**: `gpu_memory_utilization` from 0.95 to 0.88 to avoid OOM with MTP

5. **Orchestrator-8B chat template added**:
   ```python
   ORCHESTRATOR_CHAT_TEMPLATE = os.path.join(
       os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
       "tool_chat_template_llama3.1_json.jinja"
   )
   ```

### 2.2 llm_utils.py

1. **Qwen3 thinking mode disabled for experts**:
   ```python
   # For expert Qwen3 models, disable thinking mode
   expert_enable_thinking = not ('qwen3' in mode_to_call.lower())
   ```

2. **System prompt updated** for non-thinking Qwen3:
   - Removed: `Wrap thinking process between <think> </think>`
   - Kept: `message between <message> </message> and the tool call between <tool_call> </tool_call>`
   - Added: `/no_think` suffix to user messages

3. **Streaming profiling added**:
   - `prefill_ms`: Time to first token
   - `decode_ms`: Decode phase time
   - `input_tokens`, `output_tokens`: Token counts

### 2.3 analyze_profile.py

1. **Enhanced percentiles**: Added p10, p25, p40
2. **SLA metric**: `<5s%` (percentage of calls with latency < 5000ms)
3. **PNG chart generation**: Matplotlib histograms for each model
4. **Markdown report**: `experiment_report_<timestamp>.md`

### 2.4 oss_eval_node1.sh

1. **Chat template added** for Orchestrator-8B:
   ```bash
   CHAT_TEMPLATE="/home/junxiong/haokang/ToolOrchestra/evaluation/tool_chat_template_llama3.1_json.jinja"
   vllm serve ... --chat-template $CHAT_TEMPLATE
   ```

## 3. Problems Encountered & Solutions

### 3.1 `--chat-template-kwargs` not supported

**Error**:
```
vllm: error: unrecognized arguments: --chat-template-kwargs {"enable_thinking": false}
```

**Cause**: vLLM 0.12.0 doesn't support `--chat-template-kwargs`

**Solution**: Use `/no_think` suffix in system prompt instead of chat template kwargs

### 3.2 CUDA OOM for Qwen3-Next-80B

**Error**:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.74 GiB.
GPU 0 has a total capacity of 79.18 GiB of which 612.50 MiB is free.
```

**Cause**: MTP speculative decoding uses extra memory

**Solution**: Reduce `gpu_memory_utilization` from 0.95 to 0.88

### 3.3 `unexpected tokens remaining in message header`

**Error**:
```
openai.APIError: unexpected tokens remaining in message header:
Some("already attempted return; order status was delivered but not yet updated?...")
```

**Cause**: Orchestrator-8B outputs internal reasoning via a "commentary" channel:
```
<|end|><|start|>assistant<|channel|>commentary
```
The model's tokenizer_config.json uses ChatML format (`<|im_start|>`, `<|im_end|>`), but the model sometimes outputs different tokens.

**Solution**: Use `--chat-template tool_chat_template_llama3.1_json.jinja` which uses Llama 3.1 format

**Error Rate**: ~1.8-3.4% (with retry mechanism, evaluation continues)

### 3.4 SLURM allocates "busy" nodes

**Problem**: `sbatch` may allocate nodes that appear idle in SLURM but have running GPU processes from previous jobs

**Solution**: Added nvidia-smi GPU cleanup check at job start:
```bash
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$GPU_PROCESSES" -gt 0 ]; then
    # Kill stale processes owned by current user
    kill -9 $pid
fi
```

### 3.5 $REPO_PATH not expanded in command line

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/data/tau2/domains/mock/tasks.json'
```

**Cause**: Environment variable `$REPO_PATH` not expanded when passed as argument

**Solution**: Use full absolute path instead of environment variable

## 4. Best Practices

### 4.1 vLLM Deployment

1. **Always specify `--chat-template`** for Orchestrator-8B:
   ```bash
   vllm serve $CKPT_DIR \
       --enable-auto-tool-choice \
       --tool-call-parser hermes \
       --chat-template /path/to/tool_chat_template_llama3.1_json.jinja
   ```

2. **Use conservative `gpu_memory_utilization`** for models with speculative decoding:
   - Standard models: 0.95
   - Models with MTP/speculator: 0.88

3. **Add GPU cleanup check** at SLURM job start to handle stale processes

4. **Use separate VLLM_CACHE_ROOT** for each vLLM instance to avoid cache conflicts

### 4.2 Qwen3 Expert Models

1. **Disable thinking mode** for faster responses:
   - Add `/no_think` to user messages
   - Remove `<think>` from system prompt
   - Keep `<message>` and `<tool_call>` format instructions

2. **Use non-thinking system prompt**:
   ```
   You are dedicated to provide the best service.
   Wrap message between <message> </message> and the tool call between <tool_call> </tool_call>.
   ```

### 4.3 SLURM Scripts

1. **Use 24-hour time limit** for full evaluations
2. **Print hostname -i** at job start for service discovery
3. **Add sleep between vLLM launches**: 30s for small models, 60s for large models
4. **Use `sleep 15000`** at end to keep job running

### 4.4 Evaluation

1. **Run smoke test first** with mock domain and DEBUG level
2. **Use PROFILE log level** for full evaluation to capture latency metrics
3. **Use `| tee` for long-running commands** to see output in real-time
4. **Monitor with `tail -f`** and check `[TAU2_TASK_COMPLETE]` for progress

## 5. Current Experiment Status

**Started**: 2025-12-30 17:32:15
**Domains**: retail, telecom, airline (278 total tasks)

### SLURM Jobs

| Job ID | Node | IP | Status |
|--------|------|----|---------|
| 13463 | research-secure-20 | 172.27.25.244 | Running |
| 13464 | research-secure-03 | 172.27.20.172 | Running |
| 13466 | research-secure-04 | 172.27.25.31 | Running |

### Endpoints Verified

- Orchestrator: 1900-1903 on 172.27.25.244
- Qwen3-32B-FP8: 1904-1907 on 172.27.25.244
- gpt-oss-20b: 1910-1913 on 172.27.20.172
- Qwen3-Next-80B-FP8: 1920-1923 on 172.27.25.31

## 6. Files Modified

| File | Changes |
|------|---------|
| `run_oss.py` | SLURM time, GPU cleanup, chat template, Qwen3-32B fix, Qwen3-Next-80B memory |
| `tau2/utils/llm_utils.py` | Qwen3 non-thinking mode, streaming profiling |
| `analyze_profile.py` | Enhanced metrics, PNG charts, markdown report |
| `oss_eval_node1.sh` | Chat template for Orchestrator-8B |

## 7. Output Files

| File | Purpose |
|------|---------|
| `model_config_oss.json` | vLLM endpoint configuration |
| `logs_oss_full/fullrun.log` | Main evaluation log |
| `logs_oss_full/tau2_*.log` | Per-domain profiling logs |
| `outputs_oss_full/*.json` | Evaluation results |
| `profile_analysis.json` | Aggregated latency stats |
| `profile_charts/*.png` | Latency distribution charts |
| `experiment_report_*.md` | Detailed markdown report |

## 8. Monitoring Commands

```bash
# Check progress
grep "OVERALL_ETA" logs_oss_full/fullrun.log | tail -1

# Count completed tasks
grep -c "TAU2_TASK_COMPLETE" logs_oss_full/fullrun.log

# Check error rate
grep -c "unexpected tokens remaining" logs_oss_full/fullrun.log
grep -c "Retrying in 5 seconds" logs_oss_full/fullrun.log

# Real-time monitoring
tail -f logs_oss_full/fullrun.log

# Analyze when complete
python analyze_profile.py logs_oss_full/tau2_*.log
```

# SLURM Cluster Lessons Learned

## /tmp Directory Issues

### Problem
The `/tmp` directory on compute nodes is **shared** among all users. When scripts try to clear torch cache directories with:
```bash
rm -rf /tmp/torchinductor_*
```

This will attempt to delete other users' cache directories, resulting in thousands of permission denied errors:
```
rm: cannot remove '/tmp/torchinductor_syanamandra/...': Permission denied
rm: cannot remove '/tmp/torchinductor_waitong/...': Permission denied
```

### Solution
Instead of using `rm -rf /tmp/torchinductor_*`, use user-specific cache directories:
```bash
# Clear only your own cache
rm -rf ~/.cache/torch_extensions
rm -rf ~/.cache/torch/inductor
rm -rf ~/cache/vllm/*

# Use isolated per-process cache directories
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_${USER}_gpu${i}"
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${USER}_gpu${i}"
mkdir -p $TORCHINDUCTOR_CACHE_DIR $VLLM_CACHE_ROOT
```

---

## Idle Nodes with Occupied GPUs

### Problem
Some SLURM nodes appear "idle" in `sinfo` but actually have processes occupying GPU memory from previous jobs that didn't clean up properly.

### Known Bad Nodes (as of 2025-12-28)
- research-secure-02
- research-secure-03
- research-secure-07
- research-secure-09

### Solution
1. **Always verify GPU status via tmux before launching vLLM:**
   ```bash
   # After allocating a node
   tmux send-keys -t session_name "srun --jobid=<JOBID> nvidia-smi --query-gpu=index,memory.used --format=csv" Enter
   ```

2. **Use exclude list when allocating nodes:**
   ```bash
   salloc -N 1 -G 8 --exclude=research-secure-02,research-secure-03,research-secure-07,research-secure-09 ...
   ```

---

## Torch Cache Corruption

### Problem
When multiple vLLM processes share the same torch compile cache directory, corruption can occur:
```
RuntimeError: Bytes object is corrupted, checksum does not match
```

### Solution
Use isolated cache directories for each vLLM process:
```bash
for i in 0 1 2 3; do
    export CUDA_VISIBLE_DEVICES=$i
    export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_gpu$i"
    export VLLM_CACHE_ROOT="/tmp/vllm_cache_gpu$i"
    mkdir -p $TORCHINDUCTOR_CACHE_DIR $VLLM_CACHE_ROOT
    vllm serve ... &
done
```

---

## tmux Best Practices

### Sending Commands to Compute Nodes
Always use tmux to send commands, not `srun` with timeout:
```bash
# GOOD: Use tmux send-keys
tmux send-keys -t session_name "srun --jobid=<JOBID> <command>" Enter

# BAD: srun with timeout can hang
timeout 30 srun --jobid=<JOBID> <command>
```

### Check Output
```bash
tmux capture-pane -t session_name -p | tail -50
```

---

## Preserving Torch Compile Cache

### Problem
First-time torch.compile for large models can take 5-10+ minutes. Without cache preservation, every restart requires full recompilation.

### Solution
Use persistent cache directories that survive across runs:
```bash
# Use a persistent location (not /tmp which may be cleared)
export TORCHINDUCTOR_CACHE_DIR="/home/$USER/cache/inductor_cache_${MODEL_NAME}"
export VLLM_CACHE_ROOT="/home/$USER/cache/vllm_cache_${MODEL_NAME}"
mkdir -p $TORCHINDUCTOR_CACHE_DIR $VLLM_CACHE_ROOT

# For per-GPU isolation with persistence:
export TORCHINDUCTOR_CACHE_DIR="/home/$USER/cache/inductor_cache_gpu${i}"
export VLLM_CACHE_ROOT="/home/$USER/cache/vllm_cache_gpu${i}"
```

### Note
- First run: ~5-10 minutes for torch.compile
- Subsequent runs: seconds (using cached compiled graphs)
- Cache location from logs: `Using cache directory: /path/to/cache/torch_compile_cache/...`

---

## Speculative Decoding Drafter Models

### Model-Drafter Pairings
| Target Model | Drafter Model | Method | Notes |
|-------------|---------------|--------|-------|
| `openai/gpt-oss-120b` | `nvidia/gpt-oss-120b-Eagle3-v2` | eagle3 | TP8 recommended, 0.85 mem util |
| `openai/gpt-oss-20b` | `RedHatAI/gpt-oss-20b-speculator.eagle3` | eagle3 | TP8 for safety with drafter |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | (built-in MTP) | qwen3_next_mtp | Native MTP support |

### Configuration Examples
```bash
# GPT-OSS-120B with EAGLE3
--speculative-config '{"model": "nvidia/gpt-oss-120b-Eagle3-v2", "num_speculative_tokens": 3, "method": "eagle3", "draft_tensor_parallel_size": 1}'

# GPT-OSS-20B with EAGLE3
--speculative-config '{"model": "RedHatAI/gpt-oss-20b-speculator.eagle3", "num_speculative_tokens": 3, "method": "eagle3"}'

# Qwen3-Next-80B with native MTP
--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
```

### Memory Considerations
- Speculative decoding requires additional memory for drafter model + CUDA graphs
- Recommended: `--gpu-memory-utilization 0.85` (not 0.95) when using speculative decoding
- Use higher TP (e.g., TP=8) to spread memory across more GPUs

---

## vLLM Tool Parser 版本兼容性

### 各模型的正确 Tool Parser

| 模型 | 最低 vLLM 版本 | Tool Parser | 来源 |
|------|---------------|-------------|------|
| **openai/gpt-oss-120b** | >= 0.10.2 | `openai` | [vLLM GPT-OSS Recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) |
| **openai/gpt-oss-20b** | >= 0.10.2 | `openai` | [OpenAI Cookbook](https://cookbook.openai.com/articles/gpt-oss/run-vllm) |
| **Qwen3-Next-80B-A3B** | >= 0.10.2 | `hermes` | [vLLM Qwen3-Next Blog](https://blog.vllm.ai/2025/09/11/qwen3-next.html) |
| **Nemotron-Orchestrator-8B** | 0.9.2+ | `hermes` | [ToolOrchestra](https://github.com/NVlabs/ToolOrchestra) |

### vLLM 版本兼容性

| 版本 | gpt-oss | Qwen3-Next | Orchestrator | 备注 |
|------|---------|------------|--------------|------|
| 0.9.2 | ❌ | ❌ | ✅ | 缺少 `openai` parser，不支持 Qwen3-Next |
| 0.10.2 | ✅ | ✅ | ✅ | 首个完整支持版本 |
| 0.11.x | ✅ | ✅ | ✅ | **推荐**，async scheduling 更稳定 |
| 0.12.0 | ⚠️ | ⚠️ | ⚠️ | 有 token_ids bug |

### vLLM 0.12.0 Tool Parser Bug

**ALL tool parsers** in vLLM 0.12.0 fail with 500 Internal Server Error:
```
{"error":{"message":"XXXToolParser.extract_tool_calls() got an unexpected keyword argument 'token_ids'"}}
```

**解决方案**: 降级到 vLLM 0.10.2 或 0.11.x

### 正确配置示例

```bash
# GPT-OSS 模型必须用 openai parser!
vllm serve openai/gpt-oss-120b --tool-call-parser openai --enable-auto-tool-choice

# Qwen3-Next 用 hermes parser
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --tool-call-parser hermes --enable-auto-tool-choice

# Orchestrator 用 hermes parser
vllm serve nvidia/Nemotron-Orchestrator-8B --tool-call-parser hermes --enable-auto-tool-choice
```

### Available Tool Parsers (vLLM 0.10.2+)
```
deepseek_v3, deepseek_v31, ernie45, glm45, granite, granite-20b-fc, hermes,
hunyuan_a13b, internlm, jamba, kimi_k2, llama3_json, llama4_json, llama4_pythonic,
longcat, minimax, minimax_m2, mistral, olmo3, openai, phi4_mini_json, pythonic,
qwen3_coder, qwen3_xml, seed_oss, step3, xlam
```

---

## Log Redirection for vLLM Servers

### Problem
- `tmux capture-pane` only keeps ~1000 lines of history
- Server logs on compute nodes are hard to access for debugging
- Need persistent logs for profiling and debugging

### Solution
Redirect vLLM output to shared storage accessible from head node:
```bash
LOG_DIR=/home/$USER/path/to/logs
mkdir -p $LOG_DIR

vllm serve model_name ... 2>&1 | tee ${LOG_DIR}/server_port${port}.log &
```

### Cache Directory for Faster Startup
Use shared storage for torch compile cache (survives across restarts):
```bash
CACHE_DIR=/home/$USER/cache/vllm_compile
mkdir -p $CACHE_DIR

export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/inductor_${model}_${i}"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm_${model}_${i}"
```

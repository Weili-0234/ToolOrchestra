# Baseline vLLM 实验交接文档

## 1. 实验目标 (Goal)

补充测试 **标准 vLLM** (无 ThunderReact router，无 Continuum scheduling policy) 在 tau2-bench 上的性能，与之前完成的 ThunderReact/Continuum 实验对比。

### 实验矩阵

| Setting | Concurrency | 状态 |
|---------|-------------|------|
| Baseline C=32 | 32 | ✅ 完成 (`/workspace/ToolOrchestra/outputs/baseline_c32_20260105_062226`) |
| Baseline C=64 | 64 | ✅ 完成 (`/workspace/ToolOrchestra/outputs/baseline_c64_20260105_081018`) |
| Baseline C=128 | 128 | ✅ 完成 (`/workspace/ToolOrchestra/outputs/baseline_c128_20260105_101158`) |

### 已完成的实验 (供对比)

对比实验详见 `docs/experiments/2026-01-04_5090_tau2_global_thunderreact_vs_continuum.md`，主要 run 目录如下：
- ThunderReact:
  - `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c32`
  - `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c64`
  - `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c128` *(手动终止；仍可用于 active-window throughput)*
- Continuum:
  - `/workspace/ToolOrchestra/outputs/continuum_c32_20260104_191004`
  - `/workspace/ToolOrchestra/outputs/continuum_c64_20260104_205216`
  - `/workspace/ToolOrchestra/outputs/continuum_c128_20260104_223636` *(手动终止；仍可用于 active-window throughput)*

---

## 2. 环境准备

### 2.1 SSH 隧道 (必须先启动)

在 **tmux session "0"** 中运行隧道脚本，保持 H100 expert LLMs 的连接：

```bash
# 在 RunPod 5090 机器上
tmux attach -t 0
# 确认 autossh 正在运行，转发以下端口:
# 1910-1913: gpt-oss-20b (DP=4)
# 1904-1905: Qwen3-32B-FP8 (DP=2)
# 1920-1921: Qwen3-Next-80B-A3B-Instruct-FP8 (DP=2)
```

检查隧道是否正常：
```bash
ss -lntp | egrep ':(1910|1911|1912|1913|1904|1905|1920|1921) ' | wc -l
# 应该输出 8 (每个端口两条记录则为16)
```

### 2.2 环境变量

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm1
cd /workspace/ToolOrchestra
source setup_envs.sh

# 确认 CKPT_DIR 已设置
echo $CKPT_DIR
# 应输出: /workspace/ckpt/nvidia/Nemotron-Orchestrator-8B
```

---

## 3. 启动实验的正确方式

### ⚠️ 重要：Cursor 的限制

Cursor 中后台运行的命令如果超过30分钟会 timeout，并且会 kill 掉该 terminal 中启动的所有进程。

**正确做法**：使用 **tmux** 来运行实验，而不是 Cursor 的后台 terminal。

### 3.1 启动单个实验 (以 C=32 为例)

```bash
# 1. 创建/进入 tmux session
tmux new -s baseline_exp

# 2. 设置环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm1
cd /workspace/ToolOrchestra
source setup_envs.sh

# 3. 创建输出目录
TS=$(date +%Y%m%d_%H%M%S)
CONCURRENCY=32  # 改成 64 或 128 跑其他实验
OUTPUT_DIR="/workspace/ToolOrchestra/outputs/baseline_c${CONCURRENCY}_${TS}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
echo "OUTPUT_DIR: $OUTPUT_DIR"

# 4. 启动 vLLM (后台)
CUDA_VISIBLE_DEVICES=0 vllm serve "$CKPT_DIR" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 1900 \
  --gpu-memory-utilization 0.95 \
  > "${LOG_DIR}/vllm_baseline_${TS}.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# 5. 等待 vLLM 就绪 (约 1-2 分钟)
echo "等待 vLLM 就绪..."
while ! curl -sf http://127.0.0.1:1900/health >/dev/null 2>&1; do
  sleep 5
  echo -n "."
done
echo " Ready!"

# 6. 启动 KV cache sampler (后台)
bash scripts/rollout/collect_kv_cache_timeseries.sh 127.0.0.1 \
  "${OUTPUT_DIR}/kv_cache_timeseries.csv" 5 1900 \
  > "${LOG_DIR}/kv_sampler_${TS}.log" 2>&1 &
KV_PID=$!
echo "KV sampler PID: $KV_PID"

# 7. 创建 model config
MODEL_CONFIG_PATH="${OUTPUT_DIR}/model_config_5090_baseline.json"
cat > "${MODEL_CONFIG_PATH}" <<EOF
{
  "${CKPT_DIR}": [{"ip_addr": "127.0.0.1", "port": "1900"}],
  "openai/gpt-oss-20b": [
    {"ip_addr": "127.0.0.1", "port": "1910"},
    {"ip_addr": "127.0.0.1", "port": "1911"},
    {"ip_addr": "127.0.0.1", "port": "1912"},
    {"ip_addr": "127.0.0.1", "port": "1913"}
  ],
  "Qwen/Qwen3-32B-FP8": [
    {"ip_addr": "127.0.0.1", "port": "1904"},
    {"ip_addr": "127.0.0.1", "port": "1905"}
  ],
  "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": [
    {"ip_addr": "127.0.0.1", "port": "1920"},
    {"ip_addr": "127.0.0.1", "port": "1921"}
  ],
  "oss_expert_mapping": {
    "expert-1": "openai/gpt-oss-20b",
    "expert-2": "Qwen/Qwen3-32B-FP8",
    "expert-3": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
  },
  "vllm_model_config_path": "${MODEL_CONFIG_PATH}"
}
EOF

# 8. 运行评估 (前台，用 tee 保存日志)
cd evaluation/tau2-bench
python run_oss.py \
  --agent-model "$CKPT_DIR" \
  --skip-server-start \
  --schedule-mode global \
  --model-config-path "${MODEL_CONFIG_PATH}" \
  --domains retail telecom airline \
  --max-concurrency ${CONCURRENCY} \
  --log-level PROFILE \
  --output-dir "${OUTPUT_DIR}/outputs" \
  --log-dir "${LOG_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/driver.log"

# 9. 实验完成后，清理
kill $VLLM_PID $KV_PID 2>/dev/null
echo "实验完成！输出在: $OUTPUT_DIR"
```

### 3.2 按顺序跑完3个实验

```bash
# 在 tmux 中依次运行:
# 1. 先跑 C=32 (CONCURRENCY=32)
# 2. 等完成后，改 CONCURRENCY=64，重新执行上述步骤
# 3. 最后改 CONCURRENCY=128

# 或者写一个循环:
for CONC in 32 64 128; do
  echo "========== Starting Baseline C=${CONC} =========="
  # ... (上面的步骤 3-9)
done
```

---

## 4. 监控脚本

### 4.1 快速检查 (任意时刻运行)

```bash
cd /workspace/ToolOrchestra

# vLLM /health + /v1/models + /metrics(kv_cache_usage)
# SSH tunnel ports
# GPU util/mem
# tau2-bench: 自动从 run_oss.py 进程解析 run_dir + driver.log 进度
bash scripts/5090/quick_check_tau2.sh

# 可选：额外检查 expert /health (1910/1904/1920)
CHECK_EXPERT_HEALTH=1 bash scripts/5090/quick_check_tau2.sh
```

### 4.2 持续监控 (在 tmux 中运行)

```bash
cd /workspace/ToolOrchestra

# 每 5 分钟输出一次 quick_check（可通过 INTERVAL_SEC 调整）
INTERVAL_SEC=300 bash scripts/5090/monitor_loop_tau2.sh
```

---

## 5. 分析 Workflow

### 5.1 Active Window 定义

- **开始时间**: 第一次 KV cache usage > 70% 的时刻
- **结束时间**: 最后一次 KV cache usage > 70% 的时刻

### 5.2 计算 Throughput 指标

推荐直接用已保存脚本（使用 `kv_cache_timeseries.csv` + `driver.log`）：

```bash
# 生成 TR / CT / Baseline 三方对比表（tasks/min + steps/s）
python3 scripts/analysis/active_window/compare_tr_ct_bl_active_window.py \
  --threshold 70 \
  --tr32 outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c32 \
  --tr64 outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c64 \
  --tr128 outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c128 \
  --ct32 outputs/continuum_c32_20260104_191004 \
  --ct64 outputs/continuum_c64_20260104_205216 \
  --ct128 outputs/continuum_c128_20260104_223636 \
  --bl32 outputs/baseline_c32_20260105_062226 \
  --bl64 outputs/baseline_c64_20260105_081018 \
  --bl128 outputs/baseline_c128_20260105_101158

# 单个 run 的 window 外 breakdown（查看长尾：win_end -> last_task）
python3 scripts/analysis/active_window/window_outside_breakdown.py \
  --threshold 70 --name BL128 --run-dir outputs/baseline_c128_20260105_101158

# C=128 bottleneck 快速对比（active window 内的长尾 / step-rate）
python3 scripts/analysis/active_window/compare_bottleneck_active_window.py \
  --threshold 70 \
  --run TR128=outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c128 \
  --run CT128=outputs/continuum_c128_20260104_223636 \
  --run BL128=outputs/baseline_c128_20260105_101158
```

### 5.4 汇总对比表

完成所有 Baseline 实验后，更新这个表格：

Active window: **first KV>70% → last KV>70%**.

| Scenario | ThunderReact | Continuum | Baseline | Winner |
|----------|--------------|-----------|----------|--------|
| C=32 | 3.77 tasks/min | 3.55 tasks/min | 3.36 tasks/min | ThunderReact |
| C=64 | 4.13 tasks/min | 2.50 tasks/min | 3.52 tasks/min | ThunderReact |
| C=128 | 3.24 tasks/min | 2.41 tasks/min | 3.51 tasks/min | Baseline |

Steps/s（同样基于 active window）：

| Scenario | ThunderReact | Continuum | Baseline | Winner |
|----------|--------------|-----------|----------|--------|
| C=32 | 2.97 steps/s | 2.83 steps/s | 2.95 steps/s | ThunderReact |
| C=64 | 3.37 steps/s | 1.89 steps/s | 2.82 steps/s | ThunderReact |
| C=128 | 2.52 steps/s | 1.97 steps/s | 2.83 steps/s | Baseline |

---

## 6. 注意事项

1. **隧道断开**: 如果 SSH 隧道断开，expert 调用会失败。检查 tmux session "0" 中的 autossh 状态。

2. **Context length exceeded**: 某些复杂任务可能导致上下文超长错误，会无限重试。如果发现某个任务卡住（进度不动），可能需要手动终止。

3. **实验输出**: 每次实验的输出包括：
   - `driver.log` - 主日志
   - `kv_cache_timeseries.csv` - KV cache 时序
   - `outputs/metrics_global.json` - 汇总指标
   - `outputs/results_*.json` - 每个任务的结果

4. **GPU 内存**: 确保 GPU 空闲再启动新实验。用 `nvidia-smi` 检查。

---

## 7. 联系方式

如有问题，联系 haokang@...

---

*文档创建时间: 2026-01-05 04:xx UTC*

*更新: 2026-01-05 11:xx UTC（Baseline 全部跑完 + active-window 对比表已补齐）*

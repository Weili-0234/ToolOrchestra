# 5090 tau2-bench (global) — ThunderReact vs Continuum vs Baseline(vLLM) Worklog + Repro

> 目标：让后续读者 **只看这一份文档** 就能在 RunPod RTX 5090 机器上完整复现 3 组实验（ThunderReact / Continuum / Baseline vLLM）并用同一套 profiling 方法得到相同结论。
>
> 参考来源：
> - 原始计划：`/workspace/magical-swimming-naur.md`
> - TR/CT 复现实验记录：`docs/experiments/2026-01-04_5090_tau2_global_thunderreact_vs_continuum.md`
> - Baseline 交接文档：`docs/experiments/BASELINE_EXPERIMENT_HANDOFF.md`

---

## TL;DR（结论）

吞吐统计使用 **active window**（`kv_cache_usage_perc` 第一次 >70% 到最后一次 >70%）：

| Concurrency | ThunderReact | Continuum | Baseline(vLLM) | Winner |
|---:|---:|---:|---:|:--|
| 32 | 3.77 tasks/min | 3.55 tasks/min | 3.36 tasks/min | ThunderReact |
| 64 | 4.13 tasks/min | 2.50 tasks/min | 3.52 tasks/min | ThunderReact |
| 128 | 3.24 tasks/min | 2.41 tasks/min | 3.51 tasks/min | Baseline |

| Concurrency | ThunderReact | Continuum | Baseline(vLLM) | Winner |
|---:|---:|---:|---:|:--|
| 32 | 2.97 steps/s | 2.83 steps/s | 2.95 steps/s | ThunderReact |
| 64 | 3.37 steps/s | 1.89 steps/s | 2.82 steps/s | ThunderReact |
| 128 | 2.52 steps/s | 1.97 steps/s | 2.83 steps/s | Baseline |

> 说明：`metrics_global.json` 里的 `tasks_per_min` 是 **全程平均**，长尾/等待会拉低；active-window 口径更稳健（避免 “run stuck / tail domination”）。

---

## 0. 实验矩阵

固定设置：

- **Bench**：`evaluation/tau2-bench/run_oss.py`
- **schedule mode**：`global`
- **domains**：`retail telecom airline`（合计 278 tasks：retail 114 + telecom 114 + airline 50）
- **log-level**：`PROFILE`（必须，否则 step_complete 解析不到）
- **Experts**：Together H100 集群的 3 个 expert LLM（通过 SSH tunnel 映射到 5090 localhost 端口）
- **Orchestrator**：`Nemotron-Orchestrator-8B` 由 5090 上 `vllm serve` 提供（OpenAI-compatible /v1/chat/completions）

对比维度：

- **Scheduler/Serving**：
  - ThunderReact：router + vLLM backend（端口 8000/8100）
  - Continuum：vLLM `--scheduling-policy continuum`（端口 1900）
  - Baseline：标准 vLLM（无 router，无 continuum policy）（端口 1900）
- **Concurrency**：32 / 64 / 128

---

## 1. 环境与前置条件

### 1.1 机器/仓库

- Repo：`/workspace/ToolOrchestra`
- 建议记录当前代码版本：
  ```bash
  cd /workspace/ToolOrchestra
  git rev-parse HEAD
  ```

### 1.2 必要环境变量（最少集）

必须：

- `CKPT_DIR`：Orchestrator checkpoint 路径（Nemotron-Orchestrator-8B）
- `OPENAI_API_KEY`：用于 tau2 user simulator（默认配置中使用 `gpt-5`）

推荐直接：

```bash
cd /workspace/ToolOrchestra
source setup_envs.sh
echo "CKPT_DIR=${CKPT_DIR}"
```

> 注意：开源/复现时请自行提供 API Key；不要依赖仓库里可能存在的默认值。

### 1.3 Conda 环境

需要 2 个环境（TR 与 Baseline 共用 `vllm1`；Continuum 用 `vllm-continuum`）：

- `vllm1`
- `vllm-continuum`

launcher 脚本会自己 `conda activate`，但建议先手动确认能 import vllm。

### 1.4 Experts + SSH tunnels（5090 访问 H100）

端口映射（5090 localhost）：

- expert-1 `openai/gpt-oss-20b`：1910–1913（DP=4）
- expert-2 `Qwen/Qwen3-32B-FP8`：1904–1905（DP=2）
- expert-3 `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`：1920–1921（DP=2）

在 **tmux session "0"** 启动隧道并保持运行：

```bash
tmux new -s 0
cd /workspace/ToolOrchestra
bash scripts/5090/tunnel_experts_via_head.sh
```

验证：

```bash
ss -lntp | egrep ':(1910|1911|1912|1913|1904|1905|1920|1921) '
curl -sS --max-time 5 http://127.0.0.1:1910/health
curl -sS --max-time 5 http://127.0.0.1:1904/health
curl -sS --max-time 5 http://127.0.0.1:1920/health
```

更多背景见：`TECH_MEMO_5090_orch8b_h100_experts_2026-01-03.md`

---

## 2. 预检查（每次跑实验前）

使用 repo 内置脚本：

```bash
cd /workspace/ToolOrchestra
bash scripts/5090/pre_experiment_check.sh
```

如果你要用 runner 脚本（`run_remaining_matrix.sh` / `run_baseline_matrix.sh`），它们默认调用 `/tmp/pre_experiment_check.sh`，建议加一个软链：

```bash
ln -sf /workspace/ToolOrchestra/scripts/5090/pre_experiment_check.sh /tmp/pre_experiment_check.sh
chmod +x /tmp/pre_experiment_check.sh
```

---

## 3. 如何启动 9 个实验（推荐方式）

### 3.1 建一个 OUTPUT_BASE（方便归档）

```bash
export EXPERIMENT_ID="5090_tau2_global_tr_ct_baseline_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_BASE="/workspace/ToolOrchestra/outputs/${EXPERIMENT_ID}"
mkdir -p "${OUTPUT_BASE}"
echo "${EXPERIMENT_ID}" > "${OUTPUT_BASE}/experiment_id.txt"
```

### 3.2 ThunderReact（TR）

TR 需要先跑一次 `C=32`（runner 只会跑剩余的 5 个）：

```bash
cd /workspace/ToolOrchestra
source setup_envs.sh
export DOMAINS="retail telecom airline"
export LOG_LEVEL="PROFILE"

bash scripts/5090/launch_thunderreact.sh 32 2>&1 | tee "${OUTPUT_BASE}/tr32_console.log"
```

随后跑剩余 TR64 / TR128 / CT32 / CT64 / CT128：

```bash
cd /workspace/ToolOrchestra
export DOMAINS="retail telecom airline"
export LOG_LEVEL="PROFILE"

bash scripts/5090/run_remaining_matrix.sh "${OUTPUT_BASE}" 2>&1 | tee "${OUTPUT_BASE}/runner_remaining.log"
```

### 3.3 Baseline vLLM（BL）

推荐直接用 baseline runner（顺序跑 32/64/128）：

```bash
cd /workspace/ToolOrchestra
source setup_envs.sh
export DOMAINS="retail telecom airline"
export LOG_LEVEL="PROFILE"

bash scripts/5090/run_baseline_matrix.sh "${OUTPUT_BASE}" 2>&1 | tee "${OUTPUT_BASE}/runner_baseline.log"
```

> 重要：`scripts/5090/launch_baseline.sh` 已内置 `ulimit -n` 提升（避免 tmux 默认 1024 FD 导致 C=128 “Too many open files” 崩溃）。

---

## 4. 运行时监控（建议 5 分钟一次）

### 4.1 Baseline / Continuum（端口 1900）推荐脚本

```bash
cd /workspace/ToolOrchestra
bash scripts/5090/quick_check_tau2.sh
```

这个检查包含：

- vLLM：`/health`、`/v1/models`（含 model_id）、`/metrics`（KV cache usage + requests running/waiting）
- tunnel 端口数量
- GPU util + memory
- tau2-bench：从 `run_oss.py` 进程解析 run_dir，然后统计 `driver.log` 中：
  - `FINISHED SIMULATION` 数
  - `Global evaluation complete` 行数
  - last_task / last_step 的时间戳（用于判断“是否真的卡住”）

可选：额外检查 experts：

```bash
CHECK_EXPERT_HEALTH=1 bash scripts/5090/quick_check_tau2.sh
```

持续循环：

```bash
INTERVAL_SEC=300 bash scripts/5090/monitor_loop_tau2.sh
```

### 4.2 ThunderReact（端口 8100/8000）

TR 的 vLLM backend 在 8100，router 在 8000：

```bash
# 检查 vLLM backend 的 health + kv cache（用同一个 quick_check，只是换端口）
ORCH_PORT=8100 bash scripts/5090/quick_check_tau2.sh

# 检查 router backends
curl -sf --max-time 3 http://127.0.0.1:8000/backends | head
```

### 4.3 “GPU util=0%”怎么解释（不是必然 bug）

常见原因是：driver 正在等待 **远端调用**（user simulator `gpt-5` / H100 expert），而不是在调用 5090 上的 vLLM；
此时：

- `nvidia-smi`：util 可能为 0%，但显存仍高（模型常驻）
- `driver.log`：会看到 `Calling OpenAI model=gpt-5` 或 expert call
- vLLM：`/health` OK，`kv_cache_usage` 可能很低

要区分“等待 vs 崩溃”，以 `scripts/5090/quick_check_tau2.sh` 的 `/health` + last_step_ts 为准。

---

## 5. 输出结构与验收标准

每个 run 目录（TR/CT/BL）应该至少包含：

- `driver.log`
- `kv_cache_timeseries.csv`（必须存在，否则无法做 active-window profiling）
- `outputs/metrics_global.json`
- `outputs/all_domains/*.json`（应为 278 个 `*__trial0.json` 文件）

快速验收：

```bash
RUN_DIR=/path/to/run_dir
test -f "${RUN_DIR}/outputs/metrics_global.json"
test -f "${RUN_DIR}/kv_cache_timeseries.csv"
ls -1 "${RUN_DIR}/outputs/all_domains/"*__*__trial0.json | wc -l
grep -E "Global evaluation complete" -n "${RUN_DIR}/driver.log" | tail -1
```

---

## 6. Profiling & 分析（如何得到 TL;DR 表格）

### 6.1 Active window 定义（核心口径）

- **window_start**：KV cache usage 第一次 `> 70%`
- **window_end**：KV cache usage 最后一次 `> 70%`

KV cache usage 来自 `kv_cache_timeseries.csv` 内的 `vllm:kv_cache_usage_perc`（值为 0..1 的 fraction；脚本内部会转换为百分比）。

### 6.2 一条命令产出 TR/CT/BL 9 组对比表

脚本：`scripts/analysis/active_window/compare_tr_ct_bl_active_window.py`

如果你用上面的 runner，目录一般是：

- `${OUTPUT_BASE}/thunderreact_c{32,64,128}`
- `${OUTPUT_BASE}/continuum_c{32,64,128}`
- `${OUTPUT_BASE}/baseline_c{32,64,128}`

运行：

```bash
cd /workspace/ToolOrchestra
python3 scripts/analysis/active_window/compare_tr_ct_bl_active_window.py \
  --threshold 70 \
  --tr32 "${OUTPUT_BASE}/thunderreact_c32" \
  --tr64 "${OUTPUT_BASE}/thunderreact_c64" \
  --tr128 "${OUTPUT_BASE}/thunderreact_c128" \
  --ct32 "${OUTPUT_BASE}/continuum_c32" \
  --ct64 "${OUTPUT_BASE}/continuum_c64" \
  --ct128 "${OUTPUT_BASE}/continuum_c128" \
  --bl32 "${OUTPUT_BASE}/baseline_c32" \
  --bl64 "${OUTPUT_BASE}/baseline_c64" \
  --bl128 "${OUTPUT_BASE}/baseline_c128"
```

### 6.3 长尾解释（window 外的任务/步数）

脚本：`scripts/analysis/active_window/window_outside_breakdown.py`

```bash
python3 scripts/analysis/active_window/window_outside_breakdown.py \
  --threshold 70 --name BL128 --run-dir "${OUTPUT_BASE}/baseline_c128"
```

输出包含：

- `first_task / win_start / win_end / last_task`
- `gap(win_end->last_task)`：用于量化长尾（KV<70% 但任务仍在完成）
- tasks/steps before/in/after 三段计数

### 6.4 C=128 bottleneck（为什么高并发下 Baseline 反而更稳）

脚本：`scripts/analysis/active_window/compare_bottleneck_active_window.py`

它会在 **active window** 内统计：
- `steps/s` 总吞吐 + `agent->env/s`（更接近 Orchestrator vLLM 的真实吞吐）
- `llm_call` 的长尾（>300s / >1200s）以及 top slow calls
- `user_sim` / `expert_call` 的延迟分布（用于排除外部瓶颈）

```bash
python3 scripts/analysis/active_window/compare_bottleneck_active_window.py \
  --threshold 70 \
  --run TR128=outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c128 \
  --run CT128=outputs/continuum_c128_20260104_223636 \
  --run BL128=outputs/baseline_c128_20260105_101158
```

核心现象（本次数据）：
- `user_sim` / `expert_call` 的 `p50/p90` 在三者之间非常接近；差异主要来自 Orchestrator 的 `llm_call` 长尾。
- TR/CT 在 C=128 active-window 内有大量超长 `llm_call`（分钟级/半小时级），会拉低 tasks/min & steps/s，并造成 “GPU util=0 但 eval 卡住” 的错觉。
- Baseline C=128 的 `llm_call` 长尾显著更轻（>300s 的 call 极少），因此整体更稳、吞吐最好。

### 6.5 指标核对（以 BL128 为例）

本 repo 的 active-window throughput 计算方式（脚本见 `scripts/analysis/active_window/compare_tr_ct_bl_active_window.py`）：

- `window_start` / `window_end`：从 `kv_cache_timeseries.csv` 取 **第一次** `kv_cache_usage_perc > 70%` 与 **最后一次** `> 70%`
- `tasks_in_window`：`driver.log` 中 `FINISHED SIMULATION` 的时间戳落在 `[window_start, window_end]` 内的计数
- `steps_in_window`：`driver.log` 中 `[PROFILE] ... type=step_complete` 的时间戳落在 `[window_start, window_end]` 内的计数
- `tasks/min = tasks_in_window / (window_s / 60)`
- `steps/s  = steps_in_window / window_s`

对 `outputs/baseline_c128_20260105_101158`：

- `win_start=10:14:23`, `win_end=11:28:56` → `window_s=4473s (74.5min)`
- `tasks_in_window=262` → `262/(4473/60)=3.51 tasks/min`
- `steps_in_window=12658` → `12658/4473=2.83 steps/s`
- window 外完成量（同阈值 70%）：tasks `before/in/after=1/262/12`，steps `before/in/after=432/12658/330`

KV sampler 是否污染（本次 BL128 是干净的）：

- `outputs/baseline_c128_20260105_101158/kv_cache_timeseries.csv` 首行时间 `10:12:42`，末行时间 `11:45:34`，且 `port=1900` 唯一 → 覆盖了 active window 以及 run 的 tail

### 6.6 为什么 C=128 下 Baseline(vLLM) 反而最好（更“底层”的原因）

先强调一个容易误判的点：C=128 的差异，主要不是 “平均 llm_call 更快”，而是 **“尾部更短 + 更 work‑conserving（不让大量 program 在 router/queue 里饿死）”**。

用本次数据说话（active window 内，脚本输出）：

- Baseline：`llm_call p50≈81.6s`（不快），但 `p99≈143.9s`、`max≈515.7s`（尾部很轻）
- TR/CT：`llm_call p50` 更快（TR≈21s / CT≈29s），但 `p99` 到 **数十分钟**（TR≈1725s / CT≈1543s）
- `user_sim` 与 `expert_call` 延迟分布三者很接近 → **瓶颈集中在 Orchestrator 路径（尤其是排队/阻塞导致的长尾）**

解释为什么会这样：

1) **Baseline vLLM 更像 “token‑level processor sharing”**  
   vLLM 的 continuous batching 会把大量请求一起推进：单个请求延迟可能更大，但请求之间更公平、更稳定；在高并发下反而能把 GPU 吃满、减少极端 starvation → tasks/min 与 steps/s 更稳。

2) **ThunderReact 的 router 在单 backend 场景会引入“非 work‑conserving 的阻塞”**  
   当某个 program 被 pause 时，请求会在 router 端 `await resume_event.wait()` 直接挂起（而不是把请求送到 backend 排队）。这会把 “等待时间” 直接计入 driver 的 `llm_call duration_ms`，并且可能无限久（不受 vLLM 900s client timeout 约束）。  
   同时 pause/resume 的排序偏置：
   - pause：优先 pause `step_count` 低的 program（更像新/早期任务）
   - resume：优先 resume `step_count` 高的 program（更像老/后期任务）  
   在 C=128 下，这个组合很容易让一批早期任务长时间拿不到服务（长尾爆炸），整体任务完成速度下降。

3) **Continuum policy 的表现更像 “队列里有一批请求被饿死/排队过久”**  
   在 CT128 的 top slow calls 里，超长 `llm_call` 多发生在 `step=1`，直观上像 “新任务 admission 被延后到很久以后才被服务”。

能直接用于改 ThunderReact 的 insight（给算法设计看的）：

- 如果目标是 **tasks/min（完成任务数）**，最怕的是 “少数 program 极端长尾把完成数卡住”。所以需要：  
  1) **保证 work‑conserving**：GPU 空闲时一定要有请求在跑，避免 router 侧把请求挂住。  
  2) **反 starvation**：resume/pause 不能只按 step_count；需要“aging/credit”保证所有 program 都能持续获得服务。  
  3) **只在 admission 上做强约束**：KV 高时，优先限制“新 program 的进入（step=1）”，但让已开始的 program 继续跑到释放点；否则会出现 “占着 concurrency 但不推进” 的死等。

---

## 7. 本次 worklog 的“已跑完” artifacts（可复查用）

### 7.1 ThunderReact / Continuum（历史 run）

详见：`docs/experiments/2026-01-04_5090_tau2_global_thunderreact_vs_continuum.md`

主要路径：

- TR：`outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c32` / `.../thunderreact_c64` / `.../thunderreact_c128`
- CT：`outputs/continuum_c32_20260104_191004` / `outputs/continuum_c64_20260104_205216` / `outputs/continuum_c128_20260104_223636`

### 7.2 Baseline vLLM（本次新增）

- BL32：`outputs/baseline_c32_20260105_062226`
  - `Global evaluation complete: ok=263 error=15 total=278 elapsed_s=5516.4 tasks_per_min=2.86`
  - active window：`2026-01-05 06:41:13 UTC` → `2026-01-05 07:49:02 UTC`
- BL64：`outputs/baseline_c64_20260105_081018`
  - `Global evaluation complete: ok=269 error=9 total=278 elapsed_s=5168.6 tasks_per_min=3.12`
  - active window：`2026-01-05 08:27:08 UTC` → `2026-01-05 09:38:23 UTC`
- BL128：`outputs/baseline_c128_20260105_101158`
  - `Global evaluation complete: ok=275 error=3 total=278 elapsed_s=5566.3 tasks_per_min=2.96`
  - active window：`2026-01-05 10:14:23 UTC` → `2026-01-05 11:28:56 UTC`
  - window 外（threshold=70）：tasks `before/in/after=1/262/12`，steps `before/in/after=432/12658/330`，gap(win_end→last_task)=`16.6 min`

> Baseline 的 error 主要来自 expert 端偶发 500（见 `driver.log` 中 `Expert LLM call failed permanently`），不是 5090 vLLM 本身崩溃。

---

## 8. 常见坑与修复（建议保留在复现文档中）

1. **KV sampler 写错文件（污染 active-window）**  
   原因：手动启动 sampler 后忘记 kill，导致下一轮实验仍写入上一轮的 `kv_cache_timeseries.csv`。  
   规避：用 `scripts/5090/launch_*.sh`（trap cleanup），或确认采样进程 PID 与输出文件匹配。

2. **C=128: tmux 默认 `ulimit -n=1024` 导致 vLLM 崩溃**  
   表现：`OSError: [Errno 24] Too many open files`，vLLM `/health` fail。  
   修复：`scripts/5090/launch_baseline.sh` 已强制提高 `ulimit -n`。

3. **“卡住”但其实在慢慢跑**  
   GPU util 可能为 0%，但 `last_step_ts` 仍在推进；通常是在等 user sim 或 experts（这段时间 vLLM queue 可能为空，KV cache usage 也会很低）。  
   用 `scripts/5090/quick_check_tau2.sh` 的 last_task/last_step 时间戳判断更靠谱。

4. **context length exceeded / infinite retry**（TR/CT 128 更常见）  
   表现：`driver.log` 中重复报错且进度长时间不动。  
   处理：可手动终止；active-window 仍可用于 throughput 对比（只要 `driver.log` + `kv_cache_timeseries.csv` 已写出）。

---

## 9. 一键复现（最短 checklist）

```bash
# 0) tunnels (tmux session 0)
tmux new -s 0
cd /workspace/ToolOrchestra
bash scripts/5090/tunnel_experts_via_head.sh

# 1) experiment session
tmux new -s exp
cd /workspace/ToolOrchestra
source setup_envs.sh
ln -sf /workspace/ToolOrchestra/scripts/5090/pre_experiment_check.sh /tmp/pre_experiment_check.sh
chmod +x /tmp/pre_experiment_check.sh

export EXPERIMENT_ID="5090_tau2_global_tr_ct_baseline_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_BASE="/workspace/ToolOrchestra/outputs/${EXPERIMENT_ID}"
mkdir -p "${OUTPUT_BASE}"

export DOMAINS="retail telecom airline"
export LOG_LEVEL="PROFILE"

# 2) TR32 (manual)
bash scripts/5090/launch_thunderreact.sh 32 2>&1 | tee "${OUTPUT_BASE}/tr32_console.log"

# 3) TR64/TR128/CT32/CT64/CT128
bash scripts/5090/run_remaining_matrix.sh "${OUTPUT_BASE}" 2>&1 | tee "${OUTPUT_BASE}/runner_remaining.log"

# 4) BL32/BL64/BL128
bash scripts/5090/run_baseline_matrix.sh "${OUTPUT_BASE}" 2>&1 | tee "${OUTPUT_BASE}/runner_baseline.log"

# 5) compare (active window)
python3 scripts/analysis/active_window/compare_tr_ct_bl_active_window.py \
  --threshold 70 \
  --tr32 "${OUTPUT_BASE}/thunderreact_c32" --tr64 "${OUTPUT_BASE}/thunderreact_c64" --tr128 "${OUTPUT_BASE}/thunderreact_c128" \
  --ct32 "${OUTPUT_BASE}/continuum_c32" --ct64 "${OUTPUT_BASE}/continuum_c64" --ct128 "${OUTPUT_BASE}/continuum_c128" \
  --bl32 "${OUTPUT_BASE}/baseline_c32" --bl64 "${OUTPUT_BASE}/baseline_c64" --bl128 "${OUTPUT_BASE}/baseline_c128"

# 6) bottleneck (active window)
python3 scripts/analysis/active_window/compare_bottleneck_active_window.py \
  --threshold 70 \
  --run TR128="${OUTPUT_BASE}/thunderreact_c128" \
  --run CT128="${OUTPUT_BASE}/continuum_c128" \
  --run BL128="${OUTPUT_BASE}/baseline_c128"
```

---

*Doc updated: 2026-01-05*

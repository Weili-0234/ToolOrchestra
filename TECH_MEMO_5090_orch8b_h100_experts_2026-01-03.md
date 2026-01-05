# 技术备忘录：Together H100 Experts + RunPod 5090 Orchestrator-8B（Inference 实验准备）

目标：在 **RunPod 5090 单卡** 上 `vllm serve` Orchestrator-8B，并调用 **Together H100 集群 compute node** 上部署的 3 个 expert LLM，测max_concurrency={32,64,128} 下：
- **step-wise throughput**（steps/sec）
- **task-wise throughput**（tasks/min）
- **5090 单卡 vLLM backend 的 KV cache util 随时间变化**（time series）

本 memo 记录你已经做过/验证过的步骤，以及后续实验的最小可复现流程（避免来回 double check）。

---

## 1) H100 集群侧：SSH 入口（Mac/5090 都可用）

你的 SSH config：

```sshconfig
Host H100-Together
  HostName research-secure-hn.cloud.together.ai
  User junxiong
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

已验证：在 5090 RunPod shell 中可执行（首次可能要 accept host key）：

```bash
ssh -o StrictHostKeyChecking=accept-new H100-Together 'hostname; date'
```

备注：你观测到单次 `ssh 'hostname; date'` 约 **22s**（连接握手较慢），这会影响端口转发“ready”时机（见后文）。

---

## 2) H100 集群侧：3 个 Expert 的当前部署信息（compute node 内网 IP）

以下是本 session 中确认“能 `/health`、`/v1/models`、`/v1/chat/completions`”的那一批。

### expert-1
- **Model**: `openai/gpt-oss-20b`
- **SLURM job id**: `13582`
- **Node**: `research-secure-16`
- **IP**: `172.27.27.153`
- **Ports (DP backends)**: `1910, 1911, 1912, 1913`

### expert-2
- **Model**: `Qwen/Qwen3-32B-FP8`
- **SLURM job id**: `13586`
- **Node**: `research-secure-20`
- **IP**: `172.27.25.244`
- **Ports (DP backends)**: `1904, 1905`

### expert-3
- **Model**: `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- **SLURM job id**: `13588`
- **Node**: `research-secure-12`
- **IP**: `172.27.19.213`
- **Ports (DP backends)**: `1920, 1921`

---

## 3) 输出格式/返回结构：重要已知行为（避免误判）

- **`openai/gpt-oss-20b`**：`/v1/chat/completions` 可能出现
  - `choices[0].message.content == null`
  - 文本在 `choices[0].message.reasoning` / `reasoning_content`
  - 这是模型/serving 的返回字段差异，不是 Orchestrator 的 chat template 导致的。

- **Qwen3 系列**：可能在 `content` 里吐 `<think>...`（如果不加 `/no_think` 或不做 prompt 约束）；并且在 `max_tokens` 太小的时候会因为 `finish_reason="length"` 截断在 `<think>` 阶段，导致看起来“没按要求输出”。

---

## 4) RunPod 5090 侧：端口转发（让 5090 本地访问 Experts）

### 4.1 为什么需要转发
RunPod 5090 无法直接路由到 H100 compute node 的 `172.27.*` 内网 IP。当前可用路径是：
`5090 -> SSH 到 head node -> head node 内网访问 compute node experts`

### 4.2 5090 上启动 8 个端口的本地转发（推荐命令）

```bash
ssh -fN -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 1910:172.27.27.153:1910 -L 1911:172.27.27.153:1911 -L 1912:172.27.27.153:1912 -L 1913:172.27.27.153:1913 \
  -L 1904:172.27.25.244:1904 -L 1905:172.27.25.244:1905 \
  -L 1920:172.27.19.213:1920 -L 1921:172.27.19.213:1921 \
  H100-Together
```

说明：
- `-N`：只做转发不执行命令
- `-f`：等转发建立成功后放到后台返回（因此会等待较久；你实测 ssh 握手 ~22s，属正常）
- `-o ExitOnForwardFailure=yes`：如果任一端口无法建立，会直接失败退出（避免“假连通”）

### 4.3 确认转发已 ready（以 LISTEN 为准）

```bash
ss -lntp | egrep ':(1910|1911|1912|1913|1904|1905|1920|1921) '
```

你已验证：上述端口均由 `ssh(pid=4236)` LISTEN。

### 4.4 5090 本地验证（所有请求都打 127.0.0.1）

```bash
curl -sS --max-time 5 http://127.0.0.1:1910/v1/models
curl -sS --max-time 5 http://127.0.0.1:1904/v1/models
curl -sS --max-time 5 http://127.0.0.1:1920/v1/models
```

常见误判：
- 刚启动转发后立刻 curl 可能 `connection refused`，原因是 ssh 还在握手/未开始 bind，本地端口尚未 LISTEN。

---

## 5) 5090 侧：Orchestrator-8B vLLM serve + driver

### 5.1 5090 启动脚本现状（需要你决策/补齐的点）
repo 里已有 `scripts/5090/launch_{baseline,thunderreact,continuum}.sh`，但它们当前 **没有传 `--chat-template`**。

实验建议：
- Orchestrator-8B 建议统一使用：
  - `--chat-template <ToolOrchestra>/evaluation/tool_chat_template_llama3.1_json.jinja`
  - `--enable-auto-tool-choice`
  - `--tool-call-parser <与你 vLLM 版本匹配的 parser>`

否则可能复现“chat template / message header parse error”类问题。

### 5.2 model_config 原则（关键）
当 driver 在 5090 上跑时：
- **experts 一律写成本地转发端口**：`127.0.0.1:<1910/1904/1920...>`
- 不要在 5090 的 model_config 里写 `172.27.*`（否则 5090 直接路由失败）

你可以用 `scripts/gen_model_config.py` 生成一个包含 `oss_expert_mapping` 的 config，但对 5090 “单 orchestrator endpoint”可能还需要你手写/改脚本支持（原脚本默认 baseline/continuum 按 DP=8 生成 1900-1907）。

---

## 6) 5090 KV cache util time series（Orchestrator vLLM）

仓库已有脚本：
- `scripts/rollout/collect_kv_cache_timeseries.sh`

默认假设 ThunderReact rollout（采 8100-8107），但你可以通过 ports 参数改成 5090 单卡端口（常见为 1900）：

```bash
bash oss-ToolOrchestra/scripts/rollout/collect_kv_cache_timeseries.sh \
  127.0.0.1 /path/to/out_kv.csv 5 "1900"
```

---

## 7) max_concurrency=128 的稳定性注意事项

- **长连接/断线**：SSH 转发是单条常驻连接，建议至少设置 keepalive（你已在命令里加了 `ServerAlive*`）。
- **转发 ready 再测**：以 `ss -lntp` LISTEN 为准，避免“22s 握手窗口”内误判。
- **吞吐瓶颈来源**：128 并发时，head node 转发能力/连接数/CPU 可能成为瓶颈；更理想的方案是 VPN（Tailscale/WireGuard 子网路由）让 5090 直接访问 `172.27.*`，但需要集群侧权限/策略允许。

---

## 8) 快速自检 checklist（5090 上）

1) 转发还在：
```bash
ss -lntp | egrep ':(1910|1911|1912|1913|1904|1905|1920|1921) '
```

2) experts 可用（本地端口）：
```bash
curl -sS --max-time 5 http://127.0.0.1:1910/v1/models
curl -sS --max-time 5 http://127.0.0.1:1904/v1/models
curl -sS --max-time 5 http://127.0.0.1:1920/v1/models
```

3) Orchestrator vLLM 起服务后：
```bash
curl -sS --max-time 5 http://127.0.0.1:1900/v1/models
curl -sS --max-time 5 http://127.0.0.1:1900/metrics | head -n 20
```



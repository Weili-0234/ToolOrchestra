# τ²-Bench 本地运行快速开始

5 分钟快速启动指南 🚀

## 步骤 1️⃣: 安装 Conda 环境

```bash
# 创建 vllm1 环境
conda create -n vllm1 python=3.12 -y
conda activate vllm1

# 安装依赖
pip install torch
pip install "transformers<4.54.0"
pip install vllm==0.9.2

# 安装 tau2-bench
cd evaluation/tau2-bench
pip install -e .
```

## 步骤 2️⃣: 配置环境变量

编辑 `setup_envs.sh`，设置以下关键变量：

```bash
# 1. 设置模型路径
export CKPT_DIR="/path/to/your/Orchestrator-8B"

# 2. 设置 OpenAI API Key (用于 user simulation)
export OPENAI_API_KEY="sk-..."

# 3. 设置 Nebius API Key (用于 Qwen3-32B，避免本地部署)
export NEBIUS_API_KEY="v1...."

# 4. (可选) 其他 API keys
export ANTHROPIC_API_KEY="sk-ant-..."
```

然后加载环境变量：

```bash
cd /path/to/ToolOrchestra
source setup_envs.sh
```

**💡 关于 Nebius API**: 如果设置了 `NEBIUS_API_KEY`，Qwen3-32B (expert-3) 会自动通过 Nebius API 调用，无需在本地启动 Qwen3-32B 的 vLLM 服务器。详见 [NEBIUS_INTEGRATION.md](../../NEBIUS_INTEGRATION.md)。

## 步骤 3️⃣: 下载模型

```bash
# 下载 Orchestrator-8B 模型
git clone https://huggingface.co/nvidia/Nemotron-Orchestrator-8B $CKPT_DIR
```

## 步骤 4️⃣: 运行评测

```bash
cd evaluation/tau2-bench/
python run_local.py --agent-model $CKPT_DIR
```

**就这么简单！** 🎉

---

## 常见场景

### 场景 1: 只有 1 个 GPU

```bash
python run_local.py \
  --agent-model $CKPT_DIR \
  --num-servers 1 \
  --domains retail  # 先测一个域
```

### 场景 2: 有 4+ 个 GPU

```bash
python run_local.py \
  --agent-model $CKPT_DIR \
  --num-servers 4 \
  --domains retail telecom airline  # 测所有域
```

### 场景 3: 快速测试（只跑少量样本）

```bash
# 修改 task file 或使用 --num-trials 参数
python run_local.py \
  --agent-model $CKPT_DIR \
  --domains retail \
  --num-trials 1 \
  --max-steps 50
```

---

## 检查清单 ✅

运行前确保：

- [ ] ✅ Conda 环境 `vllm1` 已激活
- [ ] ✅ `CKPT_DIR` 已设置且模型已下载
- [ ] ✅ `OPENAI_API_KEY` 已设置
- [ ] ✅ `NEBIUS_API_KEY` 已设置（推荐，用于 Qwen3-32B）
- [ ] ✅ `REPO_PATH` 已设置（或在仓库根目录运行）
- [ ] ✅ 至少有 1 个可用 GPU

验证命令：
```bash
# 检查 GPU
nvidia-smi

# 检查环境变量
echo $CKPT_DIR
echo $OPENAI_API_KEY
echo $REPO_PATH

# 检查 conda 环境
conda info --envs | grep vllm1
```

---

## 预期输出

运行成功后，你会看到：

```
[2025-12-23 10:00:00] Starting 4 vLLM server(s)...
[2025-12-23 10:00:00] Starting vLLM server: /path/to/model
[2025-12-23 10:00:00]   GPU: 0, Port: 1900
[2025-12-23 10:01:00] Starting vLLM server: /path/to/model
[2025-12-23 10:01:00]   GPU: 1, Port: 1901
...
[2025-12-23 10:15:00] ✓ Server on port 1900 is ready (took 300s)
[2025-12-23 10:15:00] ✓ Server on port 1901 is ready (took 240s)
...
[2025-12-23 10:15:30] Model configuration written to model_config_local.json
[2025-12-23 10:15:30] ========== Starting evaluation: RETAIL ==========
...
[2025-12-23 12:00:00] ========== Finished RETAIL evaluation successfully ==========
...
============================================================
EVALUATION SUMMARY
============================================================
RETAIL: SUCCESS
TELECOM: SUCCESS
AIRLINE: SUCCESS
============================================================
```

结果保存在：
- `outputs/retail.json`
- `outputs/telecom.json`
- `outputs/airline.json`

---

## 故障排查

### 问题: "CUDA out of memory"

**解决**: 减少服务器数量
```bash
python run_local.py --agent-model $CKPT_DIR --num-servers 1
```

### 问题: "Server failed to start"

**解决**: 检查日志
```bash
cat logs/vllm_port_1900_*.err
```

### 问题: "OPENAI_API_KEY not found"

**解决**: 设置环境变量
```bash
export OPENAI_API_KEY="sk-..."
source setup_envs.sh
```

---

## 下一步

- 📖 详细文档: 查看 [RUN_LOCAL_GUIDE.md](RUN_LOCAL_GUIDE.md)
- 🔧 自定义配置: `python run_local.py --help`
- 📊 分析结果: 查看 `outputs/*.json` 文件

---

## 获取帮助

```bash
# 查看所有选项
python run_local.py --help

# 测试环境设置
source setup_envs.sh

# 检查依赖
pip list | grep -E "vllm|transformers|torch"
```

祝评测顺利！🎯

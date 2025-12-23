# τ²-Bench 本地运行指南

本指南说明如何在本地环境（非 SLURM 集群）运行 τ²-Bench 评测。

## 📋 前置要求

### 1. Conda 环境

创建并激活 `vllm1` 环境：

```bash
conda create -n vllm1 python=3.12 -y
conda activate vllm1
pip install torch
pip install "transformers<4.54.0"
pip install vllm==0.9.2
cd evaluation/tau2-bench
pip install -e .
```

正在尝试下面这个能不能work
```bash
conda activate vllm1 && pip uninstall -y vllm torch transformers && pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 && pip install vllm==0.10.1 transformers
pip install hf_transfer # vllm 10.1 往后会用到 hf_transfer
```

### 2. 环境变量

设置以下必需的环境变量（可以添加到 `setup_envs.sh`）：

```bash
# 模型和数据路径
export CKPT_DIR="/path/to/your/agent/model"     # Agent 模型检查点路径
export REPO_PATH="/path/to/ToolOrchestra"       # 仓库根目录
export HF_HOME="/path/to/huggingface"           # HuggingFace 缓存目录

# API Keys (用于 user simulation 和可能的 judge model)
export OPENAI_API_KEY="your_openai_api_key"     # 用于 gpt-4o 作为 user-llm
export ANTHROPIC_API_KEY="your_anthropic_key"   # 可选，如果使用 Claude
```

加载环境变量：
```bash
source setup_envs.sh
```

### 3. GPU 要求

- **最少**: 1 个 GPU（只启动 1 个 agent server）
- **推荐**: 4+ 个 GPU（可以并行运行多个 agent server，加快评测速度）

## 🚀 使用方法

### 基本使用

最简单的使用方式：

```bash
cd evaluation/tau2-bench/
python run_local.py --agent-model $CKPT_DIR
```

这将会：
1. 启动 4 个 vLLM agent 服务器（如果有足够的 GPU）
2. 依次评测 retail、telecom、airline 三个域
3. 结果保存在 `outputs/` 目录

### 常用选项

#### 指定服务器数量
```bash
# 只启动 1 个服务器（适合 GPU 资源有限的情况）
python run_local.py --agent-model $CKPT_DIR --num-servers 1

# 启动 8 个服务器（如果有 8 个 GPU）
python run_local.py --agent-model $CKPT_DIR --num-servers 8
```

#### 指定要评测的域
```bash
# 只评测 retail 域
python run_local.py --agent-model $CKPT_DIR --domains retail

# 评测 retail 和 telecom
python run_local.py --agent-model $CKPT_DIR --domains retail telecom
```

#### 指定 user-llm
```bash
# 使用 gpt-4o（默认）
python run_local.py --agent-model $CKPT_DIR --user-llm gpt-4o

# 使用 Claude
python run_local.py --agent-model $CKPT_DIR --user-llm claude-3-5-sonnet-20241022
```

#### 指定 GPU
```bash
# 使用特定的 GPU (例如 GPU 2 和 3)
python run_local.py --agent-model $CKPT_DIR --num-servers 2 --gpu-ids 2 3
```

#### 调整启动参数
```bash
# 减少服务器启动间隔（默认 60 秒）
python run_local.py --agent-model $CKPT_DIR --stagger-delay 30

# 增加服务器启动超时时间（默认 600 秒）
python run_local.py --agent-model $CKPT_DIR --server-timeout 900
```

### 高级用法

#### 使用已经运行的服务器

如果你已经手动启动了 vLLM 服务器，可以跳过启动步骤：

```bash
# 先手动创建配置文件 model_config_local.json
cat > model_config_local.json << EOF
{
  "/path/to/model": [
    {"ip_addr": "127.0.0.1", "port": "1900"},
    {"ip_addr": "127.0.0.1", "port": "1901"}
  ],
  "vllm_model_config_path": "model_config_local.json"
}
EOF

# 然后运行评测
python run_local.py --agent-model $CKPT_DIR --skip-server-start
```

#### 完整示例命令

```bash
python run_local.py \
  --agent-model /data/models/Orchestrator-8B \
  --user-llm gpt-4o \
  --num-servers 4 \
  --domains retail telecom airline \
  --max-steps 200 \
  --num-trials 1 \
  --output-dir outputs/run1 \
  --log-dir logs/run1
```

## 📊 输出说明

### 日志文件

- **服务器日志**: `logs/vllm_port_XXXX_*.out` 和 `*.err`
  - 包含每个 vLLM 服务器的启动日志和错误信息

### 结果文件

- **评测结果**: `outputs/{domain}.json`
  - 每个域的详细评测结果

### 配置文件

- **模型配置**: `model_config_local.json`
  - 自动生成的模型服务器配置文件
  - tau2-bench 使用此文件连接到 vLLM 服务器

## 🔧 故障排查

### 问题 1: GPU 内存不足

**错误**: `CUDA out of memory`

**解决方案**:
- 减少并行服务器数量: `--num-servers 1`
- 或使用更小的模型

### 问题 2: 服务器启动超时

**错误**: `Server on port XXXX failed to start within 600s`

**解决方案**:
- 增加超时时间: `--server-timeout 1200`
- 检查 `logs/` 目录中的日志文件找出原因
- 确保模型路径正确且可访问

### 问题 3: REPO_PATH 未设置

**错误**: `ERROR: Task file not found`

**解决方案**:
```bash
export REPO_PATH="/path/to/ToolOrchestra"
```

### 问题 4: API Key 未设置

**错误**: `API key not found` 或 `Authentication failed`

**解决方案**:
```bash
export OPENAI_API_KEY="your_key_here"
# 或
export ANTHROPIC_API_KEY="your_key_here"
```

### 问题 5: 端口被占用

**错误**: `Address already in use`

**解决方案**:
- 使用不同的起始端口: `--start-port 2000`
- 或停止占用端口的进程:
  ```bash
  lsof -ti:1900 | xargs kill -9
  ```

## 📝 与原版 run.py 的区别

| 特性 | run.py (SLURM) | run_local.py (本地) |
|------|----------------|---------------------|
| 调度系统 | SLURM | 直接启动进程 |
| 服务器 IP | 集群节点 IP | 127.0.0.1 (localhost) |
| Judge Model | 启动本地 Qwen3-32B | 通过 API 调用 |
| GPU 分配 | SLURM 管理 | 手动指定 GPU ID |
| 日志 | SLURM 作业输出 | 本地文件 |
| 循环运行 | 是（持续监控） | 否（运行一次） |

## 💡 性能优化建议

1. **并行服务器**: 如果有多个 GPU，启动多个服务器可以加快评测速度
   ```bash
   --num-servers 4  # 4 个 GPU，每个运行一个服务器
   ```

2. **减少启动延迟**: 如果 GPU 内存充足，可以减少启动间隔
   ```bash
   --stagger-delay 30  # 从默认的 60 秒减少到 30 秒
   ```

3. **分批评测**: 如果时间有限，可以先评测单个域
   ```bash
   --domains retail  # 只评测 retail，之后再评测其他域
   ```

4. **调整并发**: 通过启动多个服务器实例来提高吞吐量
   - 1 个 GPU: `--num-servers 1`
   - 4 个 GPU: `--num-servers 4`
   - 8 个 GPU: `--num-servers 8`

## 🎯 快速开始检查清单

- [ ] Conda 环境 `vllm1` 已创建并激活
- [ ] 已安装所需依赖 (`pip install -e .`)
- [ ] 环境变量已设置 (`CKPT_DIR`, `REPO_PATH`, `OPENAI_API_KEY`)
- [ ] Agent 模型已下载到 `$CKPT_DIR`
- [ ] 至少有 1 个可用 GPU
- [ ] 运行测试命令验证设置:
  ```bash
  python run_local.py --agent-model $CKPT_DIR --domains retail --num-trials 1
  ```

## 📞 获取帮助

查看所有可用选项：
```bash
python run_local.py --help
```

如有问题，请检查：
1. 服务器日志: `logs/vllm_port_*.out` 和 `*.err`
2. 环境变量: `echo $CKPT_DIR $REPO_PATH $OPENAI_API_KEY`
3. GPU 状态: `nvidia-smi`

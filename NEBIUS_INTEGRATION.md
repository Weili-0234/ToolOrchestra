# Nebius API Integration for Qwen3-32B

本文档说明如何使用 Nebius API 作为 Qwen3-32B 模型的后端，避免在本地启动 vLLM 服务器。

## 🎯 功能说明

当在 τ²-Bench 评测中需要调用 Qwen3-32B 模型时（作为 expert-3 或 judge model），系统会自动检测是否设置了 `NEBIUS_API_KEY`：

- ✅ **如果设置了 NEBIUS_API_KEY**: 自动使用 Nebius API 调用 `Qwen/Qwen3-32B-fast`
- ⚠️ **如果未设置**: 回退到本地 vLLM 服务器（需要在 `model_config.json` 中配置）

## 🔧 配置步骤

### 1. 获取 Nebius API Key

访问 [Nebius Token Factory](https://tokenfactory.nebius.com) 获取你的 API key。

### 2. 设置环境变量

在 `setup_envs.sh` 中已经包含了 Nebius API Key 的配置：

```bash
# Nebius API Key (for Qwen3-32B-fast on Nebius backend)
# If set, Qwen3-32B calls will automatically use Nebius API instead of local vLLM
export NEBIUS_API_KEY="your_nebius_api_key_here"
```

**或者**，直接在终端中设置：

```bash
export NEBIUS_API_KEY="v1.your_api_key_here"
```

### 3. 加载环境变量

```bash
source setup_envs.sh
```

### 4. 验证配置

运行 `source setup_envs.sh` 后，你应该能看到：

```
========================================
ToolOrchestra Environment Configuration
========================================
...
API Keys:
  ...
  NEBIUS_API_KEY:    v1.CmQKHH... (Qwen3-32B)
  ...
========================================
```

## 📊 使用场景

### 场景 1: τ²-Bench 评测中的 Expert-3

在 τ²-Bench 评测中，当 agent 调用 `call_expert` 工具并选择 `expert-3` 时，系统会自动使用 Qwen3-32B：

```python
# tau2/utils/llm_utils.py, line 489
if one_tool_call_arguments['expert']=='expert-3':
    mode_to_call = 'Qwen/Qwen3-32B'
```

如果设置了 `NEBIUS_API_KEY`，这个调用会自动路由到 Nebius API。

### 场景 2: 任何直接调用 Qwen3-32B 的地方

在代码中任何使用 `get_llm_response()` 并指定 model 为 `Qwen/Qwen3-32B` 的地方，都会自动使用 Nebius API（如果设置了 key）。

## 🔍 工作原理

修改位置：`LLM_CALL.py` 第 361-424 行

```python
elif 'qwen' in model.lower() or model_type=='vllm':
    # Check if we should use Nebius API for Qwen3-32B
    nebius_api_key = os.getenv("NEBIUS_API_KEY")
    use_nebius = nebius_api_key and 'qwen3-32b' in model.lower()

    if use_nebius:
        # Use Nebius API for Qwen3-32B
        nebius_client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=nebius_api_key
        )
        nebius_model = "Qwen/Qwen3-32B-fast"

        chat_completion = nebius_client.chat.completions.create(
            model=nebius_model,
            messages=messages,
            max_tokens=max_length,
            temperature=temperature,
            tools=tools
        )
    else:
        # Use local vLLM server (original behavior)
        ...
```

### 检测逻辑

1. **环境变量检测**: 检查 `NEBIUS_API_KEY` 是否设置
2. **模型名称检测**: 检查模型名称中是否包含 `qwen3-32b`（不区分大小写）
3. **自动路由**:
   - 如果两个条件都满足 → 使用 Nebius API
   - 否则 → 使用本地 vLLM 服务器

## ✅ 优势

使用 Nebius API 而非本地 vLLM 服务器的优势：

1. **无需本地 GPU**: 不需要在本地启动 Qwen3-32B 的 vLLM 服务器
2. **节省资源**: 释放本地 GPU 资源用于运行 agent model
3. **简化部署**: 不需要配置和管理额外的 vLLM 服务器实例
4. **灵活切换**: 通过环境变量轻松切换本地/云端部署

## 🚀 运行 τ²-Bench

使用 Nebius API 运行 τ²-Bench 评测：

```bash
# 1. 设置环境变量（包括 NEBIUS_API_KEY）
source setup_envs.sh

# 2. 激活 conda 环境
conda activate vllm1

# 3. 运行评测（只需启动 agent model 服务器）
cd evaluation/tau2-bench/
python run_local.py --agent-model $CKPT_DIR
```

**注意**: 由于 Qwen3-32B 通过 Nebius API 调用，你**不需要**在本地启动 Qwen3-32B 的 vLLM 服务器。

## 🔄 与原版对比

| 特性 | 原版 (本地 vLLM) | 使用 Nebius API |
|------|------------------|-----------------|
| Qwen3-32B 部署 | 需要本地 vLLM 服务器 | 云端 API 调用 |
| GPU 需求 | Agent model + Qwen3-32B | 仅 Agent model |
| 配置复杂度 | 需要配置多个 vLLM 实例 | 只需设置 API key |
| 网络依赖 | 无 | 需要网络连接 |
| 成本 | GPU 资源成本 | API 调用成本 |

## 📝 API 调用示例

Nebius API 的调用方式与 OpenAI API 完全兼容：

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-32B-fast",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ],
    temperature=1.0,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

## 🐛 故障排查

### 问题 1: API 调用失败

**错误**: `Error calling Nebius API: ...`

**解决方案**:
1. 检查 `NEBIUS_API_KEY` 是否正确设置
2. 验证 API key 是否有效
3. 检查网络连接

```bash
# 验证环境变量
echo $NEBIUS_API_KEY

# 测试 API 连接
curl -H "Authorization: Bearer $NEBIUS_API_KEY" \
  https://api.tokenfactory.nebius.com/v1/models
```

### 问题 2: 仍然调用本地 vLLM

**原因**: 模型名称不匹配或环境变量未设置

**解决方案**:
1. 确保模型名称包含 `Qwen3-32B`（不区分大小写）
2. 确认 `NEBIUS_API_KEY` 已正确设置并加载

```bash
# 重新加载环境变量
source setup_envs.sh

# 检查是否已设置
env | grep NEBIUS
```

### 问题 3: 如何临时禁用 Nebius API

如果想临时使用本地 vLLM 而不是 Nebius API：

```bash
# 临时取消设置环境变量
unset NEBIUS_API_KEY

# 或者运行时不加载
NEBIUS_API_KEY="" python run_local.py --agent-model $CKPT_DIR
```

## 📚 相关文件

- **核心实现**: [`LLM_CALL.py`](LLM_CALL.py#L361-L424)
- **环境配置**: [`setup_envs.sh`](setup_envs.sh#L39-L42)
- **使用示例**: [`evaluation/tau2-bench/tau2/utils/llm_utils.py`](evaluation/tau2-bench/tau2/utils/llm_utils.py#L489-L504)

## 💡 最佳实践

1. **开发环境**: 使用 Nebius API 节省本地 GPU 资源
2. **生产环境**: 根据成本和性能需求选择本地或云端部署
3. **混合部署**: Agent model 本地运行，Qwen3-32B 使用云端 API
4. **备份方案**: 保留本地 vLLM 配置作为备份（当 API 不可用时）

## 🔐 安全提示

- ⚠️ 不要在公开仓库中提交包含 API key 的 `setup_envs.sh`
- ✅ 使用 `.gitignore` 排除包含敏感信息的文件
- ✅ 定期轮换 API key
- ✅ 为不同环境使用不同的 API key

---

如有问题，请参考：
- Nebius 官方文档: https://tokenfactory.nebius.com/docs
- ToolOrchestra 文档: [README.md](README.md)
- 本地运行指南: [evaluation/tau2-bench/RUN_LOCAL_GUIDE.md](evaluation/tau2-bench/RUN_LOCAL_GUIDE.md)

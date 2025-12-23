# Nebius API 集成修改总结

本文档总结了为支持 Nebius API 调用 Qwen3-32B 所做的所有修改。

## 📝 修改的文件

### 1. **LLM_CALL.py** (核心修改)

**位置**: [LLM_CALL.py](LLM_CALL.py#L361-L424)

**修改内容**:
- 在 `get_llm_response()` 函数中添加了 Nebius API 支持
- 当检测到 `NEBIUS_API_KEY` 环境变量且模型为 Qwen3-32B 时，自动使用 Nebius API
- 否则回退到原来的本地 vLLM 服务器逻辑

**关键代码**:
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
        # ... API 调用逻辑
    else:
        # Use local vLLM server (原有逻辑)
        # ...
```

**影响范围**:
- 所有通过 `get_llm_response()` 调用 Qwen3-32B 的地方
- τ²-Bench 中 expert-3 的调用
- 任何显式调用 `Qwen/Qwen3-32B` 模型的代码

---

### 2. **setup_envs.sh** (环境配置)

**位置**: [setup_envs.sh](setup_envs.sh#L39-L42)

**修改内容**:
- 添加了 `NEBIUS_API_KEY` 环境变量配置
- 添加了说明注释
- 在环境验证输出中显示 Nebius API Key 状态

**新增内容**:
```bash
# Nebius API Key (for Qwen3-32B-fast on Nebius backend)
# If set, Qwen3-32B calls will automatically use Nebius API instead of local vLLM
# Get your key at: https://tokenfactory.nebius.com
export NEBIUS_API_KEY="v1...."

# 在验证输出中添加
echo "  NEBIUS_API_KEY:    ${NEBIUS_API_KEY:0:10}... (Qwen3-32B)"
```

---

## 📄 新增的文件

### 1. **NEBIUS_INTEGRATION.md** (详细文档)

**位置**: [NEBIUS_INTEGRATION.md](NEBIUS_INTEGRATION.md)

**内容**:
- Nebius API 集成的完整说明
- 配置步骤
- 使用场景
- 工作原理
- 故障排查
- 最佳实践

---

### 2. **test_nebius_api.py** (测试脚本)

**位置**: [test_nebius_api.py](test_nebius_api.py)

**功能**:
- 测试 Nebius API 连接
- 验证基本的 chat completion
- 测试 tool calling 功能
- 验证与 LLM_CALL.py 的集成

**使用方法**:
```bash
# 基本测试
python test_nebius_api.py

# 只测试 API，跳过集成测试
python test_nebius_api.py --skip-integration
```

---

### 3. **NEBIUS_CHANGES_SUMMARY.md** (本文档)

**位置**: [NEBIUS_CHANGES_SUMMARY.md](NEBIUS_CHANGES_SUMMARY.md)

**内容**: 所有修改的总结

---

## 📚 更新的文档

### 1. **QUICKSTART_LOCAL.md**

**位置**: [evaluation/tau2-bench/QUICKSTART_LOCAL.md](evaluation/tau2-bench/QUICKSTART_LOCAL.md)

**更新内容**:
- 在步骤 2 中添加了 `NEBIUS_API_KEY` 配置说明
- 在检查清单中添加了 Nebius API Key 验证项
- 添加了 Nebius API 的使用提示

---

## 🔍 工作流程

### 调用路径

当 τ²-Bench 评测运行时，调用 Qwen3-32B 的完整路径：

```
tau2-bench evaluation
    ↓
tau2/utils/llm_utils.py::generate()
    ↓
LLM_CALL.py::get_llm_response()
    ↓
检测 NEBIUS_API_KEY && 'qwen3-32b' in model
    ↓
┌─────────────────┬──────────────────┐
│  有 API Key     │   无 API Key     │
│                 │                  │
│  Nebius API     │   本地 vLLM      │
│  (新增逻辑)     │   (原有逻辑)     │
└─────────────────┴──────────────────┘
```

### 自动检测逻辑

```python
nebius_api_key = os.getenv("NEBIUS_API_KEY")
use_nebius = nebius_api_key and 'qwen3-32b' in model.lower()
```

**触发条件**:
1. `NEBIUS_API_KEY` 环境变量已设置
2. 模型名称包含 `qwen3-32b`（不区分大小写）

**匹配的模型名称**:
- `Qwen/Qwen3-32B` ✅
- `qwen3-32b` ✅
- `Qwen3-32B-Instruct` ✅
- `Qwen/Qwen2.5-32B` ❌ (不匹配)

---

## 🎯 使用场景

### 场景 1: τ²-Bench 中的 Expert-3

在 `tau2/utils/llm_utils.py` 第 489 行：

```python
if one_tool_call_arguments['expert']=='expert-3':
    mode_to_call = 'Qwen/Qwen3-32B'
```

当 agent 调用 expert-3 时，如果设置了 `NEBIUS_API_KEY`，会自动使用 Nebius API。

### 场景 2: 直接调用 Qwen3-32B

任何代码中调用：

```python
from LLM_CALL import get_llm_response

response = get_llm_response(
    model="Qwen/Qwen3-32B",
    messages=[...],
    temperature=1.0,
    model_type='vllm'
)
```

会自动检测并使用 Nebius API（如果设置了 key）。

---

## ✅ 验证步骤

### 1. 设置环境变量

```bash
export NEBIUS_API_KEY="v1.your_api_key_here"
source setup_envs.sh
```

### 2. 运行测试脚本

```bash
python test_nebius_api.py
```

预期输出：
```
============================================================
Testing Nebius API Integration for Qwen3-32B
============================================================

✓ NEBIUS_API_KEY is set: v1.CmQKHH...

[1/3] Initializing Nebius client...
  ✓ Client initialized successfully

[2/3] Testing basic chat completion...
  ✓ Received response: Hello from Nebius!
  ✓ Token usage: 15 prompt + 5 completion = 20 total

[3/3] Testing tool calling capability...
  ✓ Tool calling is supported
  ✓ Tool called: get_weather

============================================================
✅ All tests passed! Nebius API is working correctly.
============================================================
```

### 3. 运行 τ²-Bench 评测

```bash
cd evaluation/tau2-bench/
python run_local.py --agent-model $CKPT_DIR
```

在日志中应该能看到 Nebius API 的调用（不会有本地 vLLM 的 Qwen3-32B 连接错误）。

---

## 🔄 回退到本地 vLLM

如果需要临时回退到本地 vLLM：

```bash
# 方法 1: 取消设置环境变量
unset NEBIUS_API_KEY

# 方法 2: 运行时覆盖
NEBIUS_API_KEY="" python run_local.py --agent-model $CKPT_DIR

# 方法 3: 修改 setup_envs.sh
# 注释掉或删除 NEBIUS_API_KEY 行
```

---

## 🐛 已知问题和限制

### 1. 网络依赖

使用 Nebius API 需要稳定的网络连接。如果网络不稳定，建议使用本地 vLLM。

### 2. API 限流

Nebius API 可能有调用频率限制，高并发场景需要注意。

### 3. 模型版本

Nebius 使用 `Qwen/Qwen3-32B-fast`，可能与本地 vLLM 的模型版本略有差异。

### 4. 工具调用

需要验证 Nebius API 的 tool calling 功能是否与本地 vLLM 完全一致。

---

## 📊 性能对比

| 指标 | 本地 vLLM | Nebius API |
|------|-----------|------------|
| **延迟** | 低（本地推理） | 中（网络 + 推理） |
| **GPU 需求** | 高（需要部署 Qwen3-32B） | 无 |
| **成本** | GPU 资源成本 | API 调用成本 |
| **扩展性** | 受限于本地 GPU | 高（云端扩展） |
| **稳定性** | 高（本地控制） | 依赖网络 |

---

## 🔒 安全考虑

1. **API Key 管理**: 不要在公开仓库中提交 API key
2. **访问控制**: 定期轮换 API key
3. **数据隐私**: 了解 Nebius 的数据处理政策
4. **备份方案**: 保留本地 vLLM 配置作为备份

---

## 📞 相关资源

- **Nebius 官方文档**: https://tokenfactory.nebius.com/docs
- **ToolOrchestra README**: [README.md](README.md)
- **本地运行指南**: [evaluation/tau2-bench/RUN_LOCAL_GUIDE.md](evaluation/tau2-bench/RUN_LOCAL_GUIDE.md)
- **快速开始**: [evaluation/tau2-bench/QUICKSTART_LOCAL.md](evaluation/tau2-bench/QUICKSTART_LOCAL.md)

---

## 💡 总结

本次集成的核心目标是：**让 Qwen3-32B 的调用更加灵活，支持云端和本地两种部署方式**。

**关键优势**:
1. ✅ **零配置切换**: 只需设置/取消环境变量即可切换
2. ✅ **向后兼容**: 不影响现有的本地 vLLM 部署
3. ✅ **透明集成**: 应用层代码无需修改
4. ✅ **节省资源**: 无需在本地部署 Qwen3-32B

**实现原则**:
- 最小侵入性修改
- 保持向后兼容
- 优先使用云端 API（如果配置了）
- 自动回退到本地部署

---

**修改完成日期**: 2025-12-23
**修改人**: Claude Code Assistant
**版本**: v1.0

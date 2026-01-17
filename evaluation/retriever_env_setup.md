# Retriever 环境配置方案（Blackwell RTX 5090）

## ✅ 验证通过的配置

### 环境信息
- **GPU**: NVIDIA GeForce RTX 5090 (Blackwell sm_120)
- **Python**: 3.12
- **CUDA**: 12.8
- **PyTorch**: 2.7.1 (正式版，非 Nightly)
- **Container Image**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

### 完整安装步骤

```bash
# 0) 进入 conda（不要用 conda run）
# 根据 cursor chat 里的踩坑记录：用 conda.sh 激活最稳（而不是 source env/bin/activate）。
source /root/miniconda3/etc/profile.d/conda.sh

# 1) 创建环境
conda create -n retriever python=3.12 -y
conda activate retriever

# (可选) 把 HF cache 放在 /workspace，避免写爆 /root overlay
# export HF_HOME=/workspace/cache/huggingface

# 2) 先用 conda-forge 固定科学栈版本（关键：避免 pip 拉 numpy==2.x 导致 faiss/scipy ABI 冲突）
# 触发过的问题：torchvision 通过 pip 拉了 numpy==2.3.5，导致 scipy/transformers 报 ABI mismatch。
conda install -y -c conda-forge --force-reinstall \
  "numpy<2" "scipy<2" "scikit-learn<2" numpy-base

# 3) 安装 PyTorch 2.7.1 + CUDA 12.8 (支持 Blackwell)
# 注意：这里不要用 pip -U，也不要让 pip 升级 numpy（上一步已固定 numpy<2）。
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

# 4) 安装 Faiss-GPU (通过 Conda)
conda install -y -c pytorch -c nvidia faiss-gpu

# 5) 安装 Flash Attention 2 (预编译轮子)
pip install packaging ninja psutil
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 6) 安装其他依赖
pip install transformers datasets uvicorn fastapi tavily-python hf_transfer
```

---

## 🔧 Workaround：如果你已经踩到 `numpy/scipy` 与 `faiss-gpu` 冲突

来自 cursor chat 的可复现根因：`pip install torchvision ...` 会在新环境里拉最新 `numpy==2.x`，然后导致 `scipy`/`faiss-gpu`/`transformers` 出现 ABI 不一致（典型报错类似 `ValueError: All ufuncs must have type numpy.ufunc ...`）。

推荐修复方式（**不使用 conda run**）：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate retriever

# 1) 先把 pip 拉进来的 numpy/scipy/sklearn 清掉
python -m pip uninstall -y numpy scipy scikit-learn || true

# 2) 用 conda-forge 强制重装一致版本（numpy<2）
conda install -y -c conda-forge --force-reinstall \
  "numpy<2" "scipy<2" "scikit-learn<2" numpy-base

# 3) 快速 sanity check
python - <<'PY'
import numpy, scipy, sklearn
print("numpy", numpy.__version__, numpy.__file__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
PY
```

如果你后续又跑了 `pip install -U ...` 并再次把 numpy 升到 2.x，重复上面的修复即可。

---

## 🧩 Workaround：HF_HUB_ENABLE_HF_TRANSFER=1 导致启动崩溃

在 cursor chat 里出现过：
`ValueError: Fast download using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available`

你有两个选择：

1) 安装 `hf_transfer`：

```bash
pip install hf_transfer
```

2) 或禁用它：

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
```

## 🧪 验证结果

### PyTorch + CUDA
✅ CUDA 可用，张量运算正常

### Faiss-GPU
✅ GPU 索引创建和搜索正常
- 测试数据：100,000 个 64 维向量
- 查询：10,000 次 k=4 最近邻搜索

### Flash Attention
✅ 前向传播和反向传播正常
- 批大小：2
- 序列长度：128
- 注意力头数：4
- 头维度：64

### 其他依赖
✅ transformers, datasets, pyserini, uvicorn, fastapi, tavily 全部导入成功

## 📋 关键软件包版本

```
torch==2.7.1+cu128
torchvision==0.22.1+cu128
torchaudio==2.7.1+cu128
flash-attn==2.8.3
faiss-gpu==1.12.0
triton==3.3.1
```

## 💡 关键要点

1. **PyTorch 2.7.1** 是首个正式支持 Blackwell (sm_120) 的版本，无需使用 Nightly 版本
2. **CUDA 12.8** 是 Blackwell 架构所需的最低 CUDA 版本
3. **Faiss-GPU** 通过 Conda 安装可与 pip 安装的 PyTorch 共存
4. **Flash Attention** 需要使用与 PyTorch 2.7 匹配的预编译轮子，从源码编译会遇到文件系统链接错误
5. 该配置与同机器上的 vllm1 环境保持一致（PyTorch 2.7.1 + CUDA 12.8）

## 📝 测试脚本

测试脚本已保存在 `/workspace/test_env.py`，可随时运行验证环境：

```bash
python test_env.py
```

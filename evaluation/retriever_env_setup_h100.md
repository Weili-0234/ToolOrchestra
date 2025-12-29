# Retriever Environment Setup for H100 (CUDA 12.9)

## 适用环境
- GPU: NVIDIA H100 (Hopper Architecture)
- CUDA Runtime: 12.9
- Driver: 575.57.08+
- Python: 3.12

## 安装步骤

```bash
# 1) 创建环境
conda create -n retriever python=3.12 -y
conda activate retriever

# (可选) 设置 HuggingFace 缓存路径
export HF_HOME=/home/junxiong/haokang/hf_home

# 2) 先用 conda-forge 固定科学栈版本（关键：避免 numpy 2.x ABI 冲突）
# 这一步确保后续 pip 安装不会升级 numpy 到 2.x，导致 faiss/scipy ABI 不兼容
conda install -y -c conda-forge --force-reinstall \
  "numpy<2" "scipy<2" "scikit-learn<2" numpy-base

# 3) 安装 PyTorch 2.7.1 + CUDA 12.8
# 注意：PyTorch 2.7.1 支持 CUDA 12.8，在 12.9 runtime 上完全兼容
# CUDA 向后兼容：用 12.8 编译的 PyTorch 可以在 12.9 driver 上运行
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

# 4) 安装 Faiss-GPU (通过 Conda，自动匹配 CUDA)
# conda 的 faiss-gpu 会自动检测系统 CUDA 版本并安装兼容的二进制包
conda install -y -c pytorch -c nvidia faiss-gpu=1.9.0

# 5) 安装 Flash Attention 2
# 方案 A: 使用预编译轮子（推荐，适配 PyTorch 2.7 + CUDA 12.4）
pip install packaging ninja psutil
pip install flash-attn --no-build-isolation

# 方案 B: 如果方案 A 失败，从源码编译（需要约 10 分钟）
# export FLASH_ATTENTION_FORCE_BUILD=TRUE
# pip install flash-attn --no-build-isolation

# 6) 安装其他依赖
pip install transformers datasets pyserini uvicorn fastapi tavily-python hf_transfer

# 7) (可选) 如果需要更高性能，安装 FlashInfer
# pip install flashinfer-python -i https://flashinfer.ai/whl/cu128/torch2.7/
```

## 验证安装

```bash
python -c "
import torch
import faiss
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version (Compiled): {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'Faiss: {faiss.__version__}')

# 测试 faiss-gpu
res = faiss.StandardGpuResources()
print('✓ Faiss GPU initialized successfully')

# 测试 Flash Attention (可选)
try:
    from flash_attn import flash_attn_func
    print('✓ Flash Attention 2 available')
except ImportError:
    print('⚠ Flash Attention 2 not installed (optional)')
"
```

## 关键兼容性说明

### 1. CUDA 版本策略
- **编译版本 vs 运行时版本**：PyTorch 用 CUDA 12.8 **编译**，但可以在 CUDA 12.9 **runtime** 上运行
- NVIDIA 保证 CUDA 向后兼容（backward compatible）：新驱动支持旧 CUDA toolkit 编译的程序
- **不要**尝试安装 nightly build（`cu129`），不够稳定

### 2. Faiss-GPU 兼容性
- conda 的 `faiss-gpu=1.9.0` 已适配 CUDA 12.x 系列
- 如果 conda 安装失败，可以尝试：
  ```bash
  pip install faiss-gpu==1.9.0
  ```

### 3. Flash Attention 版本选择
| Flash Attn Version | PyTorch | CUDA | 状态 |
|-------------------|---------|------|------|
| 2.8.3 | 2.7.x | 12.8+ | ✓ 推荐 |
| 2.7.x | 2.4.x | 12.4 | ✓ 稳定（旧版） |
| 2.8.x | 2.5+ | 12.1+ | ✓ 通用 |

### 4. 常见问题排查

**Q: `ImportError: numpy.core.multiarray failed to import`**  
A: numpy 版本冲突。解决：
```bash
conda install -y -c conda-forge "numpy<2" --force-reinstall
pip install --no-deps transformers datasets  # 避免拉取新 numpy
```

**Q: `RuntimeError: CUDA error: no kernel image is available`**  
A: 驱动版本过低。确认：
```bash
nvidia-smi  # Driver 应 ≥ 575.57.08
```

**Q: Flash Attention 编译失败**  
A: 环境不完整。安装编译依赖：
```bash
conda install -y -c conda-forge gxx_linux-64 cuda-nvcc
export CUDA_HOME=/usr/local/cuda-12.9  # 或你的 CUDA 安装路径
```

## 性能优化建议

1. **启用 TF32**（H100 默认支持）：
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```

2. **使用 HF Transfer 加速下载**：
   ```bash
   export HF_HUB_ENABLE_HF_TRANSFER=1
   ```

3. **FlashInfer 替代 Flash Attention**（可选）：
   - 更快的推理速度（专为 Hopper/Ampere 优化）
   - 安装：`pip install flashinfer-python -i https://flashinfer.ai/whl/cu128/torch2.7/`

## 与 Blackwell 版本的差异

| 配置项 | Blackwell (CUDA 12.8) | Hopper (CUDA 12.9) |
|--------|----------------------|-------------------|
| PyTorch | 2.7.1 (cu128) | 2.7.1 (cu128) |
| Faiss | conda (自动) | conda (自动) |
| Flash Attn | 预编译 whl | `pip install` |
| 关键区别 | 需要特定 whl | 使用通用版本 |

## 后续步骤

安装完成后，可以运行以下命令测试检索功能：

```bash
cd evaluation
python retrieval_hle.py --test-mode
```

---

**维护者**: junxiong  
**最后更新**: 2025-12-28


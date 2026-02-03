### get started on the edge (5090)

#### plan B: vLLM 0.10.1（Blackwell GPU 支持）

```bash
conda create -n vllm1 python=3.12 -y
conda activate vllm1

# 卸载旧版本
pip uninstall -y vllm torch transformers

# 安装 PyTorch 2.7.1 (CUDA 12.8)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 安装 vLLM 0.10.1 和 transformers
pip install vllm==0.10.1 transformers

# vLLM 0.10.1+ 需要 hf_transfer 加速下载
pip install hf_transfer matplotlib
```

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

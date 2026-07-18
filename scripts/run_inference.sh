#!/usr/bin/env bash
# ============================================================
#  Qwen3.5-4B · vLLM 批量推理启动脚本
#
#  为什么需要这个 wrapper:
#  Qwen3.5 采用混合线性注意力 (Gated DeltaNet + full attention),
#  vLLM/Triton/FlashInfer 需要在运行时 JIT 编译若干 CUDA/C 内核。
#  本机默认 PATH 上没有 C 编译器,也没有系统级 CUDA_HOME,因此这里:
#    1. 指向 venv 内自带的 CUDA 13 工具链 (nvcc/头文件/库)
#    2. 用 conda 安装的 gcc-15 作为 host 编译器 (以标准名 cc/gcc/g++ 暴露)
#  经验证 nvcc 13.2 + gcc 15.2 可正常编译 sm_120 (RTX 5070 Ti) 内核。
#
#  用法:
#    ./run_inference.sh --backend vllm --limit 8          # 冒烟测试
#    ./run_inference.sh --backend vllm                     # 全量 1200 份
#    ./run_inference.sh --backend vllm --enforce_eager     # 关闭 CUDA graph
# ============================================================
set -euo pipefail

VENV="$HOME/.venvs/vllm35"
CU="$VENV/lib/python3.13/site-packages/nvidia/cu13"
SHIM="$VENV/toolshims"

# CUDA 工具链 (venv 内自带,供 FlashInfer / nvcc JIT 使用)
export CUDA_HOME="$CU"
export CUDA_PATH="$CU"

# host C/C++ 编译器 (供 Triton JIT 及 nvcc -ccbin 使用)
export CC="$SHIM/cc"
export CXX="$SHIM/c++"

# 标准编译器名 (cc/gcc/g++) + nvcc + ninja(venv/bin) 需在 PATH 上
export PATH="$SHIM:$CU/bin:$VENV/bin:$HOME/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH="$CU/lib:${LD_LIBRARY_PATH:-}"

# FlashInfer 的采样内核需要 JIT 编译,但 venv 内 pip 版 CUDA 各组件版本不一致
# (nvcc 13.2 / runtime 头 13.0 / cccl 13.3),CCCL 兼容性检查会失败。
# 这里关闭 FlashInfer 采样器,改用 vLLM 原生 (torch 预编译) 采样,无需 JIT。
# 注意力仍走 FlashInfer 预编译 cubin,不受影响。
export VLLM_USE_FLASHINFER_SAMPLER=0

# JIT 产物缓存,避免每次重编译
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton_pboc}"

cd "$(dirname "$0")"
exec "$VENV/bin/python" qwen_inference.py "$@"

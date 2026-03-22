#!/bin/bash
# 8 卡 B300 简单 DDP 训练
# 若 torchrun 子进程看不到 GPU，请先用单卡验证: python training/scripts/single_gpu_train.py
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/simple_ddp_train.py"
ENV="${NIGHTLY_ENV:-$HOME/miniconda3/envs/pt-nightly-cu130}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WANDB_MODE=offline

# 部分环境需以下变量才能正确初始化 CUDA
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MODULE_LOADING=LAZY

echo "========================================================================"
echo "  Simple 8-GPU DDP training"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "========================================================================"

"${ENV}/bin/torchrun" \
    --standalone \
    --nproc_per_node=8 \
    --master_port="${MASTER_PORT}" \
    "$TRAIN_SCRIPT"

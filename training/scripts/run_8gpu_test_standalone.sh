#!/bin/bash
# ============================================================================
# 8 GPU DDP 训练测试 — 单文件、无外部依赖
# 基于 run_simple_8gpu.sh，确保 CUDA 与 PyTorch 版本一致（Nightly cu130）
#
# 用法: bash run_8gpu_test_standalone.sh
#  或:  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_8gpu_test_standalone.sh
#  或:  NGPUS=4 bash run_8gpu_test_standalone.sh  # 4 卡测试
# ============================================================================

set -e

ENV="${NIGHTLY_ENV:-$HOME/miniconda3/envs/pt-nightly-cu130}"
NGPUS="${NGPUS:-8}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WANDB_MODE=offline

# LD_LIBRARY_PATH：仅加入 conda 环境 NCCL，由 PyTorch cu130 自带 CUDA 13.0
if [ -d "${ENV}/lib/python3.11/site-packages/nvidia/nccl/lib" ]; then
    export LD_LIBRARY_PATH="${ENV}/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
fi

# 部分环境需此变量才能正确初始化 CUDA
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MODULE_LOADING=LAZY

echo "=========================================================================="
echo "  ${NGPUS}-GPU DDP 测试 (standalone, 无外部依赖)"
echo "  ENV=$ENV"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================================================="

# 内联 Python 写入临时文件，由 torchrun 启动 8 进程
TMP_PY=$(mktemp /tmp/8gpu_test_XXXXXX.py)
trap "rm -f $TMP_PY" EXIT

cat > "$TMP_PY" << 'PYEOF'
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
        print(f"GPUs: {world_size} x {torch.cuda.get_device_name(0)}")

    model = torch.nn.Linear(256, 10).to(device)
    model = DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(10):
        x = torch.randn(32, 256, device=device)
        y = model(x)
        loss = y.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    dist.destroy_process_group()
    if rank == 0:
        print("8-GPU DDP test PASSED")

if __name__ == "__main__":
    main()
PYEOF

"${ENV}/bin/torchrun" \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    --master_port="${MASTER_PORT}" \
    "$TMP_PY"

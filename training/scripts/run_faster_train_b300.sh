#!/bin/bash
# ============================================================================
# B300 上运行 faster_train.py（基于成功示例：本地 pt-nightly-cu130 + 8 GPU）
#
# 若 CUDA 初始化失败，可尝试：
#   1. source 本脚本（与直接 python -c 相同的 shell 环境）：
#      source training/scripts/run_faster_train_b300.sh
#   2. 不修改 LD_LIBRARY_PATH，完全继承当前环境：
#      SKIP_LD_LIBRARY_PATH=1 USE_MANUAL_LAUNCH=1 bash ...
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${TRAIN_DIR}/faster_train.py"
YAML_CONFIG="${TRAIN_DIR}/config/exp1_dsai.yaml"
# 确保在 training 目录下运行，以便 Python 能找到 utils、networks 等模块
cd "$TRAIN_DIR"

# 使用本地 conda 环境（CUDA 已验证可用）
NIGHTLY_ENV="${NIGHTLY_ENV:-$HOME/miniconda3/envs/pt-nightly-cu130}"
export PATH="${NIGHTLY_ENV}/bin:${PATH}"

# GPU 选择（默认使用前 4 张：0,1,2,3；可改为 4,5,6,7 或 0-7 使用全部 8 张）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE=offline

# LD_LIBRARY_PATH：与 b300_training_nightly.sh 一致，仅加入 conda 环境 NCCL
# PyTorch cu130 自带 CUDA 13.0 运行时，不混入系统 cuda-13.1，确保 torch 与 CUDA 版本一致
if [ "${SKIP_LD_LIBRARY_PATH:-0}" != "1" ]; then
    NCCL_LIB="${NIGHTLY_ENV}/lib/python3.11/site-packages/nvidia/nccl/lib"
    export LD_LIBRARY_PATH="${NCCL_LIB}:${LD_LIBRARY_PATH:-}"
fi
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=5
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME="^lo,docker0"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512
export OMP_NUM_THREADS=8
# 部分环境需此变量才能正确检测 GPU（可选，若仍失败可取消注释）
# export PYTORCH_NVML_BASED_CUDA_CHECK=1
ulimit -l unlimited 2>/dev/null || true

# 参数
NGPUS="${NGPUS:-4}"
RUN_NUM="${RUN_NUM:-500}"
MASTER_PORT="${MASTER_PORT:-29600}"
# 设为 1 时用 shell 直接启动多进程（绕过 torchrun 子进程环境问题）
USE_MANUAL_LAUNCH="${USE_MANUAL_LAUNCH:-0}"

TRAIN_ARGS=(
    --yaml_config="$YAML_CONFIG"
    --run_num="$RUN_NUM"
    --amp-dtype fp16
    --ddp-bucket-cap-mb 512
    --ddp-fp16-compress
    --batch-size-override 8
    --epochs 1
    --max-steps 50
    --fresh_start
    --ddp-static-graph
    --max-grad-norm 1.0
    --log-every-n-steps 100
    --metrics-every 500
    --accum-steps 1
)

echo "========================================================================"
echo "  B300 faster_train.py — $NGPUS GPU"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Launch: $([ "$USE_MANUAL_LAUNCH" = "1" ] && echo "manual (shell spawn)" || echo "torchrun")"
echo "  $(date)"
echo "========================================================================"

# 可选：预检查 torchrun 子进程能否看到 GPU（RUN_FASTER_TRAIN_DEBUG=1 时执行）
if [ "${RUN_FASTER_TRAIN_DEBUG:-0}" = "1" ]; then
    echo ">>> 预检查: torchrun 1 进程 CUDA 测试 ..."
    "${NIGHTLY_ENV}/bin/torchrun" --standalone --nproc_per_node=1 \
        "${SCRIPT_DIR}/cuda_torchrun_test.py" || { echo ">>> 预检查失败，请检查 LD_LIBRARY_PATH 和 CUDA 安装"; exit 1; }
    echo ""
fi

if [ "$USE_MANUAL_LAUNCH" = "1" ]; then
    # 诊断：python -c vs python .py 文件
    echo ">>> 诊断1: python -c 能否看到 GPU?"
    RANK=0 LOCAL_RANK=0 "${NIGHTLY_ENV}/bin/python" -c "
import torch
n = torch.cuda.device_count()
print(f'>>> 诊断1 结果: device_count={n}')
assert n > 0
" || { echo ">>> 诊断1 失败"; exit 1; }
    echo ">>> 诊断2: python minimal_cuda_test.py 能否看到 GPU?"
    RANK=0 LOCAL_RANK=0 "${NIGHTLY_ENV}/bin/python" "${TRAIN_DIR}/minimal_cuda_test.py" || { echo ">>> 诊断2 失败: .py 文件运行 CUDA 失败"; exit 1; }
    echo ">>> 使用 manual launch（shell 直接 spawn，规避 torchrun 子进程 CUDA 问题）"
    export MASTER_ADDR="${MASTER_ADDR:-localhost}"
    export MASTER_PORT
    export WORLD_SIZE="$NGPUS"
    for r in $(seq 0 $((NGPUS - 1))); do
        RANK=$r LOCAL_RANK=$r "${NIGHTLY_ENV}/bin/python" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}" "$@" &
    done
    wait
    ret=$?
    [[ "${BASH_SOURCE[0]}" != "${0}" ]] && return $ret 2>/dev/null || exit $ret
fi

"${NIGHTLY_ENV}/bin/torchrun" \
    --standalone \
    --nproc_per_node="$NGPUS" \
    --master_port="$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    "${TRAIN_ARGS[@]}" \
    "$@"

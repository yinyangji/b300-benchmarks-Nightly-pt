#!/bin/bash
# ============================================================
# NVLink 5 Stress Benchmark — NVIDIA B300 SXM6 (4 GPU)
#
# Usage:
#   bash benchmarks/run_nvlink_stress.sh              # GPUs 4,5,6,7 (default)
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash benchmarks/run_nvlink_stress.sh
#
# Uses the pt-nightly-cu130 env (NCCL 2.29.3) for best results.
# Falls back to conda s2s env (NCCL 2.28.9) if nightly not found.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

# ── Pick Python env (prefer nightly for NCCL 2.29.3) ─────────────────────────
NIGHTLY_ENV="/scratch/rdesouz4/envs/pt-nightly-cu130"
CONDA_ENV="/scratch/rdesouz4/envs/s2s"

if [ -x "${NIGHTLY_ENV}/bin/torchrun" ]; then
    TORCHRUN="${NIGHTLY_ENV}/bin/torchrun"
    PYTHON="${NIGHTLY_ENV}/bin/python"
    ENV_NAME="pt-nightly-cu130 (NCCL 2.29.3)"
    export CONDA_ENV="${NIGHTLY_ENV}"
elif [ -x "${CONDA_ENV}/bin/torchrun" ]; then
    TORCHRUN="${CONDA_ENV}/bin/torchrun"
    PYTHON="${CONDA_ENV}/bin/python"
    ENV_NAME="s2s conda (NCCL 2.28.9)"
    export CONDA_ENV="${CONDA_ENV}"
else
    TORCHRUN="$(which torchrun)"
    PYTHON="$(which python3)"
    ENV_NAME="system"
fi

# ── NCCL tuning for NVLink 5 / Blackwell ─────────────────────────────────────
export NCCL_BUFFSIZE=16777216
export NCCL_MAX_NCHANNELS=32
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1      # force NVLink, disable InfiniBand

# ── Run ───────────────────────────────────────────────────────────────────────
mkdir -p "${REPO_ROOT}/results"

echo ""
echo "================================================================"
echo "  NVLink 5 Stress Benchmark — NVIDIA B300 SXM6"
echo "  GPUs  : ${CUDA_VISIBLE_DEVICES} (${N_GPUS} GPUs)"
echo "  Env   : ${ENV_NAME}"
echo "  $(date)"
echo "================================================================"
echo ""

"${TORCHRUN}" \
    --nproc_per_node="${N_GPUS}" \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    "${SCRIPT_DIR}/nvlink_stress_b300.py" \
    2>&1 | tee "${REPO_ROOT}/results/nvlink_stress_b300.log"

echo ""
echo "================================================================"
echo "  Log    : results/nvlink_stress_b300.log"
echo "  JSON   : results/nvlink_stress_b300.json"
echo "================================================================"

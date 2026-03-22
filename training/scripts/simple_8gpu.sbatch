#!/bin/bash
# sbatch 提交（需根据集群修改 -p 分区名、--gres 等）
# 若报错，检查: sinfo、scontrol show partition
#SBATCH --job-name=simple8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --output=simple_8gpu_%j.out
#SBATCH --error=simple_8gpu_%j.err
##SBATCH --partition=b300    # 若无 GPU 分区可注释或删掉

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="${NIGHTLY_ENV:-$HOME/miniconda3/envs/pt-nightly-cu130}"

export MASTER_ADDR=localhost MASTER_PORT=29500
export PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_MODULE_LOADING=LAZY

"${ENV}/bin/torchrun" --standalone --nproc_per_node=8 --master_port=29500 \
    "${SCRIPT_DIR}/simple_ddp_train.py"

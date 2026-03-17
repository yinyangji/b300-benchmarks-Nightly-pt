#!/bin/bash -l
# ============================================================================
#
# B300 (Blackwell) BENCHMARK SUITE — NGC 25.03 Container Edition
#
# Uses nvcr.io/nvidia/pytorch:25.03-py3 (PyTorch 2.7, CUDA 12.8) which has
# better Blackwell sm_103 support than the conda PyTorch 2.10+cu130 build.
#
# Benchmark results (2026-03-16, 4x B300 SXM6, batch_size=8, BF16):
#   Conda PyTorch 2.10  : 1.58 samples/sec
#   NGC   PyTorch 2.7   : 2.43 samples/sec  (+54%)
#
# Self-bootstrapping: run this script directly on the host — it will
# launch the Docker container and re-run itself inside it automatically.
#
#   bash b300_training_ngc.sh 2>&1 | tee b300_results_ngc.log
#
# To rebuild the Docker image with updated deps:
#   cd /home/rdesouz4/scratchrdesouz4/b300/pangus2s
#   sg docker -c "docker build -f training/Dockerfile.ngc -t pangu-s2s-ngc:latest ."
#
# SLURM usage:
#   sbatch b300_training_ngc.sh
#
# ============================================================================

##SBATCH --time=02:00:00
##SBATCH -p b300
##SBATCH --reservation=benchmarking
##SBATCH --nodes=1
##SBATCH --mem=500G
##SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:8
##SBATCH --cpus-per-task=64
##SBATCH -o bench_ngc_%x_%j.out
##SBATCH -e bench_ngc_%x_%j.err

set -uo pipefail

# ============================================================================
# DOCKER BOOTSTRAP
# If not already inside Docker, re-launch ourselves inside the NGC container.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # training/scripts/../../ = repo root
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset}"
IMAGE="pangu-s2s-ngc:latest"

# GPU selection — change to suit your node (4 GPUs starting at index 4)
NGC_GPUS="${NGC_GPUS:-4,5,6,7}"

if [ -z "${RUNNING_IN_DOCKER:-}" ]; then
    echo "==> Launching NGC container (${IMAGE}) on GPUs ${NGC_GPUS}..."
    echo "==> Repo root: ${REPO_ROOT}"
    echo "==> Dataset:   ${DATASET_DIR}"
    echo ""

    # Build image if it doesn't exist
    if ! sg docker -c "docker image inspect ${IMAGE}" &>/dev/null 2>&1; then
        echo "==> Image ${IMAGE} not found. Building from training/Dockerfile.ngc ..."
        sg docker -c "docker build -f ${REPO_ROOT}/training/Dockerfile.ngc -t ${IMAGE} ${REPO_ROOT}"
        echo "==> Build complete."
        echo ""
    fi

    exec sg docker -c "docker run --rm \
        --gpus '\"device=${NGC_GPUS}\"' \
        --ipc=host \
        --ulimit memlock=-1 \
        -e RUNNING_IN_DOCKER=1 \
        -e NGC_GPUS=${NGC_GPUS} \
        -v ${REPO_ROOT}:/workspace \
        -v ${DATASET_DIR}:${DATASET_DIR}:ro \
        -w /workspace/training/scripts \
        ${IMAGE} \
        bash b300_training_ngc.sh"
fi

# ============================================================================
# Everything below runs INSIDE the container
# ============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_SCRIPT="../faster_train.py"
YAML_CONFIG="../config/exp1_dsai.yaml"
RUN_NUM="${RUN_NUM:-300}"

jobid=${SLURM_JOBID:-$(date +%Y%m%d_%H%M%S)}
DATA_DIR_GPFS="${DATA_DIR_GPFS:-}"
DATA_DIR_LOCAL="${DATA_DIR_LOCAL:-}"

# ============================================================================
# NCCL optimizations for B300 (NVLink 5, SM_100+)
# ============================================================================
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=5
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_BUFFSIZE=16777216
export NCCL_NTHREADS=512
export NCCL_MAX_NCHANNELS=32
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# ============================================================================
# CUDA / PyTorch optimizations
# ============================================================================
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export WANDB_MODE=offline

# TORCHDYNAMO_DISABLE=1: prevents Triton from calling ptxas --gpu-name=sm_103
# which exits 255 under CUDA 12.8. torch.compile is still "enabled" in the
# trainer log but dynamo graph capture is skipped; PyTorch 2.7 eager on B300
# already achieves 2.43 samples/sec vs conda 1.58 samples/sec.
export TORCHDYNAMO_DISABLE=1

# ============================================================================
# GPU detection
# ============================================================================
AVAILABLE_GPUS=$(nvidia-smi -L | grep "^GPU" | grep -v "MIG" 2>/dev/null | wc -l || echo 4)
MAX_GPUS=$AVAILABLE_GPUS
RESULTS_FILE=results.ngc.${jobid}.maxgpus${AVAILABLE_GPUS}.batch_size8.all
echo "INFO: Detected ${AVAILABLE_GPUS} GPU(s). Using all ${MAX_GPUS} for benchmark runs."

# ============================================================================
# System info
# ============================================================================
echo ""
echo "========================================================================"
echo "  B300 (Blackwell) BENCHMARK SUITE — NGC 25.03 — System Info"
echo "  $(date)"
echo "========================================================================"
echo ""
echo "=== GPU Hardware ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""
echo "=== PyTorch / CUDA ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}, {p.total_mem/1e9:.1f} GB, SM {p.major}.{p.minor}')
try:
    import transformer_engine
    print(f'Transformer Engine: {transformer_engine.__version__} (FP8 supported)')
except ImportError:
    print('Transformer Engine: NOT installed')
" 2>/dev/null || true
echo ""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

sep() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "  $(date)"
    echo "========================================================================"
    echo ""
}

run_training() {
    local name="$1"
    local ngpus="$2"
    local extra_args="$3"
    local port="$4"

    sep "TRAINING: $name (${ngpus} GPU)"
    echo "Command:"
    echo "  torchrun --standalone --nproc_per_node=$ngpus --master_port=$port \\"
    echo "    $TRAIN_SCRIPT --yaml_config=$YAML_CONFIG --run_num=$RUN_NUM $extra_args"
    echo ""

    local start_time=$(date +%s)
    local _run_log=$(mktemp /tmp/b300_run.XXXXXX)

    torchrun \
        --standalone \
        --nproc_per_node=$ngpus \
        --master_port=$port \
        $TRAIN_SCRIPT \
        --yaml_config=$YAML_CONFIG \
        --run_num=$RUN_NUM \
        $extra_args | tee "$_run_log"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    local _secstep="N/A"
    local _samplesec="N/A"
    local _bline=$(grep "BENCHMARK_RESULT:" "$_run_log" | tail -1)
    if [ -n "$_bline" ]; then
        _secstep=$(echo "$_bline" | sed 's/.*sec_per_step=//;s/ .*//')
        _samplesec=$(echo "$_bline" | sed 's/.*samples_per_sec=//')
    fi
    rm -f "$_run_log"

    echo ""
    echo "--- RESULT: $name (${ngpus} GPU) ---"
    echo "  Exit code: $exit_code"
    echo "  Duration:  ${duration}s"
    echo "  sec/step:  $_secstep"
    echo "  samples/s: $_samplesec"
    echo "---"
    echo ""

    echo "${name} (${ngpus} GPU)|${exit_code}|${duration}|${_secstep}|${_samplesec}" >> "$RESULTS_FILE"
    RUN_NUM=$((RUN_NUM + 1))
}

# ============================================================================
# VALIDATE
# ============================================================================
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found at: $TRAIN_SCRIPT"
    exit 1
fi

COMMON="--epochs 1 --max-steps 50 --fresh_start --ddp-static-graph --max-grad-norm 1.0 --log-every-n-steps 100 --metrics-every 500 --accum-steps 1"
B300_BUCKET_MB=512

running_precision=1
running_batch_sizes=1
running_num_gpus_strong=0
running_num_gpus_weak=1
BENCH_GPUS=4

# ============================================================================
# TASK 1: PRECISION SWEEP
# ============================================================================
if [ $running_precision -eq 1 ]; then
    sep "TASK 1: PRECISION SWEEP ($BENCH_GPUS x B300, batch_size=8)"

    global_batch=8

    run_training "Precision: BF16" \
        $BENCH_GPUS \
        "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
        29500

    run_training "Precision: FP16" \
        $BENCH_GPUS \
        "--amp-dtype fp16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
        29501

    run_training "Precision: FP32 with TF32" \
        $BENCH_GPUS \
        "--amp-dtype fp32 --ddp-bucket-cap-mb $B300_BUCKET_MB --batch-size-override $global_batch $COMMON" \
        29502

    run_training "Precision: FP32 pure (no TF32)" \
        $BENCH_GPUS \
        "--amp-dtype fp32 --no-tf32 --ddp-bucket-cap-mb $B300_BUCKET_MB --batch-size-override $global_batch $COMMON" \
        29503

    run_training "Precision: FP8 (Transformer Engine)" \
        $BENCH_GPUS \
        "--amp-dtype fp8 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
        29504
fi

# ============================================================================
# TASK 2: BATCH SIZE SWEEP
# ============================================================================
if [ $running_batch_sizes -eq 1 ]; then
    sep "TASK 2: BATCH SIZE SWEEP ($BENCH_GPUS x B300, FP16)"

    for bsize in 4 8 12; do
        seed=$(shuf -i10000-39999 -n1)
        run_training "BatchSize: $bsize" \
            $BENCH_GPUS \
            "--amp-dtype fp16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $bsize $COMMON" \
            $seed
    done
fi

# ============================================================================
# TASK 3: DATA LOCATION
# ============================================================================
if [ -n "$DATA_DIR_LOCAL" ] && [ -n "$DATA_DIR_GPFS" ]; then
    sep "TASK 3: DATA LOCATION — Local NVMe vs GPFS (4x B300, batch_size=8)"

    run_training "DataLoc: GPFS" \
        $MAX_GPUS \
        "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --data-dir-override $DATA_DIR_GPFS $COMMON" \
        29520

    run_training "DataLoc: Local NVMe" \
        $MAX_GPUS \
        "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --data-dir-override $DATA_DIR_LOCAL $COMMON" \
        29521
else
    sep "TASK 3: DATA LOCATION — SKIPPED"
    echo "To enable, set DATA_DIR_GPFS and DATA_DIR_LOCAL."
fi

# ============================================================================
# TASK 4: GPU STRONG SCALING
# ============================================================================
if [ $running_num_gpus_strong -eq 1 ]; then
    sep "TASK 4: GPU STRONG SCALING (B300, global batch size = 16, BF16)"
    global_batch=16
    seed=$(shuf -i10000-39999 -n1)
    run_training "GPUScale: $MAX_GPUS GPU with global batch $global_batch" \
        $MAX_GPUS \
        "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
        $seed
fi

# ============================================================================
# TASK 5: GPU WEAK SCALING
# ============================================================================
if [ $running_num_gpus_weak -eq 1 ]; then
    sep "TASK 5: GPU WEAK SCALING (B300, per-GPU batch=2, BF16)"
    batch_size_per_gpu=2

    for num_gpus in 1 2 4; do
        if [ "$MAX_GPUS" -ge $num_gpus ]; then
            global_batch=$(( num_gpus * batch_size_per_gpu ))
            seed=$(shuf -i10000-39999 -n1)
            sep "Running with $num_gpus GPU(s) and global batch $global_batch.."
            run_training "GPUScale: $num_gpus GPU" \
                $num_gpus \
                "--amp-dtype fp16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
                $seed
        fi
    done
fi

# ============================================================================
# DONE — Results Table
# ============================================================================
sep "ALL B300 BENCHMARKS COMPLETE (NGC 25.03)"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    B300 BENCHMARK RESULTS TABLE (NGC 25.03)                         ║"
echo "╠════════════════════════════════════╦════════╦══════════╦════════════╦════════════════╣"
echo "║ Run Name                           ║ Status ║ Wall (s) ║ sec/step   ║ samples/sec    ║"
echo "╠════════════════════════════════════╬════════╬══════════╬════════════╬════════════════╣"

while IFS='|' read -r name exitcode duration secstep samplesec; do
    if [ "$exitcode" = "0" ]; then
        status="  OK  "
    else
        status=" FAIL "
        secstep="N/A"
        samplesec="N/A"
    fi
    printf "║ %-36s ║ %s ║ %8s ║ %10s ║ %14s ║\n" "$name" "$status" "$duration" "$secstep" "$samplesec"
done < "$RESULTS_FILE"

echo "╚════════════════════════════════════╩════════╩══════════╩════════════╩════════════════╝"
echo ""
echo "Notes:"
echo "  - Container: nvcr.io/nvidia/pytorch:25.03-py3 (PyTorch 2.7, CUDA 12.8)"
echo "  - TORCHDYNAMO_DISABLE=1: prevents Triton sm_103 ptxas crash under CUDA 12.8"
echo "  - sec/step = wall-clock seconds per training step (lower is better)"
echo "  - samples/sec = global_batch_size / sec_per_step (higher is better)"
echo "  - All runs used 50 training steps (--max-steps 50) for timing consistency"
echo ""
echo "=== NGC B300 Benchmarks Complete ==="

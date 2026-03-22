#!/bin/bash -l
# ============================================================================
#
# B300 (Blackwell) BENCHMARK SUITE — PyTorch Nightly + CUDA 13.0
#
# Tests whether nightly PyTorch includes native sm_103 cubins, working
# torch.compile/Triton for B300, and FP4 via torchao (14 PFLOPS B300 feature).
#
# Runs the same Tasks 1/2/5 as b300_training_dsai.sh for direct comparison,
# plus:
#   TASK 6 — torch.compile sweep (if Triton sm_103 works in nightly)
#   TASK 7 — FP4 via torchao (B300 flagship precision, 14 PFLOPS)
#
# Self-bootstrapping: creates conda env pt-nightly-cu130 if not present.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash b300_training_nightly.sh 2>&1 | tee nightly_results.log
#
# 若出现 FileNotFoundError: .../1979_0000.h5，需指定数据集路径：
#   DATASET_DIR=/path/to/era5_dataset bash b300_training_nightly.sh
#
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../faster_train.py"
YAML_CONFIG="${SCRIPT_DIR}/../config/exp1_dsai.yaml"

# 数据集路径：exp1_dsai.yaml 中 data_dir 默认为 rdesouz4 用户路径，需通过 DATASET_DIR 覆盖
# Usage: DATASET_DIR=/path/to/era5_dataset bash b300_training_nightly.sh
DATASET_DIR="${DATASET_DIR:-}"

NIGHTLY_ENV="${NIGHTLY_ENV:-/scratch/rdesouz4/envs/pt-nightly-cu130}"
PYTHON="${NIGHTLY_ENV}/bin/python"
PIP="${NIGHTLY_ENV}/bin/pip"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export WANDB_MODE=offline

RUN_NUM="${RUN_NUM:-400}"
jobid="${SLURM_JOBID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_FILE="${SCRIPT_DIR}/results.nightly.${jobid}.txt"

# ============================================================================
# STEP 0 — Create conda env if missing
# ============================================================================

if [ ! -f "${PYTHON}" ]; then
    echo "==> Conda env not found at ${NIGHTLY_ENV}"
    echo "==> Creating pt-nightly-cu130 env with Python 3.11 ..."
    set +u
    module load anaconda3/2024.02-1 2>/dev/null || true
    conda create -y -p "${NIGHTLY_ENV}" python=3.11
    set -u
    echo "==> Installing PyTorch nightly cu130 ..."
    "${NIGHTLY_ENV}/bin/pip" install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu130 \
        --quiet
    echo "==> Installing Pangu S2S + benchmark dependencies ..."
    "${NIGHTLY_ENV}/bin/pip" install --quiet \
        wandb einops timm "ruamel.yaml" xarray h5py \
        cartopy cftime pandas psutil tqdm netcdf4 h5netcdf \
        torchao
    echo "==> Env ready."
fi

# ============================================================================
# STEP 1 — Validate sm_103 support (key question)
# ============================================================================

echo ""
echo "========================================================================"
echo "  PyTorch Nightly — sm_103 / B300 Validation"
echo "  $(date)"
echo "========================================================================"

SM103_NATIVE=$("${PYTHON}" -c "
import torch
arch_list = torch.cuda.get_arch_list()
sm103 = 'sm_103' in arch_list
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA:           {torch.version.cuda}')
print(f'GPU:            {torch.cuda.get_device_name(0)}')
print(f'Arch list:      {arch_list}')
print(f'sm_103 native:  {sm103}')
print(f'SM103_RESULT:   {\"YES\" if sm103 else \"NO\"}')
" 2>&1 | tee /dev/stderr | grep "SM103_RESULT:" | awk '{print $2}')

echo ""
if [ "${SM103_NATIVE:-NO}" = "YES" ]; then
    echo ">>> sm_103 IS in arch list — nightly has native B300 cubins!"
    echo ">>> torch.compile / Triton sm_103 test will proceed."
    COMPILE_FLAG="--torch-compile"
else
    echo ">>> sm_103 NOT in arch list — nightly still uses sm_100 fallback."
    echo ">>> torch.compile test will run but may be slow or fall back."
    COMPILE_FLAG=""
fi
echo ""

# Check if Triton can target sm_103
echo "--- Triton / ptxas sm_103 probe ---"
"${PYTHON}" -c "
import subprocess, sys
r = subprocess.run(['ptxas', '--gpu-name=sm_103', '--version'],
                   capture_output=True, text=True)
if r.returncode == 0:
    print('ptxas sm_103: OK —', r.stdout.strip().split('\n')[0])
else:
    print('ptxas sm_103: FAIL (exit', r.returncode, ') — torch.compile will fall back')
" 2>&1 || true
echo ""

# ============================================================================
# NCCL / CUDA env (same tuning as dsai script)
# ============================================================================

CONDA_ENV="${NIGHTLY_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=5
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME="^lo,docker0"
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_BUFFSIZE=16777216
export NCCL_NTHREADS=512
export NCCL_MAX_NCHANNELS=32
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF

export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512
export OMP_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited

# Note: TORCHDYNAMO_DISABLE is intentionally NOT set here — we want torch.compile to run.
# If Triton crashes on sm_103, re-run with: TORCHDYNAMO_DISABLE=1 bash b300_training_nightly.sh

# ============================================================================
# Helpers
# ============================================================================

sep() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "  $(date)"
    echo "========================================================================"
    echo ""
}

BENCH_GPUS=4
B300_BUCKET_MB=512
DATA_DIR_ARGS=""
if [ -n "${DATASET_DIR}" ]; then
    DATA_DIR_ARGS="--data-dir-override ${DATASET_DIR}"
    echo "==> Using DATASET_DIR: ${DATASET_DIR}"
fi
COMMON="--epochs 1 --max-steps 50 --fresh_start --ddp-static-graph --max-grad-norm 1.0 --log-every-n-steps 100 --metrics-every 500 --accum-steps 1 ${DATA_DIR_ARGS}"

run_training() {
    local name="$1"
    local ngpus="$2"
    local extra_args="$3"
    local port="$4"

    sep "TRAINING: $name (${ngpus} GPU)"
    echo "  torchrun --standalone --nproc_per_node=$ngpus --master_port=$port \\"
    echo "    $TRAIN_SCRIPT --yaml_config=$YAML_CONFIG --run_num=$RUN_NUM $extra_args"
    echo ""

    local start_time end_time duration
    start_time=$(date +%s)
    local _run_log
    _run_log=$(mktemp /tmp/nightly_run.XXXXXX)

    "${NIGHTLY_ENV}/bin/torchrun" \
        --standalone \
        --nproc_per_node="$ngpus" \
        --master_port="$port" \
        "$TRAIN_SCRIPT" \
        --yaml_config="$YAML_CONFIG" \
        --run_num="$RUN_NUM" \
        $extra_args 2>&1 | tee "$_run_log"

    local exit_code="${PIPESTATUS[0]}"
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    local _secstep="N/A" _samplesec="N/A" _bline
    _bline=$(grep "BENCHMARK_RESULT:" "$_run_log" | tail -1)
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

    echo "${name} (${ngpus} GPU)|${exit_code}|${duration}|${_secstep}|${_samplesec}" >> "$RESULTS_FILE"
    RUN_NUM=$((RUN_NUM + 1))
}

# ============================================================================
# System info
# ============================================================================

echo ""
echo "========================================================================"
echo "  B300 Nightly BENCHMARK SUITE — System Info"
echo "  $(date)"
echo "========================================================================"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""
"${PYTHON}" -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}, {p.total_mem/1e9:.1f} GB, SM {p.major}.{p.minor}')
try:
    import torchao; print(f'torchao: {torchao.__version__}')
except ImportError:
    print('torchao: NOT installed (FP4 task will be skipped)')
try:
    import transformer_engine as te; print(f'TransformerEngine: {te.__version__}')
except ImportError:
    print('TransformerEngine: NOT installed')
" 2>/dev/null || true
echo ""

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

# 数据集预检查（exp1_dsai.yaml 默认 data_dir 为 rdesouz4 用户路径，若不存在需设置 DATASET_DIR）
DATA_DIR_TO_CHECK="${DATASET_DIR:-/home/rdesouz4/scratchrdesouz4/b300/pangus2s/dataset}"
if [ ! -f "${DATA_DIR_TO_CHECK}/1979_0000.h5" ]; then
    echo "ERROR: Dataset file not found: ${DATA_DIR_TO_CHECK}/1979_0000.h5"
    echo "  exp1_dsai.yaml 默认路径可能不存在于当前节点。请设置 DATASET_DIR："
    echo "  DATASET_DIR=/path/to/era5_dataset bash $0"
    exit 1
fi
echo "==> Dataset: ${DATA_DIR_TO_CHECK}"

# ============================================================================
# TASK 1 — Precision Sweep (4 GPU, batch_size=8)
# Same as dsai/ngc — direct comparison baseline
# ============================================================================

sep "TASK 1: PRECISION SWEEP (${BENCH_GPUS}x B300, batch_size=8) — NIGHTLY"

run_training "Nightly: BF16" \
    $BENCH_GPUS \
    "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override 8 $COMMON" \
    29600

run_training "Nightly: FP16" \
    $BENCH_GPUS \
    "--amp-dtype fp16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override 8 $COMMON" \
    29601

run_training "Nightly: FP32+TF32" \
    $BENCH_GPUS \
    "--amp-dtype fp32 --ddp-bucket-cap-mb $B300_BUCKET_MB --batch-size-override 8 $COMMON" \
    29602

run_training "Nightly: FP8 (TE)" \
    $BENCH_GPUS \
    "--amp-dtype fp8 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override 8 $COMMON" \
    29603

# ============================================================================
# TASK 2 — Batch Size Sweep (4 GPU, FP16)
# ============================================================================

sep "TASK 2: BATCH SIZE SWEEP (${BENCH_GPUS}x B300, FP16) — NIGHTLY"

for bsize in 4 8 12; do
    seed=$(shuf -i10000-39999 -n1)
    run_training "Nightly: BatchSize $bsize" \
        $BENCH_GPUS \
        "--amp-dtype fp16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $bsize $COMMON" \
        "$seed"
done

# ============================================================================
# TASK 5 — GPU Weak Scaling (per-GPU batch=2, FP16)
# ============================================================================

sep "TASK 5: GPU WEAK SCALING (per-GPU batch=2, FP16) — NIGHTLY"

for num_gpus in 1 2 4; do
    global_batch=$(( num_gpus * 2 ))
    seed=$(shuf -i10000-39999 -n1)
    run_training "Nightly: Weak Scale ${num_gpus}GPU" \
        "$num_gpus" \
        "--amp-dtype fp16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
        "$seed"
done

# ============================================================================
# TASK 6 — torch.compile (new vs dsai/ngc — may work with nightly Triton)
# ============================================================================

sep "TASK 6: torch.compile SWEEP (${BENCH_GPUS}x B300) — NIGHTLY ONLY"

echo "Note: If ptxas fails for sm_103, runs will fall back or crash."
echo "      Re-run with TORCHDYNAMO_DISABLE=1 to get the uncompiled baseline."
echo ""

# BF16 with compile
run_training "Nightly: BF16 + compile" \
    $BENCH_GPUS \
    "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override 8 --torch-compile $COMMON" \
    29610

# FP16 with compile
run_training "Nightly: FP16 + compile" \
    $BENCH_GPUS \
    "--amp-dtype fp16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override 8 --torch-compile $COMMON" \
    29611

# ============================================================================
# TASK 7 — FP4 via torchao (B300 flagship: 14 PFLOPS)
# ============================================================================

sep "TASK 7: FP4 via torchao (B300 flagship — 14 PFLOPS Tensor Cores)"

"${PYTHON}" -c "import torchao" 2>/dev/null && FP4_AVAILABLE=1 || FP4_AVAILABLE=0

if [ "$FP4_AVAILABLE" = "1" ]; then
    echo "torchao found — running FP4 micro-benchmark ..."
    "${PYTHON}" - <<'PYEOF'
import torch, time, os
import torchao

device = "cuda:0"
dtype  = torch.float16   # input dtype
M, N, K = 4096, 4096, 4096

try:
    # torchao float4 quantization (requires sm_100+/sm_103)
    from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
    from torchao.prototype.quantization.fp4 import fp4_weight_only

    a = torch.randn(M, K, device=device, dtype=dtype)
    lin = torch.nn.Linear(K, N, bias=False, dtype=dtype).to(device)

    # Quantize weights to FP4
    quantize_(lin, fp4_weight_only())
    lin_compiled = torch.compile(lin) if os.environ.get("TORCHDYNAMO_DISABLE") != "1" else lin

    # Warmup
    for _ in range(5):
        _ = lin_compiled(a)
    torch.cuda.synchronize()

    N_ITERS = 50
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = lin_compiled(a)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tflops = 2 * M * N * K * N_ITERS / elapsed / 1e12
    print(f"FP4 GEMM ({M}x{N}x{K}): {tflops:.1f} TFLOPS  (B300 peak: ~14000 TFLOPS)")
    print(f"BENCHMARK_RESULT: precision=FP4_torchao tflops={tflops:.1f}")

except Exception as e:
    print(f"FP4 test failed: {e}")
    print("Note: FP4 requires sm_100+ cubins and CUTLASS 4.4+ — may need newer nightly.")
PYEOF
    echo "FP4 (${BENCH_GPUS} GPU)|0|0|N/A|see_above" >> "$RESULTS_FILE"
else
    echo "torchao not installed — skipping FP4 task."
    echo "Install with: pip install torchao"
fi

# ============================================================================
# RESULTS TABLE
# ============================================================================

sep "ALL NIGHTLY BENCHMARKS COMPLETE"

# Comparison table header
echo ""
echo "╔══════════════════════════════════════╦══════════╦════════════╦════════════════╗"
echo "║ Run Name                             ║ sec/step ║ samples/s  ║ vs dsai conda  ║"
echo "╠══════════════════════════════════════╬══════════╬════════════╬════════════════╣"

# Reference values from dsai conda baseline (Section 5 of report)
declare -A BASELINE
BASELINE["Nightly: BF16"]="5.06/1.58"
BASELINE["Nightly: FP16"]="8.02/1.00"
BASELINE["Nightly: FP32+TF32"]="8.53/0.94"
BASELINE["Nightly: FP8 (TE)"]="7.13/1.12"
BASELINE["Nightly: BatchSize 4"]="5.29/0.76"
BASELINE["Nightly: BatchSize 8"]="7.91/1.01"
BASELINE["Nightly: BatchSize 12"]="10.25/1.17"
BASELINE["Nightly: Weak Scale 1GPU"]="—/0.31"
BASELINE["Nightly: Weak Scale 2GPU"]="—/0.57"
BASELINE["Nightly: Weak Scale 4GPU"]="—/1.12"

while IFS='|' read -r name exitcode duration secstep samplesec; do
    ref="${BASELINE[$name]:-—}"
    if [ "$exitcode" = "0" ]; then
        printf "║ %-38s ║ %8s ║ %10s ║ %-14s ║\n" \
            "$name" "$secstep" "$samplesec" "conda:$ref"
    else
        printf "║ %-38s ║ %8s ║ %10s ║ %-14s ║\n" \
            "$name" "FAIL" "FAIL" "conda:$ref"
    fi
done < "$RESULTS_FILE"

echo "╚══════════════════════════════════════╩══════════╩════════════╩════════════════╝"
echo ""
echo "Conda baseline (dsai script): BF16=1.58 s/s, FP16=1.00 s/s (report §5)"
echo "NGC 25.03 baseline:           BF16=1.60 s/s, FP16=1.65 s/s (report §6)"
echo ""
echo "Full results: ${RESULTS_FILE}"
echo "=== Nightly Benchmarks Complete ==="

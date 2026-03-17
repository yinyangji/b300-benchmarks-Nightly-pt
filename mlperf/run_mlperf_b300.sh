#!/bin/bash
# ============================================================
# MLPerf Inference — B300 Benchmark Runner
# Runs ResNet-50 and BERT-Large offline scenarios (synthetic data)
# ============================================================
set -uo pipefail

CONDA_ENV=/scratch/rdesouz4/envs/s2s
PYTHON=${CONDA_ENV}/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

export LD_LIBRARY_PATH=${CONDA_ENV}/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"   # 4-GPU DataParallel inference
export WANDB_MODE=offline

# Install transformers if needed
${PYTHON} -c "import transformers" 2>/dev/null || \
    ${PYTHON} -m pip install transformers --quiet

mkdir -p "${LOG_DIR}/resnet50" "${LOG_DIR}/bert"

echo ""
echo "================================================================"
echo "  MLPerf Inference — NVIDIA B300 SXM6 AC"
echo "  $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n "${CUDA_VISIBLE_DEVICES:-4}p" 2>/dev/null || nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "================================================================"
echo ""

# ── ResNet-50 Offline ──────────────────────────────────────
echo ">>> [1/2] ResNet-50 FP16 Offline"
echo ""
${PYTHON} "${SCRIPT_DIR}/mlperf_b300_resnet50.py" 2>&1 | tee "${LOG_DIR}/resnet50_run.log"
echo ""

# ── BERT-Large Offline ─────────────────────────────────────
echo ">>> [2/2] BERT-Large FP16 Offline"
echo ""
${PYTHON} "${SCRIPT_DIR}/mlperf_b300_bert.py" 2>&1 | tee "${LOG_DIR}/bert_run.log"
echo ""

# ── Summary ───────────────────────────────────────────────
echo "================================================================"
echo "  RESULTS SUMMARY"
echo "================================================================"
for model in resnet50 bert; do
    log="${LOG_DIR}/${model}/mlperf_log_summary.txt"
    if [ -f "$log" ]; then
        qps=$(grep "Samples per second" "$log" | awk '{print $NF}')
        result=$(grep "Result is" "$log" | awk '{print $NF}')
        printf "  %-12s  QPS: %-10s  %s\n" "$model" "$qps" "$result"
    fi
done
echo "================================================================"
echo ""
echo "Full logs: ${LOG_DIR}/"

#!/bin/bash
# ============================================================
# MLPerf Inference — B300 Runner (NGC Container Edition)
#
# Uses nvcr.io/nvidia/pytorch:25.03-py3 as base — confirmed
# working on B300 SXM6 (sm_103). Intel MLPerf container
# (intel/intel-optimized-pytorch:mlperf-inference-*) is
# NOT compatible: it targets Intel XPU, not NVIDIA CUDA.
#
# Usage:
#   bash run_mlperf_b300_docker.sh              # GPU 4 (default)
#   CUDA_GPU=0 bash run_mlperf_b300_docker.sh   # GPU 0
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="mlperf-b300-ngc:latest"
CUDA_GPU="${CUDA_GPU:-4}"
LOG_DIR="${SCRIPT_DIR}/logs"

# Pull image if not present
if ! sg docker -c "docker image inspect ${IMAGE}" &>/dev/null 2>&1; then
    echo "==> Building ${IMAGE} from Dockerfile.mlperf ..."
    sg docker -c "docker build -f ${SCRIPT_DIR}/Dockerfile.mlperf -t ${IMAGE} ${SCRIPT_DIR}"
fi

mkdir -p "${LOG_DIR}"

echo ""
echo "================================================================"
echo "  MLPerf Inference — NVIDIA B300 SXM6 (NGC 25.03 container)"
echo "  $(date)"
echo "================================================================"
echo ""

sg docker -c "docker run --rm \
    --gpus '\"device=${CUDA_GPU}\"' \
    --ipc=host \
    --ulimit memlock=-1 \
    -e TORCHDYNAMO_DISABLE=1 \
    -v ${SCRIPT_DIR}:/workspace \
    -w /workspace \
    ${IMAGE} bash -c '
        echo \"GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\"
        echo \"PyTorch: \$(python -c \"import torch; print(torch.__version__)\")\"
        echo \"CUDA: \$(python -c \"import torch; print(torch.version.cuda)\")\"
        echo \"\"
        echo \">>> [1/2] ResNet-50 FP16 Offline\"
        python mlperf_b300_resnet50.py
        echo \"\"
        echo \">>> [2/2] BERT-Large FP16 Offline\"
        python mlperf_b300_bert.py
    '" 2>&1 | tee "${LOG_DIR}/mlperf_b300_docker_run.log"

echo ""
echo "================================================================"
echo "  RESULTS"
echo "================================================================"
for model in resnet50 bert; do
    log="${LOG_DIR}/${model}/mlperf_log_summary.txt"
    if [ -f "$log" ]; then
        qps=$(grep "Samples per second" "$log" | awk '{print $NF}')
        result=$(grep "Result is" "$log" | awk '{print $NF}')
        printf "  %-12s  QPS: %-10s  %s\n" "${model}" "${qps}" "${result}"
    fi
done
echo "================================================================"

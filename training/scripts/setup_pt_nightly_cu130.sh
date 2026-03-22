#!/bin/bash
# ============================================================================
# pt-nightly-cu130 环境一键部署脚本
# 安装 Miniconda + PyTorch Nightly (CUDA 13.0) + B300 训练依赖
# 复现 requirements-nightly-cu130.txt 一致环境
#
# 用法:
#   bash training/scripts/setup_pt_nightly_cu130.sh
#   bash training/scripts/setup_pt_nightly_cu130.sh --skip-conda   # 若已安装 conda
#
# 环境变量:
#   CONDA_ENV_NAME - conda 环境名称，默认 pt-nightly-cu130
#   MINICONDA_DIR  - Miniconda 安装目录，默认 $HOME/miniconda3
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SKIP_CONDA="${SKIP_CONDA:-0}"
for arg in "$@"; do
    [ "$arg" = "--skip-conda" ] && SKIP_CONDA=1
done

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pt-nightly-cu130}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
CONDA_ENV_PATH="${MINICONDA_DIR}/envs/${CONDA_ENV_NAME}"

echo "=========================================================================="
echo "  pt-nightly-cu130 环境部署"
echo "  环境名称: ${CONDA_ENV_NAME}"
echo "  $(date)"
echo "=========================================================================="

# ----------------------------------------------------------------------------
# 0. 前置检查：NVIDIA 驱动（B300 需 590-server-open）
# ----------------------------------------------------------------------------
echo ""
echo ">>> 检查 NVIDIA 驱动 ..."
if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
        DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        echo "    驱动版本: $DRIVER_VER"
    else
        echo "    警告: nvidia-smi 无法与驱动通信，请确认:"
        echo "      - B300 需 nvidia-driver-590-server-open（见 NOTE_B300_CUDA_TROUBLESHOOTING.md）"
        echo "      - 已执行 fix_b300_cuda.sh 并重启"
    fi
else
    echo "    未找到 nvidia-smi，请先安装 NVIDIA 驱动"
fi

# ----------------------------------------------------------------------------
# 1. 安装 Miniconda（若未安装）
# ----------------------------------------------------------------------------
if [ "$SKIP_CONDA" != "1" ] && [ ! -f "${MINICONDA_DIR}/bin/conda" ]; then
    echo ""
    echo ">>> 安装 Miniconda 到 ${MINICONDA_DIR} ..."
    mkdir -p "$(dirname "${MINICONDA_DIR}")"
    wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_DIR}"
    rm -f /tmp/Miniconda3-latest-Linux-x86_64.sh
    echo "    完成"
else
    echo ""
    echo ">>> 使用已有 Conda: ${MINICONDA_DIR}"
fi

# 初始化 conda（当前 shell）
CONDA_EXE="${MINICONDA_DIR}/bin/conda"
if [ -f "${CONDA_EXE}" ]; then
    set +e
    eval "$("${CONDA_EXE}" shell.bash hook)"
    set -e
fi

# ----------------------------------------------------------------------------
# 2. 创建 conda 环境（命名环境 pt-nightly-cu130）
# ----------------------------------------------------------------------------
echo ""
echo ">>> 创建环境 ${CONDA_ENV_NAME} (Python 3.11) ..."
if [ -f "${CONDA_ENV_PATH}/bin/python" ]; then
    echo "    环境已存在，跳过创建"
else
    conda create -y -n "${CONDA_ENV_NAME}" python=3.11 \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
        --override-channels
fi

# ----------------------------------------------------------------------------
# 2.5 激活 conda 环境并验证
# ----------------------------------------------------------------------------
echo ""
echo ">>> 激活环境 ${CONDA_ENV_NAME} ..."
conda activate "${CONDA_ENV_NAME}"

# 检查激活是否成功
if [ "${CONDA_PREFIX}" != "${CONDA_ENV_PATH}" ]; then
    echo "    错误: conda 激活失败"
    echo "    期望 CONDA_PREFIX=${CONDA_ENV_PATH}"
    echo "    实际 CONDA_PREFIX=${CONDA_PREFIX:-未设置}"
    exit 1
fi
echo "    已激活: ${CONDA_PREFIX}"

PIP="${CONDA_ENV_PATH}/bin/pip"
PYTHON="${CONDA_ENV_PATH}/bin/python"

# ----------------------------------------------------------------------------
# 3. 安装依赖（优先用 requirements-nightly-cu130.txt 精确复现）
# ----------------------------------------------------------------------------
REQ_FILE="${REPO_ROOT}/requirements-nightly-cu130.txt"
# PyTorch cu130 来自 nightly index，其余来自 PyPI
PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu130"

if [ -f "${REQ_FILE}" ]; then
    echo ""
    echo ">>> 根据 requirements-nightly-cu130.txt 安装（精确复现 pt-nightly-cu130 环境）..."
    "${PIP}" install -r "${REQ_FILE}" \
        --extra-index-url "${PYTORCH_INDEX}"
else
    echo ""
    echo ">>> 未找到 requirements-nightly-cu130.txt，安装基础 PyTorch + 训练依赖 ..."
    "${PIP}" install --pre torch torchvision torchaudio \
        --index-url "${PYTORCH_INDEX}"
    "${PIP}" install wandb einops timm "ruamel.yaml" xarray h5py \
        cartopy cftime pandas psutil tqdm netcdf4 h5netcdf matplotlib
    "${PIP}" install --pre torchao --index-url "${PYTORCH_INDEX}" 2>/dev/null || true
fi

# ----------------------------------------------------------------------------
# 5. 验证
# ----------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo "  验证环境"
echo "=========================================================================="
unset LD_LIBRARY_PATH 2>/dev/null || true
"${PYTHON}" -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'devices:  {torch.cuda.device_count()}')
if torch.cuda.device_count() > 0:
    try:
        x = torch.zeros(1, device='cuda:0')
        print('CUDA 初始化: OK')
    except RuntimeError as e:
        err = str(e)
        print('CUDA 张量创建失败 (B300 上常见 Error 802):', err[:60] + '...' if len(err) > 60 else err)
        print('  环境已部署成功。请在新终端中手动验证，详见 NOTE_B300_CUDA_TROUBLESHOOTING.md')
else:
    print('CUDA 初始化: 无 GPU 或需检查驱动')
"

echo ""
echo "=========================================================================="
echo "  部署完成"
echo "=========================================================================="
echo ""
echo "激活环境:"
echo "  conda activate ${CONDA_ENV_NAME}"
echo ""
echo "运行前建议（避免 LD_LIBRARY_PATH 冲突）:"
echo "  unset LD_LIBRARY_PATH"
echo ""
echo "快速测试:"
echo "  python -c \"import torch; x=torch.zeros(1,device='cuda:0'); print('OK')\""
echo ""

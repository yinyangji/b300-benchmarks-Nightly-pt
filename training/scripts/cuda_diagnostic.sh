#!/bin/bash
# CUDA/GPU 诊断脚本 — 收集信息用于排查 "CUDA driver initialization failed"
# 运行: bash training/scripts/cuda_diagnostic.sh 2>&1 | tee cuda_diag.log

echo "================================================================================"
echo "  CUDA/GPU 诊断报告 — $(date)"
echo "================================================================================"

echo ""
echo "=== 1. nvidia-smi (驱动与设备) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv 2>/dev/null || nvidia-smi

echo ""
echo "=== 2. 设备文件权限 ==="
ls -la /dev/nvidia* 2>/dev/null || echo "/dev/nvidia* 不存在或无权限"

echo ""
echo "=== 3. 环境 (CUDA/LD) ==="
echo "CUDA_HOME=${CUDA_HOME:-未设置}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-未设置}"
echo "PATH 中 cuda: $(echo $PATH | tr ':' '\n' | grep -i cuda || echo '无')"

echo ""
echo "=== 4. 系统 CUDA 库 ==="
ldconfig -p 2>/dev/null | grep -E "libcuda|libcudart" | head -5 || true
ls -la /usr/local/cuda*/lib64/libcudart* 2>/dev/null || true

echo ""
echo "=== 5. Python/PyTorch 环境 ==="
for env in pt-nightly-cu130 pt128 base; do
    if [ -f "$HOME/miniconda3/envs/$env/bin/python" ]; then
        echo "--- $env ---"
        "$HOME/miniconda3/envs/$env/bin/python" -c "
import sys
print('Python:', sys.executable)
try:
    import torch
    print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
    print('device_count:', torch.cuda.device_count())
    x = torch.zeros(1, device='cuda:0')
    print('torch.zeros(device=cuda:0): OK')
except Exception as e:
    print('Error:', e)
" 2>&1
    fi
done

echo ""
echo "=== 6. 是否在容器内 ==="
cat /proc/1/cgroup 2>/dev/null | head -3
[ -f /.dockerenv ] && echo "Docker 容器" || echo "非 Docker"

echo ""
echo "=== 7. TDX/Confidential Computing ==="
ls /dev/tdx* 2>/dev/null && echo "存在 TDX 设备" || echo "无 TDX 设备"
nvidia-smi -q 2>/dev/null | grep -A2 "Conf Compute" || true

echo ""
echo "================================================================================"
echo "  诊断结束 — 将 cuda_diag.log 提供给集群管理员"
echo "================================================================================"

#!/bin/bash
# B300 CUDA 初始化修复脚本
# 1. 切回 590-server-open（595 无 server 版，B300 需要 server 驱动）
# 2. 应用 uvm_disable_hmm=1 工作区（修复 Open 内核 CUDA init 已知问题）
set -e

echo "=== B300 需 server-open 驱动，595 仅 desktop 版，切回 590-server-open ==="

# 移除 595 desktop open
apt-get remove -y nvidia-driver-open nvidia-dkms-open 2>/dev/null || true

# 若从 595 切回，需先移除冲突的 lib 包
for pkg in libnvidia-cfg1 libnvidia-egl-gbm1 libnvidia-egl-wayland21 libnvidia-egl-xcb1 libnvidia-egl-xlib1 libnvidia-gpucomp; do
  dpkg -l "$pkg" 2>/dev/null | grep -q ^ii && dpkg -r "$pkg" 2>/dev/null || true
done

# 安装 590-server-open
apt-get update
apt-get install -y nvidia-driver-590-server-open nvidia-fabricmanager-590
apt-get install -y -f 2>/dev/null || true

echo ""
echo "=== 应用 uvm_disable_hmm=1 工作区 ==="
echo "options nvidia_uvm uvm_disable_hmm=1" > /etc/modprobe.d/nvidia-uvm.conf
echo "已写入 /etc/modprobe.d/nvidia-uvm.conf"
update-initramfs -u

echo ""
echo "=== 完成。请执行: sudo reboot ==="
echo "重启后验证: nvidia-smi && python -c \"import torch; x=torch.zeros(1,device='cuda:0'); print('OK')\""

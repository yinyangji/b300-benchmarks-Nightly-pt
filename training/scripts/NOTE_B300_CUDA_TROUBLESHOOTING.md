# B300 CUDA 排查说明与版本一致性检查

本文档整理 B300 上 PyTorch/CUDA 训练过程中的常见问题，**重点强调版本一致性和排查顺序**。

---

## 一、版本一致性（最高优先级）

### 1.1 必须匹配的三方版本

| 组件 | 说明 | 检查命令 |
|------|------|----------|
| **NVIDIA 驱动** | 内核模块 + 用户态库 | `nvidia-smi` 或 `cat /proc/driver/nvidia/version` |
| **CUDA Runtime** | PyTorch 自带的 libcudart | `python -c "import torch; print(torch.version.cuda)"` |
| **PyTorch CUDA 构建** | cu121 / cu128 / cu130 等 | `python -c "import torch; print(torch.__version__)"` |

### 1.2 兼容性矩阵（参考）

- **驱动 590.48**：支持 CUDA 13.x，理论上兼容 PyTorch cu130
- **驱动 535 / 550**：支持 CUDA 12.x
- **PyTorch cu128**：需驱动 ≥ 525
- **PyTorch cu130**：需驱动 ≥ 580

详见：<https://docs.nvidia.com/deploy/cuda-compatibility/>

### 1.3 典型错误与对应原因

| 错误 | 含义 | 处理方向 |
|------|------|----------|
| **Error 803** | `unsupported display driver / cuda driver combination` | 驱动与 PyTorch CUDA 版本不匹配，或内核模块与用户态版本混用 |
| **Error 304** | `OS call failed or operation not supported` | 设备访问或权限问题，常见于容器/调度环境 |
| **CUDA driver initialization failed** | `torch._C._cuda_init()` 失败 | 见下文「驱动类型 / HMM」 |

### 1.4 LD_LIBRARY_PATH 冲突

**系统 CUDA 与 conda 环境混用会导致版本错乱。**

```bash
# 若 LD_LIBRARY_PATH 指向 /usr/local/cuda-13.1/lib64
# 而 PyTorch 为 cu130，可能加载到错误版本的 libcuda / libcudart

# 建议：运行 PyTorch 前清空或仅保留必要路径
unset LD_LIBRARY_PATH
conda activate pt-nightly-cu130
python -c "import torch; x = torch.zeros(1, device='cuda:0'); print('OK')"
```

或仅在需要时显式设置 conda 环境内的库路径。

---

## 二、B300 特定约束

### 2.1 必须使用 Open 内核模块

- **B300 (PCI 10de:3182)** 仅支持 `nvidia-driver-XXX-server-open`
- 闭源 `nvidia-srv` 不支持 B300，会提示 `requires use of the NVIDIA open kernel modules`
- **不要**对 B300 安装 `nvidia-driver-590-server`（闭源）

### 2.2 595 只有 desktop 版，无 server-open

- `nvidia-driver-595-open` 为桌面版
- B300 需 server 驱动（含 Fabric Manager）
- 当前仅 `nvidia-driver-590-server-open` 适用于 B300

### 2.3 uvm_disable_hmm 工作区

Open 内核在部分环境（TDX、特定内核）下会出现 CUDA 初始化失败，可通过禁用 HMM 规避：

```bash
echo "options nvidia_uvm uvm_disable_hmm=1" | sudo tee /etc/modprobe.d/nvidia-uvm.conf
sudo update-initramfs -u
sudo reboot
```

重启后验证：`cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm` 应为 `Y` 或 `1`

---

## 三、推荐排查顺序

1. **版本一致性**  
   - 核对：驱动版本、PyTorch CUDA 版本、LD_LIBRARY_PATH  
   - 运行前 `unset LD_LIBRARY_PATH` 再测

2. **驱动类型**  
   - 确认为 `nvidia-driver-590-server-open`，非 595-open 或闭源

3. **uvm_disable_hmm**  
   - 若仍有初始化失败，应用上述 modprobe 配置并重启

4. **环境 / 调度**  
   - 若在集群中，确认通过 `srun`/`sbatch` 正确分配 GPU

---

## 四、相关脚本

| 脚本 | 用途 |
|------|------|
| `setup_pt_nightly_cu130.sh` | 一键部署 conda + PyTorch nightly cu130 + 训练依赖，复现 pt-nightly-cu130 环境 |
| `cuda_diagnostic.sh` | 收集驱动、CUDA、设备等信息 |
| `fix_b300_cuda.sh` | 从 595 切回 590-server-open 并启用 uvm_disable_hmm |
| `run_simple_8gpu.sh` | 8 卡 DDP 训练 |
| `run_faster_train_b300.sh` | faster_train B300 运行脚本 |
| `single_gpu_train.py` / `simple_ddp_train.py` | 单卡/多卡最小测试 |

---

## 五、快速验证流程

```bash
# 1. 驱动
nvidia-smi

# 2. 版本（驱动需与 PyTorch CUDA 匹配）
cat /proc/driver/nvidia/version

# 3. PyTorch（推荐先 unset LD_LIBRARY_PATH）
unset LD_LIBRARY_PATH
conda activate pt-nightly-cu130  # 或 pt128
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; x = torch.zeros(1, device='cuda:0'); print('OK')"
```

---

## 六、参考链接

- CUDA 兼容性：<https://docs.nvidia.com/deploy/cuda-compatibility/>
- Open 内核 CUDA 初始化问题：<https://github.com/NVIDIA/open-gpu-kernel-modules/issues/797>
- TDX 相关：<https://github.com/NVIDIA/open-gpu-kernel-modules/issues/531>

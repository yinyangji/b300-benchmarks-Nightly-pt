#!/usr/bin/env python3
"""单卡训练，无 DDP。"""
import torch
# 与 python -c "import torch; print(torch.cuda.device_count())" 不同，
# 下面会真正在 GPU 上分配内存，触发完整 CUDA 驱动初始化
_ = torch.zeros(1, device="cuda:0")  # 若此步失败，说明 device_count 能返回但实际无法使用
device = torch.device("cuda:0")
model = torch.nn.Linear(256, 10).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(10):
    x = torch.randn(32, 256, device=device)
    loss = model(x).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"Step {step} loss={loss.item():.4f}")

print("Done. GPU OK.")

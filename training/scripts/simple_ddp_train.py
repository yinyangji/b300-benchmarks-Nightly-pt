#!/usr/bin/env python3
"""
最简单的 8 卡 DDP 训练脚本 — 不依赖 faster_train、无复杂 import。
用于验证 B300 多 GPU 训练环境是否正常。
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 简单模型
    model = torch.nn.Linear(256, 10).to(device)
    model = DistributedDataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 跑几步
    for step in range(10):
        x = torch.randn(32, 256, device=device)
        y = model(x)
        loss = y.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Step {step} loss={loss.item():.4f}")

    dist.destroy_process_group()
    if rank == 0:
        print("Done.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Minimal script to verify CUDA is visible inside torchrun worker."""
import torch
n = torch.cuda.device_count()
print(f"Worker CUDA devices: {n}")
assert n > 0, "No GPUs visible in torchrun worker!"
print(">>> 预检查通过")

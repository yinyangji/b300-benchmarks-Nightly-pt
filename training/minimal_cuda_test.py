#!/usr/bin/env python3
"""Minimal CUDA test - same as python -c but as a .py file."""
import torch
print("device_count:", torch.cuda.device_count())

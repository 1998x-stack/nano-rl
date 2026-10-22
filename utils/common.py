"""Reproducibility and device selection for CPU-first teaching experiments."""
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    # 相同随机种子保证 numpy / random / torch 的局部实验可以复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

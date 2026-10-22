"""Ring replay buffer shared by off-policy algorithms; stores terminated, not done."""
from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def add(self, s, a, r, next_s, terminated):
        self.data.append((np.array(s, copy=True), np.array(a, copy=True),
                          float(r), np.array(next_s, copy=True), float(terminated)))

    def sample(self, batch_size: int, device="cpu"):
        batch = random.sample(self.data, batch_size)
        s, a, r, ns, terminated = map(np.stack, zip(*batch))
        # 奖励和终止掩码为 float; 动作保留原始维度供离散/连续算法分别处理。
        def tensor(x):
            return torch.as_tensor(x, device=device)
        return tensor(s).float(), tensor(a), tensor(r).float(), tensor(ns).float(), tensor(terminated).float()

"""Tiny corridor process shared by learned-model and sequence RL lessons.

Actions: left (0), right (1). Terminal when position == n-1. No external download.
The model-based agent must never call transition() *inside* its planning search.
"""
import numpy as np


class CorridorEnv:
    def __init__(self, n=5, step_reward=-0.02):
        assert n >= 3
        self.n, self.step_reward = n, step_reward
        self.position = 0

    def onehot(self, pos=None):
        v = np.zeros(self.n, dtype=np.float32)
        v[self.position if pos is None else pos] = 1.
        return v

    def transition(self, position: int, action: int):
        if not 0 <= position < self.n or action not in (0, 1):
            raise ValueError("invalid state or action")
        if position == self.n - 1:
            return position, 0., True
        nxt = min(self.n - 1, max(0, position + (1 if action else -1)))
        done = nxt == self.n - 1
        return nxt, 1. if done else self.step_reward, done

    def reset(self, seed=None):
        self.position = 0
        return self.onehot(), {}

    def step(self, action):
        self.position, r, done = self.transition(self.position, int(action))
        return self.onehot(), r, done, False, {}

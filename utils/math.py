"""Auditable shared math: GAE and PPO ratio clipping."""
import numpy as np
import torch


def discounted_returns(rewards, gamma: float):
    out, running = [], 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        out.append(running)
    return np.asarray(out[::-1], dtype=np.float32)


def generalized_advantage(rewards, values, next_values, terminated, boundaries, gamma=.99, lam=.95):
    """Truncation: bootstrap into delta, but do NOT propagate GAE across episode reset.

    terminated=1 masks the *value bootstrap*; boundaries=1 masks *lambda recursion*.
    """
    n = len(rewards)
    advantage = np.zeros(n, dtype=np.float32)
    next_gae = 0.0
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * (1.0 - terminated[t]) * next_values[t] - values[t]
        next_gae = delta + gamma * lam * (1.0 - boundaries[t]) * next_gae
        advantage[t] = next_gae
    return advantage


def ppo_surrogate(new_logprob, old_logprob, advantage, epsilon=.2):
    # p(new_action|state) / p(old_action|state); avoid drifting too far from behavior.
    ratio = (new_logprob - old_logprob).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    return torch.minimum(unclipped, clipped), ratio

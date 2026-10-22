"""03 REINFORCE: Monte Carlo policy gradient with a per-episode return baseline."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils import RunLogger, seed_everything
from utils.math import discounted_returns


class Policy(nn.Module):
    def __init__(self, obs=4, actions=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, 64), nn.Tanh(), nn.Linear(64, actions))

    def forward(self, obs):
        return Categorical(logits=self.net(obs))


def reinforce_loss(log_probs, rewards, gamma=.99):
    returns = torch.as_tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
    # Baseline 不影响策略梯度的期望，却通常能降低方差。
    advantages = returns - returns.mean()
    if returns.numel() > 1:
        advantages /= returns.std(unbiased=False).clamp_min(1e-6)
    return -(torch.stack(log_probs) * advantages).sum(), returns


def train(episodes=250, seed=42, log_enabled=True):
    seed_everything(seed)
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    log = RunLogger("03-reinforce", log_enabled)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        probs, rewards = [], []
        for _ in range(500):
            dist = policy(torch.as_tensor(obs, dtype=torch.float32))
            action = dist.sample()
            nxt, r, terminated, truncated, _ = env.step(int(action.item()))
            probs.append(dist.log_prob(action))
            rewards.append(float(r))
            obs = nxt
            if terminated or truncated:
                break
        loss, _ = reinforce_loss(probs, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log.scalar("episode/return", sum(rewards), ep)
        log.scalar("loss/policy", loss.item(), ep)
    env.close()
    log.close()
    return policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    train(a.episodes, a.seed, not a.no_log)


if __name__ == "__main__":
    main()

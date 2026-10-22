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


def reinforce_loss(log_probs, rewards, gamma=.99, baseline=0.0):
    returns = torch.as_tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
    # 使用之前 episode 统计得到的 baseline：对当前轨迹采样的动作是独立的。
    # 不减去当前轨迹自己的 returns.mean()，因为这会引入与当前动作相关的偏差。
    advantages = returns - float(baseline)
    return -(torch.stack(log_probs) * advantages).sum(), returns


def train(episodes=250, seed=42, log_enabled=True):
    seed_everything(seed)
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    log = RunLogger("03-reinforce", log_enabled)
    baseline = 0.0  # 从之前 episode 的平均 returns 构建滑动 baseline。
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
        loss, returns = reinforce_loss(probs, rewards, baseline=baseline)
        baseline = 0.9 * baseline + 0.1 * returns.mean().item()
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

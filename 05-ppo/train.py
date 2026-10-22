"""05 PPO with GAE: collect fresh on-policy data, then optimize it for a few epochs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils import RunLogger, seed_everything
from utils.math import generalized_advantage, ppo_surrogate


class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(),
                                   nn.Linear(64, action_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(),
                                    nn.Linear(64, 1))

    def forward(self, state):
        return Categorical(logits=self.actor(state)), self.critic(state).squeeze(-1)


def train(steps=8192, seed=42, log_enabled=True, horizon=256):
    seed_everything(seed)
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    log = RunLogger("05-ppo", log_enabled)
    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    total_steps = 0
    update = 0
    while total_steps < steps:
        data = {k: [] for k in ("obs", "action", "logp", "value", "reward", "next_value", "terminated", "boundary")}
        for _ in range(min(horizon, steps - total_steps)):
            st = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist, val = model(st)
                act = dist.sample()
                logp = dist.log_prob(act)
            next_obs, r, terminated, truncated, _ = env.step(int(act.item()))
            with torch.no_grad():
                _, next_val = model(torch.as_tensor(next_obs, dtype=torch.float32))
            for key, v in (("obs", obs.copy()), ("action", int(act.item())),
                           ("logp", float(logp.item())), ("value", float(val.item())),
                           ("reward", r), ("next_value", float(next_val.item())),
                           ("terminated", float(terminated)), ("boundary", float(terminated or truncated))):
                data[key].append(v)
            episode_return += r
            obs = next_obs
            total_steps += 1
            if terminated or truncated:
                log.scalar("episode/return", episode_return, total_steps)
                obs, _ = env.reset()
                episode_return = 0.0
        # 时间截断：delta 可以 bootstrap 到 next_obs，但 GAE 不跨重置后的 episode。
        advantage = generalized_advantage(data["reward"], data["value"], data["next_value"],
                                          data["terminated"], data["boundary"])
        returns = torch.as_tensor(advantage + np.asarray(data["value"]), dtype=torch.float32)
        advantage = torch.as_tensor(advantage, dtype=torch.float32)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        states = torch.as_tensor(np.asarray(data["obs"]), dtype=torch.float32)
        actions = torch.as_tensor(data["action"], dtype=torch.int64)
        old_logp = torch.as_tensor(data["logp"], dtype=torch.float32)
        # On-policy: 少量 epoch 复用这批新轨迹；下轮必须重新从更新后的策略采样。
        for _ in range(4):
            order = torch.randperm(len(states))
            for ix in order.split(64):
                dist, values = model(states[ix])
                clipped_objective, ratio = ppo_surrogate(dist.log_prob(actions[ix]), old_logp[ix], advantage[ix])
                policy_loss = -clipped_objective.mean()
                value_loss = .5 * (values - returns[ix]).square().mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + .5 * value_loss - .01 * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
        log.scalar("loss/policy", policy_loss.item(), update)
        log.scalar("loss/value", value_loss.item(), update)
        log.scalar("policy/approx_ratio", ratio.mean().item(), update)
        update += 1
    env.close()
    log.close()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    train(a.steps, a.seed, not a.no_log)


if __name__ == "__main__":
    main()

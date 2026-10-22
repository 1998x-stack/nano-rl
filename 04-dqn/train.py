"""04 DQN: replay buffer, delayed target network, terminal-aware Bellman target."""
import argparse
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ReplayBuffer, RunLogger, seed_everything


class QNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_dim))

    def forward(self, obs):
        return self.layers(obs)


def dqn_target(rewards, terminated, next_q_max, gamma=.99):
    # 只在环境真正 terminated 时停止 bootstrap；时间截断不是 MDP 终止。
    return rewards + gamma * (1 - terminated) * next_q_max


def train(steps=10000, seed=42, log_enabled=True, batch_size=64):
    seed_everything(seed)
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    q, target = QNet(), QNet()
    target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
    buffer, log = ReplayBuffer(20000), RunLogger("04-dqn", log_enabled)
    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    for step in range(steps):
        epsilon = max(.05, 1 - step / max(1, steps * .7))
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = int(q(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())
        nxt, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, nxt, terminated)
        episode_return += reward
        obs = nxt
        if len(buffer) >= batch_size:
            s, a, r, ns, done = buffer.sample(batch_size)
            prediction = q(s).gather(1, a.long().view(-1, 1)).squeeze(1)
            with torch.no_grad():
                backup = dqn_target(r, done, target(ns).max(dim=1).values)
            loss = F.smooth_l1_loss(prediction, backup)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            optimizer.step()
            log.scalar("loss/td", loss.item(), step)
        if step % 200 == 0:
            target.load_state_dict(q.state_dict())  # 固定 TD 目标，减少移动目标震荡。
        if terminated or truncated:
            log.scalar("episode/return", episode_return, step)
            obs, _ = env.reset()
            episode_return = 0.0
    env.close()
    log.close()
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    train(a.steps, a.seed, not a.no_log)


if __name__ == "__main__":
    main()

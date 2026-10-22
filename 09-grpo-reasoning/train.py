"""09 GRPO math in miniature: exact-match verified rewards for modular arithmetic.

A tiny categorical neural policy predicts the answer to (a+b) % 10.
This is NOT a language model, not chain-of-thought generation and not RLHF.
It isolates the group-relative-advantage and clipped policy update mechanisms.
"""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import RunLogger, seed_everything
from utils.math import ppo_surrogate


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 10))

    def forward(self, questions):
        return Categorical(logits=self.net(questions))


def encode_questions(a, b):
    return torch.cat([F.one_hot(a.long(), 10), F.one_hot(b.long(), 10)], dim=-1).float()


def group_advantages(rewards, epsilon=1e-4):
    """[batch, group]: mean/std computed WITHIN the same question's generations."""
    return (rewards - rewards.mean(dim=-1, keepdim=True)) / (
        rewards.std(dim=-1, keepdim=True, unbiased=False) + epsilon)


def train(steps=1000, seed=42, log_enabled=True, group_size=8):
    seed_everything(seed)
    policy = TinyPolicy()
    reference = copy.deepcopy(policy)
    for p in reference.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    log = RunLogger("09-grpo", log_enabled)
    for step in range(steps):
        a = torch.randint(0, 10, (32,))
        b = torch.randint(0, 10, (32,))
        questions = encode_questions(a, b)
        with torch.no_grad():
            old = policy(questions)
            actions = old.sample((group_size,)).transpose(0, 1).contiguous()  # [batch, group]
            old_logp = old.log_prob(actions.T).T  # frozen behavior probability per completion
            reward = (actions == ((a + b) % 10)[:, None]).float()
            advantage = group_advantages(reward)
        for _ in range(2):
            current = policy(questions)
            new_logp = current.log_prob(actions.T).T
            surrogate, _ = ppo_surrogate(new_logp, old_logp, advantage, epsilon=.2)
            # Frozen pretrained-like reference distribution; forward KL discourages drifting.
            current_logits = current.logits
            ref_logits = reference(questions).logits
            kl = torch.distributions.kl_divergence(
                Categorical(logits=current_logits), Categorical(logits=ref_logits)).mean()
            loss = -surrogate.mean() + .01 * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log.scalar("reward/exact_match", reward.mean().item(), step)
        log.scalar("loss/group_policy", loss.item(), step)
        log.scalar("group/nonzero_std", (reward.std(-1, unbiased=False) > 0).float().mean().item(), step)
    log.close()
    return policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    train(a.steps, a.seed, not a.no_log)


if __name__ == "__main__":
    main()

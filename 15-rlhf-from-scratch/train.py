"""15 RLHF pedagogical pipeline: SFT → Bradley-Terry reward model → PPO+KL.

The conditional 'LM' produces ONE answer token 0..9 to a modular addition
prompt. Preferences are SYNTHETIC exact-correctness comparisons. This captures
the stages of RLHF but is not full autoregressive sequence PPO/large language RL.
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


def prompts(batch=64):
    a, b = torch.randint(0, 10, (batch,)), torch.randint(0, 10, (batch,))
    features = torch.cat((F.one_hot(a,10), F.one_hot(b,10)), dim=-1).float()
    return features, (a+b) % 10


class TinyAnswerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 10))

    def forward(self, prompts):
        return self.net(prompts)


class PreferenceReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(30, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, x, answer):
        a = F.one_hot(answer.long(), 10).float()
        return self.net(torch.cat((x, a), -1)).squeeze(-1)


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def preference_pairs(batch=64):
    """Synthetic annotator: correct answers preferred to a randomly chosen wrong one."""
    x, good = prompts(batch)
    wrong = (good + torch.randint(1, 10, (batch,))) % 10
    # Randomize first/second order to prevent an ordering shortcut.
    flipped = torch.rand(batch) < .5
    first = torch.where(flipped, wrong, good)
    second = torch.where(flipped, good, wrong)
    preferred_first = (~flipped).float()
    return x, first, second, preferred_first


def bradley_terry_loss(scores_first, scores_second, prefers_first):
    # P(first preferred) = sigmoid(R(first) - R(second)).
    return F.binary_cross_entropy_with_logits(scores_first-scores_second,
                                               prefers_first.float())


def sft(policy, steps=200):
    opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
    for _ in range(steps):
        x, answer = prompts()
        loss = F.cross_entropy(policy(x), answer)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.detach()


def reward_model_fit(reward_model, steps=250):
    opt = torch.optim.Adam(reward_model.parameters(), lr=2e-3)
    for _ in range(steps):
        x, first, second, label = preference_pairs()
        loss = bradley_terry_loss(reward_model(x, first),
                                 reward_model(x, second), label)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.detach()


def ppo_update(policy, reference, reward_model, critic, optimizer, batch=64,
               kl_coefficient=.05, epsilon=.2, epochs=2):
    x, correct = prompts(batch)
    with torch.no_grad():
        old_dist = Categorical(logits=policy(x))
        actions = old_dist.sample()
        old_logp = old_dist.log_prob(actions)
        ref_logp = Categorical(logits=reference(x)).log_prob(actions)
        reward = reward_model(x, actions)
        # KL estimate is logπ_old(a|x)-logπ_ref(a|x), sampled from old policy.
        adjusted = reward - kl_coefficient * (old_logp-ref_logp)
        baseline = critic(x).detach()
        advantages = adjusted - baseline
    for _ in range(epochs):
        dist = Categorical(logits=policy(x))
        surrogate, ratio = ppo_surrogate(dist.log_prob(actions), old_logp,
                                         advantages, epsilon)
        value = critic(x)
        # PPO clipped surrogate with frozen old probs + fixed reference KL penalty.
        loss = -surrogate.mean() + .5 * F.mse_loss(value, adjusted) - .002 * dist.entropy().mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(policy.parameters())+list(critic.parameters()), 1.)
        optimizer.step()
    with torch.no_grad():
        accuracy = (policy(x).argmax(-1) == correct).float().mean()
    return float(loss.item()), float(reward.mean().item()), float(accuracy.item())


def train(steps=250, seed=42, log_enabled=True):
    seed_everything(seed)
    policy, reward_model, critic = TinyAnswerLM(), PreferenceReward(), Value()
    log = RunLogger("15-rlhf", log_enabled)
    sft_loss = sft(policy, steps)
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    rm_loss = reward_model_fit(reward_model, steps)
    for p in reward_model.parameters():
        p.requires_grad_(False)  # RM frozen during policy learning.
    optimizer = torch.optim.Adam(list(policy.parameters())+list(critic.parameters()), lr=5e-4)
    for step in range(steps):
        policy_loss, reward, accuracy = ppo_update(
            policy, reference, reward_model, critic, optimizer)
        log.scalar("ppo/policy_value_loss", policy_loss, step)
        log.scalar("ppo/predicted_reward", reward, step)
        # True label used ONLY for evaluation; NOT as optimization feedback here.
        log.scalar("eval/exact_answer_accuracy", accuracy, step)
    log.close()
    print("SFT loss:", round(sft_loss.item(),4),
          "BT reward loss:", round(rm_loss.item(),4))
    return policy, reference, reward_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=250,
                   help="iterations PER PHASE: SFT, reward-model fitting, PPO")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    train(a.steps, a.seed, not a.no_log)


if __name__ == "__main__":
    main()

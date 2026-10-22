"""12 MuZero-inspired latent MCTS: representation h, dynamics g, prediction f.

The environment supplies supervised training sequences ONLY. The search itself never
uses CorridorEnv.transition: it plans with learned latent dynamics + reward estimates.
Unlike the original MuZero, targets here include a simple teacher policy/value.
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CorridorEnv, RunLogger, seed_everything


class LatentModel(nn.Module):
    def __init__(self, n_states=5, latent=32):
        super().__init__()
        self.representation = nn.Sequential(nn.Linear(n_states, latent), nn.Tanh())
        self.dynamics_net = nn.Sequential(nn.Linear(latent + 2, 64),
                                          nn.ReLU(), nn.Linear(64, latent), nn.Tanh())
        self.reward_head = nn.Linear(latent + 2, 1)
        self.done_head = nn.Linear(latent + 2, 1)
        self.policy_head = nn.Linear(latent, 2)
        self.value_head = nn.Linear(latent, 1)

    def initial(self, observation):
        return self.representation(observation)

    def predict(self, hidden):
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def recurrent(self, hidden, action):
        a = F.one_hot(action.long(), num_classes=2).to(hidden.dtype)
        x = torch.cat((hidden, a), dim=-1)
        return (self.dynamics_net(x), self.reward_head(x).squeeze(-1),
                self.done_head(x).squeeze(-1))


def oracle_value(position, gamma=.95):
    """Corridor teacher: take right until terminal; used only in DATA LABELS."""
    env = CorridorEnv()
    total, discount, s = 0., 1., int(position)
    while s != env.n-1:
        s, reward, terminal = env.transition(s, 1)
        total += discount * reward
        discount *= gamma
    return total


def make_batch(batch_size=16, unroll=3, seed=42):
    rng, env = np.random.default_rng(seed), CorridorEnv()
    states, actions, rewards, dones, valid = [], [], [], [], []
    for _ in range(batch_size):
        s = int(rng.integers(0, env.n-1))
        st, ac, rew, dn, mask = [s], [], [], [], []
        ended = False
        for _ in range(unroll):
            if ended:
                # padding 不是环境交互，不应贡献 dynamics/reward/policy loss。
                a, ns, r, done, active = 0, s, 0., True, 0.
            else:
                a = int(rng.integers(2)) if rng.random() < .45 else 1
                ns, r, done = env.transition(s, a)
                active = 1.
            ac.append(a); rew.append(r); dn.append(float(done))
            mask.append(active)
            st.append(ns)
            s, ended = ns, ended or done
        states.append(st); actions.append(ac); rewards.append(rew)
        dones.append(dn); valid.append(mask)
    return (torch.tensor(states, dtype=torch.int64),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(valid, dtype=torch.float32))


def masked_mean(per_example, mask):
    """固定张量形状不意味着 padding 是真实经历；仅对有效样本求平均。"""
    return (per_example * mask).sum() / mask.sum().clamp_min(1.)


def unroll_loss(net, batch):
    states, actions, rewards, dones, valid = batch
    h = net.initial(F.one_hot(states[:, 0], 5).float())
    losses = []
    K = actions.shape[1]
    for t in range(K + 1):
        logits, values = net.predict(h)
        # policy 只能监督真实的非终止行动；value 还可学习最后的 terminal V=0。
        policy_mask = valid[:, t] if t < K else (
            valid[:, t-1] * (1 - dones[:, t-1]))
        state_mask = torch.ones_like(valid[:, 0]) if t == 0 else valid[:, t-1]
        target_policy = torch.ones(states.shape[0], dtype=torch.long)
        target_values = torch.tensor([oracle_value(s) for s in states[:, t].tolist()],
                                     dtype=torch.float32, device=values.device)
        policy_ce = F.cross_entropy(logits, target_policy, reduction="none")
        value_error = (values - target_values).square()
        losses.append(.2 * masked_mean(policy_ce, policy_mask) +
                      .5 * masked_mean(value_error, state_mask))
        if t < K:
            nxt, reward_pred, done_logits = net.recurrent(h, actions[:, t])
            obs_next = F.one_hot(states[:, t+1], 5).float()
            expected_hidden = net.initial(obs_next).detach()
            per_transition = (reward_pred - rewards[:, t]).square() + (
                .2 * F.binary_cross_entropy_with_logits(
                    done_logits, dones[:, t], reduction="none")) + (
                .1 * (nxt - expected_hidden).square().mean(-1))
            losses.append(masked_mean(per_transition, valid[:, t]))
            h = nxt
    return torch.stack(losses).sum() / (K + 1)


@dataclass
class SearchNode:
    prior: float = 1.
    hidden: torch.Tensor = None
    reward: float = 0.
    terminal: bool = False
    visits: int = 0
    value_sum: float = 0.
    children: dict = field(default_factory=dict)

    def value(self):
        return self.value_sum / self.visits if self.visits else 0.


@torch.no_grad()
def expand_node(node, net):
    if node.terminal:
        return 0.
    logits, value = net.predict(node.hidden)
    probs = logits.softmax(-1).flatten().cpu().numpy()
    node.children = {a: SearchNode(prior=float(p)) for a,p in enumerate(probs)}
    return float(value.flatten()[0].item())


def select_child(node, gamma=.95, c_puct=1.5):
    # Single-agent: positive continuation values (unlike AlphaZero's sign flip).
    return max(node.children.items(),
               key=lambda item: item[1].reward + gamma * item[1].value() +
               c_puct * item[1].prior * np.sqrt(node.visits + 1) / (1 + item[1].visits))


@torch.no_grad()
def latent_mcts(net, observation, simulations=16, max_depth=5, gamma=.95):
    assert simulations >= 1 and max_depth >= 1
    root = SearchNode(hidden=net.initial(torch.as_tensor(observation).float().view(1,-1)))
    expand_node(root, net)
    for _ in range(simulations):
        node, path, depth = root, [root], 0
        while node.children and depth < max_depth:
            action, child = select_child(node, gamma)
            if child.hidden is None:
                act = torch.tensor([action])
                child.hidden, r, done_logits = net.recurrent(node.hidden, act)
                child.reward = float(r.item())
                child.terminal = bool(torch.sigmoid(done_logits).item() > .5)
            node = child
            path.append(node)
            depth += 1
            if child.visits == 0 or child.terminal:
                break
        value = 0. if node.terminal else (expand_node(node, net)
                        if depth < max_depth and not node.children
                        else float(net.predict(node.hidden)[1].item()))
        # Reward is attached to edges; prepend it before assigning value to parent.
        for index in reversed(range(len(path))):
            path[index].visits += 1
            path[index].value_sum += value
            if index > 0:
                value = path[index].reward + gamma * value
    counts = np.asarray([root.children[a].visits for a in range(2)], dtype=np.float32)
    return counts / counts.sum(), root


def train(steps=300, seed=42, log_enabled=True):
    seed_everything(seed)
    net, opt = LatentModel(), None
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    log = RunLogger("12-muzero-latent", log_enabled)
    for step in range(steps):
        data = make_batch(16, 3, seed + step)
        loss = unroll_loss(net, data)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.)
        opt.step()
        log.scalar("train/unroll_loss", loss.item(), step)
    env = CorridorEnv()
    obs, _ = env.reset()
    probs, _ = latent_mcts(net, obs, simulations=16)
    print("root action probabilities (left, right):", probs)
    log.close()
    return net


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    train(a.steps, a.seed, not a.no_log)


if __name__ == "__main__":
    main()

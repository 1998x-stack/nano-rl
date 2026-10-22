"""14 Decision Transformer: train a return-conditioned causal action predictor.

Interleaved [return-to-go, state, action] tokens. Action predicted from STATE
tokens, *before* seeing same-step actions (causal mask). Dataset is fixed and
no RL gradients or online environmental steps happen during training.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CorridorEnv, RunLogger, seed_everything


def generate_episodes(count=256, horizon=8, seed=42):
    rng, env = np.random.default_rng(seed), CorridorEnv()
    episodes = []
    for _ in range(count):
        s, _, rewards, states, actions = 0, None, [], [], []
        for _ in range(horizon):
            a = int(rng.integers(2)) if rng.random() < .3 else 1
            states.append(s); actions.append(a)
            s, reward, done = env.transition(s, a)
            rewards.append(reward)
            if done:
                break
        rtg = np.cumsum(np.asarray(rewards, dtype=np.float32)[::-1])[::-1].copy()
        episodes.append((states, actions, rtg))
    return episodes


def pad_batch(episodes, length=8):
    batch = len(episodes)
    s = np.zeros((batch, length), dtype=np.int64)
    a = np.zeros_like(s)
    rtg = np.zeros((batch, length), dtype=np.float32)
    mask = np.zeros((batch, length), dtype=np.float32)
    for j, (st, ac, returns) in enumerate(episodes):
        valid = min(length, len(st))
        s[j, :valid], a[j, :valid], rtg[j, :valid] = (
            st[:valid], ac[:valid], returns[:valid])
        mask[j, :valid] = 1.
    return tuple(torch.as_tensor(x) for x in (s, a, rtg, mask))


class DecisionTransformer(nn.Module):
    def __init__(self, context=8, dim=48, layers=2, heads=4):
        super().__init__()
        self.context = context
        self.return_embed = nn.Linear(1, dim)
        self.state_embed = nn.Embedding(5, dim)
        self.action_embed = nn.Embedding(2, dim)
        self.time_embed = nn.Embedding(context, dim)
        block = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                           dim_feedforward=2*dim,
                                           dropout=0., batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, num_layers=layers,
                                                  enable_nested_tensor=False)
        self.head = nn.Linear(dim, 2)

    def forward(self, states, actions, returns):
        B, T = states.shape
        if T > self.context:
            raise ValueError("context window exceeded")
        times = self.time_embed(torch.arange(T, device=states.device))[None]
        rtok = self.return_embed(returns.float().unsqueeze(-1)) + times
        stok = self.state_embed(states.long()) + times
        atok = self.action_embed(actions.long()) + times
        tokens = torch.stack((rtok, stok, atok), dim=2).reshape(B, 3*T, -1)
        # True in the upper triangle means the token CANNOT attend to the future.
        causal = torch.triu(torch.ones(3*T, 3*T, device=states.device,
                                     dtype=torch.bool), diagonal=1)
        hidden = self.transformer(tokens, mask=causal)
        # Position 1+3*t is state_t. Its attention mask hides action_t.
        return self.head(hidden[:, 1::3, :])


def action_loss(model, batch):
    states, actions, rtg, valid = batch
    logits = model(states, actions, rtg)
    per_token = F.cross_entropy(logits.reshape(-1, 2), actions.flatten(),
                                reduction="none").view_as(valid)
    return (per_token * valid).sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, desired_return=.94, max_steps=8):
    env = CorridorEnv()
    obs, _ = env.reset()
    states, actions, returns = [], [], []
    current_return = desired_return
    total, steps = 0., 0
    while steps < max_steps:
        s = int(obs.argmax())
        states.append(s); returns.append(current_return)
        actions.append(0)   # dummy current action, invisible to current state token.
        window = min(model.context, len(states))
        st = torch.tensor(states[-window:])[None]
        ac = torch.tensor(actions[-window:])[None]
        rt = torch.tensor(returns[-window:])[None]
        action = int(model(st, ac, rt)[0, window-1].argmax())
        actions[-1] = action
        obs, reward, done, _, _ = env.step(action)
        total += reward
        current_return -= reward  # RTG condition decreases by observed reward.
        steps += 1
        if done:
            break
    return total, bool(done)


def train(steps=300, seed=42, log_enabled=True):
    seed_everything(seed)
    # Loading is entirely upfront:  no env.step inside the optimizer loop.
    episodes = generate_episodes(seed=seed)
    model = DecisionTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    log = RunLogger("14-decision-transformer", log_enabled)
    rng = np.random.default_rng(seed+10)
    for step in range(steps):
        ids = rng.integers(len(episodes), size=24)
        batch = pad_batch([episodes[i] for i in ids], length=8)
        loss = action_loss(model, batch)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        log.scalar("offline/action_ce", loss.item(), step)
    model.eval()
    evaluation_return, success = evaluate(model)
    print("offline episodes:", len(episodes), "eval return:", evaluation_return,
          "reached terminal:", success)
    log.close()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    train(a.steps, a.seed, not a.no_log)


if __name__ == "__main__":
    main()

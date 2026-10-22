"""13 Dreamer-inspired imagination actor-critic with a tiny stochastic RSSM.

Real corridor transitions train an encoder, deterministic GRU, stochastic
prior/posterior, and observation/reward/continuation heads. Later policy updates
use ONLY imagined latent transitions. This is NOT complete DreamerV3.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence
from utils import CorridorEnv, RunLogger, seed_everything


def normal_from_params(params):
    mean, scale_logits = params.chunk(2, dim=-1)
    return Normal(mean, F.softplus(scale_logits) + .1)


class WorldModel(nn.Module):
    def __init__(self, observations=5, deter=32, stoch=8):
        super().__init__()
        self.deter, self.stoch, self.observations = deter, stoch, observations
        self.encoder = nn.Sequential(nn.Linear(observations, 32), nn.ELU())
        self.gru = nn.GRUCell(stoch + 2, deter)
        self.prior_head = nn.Linear(deter, 2 * stoch)
        self.post_head = nn.Linear(deter + 32, 2 * stoch)
        self.obs_head = nn.Linear(deter + stoch, observations)
        self.reward_head = nn.Linear(deter + stoch, 1)
        self.done_head = nn.Linear(deter + stoch, 1)

    def posterior(self, deter_state, obs):
        return normal_from_params(self.post_head(torch.cat(
            (deter_state, self.encoder(obs)), dim=-1)))

    def initial(self, obs):
        h = torch.zeros(obs.shape[0], self.deter, device=obs.device)
        return h, self.posterior(h, obs).rsample()

    def transition(self, h, z, action):
        one_hot = F.one_hot(action.long(), 2).to(z.dtype)
        h_next = self.gru(torch.cat((z, one_hot), dim=-1), h)
        return h_next, normal_from_params(self.prior_head(h_next))

    @staticmethod
    def feature(h, z):
        return torch.cat((h, z), dim=-1)

    def predict(self, h, z):
        feat = self.feature(h, z)
        return (self.obs_head(feat), self.reward_head(feat).squeeze(-1),
                self.done_head(feat).squeeze(-1))


def collect_sequences(batch_size=24, horizon=5, seed=42):
    """Collect transitions until terminal; pad with mask 0, NOT fabricated experience."""
    rng, env = np.random.default_rng(seed), CorridorEnv()
    states, actions, rewards, terminals, masks = [], [], [], [], []
    for _ in range(batch_size):
        pos = int(rng.integers(0, env.n - 1))
        st, ac, rew, done, valid = [pos], [], [], [], []
        ended = False
        for _ in range(horizon):
            if ended:
                action, reward, terminal, active = 0, 0., True, 0.
            else:
                action = int(rng.integers(2)) if rng.random() < .5 else 1
                pos, reward, terminal = env.transition(pos, action)
                active = 1.
            st.append(pos); ac.append(action)
            rew.append(reward); done.append(float(terminal)); valid.append(active)
            ended = ended or terminal
        states.append(st); actions.append(ac)
        rewards.append(rew); terminals.append(done); masks.append(valid)
    return tuple(torch.tensor(x, dtype=dtype) for x,dtype in zip(
        (states, actions, rewards, terminals, masks),
        (torch.long, torch.long, torch.float32, torch.float32, torch.float32)))


def masked_mean(values, valid):
    """Prevent gradients from post-terminal padded rows."""
    return (values * valid).sum() / valid.sum().clamp_min(1.)


def reconstruction_loss(model, data):
    states, actions, rewards, terminals, valid = data
    obs = F.one_hot(states[:, 0], model.observations).float()
    h, z = model.initial(obs)
    losses = []
    for t in range(actions.shape[1]):
        h_new, prior = model.transition(h, z, actions[:, t])
        # Reward and termination predicted from PRIOR, not leaked next observation.
        prior_z = prior.rsample()
        _, r_hat, done_logits = model.predict(h_new, prior_z)
        next_obs = F.one_hot(states[:, t + 1], model.observations).float()
        post = model.posterior(h_new, next_obs)
        z_new = post.rsample()
        logits, _, _ = model.predict(h_new, z_new)
        kl = kl_divergence(post, prior).sum(-1).clamp(min=.1)
        per_row = (F.cross_entropy(logits, states[:, t+1], reduction="none") +
                   (r_hat - rewards[:, t]).square() +
                   .3 * F.binary_cross_entropy_with_logits(
                       done_logits, terminals[:, t], reduction="none") +
                   .1 * kl)
        losses.append(masked_mean(per_row, valid[:, t]))
        # Preserve the last real state for terminated sequences; do not roll
        # the RSSM forward over absorbing padding rows.
        active = valid[:, t].bool().unsqueeze(-1)
        h = torch.where(active, h_new, h)
        z = torch.where(active, z_new, z)
    return torch.stack(losses).mean()


@torch.no_grad()
def infer_last_state(model, data):
    """Return last REAL posterior state, ignoring any post-terminal padding."""
    states, actions, _, _, valid = data
    h, z = model.initial(F.one_hot(states[:, 0], model.observations).float())
    for t in range(actions.shape[1]):
        proposed_h, _ = model.transition(h, z, actions[:, t])
        proposed_z = model.posterior(
            proposed_h, F.one_hot(states[:, t+1], model.observations).float()).sample()
        active = valid[:, t].bool().unsqueeze(-1)
        h = torch.where(active, proposed_h, h)
        z = torch.where(active, proposed_z, z)
    return h, z


class Agent(nn.Module):
    def __init__(self, features=40):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(features, 64), nn.ELU(), nn.Linear(64, 2))
        self.critic = nn.Sequential(nn.Linear(features, 64), nn.ELU(), nn.Linear(64, 1))

    def distribution(self, feat):
        return Categorical(logits=self.actor(feat))

    def value(self, feat):
        return self.critic(feat).squeeze(-1)


def imagination_loss(model, agent, initial_state, horizon=5, gamma=.95, lam=.95):
    """REINFORCE-style actor advantage from lambda returns in imagined states."""
    h, z = (x.detach() for x in initial_state)
    logps, entropies, features = [], [], []
    imagined_reward, predicted_done, next_features = [], [], []
    for _ in range(horizon):
        feat = model.feature(h, z).detach()
        dist = agent.distribution(feat)
        action = dist.sample()
        features.append(feat)
        logps.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        with torch.no_grad():  # stop model gradients during actor/critic improvement
            h, prior = model.transition(h, z, action)
            z = prior.sample()
            _, r_hat, done_logits = model.predict(h, z)
            imagined_reward.append(r_hat)
            predicted_done.append(torch.sigmoid(done_logits))
            next_features.append(model.feature(h, z).detach())
    with torch.no_grad():
        next_v = [agent.value(x) for x in next_features]
        future = next_v[-1]
        targets = []
        for t in reversed(range(horizon)):
            future = imagined_reward[t] + gamma * (1 - predicted_done[t]) * (
                (1 - lam) * next_v[t] + lam * future)
            targets.insert(0, future)
        targets = torch.stack(targets)
    values = torch.stack([agent.value(x) for x in features])
    advantage = (targets - values).detach()
    logp = torch.stack(logps)
    entropy = torch.stack(entropies)
    actor_loss = -(logp * advantage).mean() - .003 * entropy.mean()
    critic_loss = .5 * F.mse_loss(values, targets)
    return actor_loss, critic_loss, targets.mean().detach()


def train(steps=250, seed=42, log_enabled=True):
    seed_everything(seed)
    model, agent = WorldModel(), Agent()
    model_opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    agent_opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    log = RunLogger("13-dreamer", log_enabled)
    for step in range(steps):
        batch = collect_sequences(seed=seed + step)
        model_loss = reconstruction_loss(model, batch)
        model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        model_opt.step()
        # Start imagined rollouts from observed NON-TERMINAL starts instead of
        # the final posterior, which may already represent a terminal state.
        with torch.no_grad():
            start_obs = F.one_hot(batch[0][:, 0], model.observations).float()
            starting_state = model.initial(start_obs)
        actor_loss, critic_loss, avg_target = imagination_loss(model, agent, starting_state)
        agent_opt.zero_grad()
        (actor_loss + critic_loss).backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 10.)
        agent_opt.step()
        log.scalar("model/reconstruction_total", model_loss.item(), step)
        log.scalar("imagine/actor_loss", actor_loss.item(), step)
        log.scalar("imagine/critic_loss", critic_loss.item(), step)
        log.scalar("imagine/target_return", avg_target.item(), step)
    log.close()
    print("trained RSSM / imagined actor for", steps, "updates")
    return model, agent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    train(a.steps, a.seed, not a.no_log)


if __name__ == "__main__":
    main()

"""Advanced RLHF: genuine causal token-level tiny-GPT with 2-token numeric answers.

Prompt tokens: <BOS> a <PLUS> b <EQUAL>.
Answer: tens digit + ones digit (plus EOS when passed to reward model).
Pipeline: conditional SFT, pairwise Bradley-Terry sequence reward model,
two-token PPO with per-token KL to a frozen SFT reference and value baseline.

Pedagogical: this is a real autoregressive transformer, but tiny and trained on
synthetic numerical preferences (NOT a production LLM or human annotations).
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


BOS, PLUS, EQUAL, EOS, VOCAB = 10, 11, 12, 13, 14


def questions(batch=32):
    a, b = torch.randint(0, 10, (batch,)), torch.randint(0, 10, (batch,))
    bos = torch.full_like(a, BOS)
    prompt = torch.stack((bos, a, torch.full_like(a, PLUS), b,
                          torch.full_like(a, EQUAL)), dim=1)
    total = a + b
    answers = torch.stack((total // 10, total % 10), dim=1)
    return prompt, answers


def masked_logits(logits, token_step):
    """The output grammar is [tens in {0,1}, ones in {0..9}, EOS].

    A restrictive grammar eliminates OOD special tokens during the demo;
    production language-model RLHF does not generally impose this grammar.
    """
    result = logits.clone()
    result[..., (2 if token_step == 0 else 10):] = -1e9
    return result


class MiniGPT(nn.Module):
    def __init__(self, dim=32, layers=1, heads=4):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, dim)
        self.pos = nn.Embedding(8, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, 2*dim, dropout=0., batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.lm_head = nn.Linear(dim, VOCAB)
        self.value_head = nn.Linear(dim, 1)

    def forward(self, tokens):
        B, T = tokens.shape
        if T > 8:
            raise ValueError("context too long")
        x = self.tok(tokens) + self.pos(torch.arange(T, device=tokens.device))[None]
        causal = torch.triu(torch.ones(T,T, device=tokens.device, dtype=torch.bool), diagonal=1)
        h = self.blocks(x, mask=causal)
        return self.lm_head(h), self.value_head(h).squeeze(-1)

    def evaluate(self, prompt, responses):
        # At token position 4: predict response_0; at position 5: predict response_1.
        tokens = torch.cat((prompt, responses[:, :1]), dim=1)
        logits, values = self(tokens)
        first = Categorical(logits=masked_logits(logits[:, 4], 0))
        second = Categorical(logits=masked_logits(logits[:, 5], 1))
        logprob = torch.stack((first.log_prob(responses[:,0]),
                               second.log_prob(responses[:,1])), dim=1)
        return logprob, values[:, 4:6]

    @torch.no_grad()
    def generate(self, prompt):
        context, outputs = prompt.clone(), []
        for step in range(2):
            logits, _ = self(context)
            dist = Categorical(logits=masked_logits(logits[:, -1], step))
            action = dist.sample()
            outputs.append(action)
            context = torch.cat((context, action.unsqueeze(-1)), dim=1)
        return torch.stack(outputs, dim=1)


class SequenceReward(nn.Module):
    def __init__(self, dim=24):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, dim)
        self.rnn = nn.GRU(dim, 32, batch_first=True)
        self.out = nn.Linear(32, 1)

    def forward(self, prompt, completion):
        eos = torch.full((len(prompt), 1), EOS, dtype=torch.long,
                         device=prompt.device)
        tokens = torch.cat((prompt, completion, eos), dim=1)
        _, last = self.rnn(self.embedding(tokens))
        return self.out(last[-1]).squeeze(-1)


def synthetic_preference_pairs(batch=32):
    prompt, correct = questions(batch)
    answer_num = correct[:, 0]*10 + correct[:,1]
    # Other value in [0,18] that cannot equal the true 0..18 total.
    other = (answer_num + torch.randint(1, 19, (batch,))) % 19
    wrong = torch.stack((other // 10, other % 10), dim=1)
    swap = torch.rand(batch) < .5
    first = torch.where(swap[:,None], wrong, correct)
    second = torch.where(swap[:,None], correct, wrong)
    preferred_first = (~swap).float()
    return prompt, first, second, preferred_first


def bt_pairwise_loss(reward_model, data):
    prompt, first, second, prefer_first = data
    diff = reward_model(prompt, first) - reward_model(prompt, second)
    return F.binary_cross_entropy_with_logits(diff, prefer_first)


def supervised_finetuning(policy, steps):
    opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
    for _ in range(steps):
        prompt, answers = questions()
        lp, _ = policy.evaluate(prompt, answers)
        loss = -lp.sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


def learn_preference_model(rm, steps):
    opt = torch.optim.Adam(rm.parameters(), lr=2e-3)
    for _ in range(steps):
        loss = bt_pairwise_loss(rm, synthetic_preference_pairs())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


def token_ppo_step(policy, reference, rm, optimizer, batch=32,
                   beta=.05, epsilon=.2, epochs=2):
    prompt, answer = questions(batch)
    with torch.no_grad():
        actions = policy.generate(prompt)  # old policy behavior rollout
        old_lp, old_v = policy.evaluate(prompt, actions)
        reference_lp, _ = reference.evaluate(prompt, actions)
        score = rm(prompt, actions)
        # Per-token KL shaping plus terminal sequence reward from frozen RM.
        rewards = -beta*(old_lp - reference_lp)
        rewards[:,1] += score
        targets = rewards.detach().clone()
        targets[:,0] = targets[:,0] + targets[:,1]  # γ=1 Monte-Carlo return
        adv = targets - old_v
    for _ in range(epochs):
        new_lp, values = policy.evaluate(prompt, actions)
        clipped, ratio = ppo_surrogate(new_lp, old_lp, adv, epsilon)
        policy_loss = -clipped.mean()
        value_loss = .5 * F.mse_loss(values, targets)
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        optimizer.step()
    with torch.no_grad():
        exact = (actions == answer).all(-1).float().mean()
    return float(loss), float(score.mean()), float(exact), float(ratio.mean())


def train(steps=200, seed=42, log_enabled=True):
    seed_everything(seed)
    policy = MiniGPT()
    log = RunLogger("15-token-rlhf-advanced", log_enabled)
    sft_loss = supervised_finetuning(policy, steps)
    reference = copy.deepcopy(policy).eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    rm = SequenceReward()
    rm_loss = learn_preference_model(rm, steps)
    for param in rm.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    for step in range(steps):
        loss, rm_score, accuracy, ratio = token_ppo_step(
            policy, reference, rm, optimizer)
        log.scalar("token_ppo/loss", loss, step)
        log.scalar("token_ppo/reward_model", rm_score, step)
        log.scalar("token_ppo/ratio", ratio, step)
        log.scalar("eval/exact_response", accuracy, step)
    log.close()
    print("token-SFT loss:", round(sft_loss, 4),
          "pairwise-RM loss:", round(rm_loss, 4))
    return policy, reference, rm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200,
                        help="iterations per stage SFT, preference RM, token PPO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()
    train(args.steps, args.seed, not args.no_log)


if __name__ == "__main__":
    main()

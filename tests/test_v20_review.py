"""Regression tests for v2.0 scientific correctness fixes."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def chapter(name):
    filename = ROOT / name / "train.py"
    spec = importlib.util.spec_from_file_location("v20_"+name.replace("-","_"), filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_alphazero_illegal_actions_excluded_from_policy_normalization():
    m = chapter("11-alphazero-selfplay")
    logits = torch.tensor([[3., -4., .1, .5, -.2, 0., 1., -1., 0.]],
                          dtype=torch.float32, requires_grad=True)
    state = torch.tensor([[1., 0., 0., -1., 0., 0., 0., 0., 0.]])
    logp = m.masked_policy_log_probs(logits, state)
    assert torch.allclose(logp.exp()[:, [1,2,4,5,6,7,8]].sum(-1),
                          torch.ones(1), atol=1e-6)
    # Replacing illegal logits with arbitrarily extreme values must not change loss.
    changed = logits.detach().clone()
    changed[0,0], changed[0,3] = 1e6, 1e6
    other = m.masked_policy_log_probs(changed, state)
    assert torch.allclose(logp[:, [1,2,4,5,6,7,8]],
                          other[:, [1,2,4,5,6,7,8]])
    (-logp[0, 1]).backward()
    assert logits.grad[0, 0] == 0 and logits.grad[0, 3] == 0
    assert logits.grad[0, 1] != 0


def test_alphazero_rejects_bad_search_distribution():
    m = chapter("11-alphazero-selfplay")
    net = m.PolicyValue()
    state = torch.tensor([1., 0., -1., 0., 0., 0., 0., 0., 0.])
    pi = np.zeros(9, dtype=np.float32)
    pi[0] = 1.  # cell already occupied
    with pytest.raises(ValueError, match="illegal"):
        m.policy_value_loss(net, [(state, pi, 0.)])


def test_muzero_padding_masks_terminal_followup():
    m = chapter("12-muzero-latent")
    data = m.make_batch(batch_size=8, unroll=24, seed=10)
    states, actions, rewards, dones, valid = data
    assert states.shape == (8, 25) and valid.shape == (8, 24)
    # Once a trajectory ends, all future actions are padding (mask=0).
    assert torch.all(valid[:, 1:] <= valid[:, :-1])
    assert bool((valid == 0).any())
    for batch_index, row in enumerate(valid):
        indices = torch.nonzero(row == 0).flatten()
        if len(indices):
            t = int(indices[0])
            assert bool((rewards[batch_index,t:] == 0).all())
            assert bool((states[batch_index,t:] == states[batch_index,t]).all())
    model = m.LatentModel()
    loss = m.unroll_loss(model, data)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.reward_head.weight.grad is not None


def test_zero_weight_padded_targets_do_not_change_loss():
    # A masked supervised loss must not react to labels in padded rows.
    m = chapter("12-muzero-latent")
    sample_loss = torch.tensor([1., 1e6], requires_grad=True)
    avg = m.masked_mean(sample_loss, torch.tensor([1., 0.]))
    assert avg.item() == 1.
    avg.backward()
    assert sample_loss.grad[1].item() == 0.

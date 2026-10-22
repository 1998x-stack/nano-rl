"""Dreamer v2.0 terminal masking and no imagined model gradient."""
import importlib.util
import sys
from pathlib import Path

import torch


def module():
    path = Path(__file__).resolve().parents[1] / "13-dreamer-imagination" / "train.py"
    spec = importlib.util.spec_from_file_location("v20_dreamer_model", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_no_post_terminal_fake_data_and_masking():
    m = module()
    states, actions, rewards, dones, valid = m.collect_sequences(
        batch_size=8, horizon=24, seed=5)
    assert valid.shape == (8, 24)
    assert bool((valid == 0).any())
    assert bool((valid[:,1:] <= valid[:,:-1]).all())
    for i in range(len(states)):
        padded = (valid[i] == 0).nonzero().flatten()
        if len(padded):
            at = int(padded[0])
            assert bool((rewards[i,at:] == 0).all())
            assert bool((states[i,at:] == states[i,at]).all())
    model = m.WorldModel()
    loss = m.reconstruction_loss(model, (states,actions,rewards,dones,valid))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.reward_head.weight.grad is not None
    assert torch.isfinite(model.reward_head.weight.grad).all()


def test_imagined_policy_ignores_world_model_gradients():
    m = module()
    model, agent = m.WorldModel(), m.Agent()
    sample = m.collect_sequences(batch_size=3, horizon=4)
    h, z = m.infer_last_state(model, sample)
    model.zero_grad(set_to_none=True)
    actor, critic, estimate = m.imagination_loss(model, agent, (h,z), horizon=2)
    (actor+critic).backward()
    assert all(param.grad is None for param in model.parameters())
    assert agent.actor[0].weight.grad is not None
    assert torch.isfinite(estimate)

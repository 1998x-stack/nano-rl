"""V2 invariant tests: search perspective, no oracle in latent search, stochastic
prior/posterior, causal action mask and preference model frozen references."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from utils import CorridorEnv


ROOT = Path(__file__).resolve().parents[1]


def load(chapter):
    spec = importlib.util.spec_from_file_location(
        "nano_" + chapter.replace("-", "_"), ROOT / chapter / "train.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_corridor_terminal_and_onehot():
    env = CorridorEnv()
    obs, _ = env.reset()
    assert obs.shape == (5,) and obs[0] == 1
    assert env.transition(3, 1) == (4, 1., True)
    assert env.transition(4, 0) == (4, 0., True)
    with pytest.raises(ValueError):
        env.transition(0, 2)


def test_alphazero_player_view_and_legal_moves():
    m = load("11-alphazero-selfplay")
    board = np.asarray([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=np.int8)
    assert torch.equal(m.encode(board, 1), -m.encode(board, -1))
    next_board, next_player = m.apply_move(board, 1, 2)
    assert next_player == -1 and m.status(next_board) == (1, True)
    with pytest.raises(ValueError):
        m.apply_move(board, 1, 0)
    net = m.PolicyValue()
    pi, root = m.mcts(board, 1, net, simulations=5)
    assert np.isclose(pi.sum(), 1.)
    assert pi[0] == 0 and pi[1] == 0 and pi[3] == 0
    assert sum(child.visits for child in root.children.values()) == 5


def test_alphazero_selfplay_targets_and_gradient():
    m = load("11-alphazero-selfplay")
    net = m.PolicyValue()
    samples, outcome = m.self_play(net, simulations=3)
    assert samples and outcome in (-1, 0, 1)
    for features, pi, value in samples:
        assert features.shape == (9,) and np.isclose(pi.sum(), 1.)
        assert value in (-1., 0., 1.)
    loss = m.policy_value_loss(net, samples)
    loss.backward()
    assert torch.isfinite(loss)


def test_muzero_latent_model_unroll_and_search_without_oracle(monkeypatch):
    m = load("12-muzero-latent")
    net = m.LatentModel()
    batch = m.make_batch(batch_size=4, unroll=2)
    loss = m.unroll_loss(net, batch)
    loss.backward()
    assert torch.isfinite(loss)
    # Search MUST use learned model, not known Corridor transition function.
    def forbidden(*args):
        raise AssertionError("Environment oracle leaked into MuZero search")
    monkeypatch.setattr(CorridorEnv, "transition", forbidden)
    probs, root = m.latent_mcts(net, np.eye(5, dtype=np.float32)[0],
                                simulations=5, max_depth=3)
    assert np.isclose(probs.sum(), 1.)
    assert sum(child.visits for child in root.children.values()) == 5


def test_dreamer_stochastic_world_model_and_imagination():
    m = load("13-dreamer-imagination")
    model, agent = m.WorldModel(), m.Agent()
    data = m.collect_sequences(batch_size=4, horizon=3)
    loss = m.reconstruction_loss(model, data)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.prior_head.weight.grad is not None
    initial = m.infer_last_state(model, data)
    actor, critic, ret = m.imagination_loss(model, agent, initial, horizon=3)
    assert all(torch.isfinite(x) for x in (actor, critic, ret))
    (actor+critic).backward()
    assert agent.actor[0].weight.grad is not None


def test_decision_transformer_no_current_action_leak():
    m = load("14-decision-transformer")
    model = m.DecisionTransformer().eval()
    states = torch.tensor([[0, 1, 2, 3]])
    actions = torch.tensor([[0, 0, 0, 0]])
    returns = torch.tensor([[.94, .96, .98, 1.0]])
    edited = actions.clone()
    edited[0, 1] = 1               # current action1 must be invisible at state1
    edited[0, 2] = 1
    with torch.no_grad():
        before = model(states, actions, returns)
        after = model(states, edited, returns)
    assert torch.allclose(before[:, 0], after[:, 0], atol=1e-6)
    assert torch.allclose(before[:, 1], after[:, 1], atol=1e-6)
    episodes = m.generate_episodes(count=4)
    loss = m.action_loss(model, m.pad_batch(episodes))
    assert torch.isfinite(loss)


def test_rlhf_preferences_order_and_fixed_reference():
    m = load("15-rlhf-from-scratch")
    favorable = m.bradley_terry_loss(torch.tensor([2.]), torch.tensor([-2.]),
                                      torch.tensor([1.]))
    unfavorable = m.bradley_terry_loss(torch.tensor([-2.]), torch.tensor([2.]),
                                        torch.tensor([1.]))
    assert favorable < unfavorable
    model, reference, rm, critic = m.TinyAnswerLM(), m.TinyAnswerLM(), m.PreferenceReward(), m.Value()
    reference.load_state_dict(model.state_dict())
    for p in reference.parameters():
        p.requires_grad_(False)
    for p in rm.parameters():
        p.requires_grad_(False)
    before = [p.detach().clone() for p in reference.parameters()]
    optim = torch.optim.Adam(list(model.parameters())+list(critic.parameters()), lr=1e-3)
    loss, reward, accuracy = m.ppo_update(model, reference, rm, critic, optim, batch=8)
    assert np.isfinite(loss) and np.isfinite(reward)
    assert 0 <= accuracy <= 1
    assert all(torch.equal(a, b) for a, b in zip(before, reference.parameters()))

import importlib.util
from pathlib import Path
import numpy as np
import torch
from utils import GridWorld


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / name / "train.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bellman():
    m = load("01-mdp-planning")
    v, policy, errors = m.value_iteration(GridWorld())
    v2, policy2 = m.policy_iteration(GridWorld())
    assert errors[-1] < 1e-8 and np.allclose(v, v2)
    assert np.array_equal(policy[:-1], policy2[:-1])


def test_sarsa_vs_q():
    m = load("02-td-control")
    q = np.zeros((16, 4))
    q[1, 2], q[1, 3] = 10, 2
    m.update_q(q, 0, 1, 1, 1, False, lr=1, gamma=1)
    m.update_q(q, 0, 2, 1, 1, False, lr=1, gamma=1, next_action=3)
    assert q[0, 1] == 11 and q[0, 2] == 3
    m.update_q(q, 0, 3, 1, 1, True, lr=1, gamma=1)
    assert q[0, 3] == 1


def test_reinforce_gradient():
    m = load("03-reinforce")
    p = torch.tensor([-.2, -.3, -.4], requires_grad=True)
    loss, returns = m.reinforce_loss([p[0], p[1], p[2]], [1, 1, 1])
    loss.backward()
    assert torch.isfinite(loss) and p.grad is not None and returns[-1] == 1


def test_dqn_terminal_mask():
    m = load("04-dqn")
    x = m.dqn_target(torch.ones(2), torch.tensor([1., 0.]),
                     torch.tensor([100., 2.]), .9)
    assert torch.allclose(x, torch.tensor([1., 2.8]))


def test_ppo_critic_shape():
    m = load("05-ppo")
    d, v = m.ActorCritic()(torch.randn(5, 4))
    assert d.sample().shape == (5,) and v.shape == (5,)


def test_sac_squash_and_soft_target():
    m = load("06-sac")
    action, logp = m.Actor()(torch.randn(7, 3))
    assert action.shape == logp.shape == (7, 1)
    assert torch.isfinite(logp).all() and (action.abs() <= 2.).all()
    assert m.soft_target(torch.ones(1,1), torch.ones(1,1),
                         torch.ones(1,1)*50, torch.zeros(1,1)).item() == 1


def test_dyna_learns_only_seen_transitions():
    m = load("07-dyna-q")
    q, model, rewards = m.dyna_q(episodes=3, planning_steps=2, log_enabled=False)
    assert len(model) > 0 and len(rewards) == 3
    for (s, a), transition in model.items():
        assert transition == (lambda ns,r,term: (r,ns,term))(*GridWorld().transition(s, a))


def test_offline_coverage():
    m = load("08-offline-fqi")
    data = m.collect_dataset(episodes=5, seed=3)
    q, safe, mask, losses = m.fitted_q_iteration(data, iterations=3)
    assert data.shape[1] == 5 and len(losses) == 3
    assert q.shape == (16,4) and np.isneginf(safe[~mask]).all()


def test_grpo_groups():
    m = load("09-grpo-reasoning")
    r = torch.tensor([[1., 0., 0., 1.], [1., 1., 1., 1.]])
    a = m.group_advantages(r)
    assert torch.allclose(a.mean(-1), torch.zeros(2), atol=1e-6)
    assert torch.all(a[1] == 0)
    assert m.encode_questions(torch.tensor([1, 2]),torch.tensor([3, 4])).shape == (2,20)


def test_multi_agent():
    m = load("10-multi-agent")
    assert m.coordination_reward([1,1]) == 1
    assert m.coordination_reward([1,0]) == 0
    q, rewards = m.independent_q(steps=8, log_enabled=False)
    assert q.shape == (2,2) and len(rewards) == 8

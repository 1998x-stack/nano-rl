import numpy as np
import torch
from utils import GridWorld, ReplayBuffer
from utils.math import discounted_returns, generalized_advantage, ppo_surrogate


def test_grid():
    env = GridWorld()
    assert env.transition(0, 0)[0] == 0
    assert env.transition(14, 2) == (15, 1., True)
    assert env.transition(15, 0) == (15, 0., True)


def test_returns():
    assert np.allclose(discounted_returns([1, 2, 3], .5), [2.75, 3.5, 3.])


def test_gae_time_limit_vs_terminal():
    # A time-limit truncation bootstraps but does not carry later episode advantage.
    a = generalized_advantage([1, 5], [0, 0], [2, 0], [0, 1], [1, 1], .9)
    assert np.allclose(a, [2.8, 5.])
    b = generalized_advantage([1], [0], [999], [1], [1], .9)
    assert np.allclose(b, [1.])


def test_ppo_clipping():
    p, ratio = ppo_surrogate(torch.log(torch.tensor([2.])), torch.zeros(1),
                              torch.tensor([1.]), .2)
    assert torch.allclose(p, torch.tensor([1.2]))
    assert torch.allclose(ratio, torch.tensor([2.]))
    p, _ = ppo_surrogate(torch.log(torch.tensor([.3])), torch.zeros(1),
                         torch.tensor([-1.]), .2)
    assert torch.allclose(p, torch.tensor([-.8]))


def test_replay():
    b = ReplayBuffer(3)
    for i in range(4):
        b.add(np.array([i]), i % 2, i, np.array([i+1]), i == 3)
    assert len(b) == 3
    s, a, r, ns, term = b.sample(3)
    assert s.shape == (3, 1) and term.sum().item() == 1

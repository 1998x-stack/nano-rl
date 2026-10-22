"""10 Independent Q-learning in a two-agent coordination matrix game.

Both agents pick 0/1 simultaneously. Shared reward=1 when matching actions.
The partner's changing policy makes the effective environment non-stationary.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils import RunLogger, seed_everything


def coordination_reward(actions):
    assert len(actions) == 2
    return 1.0 if actions[0] == actions[1] else 0.0


def independent_q(steps=400, seed=42, log_enabled=True):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    q = np.zeros((2, 2), dtype=np.float64)
    log = RunLogger("10-multi-agent", log_enabled)
    rate = []
    for t in range(steps):
        epsilon = max(.05, .35 * (1 - t / max(1, steps)))
        acts = [int(rng.integers(2)) if rng.random() < epsilon else int(np.argmax(q[i]))
                for i in range(2)]
        reward = coordination_reward(acts)
        # One-step stateless game: Q(a) <- Q(a) + lr * (r - Q(a)).
        for agent, act in enumerate(acts):
            q[agent, act] += .1 * (reward - q[agent, act])
        rate.append(reward)
        log.scalar("team/coordination", reward, t)
    log.close()
    return q, rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    q, rate = independent_q(a.steps, a.seed, not a.no_log)
    print("Q table:", q, "\nfinal 50 coordination rate:", np.mean(rate[-50:]))


if __name__ == "__main__":
    main()

"""02 SARSA / Q-Learning: explicit TD target and exact real-interaction accounting."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils import GridWorld, RunLogger, seed_everything


def epsilon_greedy(q, state, epsilon, rng):
    return int(rng.integers(len(q[state]))) if rng.random() < epsilon else int(np.argmax(q[state]))


def update_q(q, s, a, reward, ns, terminated, lr=.1, gamma=.95, next_action=None):
    # SARSA: on-policy Q(ns, next_action), Q-Learning: off-policy max Q(ns, ·).
    best = np.max(q[ns]) if next_action is None else q[ns, next_action]
    td_error = reward + gamma * (not terminated) * best - q[s, a]
    q[s, a] += lr * td_error
    return float(td_error)


def train(algorithm="q", episodes=400, seed=42, log_enabled=True,
          max_env_steps=None, return_stats=False, epsilon=None, lr=.1, gamma=.95):
    assert algorithm in ("q", "sarsa") and episodes >= 1
    assert max_env_steps is None or max_env_steps >= 1
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = GridWorld()
    q = np.zeros((env.n_states, env.n_actions))
    log = RunLogger("02-td-" + algorithm, log_enabled)
    returns, env_steps = [], 0
    for ep in range(episodes):
        if max_env_steps is not None and env_steps >= max_env_steps:
            break
        s, _ = env.reset()
        if epsilon is not None:
            exploration = float(epsilon)
        elif max_env_steps is not None:
            exploration = max(.05, .3 * (1 - env_steps / max_env_steps))
        else:
            exploration = max(.05, .3 * (1 - ep / max(1, episodes)))
        a = epsilon_greedy(q, s, exploration, rng)
        total = 0.
        budget_exhausted = False
        for _ in range(60):
            ns, reward, terminated, _, _ = env.step(a)
            env_steps += 1  # This is a REAL transition, never a planning backup.
            na = epsilon_greedy(q, ns, exploration, rng)
            update_q(q, s, a, reward, ns, terminated, lr=lr, gamma=gamma,
                     next_action=na if algorithm == "sarsa" else None)
            total += reward
            s, a = ns, na
            if terminated:
                break
            if max_env_steps is not None and env_steps >= max_env_steps:
                budget_exhausted = True
                break
        if not budget_exhausted:
            returns.append(float(total))
            log.scalar("episode/return", total, ep)
    log.close()
    stats = {"env_steps": env_steps, "completed_episodes": len(returns),
             "planning_backups": 0}
    return (q, returns, stats) if return_stats else (q, returns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=["q", "sarsa"], default="q")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--env-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    q, returns, stats = train(a.algorithm, a.episodes, a.seed, not a.no_log,
                               max_env_steps=a.env_steps, return_stats=True)
    print("real steps:", stats["env_steps"], "complete episodes:", len(returns))
    if returns:
        print("mean return last 20 =", round(float(np.mean(returns[-20:])), 4))
    print(GridWorld().render_ascii(q.argmax(axis=-1)))


if __name__ == "__main__":
    main()

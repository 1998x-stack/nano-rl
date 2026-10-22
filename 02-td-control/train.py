"""02 TD control: compare on-policy SARSA with off-policy Q-Learning."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils import GridWorld, RunLogger, seed_everything


def epsilon_greedy(q, state, epsilon, rng):
    return int(rng.integers(len(q[state]))) if rng.random() < epsilon else int(np.argmax(q[state]))


def update_q(q, s, a, reward, ns, terminated, lr=.1, gamma=.95, next_action=None):
    # next_action=None => Q-Learning 的 max bootstrap; 提供 next_action => SARSA。
    best = np.max(q[ns]) if next_action is None else q[ns, next_action]
    td_error = reward + gamma * (not terminated) * best - q[s, a]
    q[s, a] += lr * td_error
    return float(td_error)


def train(algorithm="q", episodes=400, seed=42, log_enabled=True):
    assert algorithm in ("q", "sarsa")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env, q = GridWorld(), np.zeros((16, 4))
    log = RunLogger("02-td-" + algorithm, log_enabled)
    returns = []
    for ep in range(episodes):
        s, _ = env.reset()
        epsilon = max(.05, .3 * (1 - ep / max(1, episodes)))
        a = epsilon_greedy(q, s, epsilon, rng)
        total = 0.0
        for _ in range(60):
            ns, reward, terminated, _, _ = env.step(a)
            na = epsilon_greedy(q, ns, epsilon, rng)  # SARSA 用实际采样的下一动作。
            update_q(q, s, a, reward, ns, terminated, next_action=na if algorithm == "sarsa" else None)
            total += reward
            s, a = ns, na
            if terminated:
                break
        returns.append(total)
        log.scalar("episode/return", total, ep)
    log.close()
    return q, returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=["q", "sarsa"], default="q")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    q, returns = train(a.algorithm, a.episodes, a.seed, not a.no_log)
    print("mean return last 20 =", round(float(np.mean(returns[-20:])), 4))
    print(GridWorld().render_ascii(q.argmax(axis=-1)))


if __name__ == "__main__":
    main()

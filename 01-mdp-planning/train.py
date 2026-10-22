"""01 Bellman backups: value iteration and policy iteration on a known tiny MDP."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from utils import GridWorld, RunLogger, seed_everything


def value_iteration(env: GridWorld, gamma=.95, tolerance=1e-9, max_iters=500):
    v = np.zeros(env.n_states)
    errors = []
    for _ in range(max_iters):
        old = v.copy()  # 同步 Bellman 备份；不能在此循环里就地覆盖旧值。
        for s in range(env.n_states):
            if s == env.goal:
                continue
            q = [r + gamma * (not done) * old[ns]
                 for a in range(env.n_actions)
                 for ns, r, done in [env.transition(s, a)]]
            v[s] = max(q)
        errors.append(float(np.max(np.abs(v - old))))
        if errors[-1] < tolerance:
            break
    return v, greedy_policy(env, v, gamma), errors


def greedy_policy(env, v, gamma=.95):
    return np.asarray([
        int(np.argmax([r + gamma * (not done) * v[ns]
                       for a in range(env.n_actions)
                       for ns, r, done in [env.transition(s, a)]]))
        for s in range(env.n_states)], dtype=np.int64)


def policy_iteration(env, gamma=.95, max_iters=100):
    policy = np.zeros(env.n_states, dtype=np.int64)
    v = np.zeros(env.n_states)
    for _ in range(max_iters):
        # 策略评估：固定动作，仅传播该策略下的未来价值。
        for _ in range(1000):
            old = v.copy()
            for s in range(env.n_states):
                ns, r, done = env.transition(s, int(policy[s]))
                v[s] = 0 if s == env.goal else r + gamma * (not done) * old[ns]
            if np.max(np.abs(v - old)) < 1e-9:
                break
        new_policy = greedy_policy(env, v, gamma)
        if np.array_equal(policy[:-1], new_policy[:-1]):
            return v, new_policy
        policy = new_policy
    return v, policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    seed_everything(a.seed)
    env = GridWorld()
    log = RunLogger("01-mdp", not a.no_log)
    v, policy, errors = value_iteration(env, max_iters=a.iterations)
    for i, error in enumerate(errors):
        log.scalar("bellman/residual", error, i)
    print("V(0) =", round(v[0], 5), "Bellman steps =", len(errors))
    print(env.render_ascii(policy))
    log.close()


if __name__ == "__main__":
    main()

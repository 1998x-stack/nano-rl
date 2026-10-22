"""07 Dyna-Q: plan from a learned one-step tabular model, not an oracle model."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils import GridWorld, RunLogger, seed_everything


def dyna_q(episodes=250, planning_steps=10, seed=42, log_enabled=True):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = GridWorld()
    q = np.zeros((env.n_states, env.n_actions))
    model = {}  # model[(s,a)] = (r,s',terminal), learned ONLY from real transitions.
    log = RunLogger("07-dyna-q", log_enabled)
    returns = []
    def backup(s, a, r, ns, terminated):
        error = r + .95 * (not terminated) * np.max(q[ns]) - q[s, a]
        q[s, a] += .15 * error

    for ep in range(episodes):
        s, _ = env.reset()
        total = 0.
        for _ in range(60):
            a = int(rng.integers(4)) if rng.random() < .20 else int(np.argmax(q[s]))
            ns, r, terminated, _, _ = env.step(a)
            backup(s, a, r, ns, terminated)  # Step 1: real TD update.
            model[s, a] = (r, ns, terminated)  # Step 2: identify one-step dynamics.
            pairs = list(model)
            for _ in range(planning_steps):
                pair = pairs[int(rng.integers(len(pairs)))]
                mr, mns, mdone = model[pair]
                backup(pair[0], pair[1], mr, mns, mdone)  # Step 3: simulated TD update.
            total += r
            s = ns
            if terminated:
                break
        returns.append(total)
        log.scalar("episode/return", total, ep)
    log.close()
    return q, model, returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=250)
    p.add_argument("--planning-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    q, model, returns = dyna_q(a.episodes, a.planning_steps, a.seed, not a.no_log)
    print("learned transitions:", len(model), "return:", round(float(np.mean(returns[-20:])), 3))
    print(GridWorld().render_ascii(q.argmax(-1)))


if __name__ == "__main__":
    main()

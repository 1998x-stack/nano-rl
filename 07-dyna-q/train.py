"""07 Dyna-Q: learned transition model, exact env steps + extra imagined TD backups."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils import GridWorld, RunLogger, seed_everything


def dyna_q(episodes=250, planning_steps=10, seed=42, log_enabled=True,
           max_env_steps=None, return_stats=False, epsilon=.2, lr=.15, gamma=.95):
    assert episodes >= 1 and planning_steps >= 0
    assert max_env_steps is None or max_env_steps >= 1
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = GridWorld()
    q = np.zeros((env.n_states, env.n_actions))
    model = {}  # Learned ONLY from real (s, a, r, s', terminated).
    log = RunLogger("07-dyna-q", log_enabled)
    returns, env_steps, planning_backups = [], 0, 0

    def backup(s, a, r, ns, terminated):
        delta = r + gamma * (not terminated) * np.max(q[ns]) - q[s, a]
        q[s, a] += lr * delta

    for ep in range(episodes):
        if max_env_steps is not None and env_steps >= max_env_steps:
            break
        s, _ = env.reset()
        total = 0.
        budget_exhausted = False
        for _ in range(60):
            a = int(rng.integers(env.n_actions)) if rng.random() < epsilon else int(np.argmax(q[s]))
            ns, r, terminated, _, _ = env.step(a)
            env_steps += 1
            backup(s, a, r, ns, terminated)  # One REAL TD update.
            model[s, a] = (r, ns, terminated)
            # Planning always draws ONLY from transitions actually observed so far.
            pairs = list(model)
            for _ in range(planning_steps):
                pair = pairs[int(rng.integers(len(pairs)))]
                mr, mns, mdone = model[pair]
                backup(pair[0], pair[1], mr, mns, mdone)
                planning_backups += 1
            total += r
            s = ns
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
             "planning_backups": planning_backups}
    return (q, model, returns, stats) if return_stats else (q, model, returns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=250)
    p.add_argument("--planning-steps", type=int, default=10)
    p.add_argument("--env-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    q, model, returns, stats = dyna_q(
        a.episodes, a.planning_steps, a.seed, not a.no_log,
        max_env_steps=a.env_steps, return_stats=True)
    print("real steps:", stats["env_steps"],
          "imagined backups:", stats["planning_backups"],
          "model transitions:", len(model))
    if returns:
        print("mean return last 20:", round(float(np.mean(returns[-20:])), 3))
    print(GridWorld().render_ascii(q.argmax(-1)))


if __name__ == "__main__":
    main()

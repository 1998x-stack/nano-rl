"""08 Tabular Fitted Q Iteration (FQI) on a fixed logged dataset.

No environment steps happen DURING fitting. A behavior-coverage mask avoids treating
unseen actions as confidently valued; this is a teaching concession, not full CQL/IQL.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils import GridWorld, RunLogger, seed_everything


def collect_dataset(episodes=500, seed=42, epsilon=.5):
    rng, env = np.random.default_rng(seed), GridWorld()
    rows = []
    # Fixed logging policy: random exploration mixed with a short-path heuristic.
    for _ in range(episodes):
        s, _ = env.reset()
        for _ in range(60):
            row, col = divmod(s, env.size)
            preferred = 2 if col < env.size-1 else 1
            a = int(rng.integers(4)) if rng.random() < epsilon else preferred
            ns, reward, terminal, _, _ = env.step(a)
            rows.append((s, a, reward, ns, terminal))
            s = ns
            if terminal:
                break
    return np.asarray(rows, dtype=np.float64)


def fitted_q_iteration(dataset, iterations=60, gamma=.95):
    """Bellman regression with a tabular function class and dataset-only support."""
    q = np.zeros((16, 4))
    count = np.zeros((16, 4))
    for row in dataset:
        count[int(row[0]), int(row[1])] += 1
    observed = count > 0
    losses = []
    for _ in range(iterations):
        old = q.copy()
        targets = {}
        # 每一轮只从旧 Q 做 bootstrapping，符合 FQI 的同步目标更新。
        for s, a, r, ns, terminal in dataset:
            s, a, ns = int(s), int(a), int(ns)
            supported = old[ns][observed[ns]]
            next_max = float(supported.max()) if supported.size else 0.0
            targets.setdefault((s, a), []).append(r + gamma * (not terminal) * next_max)
        for (s, a), y in targets.items():
            q[s, a] = np.mean(y)
        losses.append(float(np.mean((q-old)**2)))
    # Treat missing actions as unavailable at deployment, never greedily select Q=0 unseen.
    safe_q = np.where(observed, q, -np.inf)
    return q, safe_q, observed, losses


def train(episodes=500, seed=42, log_enabled=True):
    seed_everything(seed)
    data = collect_dataset(episodes, seed)
    q, safe_q, mask, losses = fitted_q_iteration(data)
    log = RunLogger("08-offline-fqi", log_enabled)
    for i, loss in enumerate(losses):
        log.scalar("fit/change", loss, i)
    log.close()
    print("samples:", len(data), "observed actions:", int(mask.sum()), "/ 64")
    print(GridWorld().render_ascii(safe_q.argmax(axis=1)))
    return data, q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    train(a.episodes, a.seed, not a.no_log)


if __name__ == "__main__":
    main()

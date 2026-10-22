"""Multi-seed teaching experiment: Q-Learning vs Dyna-Q on the SAME GridWorld.

This is NOT a publication-quality wall-clock / sample-efficiency claim.
Both agents receive the same number of EPISODES, but may take different real
environment steps; Dyna additionally runs planning backups at each real step.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    import sys
    spec = importlib.util.spec_from_file_location(
        "chapter_" + path.replace("-", "_"), ROOT / path / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def analyze(episodes=100, seeds=(0, 1, 2, 3, 4), planning_steps=10):
    assert episodes >= 1 and len(seeds) >= 1 and planning_steps >= 0
    q = load("02-td-control")
    dyna = load("07-dyna-q")
    per_seed = []
    for seed in seeds:
        _, q_rewards = q.train("q", episodes=episodes, seed=seed, log_enabled=False)
        _, _, dyna_rewards = dyna.dyna_q(
            episodes=episodes, planning_steps=planning_steps, seed=seed, log_enabled=False)
        tail = min(20, episodes)
        per_seed.append({
            "seed": int(seed),
            "q_learning_mean_last_episodes": float(np.mean(q_rewards[-tail:])),
            "dyna_q_mean_last_episodes": float(np.mean(dyna_rewards[-tail:])),
        })

    def summary(key):
        values = np.array([row[key] for row in per_seed], dtype=np.float64)
        # Standard error measures seed variance; NOT a confidence interval or formal p-value.
        return {"mean": float(values.mean()),
                "std_error": float(values.std(ddof=1)/np.sqrt(len(values)))
                if len(values)>1 else None}

    return {
        "environment": "utils.GridWorld(size=4, deterministic)",
        "unit_of_budget": "episodes, NOT matched environment steps / compute",
        "episodes_per_seed": episodes,
        "planning_backups_per_dyna_real_step": planning_steps,
        "per_seed": per_seed,
        "summary": {
            "q_learning": summary("q_learning_mean_last_episodes"),
            "dyna_q": summary("dyna_q_mean_last_episodes"),
        },
        "limitations": ["different real steps per episode", "different planning compute",
                        "toy fixed topology", "not a statistical significance test"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--planning-steps", type=int, default=10)
    p.add_argument("--output", default=None, help="Optional JSON report path")
    a = p.parse_args()
    result = analyze(a.episodes, a.seeds, a.planning_steps)
    report = json.dumps(result, indent=2, ensure_ascii=False)
    print(report)
    if a.output:
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

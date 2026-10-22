"""Equal-real-step TD versus Dyna-Q: comparable interactions, explicit planning cost.

Two update regimes:
- --env-steps: EXACT same number of real GridWorld steps; extra planning backups
  reported separately. Only *completed* episodes contribute to return stats.
- --episodes: legacy quick educational comparison; real steps can differ.

Exploration epsilon, TD learning rate and gamma are matched between algorithms.
No claim of matched wall-time (Dyna uses extra model computations).
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


def analyze(episodes=100, seeds=(0, 1, 2, 3, 4), planning_steps=10,
            env_steps=None, epsilon=.2, lr=.15, gamma=.95):
    assert episodes >= 1 and len(seeds) >= 1 and planning_steps >= 0
    assert env_steps is None or env_steps >= 1
    q, dyna = load("02-td-control"), load("07-dyna-q")
    per_seed = []
    for seed in seeds:
        # Increase episode ceiling so env_steps, not number of resets, stops runs.
        limit = max(episodes, int(env_steps)) if env_steps is not None else episodes
        _, q_rewards, q_stats = q.train(
            "q", episodes=limit, seed=seed, log_enabled=False,
            max_env_steps=env_steps, return_stats=True, epsilon=epsilon,
            lr=lr, gamma=gamma)
        _, _, dyna_rewards, dyna_stats = dyna.dyna_q(
            episodes=limit, planning_steps=planning_steps, seed=seed,
            log_enabled=False, max_env_steps=env_steps, return_stats=True,
            epsilon=epsilon, lr=lr, gamma=gamma)
        tail = 20
        per_seed.append({
            "seed": int(seed),
            "q_learning_mean_last_episodes": float(np.mean(q_rewards[-tail:]))
                if q_rewards else None,
            "dyna_q_mean_last_episodes": float(np.mean(dyna_rewards[-tail:]))
                if dyna_rewards else None,
            "q_learning": q_stats,
            "dyna_q": dyna_stats,
        })
    if env_steps is not None:
        assert all(row["q_learning"]["env_steps"] == env_steps
                   and row["dyna_q"]["env_steps"] == env_steps for row in per_seed)

    def summary(key):
        values = np.array([row[key] for row in per_seed if row[key] is not None],
                          dtype=np.float64)
        if not len(values):
            return {"mean": None, "std_error": None, "valid_seeds": 0}
        return {"mean": float(values.mean()),
                "std_error": float(values.std(ddof=1)/np.sqrt(len(values)))
                if len(values)>1 else None,
                "valid_seeds": len(values)}

    return {
        "environment": "utils.GridWorld(size=4, deterministic)",
        "budget": {"unit": "real_env_steps" if env_steps is not None else "episodes",
                   "per_seed": env_steps if env_steps is not None else episodes},
        "matched_parameters": {"epsilon":epsilon,"learning_rate":lr,"gamma":gamma},
        "planning_backups_per_dyna_real_step": planning_steps,
        "per_seed": per_seed,
        "summary": {
            "q_learning": summary("q_learning_mean_last_episodes"),
            "dyna_q": summary("dyna_q_mean_last_episodes"),
        },
        "limitations": ["only completed episode returns counted",
                        "Dyna has additional planning compute",
                        "toy fixed topology", "not a significance test"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--env-steps", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    p.add_argument("--planning-steps", type=int, default=10)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    result = analyze(a.episodes, a.seeds, a.planning_steps, env_steps=a.env_steps)
    report = json.dumps(result, indent=2, ensure_ascii=False)
    print(report)
    if a.output:
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()

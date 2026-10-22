"""Check multi-seed experiment data shape; no convergence claim."""
import importlib.util
from pathlib import Path
import numpy as np


def test_td_dyna_comparison_small():
    path = Path(__file__).resolve().parents[1] / "experiments" / "compare_td_dyna.py"
    spec = importlib.util.spec_from_file_location("comparison_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = mod.analyze(episodes=3, seeds=(1, 2), planning_steps=2)
    assert len(report["per_seed"]) == 2
    for name in ("q_learning", "dyna_q"):
        assert np.isfinite(report["summary"][name]["mean"])
        assert report["summary"][name]["std_error"] >= 0
    assert report["budget"]["unit"] == "episodes"
    assert report["matched_parameters"]["learning_rate"] == .15



def test_exact_real_step_matching_and_planning_cost():
    path = Path(__file__).resolve().parents[1] / "experiments" / "compare_td_dyna.py"
    spec = importlib.util.spec_from_file_location("compare_exact_steps", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = mod.analyze(episodes=3, env_steps=25, seeds=(0, 1),
                         planning_steps=3)
    assert report["budget"] == {"unit": "real_env_steps", "per_seed": 25}
    for row in report["per_seed"]:
        assert row["q_learning"]["env_steps"] == 25
        assert row["dyna_q"]["env_steps"] == 25
        assert row["q_learning"]["planning_backups"] == 0
        assert row["dyna_q"]["planning_backups"] == 75
        assert row["q_learning"]["completed_episodes"] <= 25
        assert row["dyna_q"]["completed_episodes"] <= 25


def test_extremely_short_budget_not_fabricated_completed_episode():
    path = Path(__file__).resolve().parents[1] / "experiments" / "compare_td_dyna.py"
    spec = importlib.util.spec_from_file_location("compare_one_step", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = mod.analyze(episodes=3, env_steps=1, seeds=(0,), planning_steps=2)
    assert report["summary"]["q_learning"]["valid_seeds"] == 0
    assert report["summary"]["q_learning"]["mean"] is None

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
    assert report["unit_of_budget"].startswith("episodes")

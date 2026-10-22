"""Documentation checks: every algorithm has an interactive lab, README and provenance."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_complete_curriculum():
    names = [
        "01-mdp-planning","02-td-control","03-reinforce","04-dqn","05-ppo",
        "06-sac","07-dyna-q","08-offline-fqi","09-grpo-reasoning","10-multi-agent"
    ]
    assert (ROOT / "index.html").exists()
    for name in names:
        folder=ROOT / name
        for file in ["README.md","train.py","index.html","papers/README.md"]:
            assert (folder/file).is_file(), (name,file)
        html=(folder/"index.html").read_text(encoding="utf-8")
        assert '<input id="slider"' in html and '<svg' in html
        assert name in html and "redraw()" in html
        assert "train.py" in (folder/"README.md").read_text(encoding="utf-8")

"""Check educational artifact completeness, and parse inline JS when node is available."""
import re
import shutil
import subprocess
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
CHAPTERS=[
    "01-mdp-planning","02-td-control","03-reinforce","04-dqn","05-ppo",
    "06-sac","07-dyna-q","08-offline-fqi","09-grpo-reasoning","10-multi-agent",
    "11-alphazero-selfplay","12-muzero-latent","13-dreamer-imagination",
    "14-decision-transformer","15-rlhf-from-scratch"
]


def test_chapter_assets():
    assert (ROOT / "index.html").exists()
    for name in CHAPTERS:
        folder=ROOT/name
        for relative in ("README.md","train.py","index.html","papers/README.md"):
            assert (folder/relative).is_file(), (name,relative)
        html=(folder/"index.html").read_text(encoding="utf-8")
        assert "<input" in html and any(x in html for x in ("<svg", 'id="board"', 'id="trace"', 'id="matrix"'))
        assert "<script>" in html and "train.py" in html
        assert "train.py" in (folder/"README.md").read_text(encoding="utf-8")


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js optional")
def test_inline_scripts_syntax(tmp_path):
    for html_path in [ROOT/"index.html"]+[ROOT/name/"index.html" for name in CHAPTERS]:
        html=html_path.read_text(encoding="utf-8")
        scripts=re.findall(r"<script>(.*?)</script>", html, flags=re.S)
        assert scripts, str(html_path)
        for i,body in enumerate(scripts):
            jsfile=tmp_path/("script_"+str(i)+".js")
            jsfile.write_text(body, encoding="utf-8")
            subprocess.run(["node", "--check", str(jsfile)], check=True, capture_output=True)

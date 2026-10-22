"""Optional paper downloader. PDFs stay local and are not checked into git.

Usage: python scripts/download_papers.py --list
       python scripts/download_papers.py --chapter 05-ppo
"""
import argparse
from pathlib import Path
from urllib.request import urlopen

PAPERS = {
    "01-mdp-planning": ["https://www.cs.rhul.ac.uk/home/tim/reading/rl/bellman1957.pdf"],
    "02-td-control": ["https://link.springer.com/content/pdf/10.1007/BF00992698.pdf"],
    "03-reinforce": ["https://link.springer.com/content/pdf/10.1007/BF00992696.pdf"],
    "04-dqn": ["https://arxiv.org/pdf/1312.5602"],
    "05-ppo": ["https://arxiv.org/pdf/1707.06347"],
    "06-sac": ["https://arxiv.org/pdf/1801.01290"],
    "07-dyna-q": ["https://link.springer.com/content/pdf/10.1007/BF00115009.pdf"],
    "08-offline-fqi": ["https://arxiv.org/pdf/2005.01643"],
    "09-grpo-reasoning": ["https://arxiv.org/pdf/2402.03300"],
    "10-multi-agent": ["https://arxiv.org/pdf/1706.02275"],
}
# Keep potentially inaccessible publisher PDFs as citations only; download only arxiv hosts.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chapter", choices=PAPERS)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    for chapter, urls in PAPERS.items():
        if args.list or not args.chapter:
            print(chapter, *urls, sep=" | ")
            continue
        if chapter != args.chapter:
            continue
        for url in urls:
            if not url.startswith("https://arxiv.org/pdf/"):
                print("Publisher PDF: use link in papers/README.md:", url)
                continue
            dest = Path(chapter) / "papers" / "paper.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading", url, "->", dest)
            with urlopen(url, timeout=25) as response:
                data = response.read(20_000_000)
            if not data.startswith(b"%PDF"):
                raise ValueError("Expected PDF magic bytes")
            dest.write_bytes(data)


if __name__ == "__main__":
    main()

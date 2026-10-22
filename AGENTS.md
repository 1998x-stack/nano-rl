# Agent / contributor guide

## Teaching invariant
Keep the implementation small enough to understand from top to bottom. **Never swap the handwritten algorithm for a black-box trainer.** Style inspiration: karpathy/nanoGPT (minimal data setup, transparent model + training loop, CLI overrides).

## Required files per chapter
- `README.md`: why, math, algorithm sketch, observations, reproduce command, failure cases, paper DOI/arXiv.
- `train.py`: readable single-process CPU-capable reference algorithm; explain each nontrivial update next to the code in Chinese.
- `index.html`: static, dependency-free, interactive animation or graph; usable without a build step.
- `papers/README.md`: source metadata and links; never silently bundle copyrighted PDFs.

## Engineering rules
- Always call `utils.seed_everything(seed)` for deterministic study runs.
- Gymnasium: `obs, info = reset()`; `obs, reward, terminated, truncated, info = step()`; **never bootstrap if terminated**. Truncation is a time limit, and may bootstrap if a value estimate of the last state is available.
- Use `loguru` for logs, `SummaryWriter` for metrics through `utils.RunLogger`.
- Keep algorithm-specific losses in the chapter, common infra in `utils/`.
- Code comments: explain **why** mathematically, not restate the Python syntax.
- Defaults cheap: smoke tests finish on CPU; don't claim benchmark convergence.
- Add tests for invariants and at least one --episodes/--steps tiny run when changing implementations.
- Do not git-commit downloads, benchmark datasets or credentials. Use provenance, license and URL in datasets catalog.
- No insecure `eval()` on untrusted data; constrain PDF download hosts to arxiv.org in helper.
- Prefer clarity to generic abstractions; avoid premature configuration factories.

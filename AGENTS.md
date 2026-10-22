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

## V2 rigorous design contract
- AlphaZero: test legal action masking and alternating-player value negation.
- MuZero: search may call **only learned** h/g/f; use monkeypatch test to reject calls to `CorridorEnv.transition`.
- Dreamer: train posterior/prior from real observations; in imagination no next observation is allowed and model gradients should be stopped for agent updates.
- Decision Transformer: action head reads *state token* position `1+3*t`, masked from same-step action; add anti-leak regression tests.
- RLHF: never expose synthetic ground-truth correct answers to PPO; train Bradley–Terry RM first, then freeze RM and SFT reference.
- Distinguish a mini demonstration from full original paper: document exact simplifications next to claims.
- Run `pytest` and 15 CPU smoke scripts in CI before declaring success; verify computed HTML scripts with Node syntax checker.
- Update both main `index.html` chapters and root `README.md` when creating a new chapter.

## v2.0 audit invariants
- AlphaZero: mask invalid actions in both search and training loss, including logsoftmax normalization.
- MuZero/Dreamer: padded post-terminal rows carry `valid=0`; no supervised gradient on padding and no imagined rollout from already terminal states.
- Matched research experiments: count real `env_steps` and imagined `planning_backups` separately, align epsilon/lr/gamma and report incomplete episode exclusions.
- Token RLHF: evaluation after PPO must sample from the updated policy on fresh prompts, not old rollout actions.
- Review fixes and remaining research debt in `REVIEW_V2.0.md` before merging.

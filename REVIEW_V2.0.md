# nano-rl · v2.0 code review and iteration

Baseline: `main` at `89f4d7cdc7d117aff931d4e11042a6a2758e714e`. Delivery: independent `v2.0` branch; `main` stays unchanged.

## Review scope
Reviewed algorithm training code, data construction, experiment budgets, regression tests, CI and documentation. This is not paper-level formal verification, not performance benchmarking and not cross-browser QA.

| Priority | Confirmed finding | Fix on v2.0 | Verification |
|---|---|---|---|
| P0 | AlphaZero search masks illegal cells, but policy-loss softmax originally did not | Mask policy logits during loss; reject illegal target mass | Gradient/invariance/invalid-target tests |
| P0 | MuZero included artificial post-terminal steps in fixed unroll | Real-step valid masks; mask CE/value/reward/done/dynamics losses | Terminal padding, finite backward |
| P0 | Dreamer trained world model on post-terminal absorbing padding | Episode-valid mask; don't advance RSSM on padded states; imagine from nonterminal start | Padding and actor/model smoke |
| P1 | Q-Learning vs Dyna-Q compared unequal real interactions when episodes differ | `--env-steps`, matched epsilon/lr/gamma, explicit model planning counts | Exact steps per agent and seed, planning count |
| P1 | Budget ending mid-episode could contaminate evaluation | Exclude incomplete episodes; null result if no full episode | Tiny-budget regression |
| P1 | Token RLHF accuracy reported old behavior policy after optimizer step | Evaluate current policy on fresh randomly generated tasks | Token RLHF tests, metric rename |
| P2 | Docs conflated equal episodes and equal real-step budgets | New example and explicit caveats | Review & CI comparison smoke |

## Reproducible equal-interaction experiment

```bash
python experiments/compare_td_dyna.py --env-steps 2000 --seeds 0 1 2 3 4 --planning-steps 10 --output runs/td_dyna_equal_steps.json
python experiments/compare_td_dyna.py --env-steps 2000 --seeds 0 1 2 3 4 --planning-steps 0 --output runs/td_no_planning.json
```

Both algorithms consume exactly the same number of real environment steps and use the same epsilon=.2, lr=.15, gamma=.95. Dyna additionally pays `planning_backups`. Statistics use last 20 **completed** episodes; if none complete, mean and SE are null rather than invented. Neither matched real steps nor smoke tests proves overall computational efficiency or convergence.

## Release requirements
- All existing unit tests and v2.0 review regression tests pass.
- All 15 small CPU training entries, advanced token RLHF and equal-step comparison pass on GitHub Actions for **the v2.0 commit**.
- All 15 chapter READMEs and interactive HTML remain available.
- No force push or modification of main.
- State limits transparently: no benchmark-scale AlphaZero / MuZero / DreamerV3 / LLM PPO reproductions.

## Next iteration — deliberately outstanding
1. Independent train/eval splits and calibrated latent-model error metrics.
2. Five-seed large-budget evaluation with aligned real steps, curves and compute accounting.
3. Save seed, package versions, git SHA, hyperparameters and evaluation logs as reproducible run manifest.
4. Proper variable-length token PPO and masks, independently held-out prompts, reward hacking diagnostics.
5. Verify browsers/accessibility; activate GitHub Pages through owner repository settings (not automatically enabled).

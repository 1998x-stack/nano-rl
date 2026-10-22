# Roadmap · From runnable kernels to paper-grade RL

## V1 → V2 (implemented here)
- [x] 01–10 algorithmic kernels, RunLogger, datasets catalog, CI smoke tests.
- [x] 11–15: AlphaZero, learned-dynamics MuZero, latent imagination Dreamer, Decision Transformer, preference-model RLHF.
- [x] End-to-end sample commands, chapters' papers provenance, interactive pages, correctness tests.
- [x] New reproducible `CorridorEnv` for latent dynamics and sequence modelling.
- [x] Advanced optional causal two-token GPT RLHF path alongside the minimal one-token baseline.

## V3 · stronger experimental science (not yet implemented)
- [ ] Common **evaluation harness**: 5 independent random seeds, bootstrap CI, learning curves with aligned env steps.
- [ ] Dyna-Q vs learned model planning vs PPO controlled budget benchmark.
- [ ] More faithful AlphaZero target value normalization, replay-window iteration, Elo vs random and previous checkpoint.
- [ ] MuZero reanalysis, value/reward support discretization, prioritized sequence replay.
- [ ] DreamerV3 symlog, categorical stochastic representations, lambda returns with continuation-model calibration.
- [ ] Offline DT benchmark with dataset quality tiers, attention ablations, return targeting.
- [ ] Expand the implemented causal two-token RLHF demonstration to variable-length sequence PPO, completion masks, full GAE and online preference/data versioning.
- [ ] RLVR verification against code/math tasks with tool sandbox and regression guardrails.
- [ ] Deterministic exported run manifests (git SHA, seed, env package versions), reproducible report.
- [ ] Accessible, multilingual visualizations (keyboard, touch, math typesetting).

## Policy on claims
V2's small experiments isolate mathematical principles and are not official AlphaZero/MuZero/DreamerV3/InstructGPT performance reproductions. Training success requires evidence from multi-seed evaluation, not a smoke test.

## v2.0 reviewed changes
- [x] AlphaZero policy loss masks all illegal moves and tests gradient exclusion.
- [x] MuZero/Dreamer use valid masks for terminal padding; Dreamer starts imagined rollouts from nonterminal observations.
- [x] Equal-real-step Q-Learning vs Dyna comparison with separate planning overhead and incomplete episode accounting.
- [x] Fresh-current-policy accuracy for token-level RLHF.
- [x] Formal review with prioritized remaining work: [REVIEW_V2.0.md](REVIEW_V2.0.md).
- [ ] Independent held-out evaluation sets and model error calibration.
- [ ] Large-budget multi-seed performance curves with fair wall-time reporting.

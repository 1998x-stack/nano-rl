# Roadmap · From runnable kernels to paper-grade RL

## V1 → V2 (implemented here)
- [x] 01–10 algorithmic kernels, RunLogger, datasets catalog, CI smoke tests.
- [x] 11–15: AlphaZero, learned-dynamics MuZero, latent imagination Dreamer, Decision Transformer, preference-model RLHF.
- [x] End-to-end sample commands, chapters' papers provenance, interactive pages, correctness tests.
- [x] New reproducible `CorridorEnv` for latent dynamics and sequence modelling.

## V3 · stronger experimental science (not yet implemented)
- [ ] Common **evaluation harness**: 5 independent random seeds, bootstrap CI, learning curves with aligned env steps.
- [ ] Dyna-Q vs learned model planning vs PPO controlled budget benchmark.
- [ ] More faithful AlphaZero target value normalization, replay-window iteration, Elo vs random and previous checkpoint.
- [ ] MuZero reanalysis, value/reward support discretization, prioritized sequence replay.
- [ ] DreamerV3 symlog, categorical stochastic representations, lambda returns with continuation-model calibration.
- [ ] Offline DT benchmark with dataset quality tiers, attention ablations, return targeting.
- [ ] Real small-token autoregressive GPT RLHF (SFT → pairwise RM → sequence PPO) with masked KL and advantage traces.
- [ ] RLVR verification against code/math tasks with tool sandbox and regression guardrails.
- [ ] Deterministic exported run manifests (git SHA, seed, env package versions), reproducible report.
- [ ] Accessible, multilingual visualizations (keyboard, touch, math typesetting).

## Policy on claims
V2's small experiments isolate mathematical principles and are not official AlphaZero/MuZero/DreamerV3/InstructGPT performance reproductions. Training success requires evidence from multi-seed evaluation, not a smoke test.

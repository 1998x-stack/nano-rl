# Experimental science · beyond passing a smoke test

## Reproducible comparison

```bash
python experiments/compare_td_dyna.py --episodes 100 --seeds 0 1 2 3 4 --planning-steps 10 --output runs/td_dyna.json
python experiments/compare_td_dyna.py --episodes 100 --seeds 0 1 2 3 4 --planning-steps 0 --output runs/dyna_no_planning.json
```

The script reports **per-seed final-20-episode mean reward**, aggregate sample mean and **standard error**. It is a method teaching experiment, not a paper benchmark. Different episodes may require different real interactions and Dyna adds model planning backups; therefore equal episode count alone does **not** prove superior sample efficiency.

## Research-grade minimum before drawing conclusions
1. Align actual **environment interaction steps**, not episode count alone; measure planning FLOPs or wall time separately.
2. Fix comparable reward normalization, episode time limits, data split and observation space.
3. At least 5 independent seeds, plus learning curves at matched interaction budget.
4. Estimate uncertainty from *seed-level* samples; don't treat trajectory steps as independent replications.
5. Log full config, PyTorch/Gymnasium version, git SHA, data source and seed in the final report.
6. Run ablations: epsilon, gamma, replay, target network, KL, group size, model unroll and MCTS simulations.
7. Report failures and non-convergence. Never infer algorithm dominance from CI smoke tests.

## V2 algorithmic correctness gates
- PUCT root legal moves, zero-sum sign flip and final self-play value targets.
- MuZero latent tree never calls actual environment transition.
- Dreamer imagination stops model gradients and uses prior-only dynamics.
- DT action token is invisible to the state token that predicts it.
- RLHF freezes reference and RM, ensuring synthetic preference labels do not enter PPO.

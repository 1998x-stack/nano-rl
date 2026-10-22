# Environment & dataset ledger

nanoGPT 使用 tiny Shakespeare 作为可一键准备的数据。这里采用相同思路：**先用确定性、可生成的小环境看见算法，再迁移到标准环境/数据集**。

| Stage | Zero-download default | Standard / optional expansion | Link / provenance |
|---|---|---|---|
| 01,02,07 | `utils.GridWorld` 4×4 / synthetic transition model | FrozenLake-v1, CliffWalking-v1 | https://gymnasium.farama.org/environments/toy_text/frozen_lake/ |
| 03–05 | Gymnasium CartPole-v1 (installed with requirements) | Acrobot-v1, LunarLander-v3 | https://gymnasium.farama.org/environments/classic_control/cart_pole/ |
| 06 | Gymnasium Pendulum-v1 | MuJoCo HalfCheetah / Hopper (optional license/runtime) | https://gymnasium.farama.org/environments/classic_control/pendulum/ |
| 08 | Fixed behavior-policy GridWorld trajectories, generated in-script | Minari offline datasets (optional) | https://minari.farama.org/ |
| 09 | Seeded integer arithmetic prompts + exact-match verifiable rewards | TRL GRPOTrainer / Hugging Face datasets | https://huggingface.co/docs/trl/grpo_trainer |
| 10 | Synthetic 2-agent coordination matrix | MPE2 simple_spread_v3 | https://mpe2.farama.org/main/environments/simple_spread/ |
| 11 | TicTacToe self-play generated in-code; no external dataset | PettingZoo TicTacToe, larger games | https://pettingzoo.farama.org/environments/classic/tictactoe/ |
| 12 | CorridorEnv supervised transition sequences | Atari observations + learned models (optional, heavy) | https://arxiv.org/abs/1911.08265 |
| 13 | CorridorEnv sequence → stochastic latent imagination | DreamerV3 open-source reference environments | https://arxiv.org/abs/2301.04104 |
| 14 | Fixed CorridorEnv episodes with return-to-go | D4RL-style offline datasets (optional) | https://arxiv.org/abs/2106.01345 |
| 15 | Synthetic modular-addition prompts, pairwise exact-match preferences | Human-labeled preference corpora (license-dependent) | https://arxiv.org/abs/2203.02155 |

**Behavior coverage matters** in offline RL: fixed data from a random policy cannot demonstrate reliable out-of-distribution Q values. Synthetic data belongs to this repo (generated, no rights issue). Original benchmark datasets have their own terms; do not redistribute by default.

## Benchmark discipline
Use at least five seeds, report mean ± standard error, count *environment steps*, include wall-clock cost, and distinguish `terminated` vs `truncated`. Default examples illustrate mechanics, not published benchmark reproduction.

## V2 train/validation boundary

- 11: rules are available to the tree search; final game outcomes are the supervised value labels. No human expert move labels.
- 12: `CorridorEnv.transition` is allowed while collecting synthetic teacher rollouts only; **not in latent_mcts**.
- 13: posterior sees real observations only in representation training; imagined rollout uses only prior samples and predicted reward/done.
- 14: offline model trains only from fixed trajectories; Transformer predicts `action_t` from `state_t` token with strict causal mask.
- 15: synthetic preference oracle is used only when generating preference pairs; PPO reward must come from the frozen reward model.

Large benchmark acquisition is deliberately optional and governed by source license and reproducibility protocol.

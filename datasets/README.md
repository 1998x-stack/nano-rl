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

**Behavior coverage matters** in offline RL: fixed data from a random policy cannot demonstrate reliable out-of-distribution Q values. Synthetic data belongs to this repo (generated, no rights issue). Original benchmark datasets have their own terms; do not redistribute by default.

## Benchmark discipline
Use at least five seeds, report mean ± standard error, count *environment steps*, include wall-clock cost, and distinguish `terminated` vs `truncated`. Default examples illustrate mechanics, not published benchmark reproduction.

# nano-rl — Reinforcement Learning, one algorithm at a time

> 用 nanoGPT 式的简洁代码，从 Bellman 方程、PPO、GRPO 学到 AlphaZero、MuZero、Dreamer、Decision Transformer 与 RLHF。每章均有可运行实验、逐行关键注释、独立 HTML 可视化，以及原始论文链接。
>
> **目标：小到能读懂，深到能改动，并且每个公式都能定位到代码。**

[![CI](https://github.com/1998x-stack/nano-rl/actions/workflows/tests.yml/badge.svg)](https://github.com/1998x-stack/nano-rl/actions/workflows/tests.yml)
[**交互式 HTML 教材**](index.html) · [GitHub Pages（需要仓库所有者首次启用）](https://1998x-stack.github.io/nano-rl/)

## 学习地图

| 章节 | 核心问题 / 方法 | 默认环境 / 数据 | 入口 |
|---|---|---|---|
| [01 · MDP / DP](01-mdp-planning/) | Bellman 备份、值迭代与策略迭代 | 自包含 GridWorld | `python 01-mdp-planning/train.py` |
| [02 · TD Control](02-td-control/) | Q-Learning vs SARSA、Bootstrapping | 自包含 GridWorld；可扩展 FrozenLake | `python 02-td-control/train.py` |
| [03 · REINFORCE](03-reinforce/) | 回报加权的策略梯度 | CartPole-v1 | `python 03-reinforce/train.py` |
| [04 · DQN](04-dqn/) | Experience Replay + Target Network | CartPole-v1 | `python 04-dqn/train.py` |
| [05 · PPO](05-ppo/) | GAE / Clipped Policy Optimization | CartPole-v1 | `python 05-ppo/train.py` |
| [06 · SAC](06-sac/) | 最大熵目标 + 双 Q 网络 | Pendulum-v1 | `python 06-sac/train.py` |
| [07 · Dyna-Q](07-dyna-q/) | 真实经验与模型规划混合 | 自包含 GridWorld | `python 07-dyna-q/train.py` |
| [08 · Offline RL](08-offline-fqi/) | 固定经验上的 Fitted Q Iteration | 自动生成 GridWorld 行为数据 | `python 08-offline-fqi/train.py` |
| [09 · GRPO](09-grpo-reasoning/) | 组内相对优势 + Clip 更新 | 程序合成的迷你算术任务 | `python 09-grpo-reasoning/train.py` |
| [10 · Multi-Agent](10-multi-agent/) | 多智能体协调和独立 Q-Learning | 自包含协同矩阵游戏 | `python 10-multi-agent/train.py` |
| [11 · AlphaZero](11-alphazero-selfplay/) | PUCT + 神经网络 + 自博弈 | TicTacToe | `python 11-alphazero-selfplay/train.py` |
| [12 · MuZero](12-muzero-latent/) | Learned representation/dynamics/prediction + latent MCTS | CorridorEnv | `python 12-muzero-latent/train.py` |
| [13 · Dreamer](13-dreamer-imagination/) | RSSM posterior/prior + imagined actor-critic | CorridorEnv | `python 13-dreamer-imagination/train.py` |
| [14 · Decision Transformer](14-decision-transformer/) | Return-to-go + causal action modeling | 固定 Corridor trajectories | `python 14-decision-transformer/train.py` |
| [15 · RLHF](15-rlhf-from-scratch/) | SFT → Preference RM → PPO + KL | Synthetic arithmetic preferences | `python 15-rlhf-from-scratch/train.py` |

**章节数字代表建议的学习顺序，不是严格历史时间轴。** 11–15 章探索搜索、学习环境模型、想象学习、序列模型和偏好对齐五条不完全可相互替代的路线。

## 快速开始

```bash
git clone https://github.com/1998x-stack/nano-rl.git
cd nano-rl
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python 01-mdp-planning/train.py
python 05-ppo/train.py --steps 2048
pytest -q
tensorboard --logdir runs
```

全部示例默认使用 CPU、固定随机种子、短训练预算以利教学；想比较真实算法性能，应增加采样预算并跨随机种子评估。完整运行依赖详见 `requirements.txt`；无网络亦可执行 01/02/07/08/09/10/11/12/13/14/15 的核心小环境实验。

## V2 深化教材（论文级知识链条）

[**设计决策、信息泄漏边界和验收标准**](DESIGN.md) · [**按研究级别分阶段升级的 ROADMAP**](ROADMAP.md)

每章独立运行并保留核心数学实现。11 章用真实游戏规则做零和 PUCT；12 章仅在数据采集阶段接触环境转移，在搜索时只能访问学习的潜在动力学；13 章训练随机 RSSM 后于 prior 中想象轨迹；14 章严格保证 state token 无法看到同一步动作答案；15 章冻结奖励模型和参考策略，将程序化偏好替代人工标注；其 [进阶实验](15-rlhf-from-scratch/token_rlhf.py) 进一步实现真正自回归的双 token GPT、序列奖励模型和逐 token PPO。**它们是忠实呈现关键机制的教学缩微实现，而不是官方论文完整复刻**。

可单独运行：

```bash
python 11-alphazero-selfplay/train.py --games 20 --simulations 16
python 12-muzero-latent/train.py --steps 200
python 13-dreamer-imagination/train.py --steps 200
python 14-decision-transformer/train.py --steps 200
python 15-rlhf-from-scratch/train.py --steps 200
python 15-rlhf-from-scratch/token_rlhf.py --steps 200  # 进阶：causal multi-token GPT
```

## 设计约定（源自 nanoGPT 的学习体验）

1. **核心先行**：每章 `train.py` 清晰写出数据采样、计算目标、梯度/表格更新、评估四步；不隐藏核心算法到第三方训练器。
2. **从公式到实现**：README 依次给出公式、关键代码、观察指标、消融实验、论文；HTML 可通过滑块改变超参观察目标函数/更新。
3. **统一但不臃肿**：`utils/` 只提供随机种子、GridWorld、经验回放、loguru + TensorBoard 等共享基建；策略/价值损失函数留在章节内部。
4. **数据可追溯**：环境/离线数据清单位于 [datasets/README.md](datasets/README.md)。不提交第三方数据集/模型权重；离线样本以固定种子可重生成。
5. **论文可获取**：每章 `papers/README.md` 给出官方论文链接，并用 [scripts/download_papers.py](scripts/download_papers.py) 下载至本地。第三方 PDF 默认不提交 git。
6. **优先正确性**：单元测试检查 Bellman 备份、终止状态、优势、概率比、奖励组归一化；CI 做 CPU smoke tests，不保证短训练收敛。

## v2.0 代码审查与质量修复

**[REVIEW_V2.0.md](REVIEW_V2.0.md)** 记录实际发现的算法问题、修改、回归测试及未解决的研究级差距。v2.0 是从 main 创建的独立实验分支，不自动合并。

## 可复现的多随机种子实验

[实验规范与陷阱](experiments/README.md) · [Q-Learning vs Dyna-Q 5 seeds 复现实验](experiments/compare_td_dyna.py)

```bash
python experiments/compare_td_dyna.py --env-steps 2000 --seeds 0 1 2 3 4 --planning-steps 10 --output runs/td_dyna_equal_steps.json
```

使用相同 epsilon、学习率和折扣因子，严格匹配真实环境交互步数，输出各 seed 完成 Episode 的回报、标准误、额外规划备份次数。**真实交互量相同不代表总计算量相同；没有完成的 Episode 不参与均值计算。**

## 检查与工程

```bash
python -m pytest tests -q
python scripts/download_papers.py --list
python scripts/download_papers.py --chapter 05-ppo
python -m http.server 8000
# 访问 http://localhost:8000/index.html
```

训练输出位于 `runs/`，由 `loguru` 输出文本日志，`tensorboard` 绘制 reward、loss、学习率等指标。脚本自动将仓库根目录加入导入路径，确保 `python 05-ppo/train.py` 可直接工作。

## 范围与免责声明

本仓库是透明的**教学实现**，不是官方论文的逐位复现，不声明优于标准实现。09 章是用离散微型策略演示 GRPO 数学机制，**不是**真实预训练 LLM 或全量 RLHF 系统。PPO 的数据仅短周期复用，属于 on-policy 更新范式；08 章固定日志数据则属于 offline RL。论文版权归原作者；请引用原文。

上游风格参考：[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)，借鉴其少文件、代码透明、数据准备步骤，不复制受版权保护的实现。贡献前阅读 [AGENTS.md](AGENTS.md)。

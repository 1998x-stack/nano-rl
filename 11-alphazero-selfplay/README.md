# 11 · AlphaZero: PUCT + self-play

[交互式 HTML 教材](index.html) · [代码 train.py](train.py) · [论文目录](papers/README.md) · [架构设计](../DESIGN.md)

> 从 10 的多智能体学习转向对称、零和、自博弈：训练目标不是手工最优行动，而是树搜索的访问次数和最终输赢。

## 1. 论文定位和任务

**年份 / 研究路线**：2017–2018。

**数据/环境**：TicTacToe, deterministic 3×3, perfect-information, zero-sum。

**论文**：[A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go Through Self-Play](https://arxiv.org/abs/1712.01815)。

重要背景：前一阶段的机制并非被取代。从 10 的多智能体学习转向对称、零和、自博弈：训练目标不是手工最优行动，而是树搜索的访问次数和最终输赢。

## 2. 数学与变量

```text
a* = argmax_a [-Q_child(s,a) + c_puct P(s,a) √N(s)/(1+N(s,a))]
```

请读出每个运算的参与者：观测、动作、概率分布、value、reward/return、监督目标。不要只背诵算法名；要知道哪些量可以 backprop，哪些是从旧策略或环境采集后冻结的数据。

## 3. 单步更新对应到核心代码

1. 用当前玩家视角编码棋盘 x=player·board
2. 策略/价值网络产生合法动作先验 P 和局面价值 V
3. PUCT 多次搜索：选择→规则引擎走子→扩展→交替翻转价值备份
4. self-play 得到 (状态, MCTS 访问分布 π, 最终玩家视角输赢 z)
5. 优化 −π·log_softmax(policy_logits)+MSE(value,z)

`train.py` 保持公开可查的更新顺序，没有依赖第三方算法训练器。统一调用 `utils.seed_everything`、`RunLogger`；当需要环境时调用 `utils.CorridorEnv`。训练结果保存在 TensorBoard 的 `runs/` 目录。

## 4. 运行命令与调试

```bash
pip install -r requirements.txt
python 11-alphazero-selfplay/train.py --help
python 11-alphazero-selfplay/train.py
tensorboard --logdir runs
python -m pytest tests -q
```

建议先在 HTML 中操作公式，再查看 `train.py` 对应函数，最后删掉一个关键项做消融实验。以上是 CPU 小预算教程，并不保证达到论文指标。

## 5. 关键数学与工程不变量

- MCTS 仅在空格落子，已终止棋局禁止再行动
- 零和树搜索在每个父子层级交换视角：child Q 前必须取负
- 最终标签 z = winner * historical_player，不是 winner 的绝对值
- 短训练预算不保证棋力胜过随机基线

## 6. 消融实验与延伸

将 simulations 从 2 改为 8、24，观察访问分布和胜率；比较 c_puct 与探索深度。

**简化边界**：用井字棋和短窗口经验训练，不实现 chess 网络、reanalyse、distributed workers 或真实 Elo；规则模拟器在树搜索内是该类方法的标准假设。

**数据来源 / 许可**：无需下载，内置 TicTacToe。合法动作由棋盘空格决定。 见 [环境数据清单](../datasets/README.md)。若扩展到真实 benchmark，请核实源许可证与评价预算。

**论文 PDF**：在仓库根目录执行 `python scripts/download_papers.py --chapter 11-alphazero-selfplay` 可选择性下载 arXiv 原文到本地 papers/，第三方论文不会提交 Git 仓库。

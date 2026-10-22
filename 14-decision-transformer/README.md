# 14 · Decision Transformer: Return-conditioned Action Modeling

[交互式 HTML 教材](index.html) · [代码 train.py](train.py) · [论文目录](papers/README.md) · [架构设计](../DESIGN.md)

> 不必用 Bellman 值函数估计或策略梯度。用历史状态/动作/剩余目标回报训练序列模型，直接预测下一动作。

## 1. 论文定位和任务

**年份 / 研究路线**：2021。

**数据/环境**：固定 CorridorEnv 轨迹数据，训练期间严格离线。

**论文**：[Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)。

重要背景：前一阶段的机制并非被取代。不必用 Bellman 值函数估计或策略梯度。用历史状态/动作/剩余目标回报训练序列模型，直接预测下一动作。

## 2. 数学与变量

```text
P(a_t | R̂_0,s_0,a_0,...,R̂_t,s_t) (causal self-attention)
```

请读出每个运算的参与者：观测、动作、概率分布、value、reward/return、监督目标。不要只背诵算法名；要知道哪些量可以 backprop，哪些是从旧策略或环境采集后冻结的数据。

## 3. 单步更新对应到核心代码

1. 用行为策略采集固定的 trajectory，计算每个时刻的 Return-to-Go
2. 将 [RTG, state, action] 按时刻交错嵌入 Transformer token stream
3. 使用上三角注意力 mask，保证 state_t 看不到 action_t 和任何未来 token
4. 仅在 state-token 位置做动作交叉熵，padding 不计入 loss
5. 评估时输入期望回报，并在接收真实奖励后更新剩余 RTG

`train.py` 保持公开可查的更新顺序，没有依赖第三方算法训练器。统一调用 `utils.seed_everything`、`RunLogger`；当需要环境时调用 `utils.CorridorEnv`。训练结果保存在 TensorBoard 的 `runs/` 目录。

## 4. 运行命令与调试

```bash
pip install -r requirements.txt
python 14-decision-transformer/train.py --help
python 14-decision-transformer/train.py
tensorboard --logdir runs
python -m pytest tests -q
```

建议先在 HTML 中操作公式，再查看 `train.py` 对应函数，最后删掉一个关键项做消融实验。以上是 CPU 小预算教程，并不保证达到论文指标。

## 5. 关键数学与工程不变量

- state_t 的 logits 在修改 action_t 后不得变化（测试严格检查）
- 使用 mask 保证 causal autoregressive，不依赖动作标签泄露
- 推理的当前 action token 是占位符，不能被当前 state token 访问
- 数据质量与期望回报范围决定何时能可靠规划

## 6. 消融实验与延伸

将期望回报从 0.2 调到 0.95，比较动作分布和 terminal 成功率；比较混合/随机行为数据。

**简化边界**：只有有限状态离散动作，context=8；不声称达到论文 Atari/连续控制性能。

**数据来源 / 许可**：从 CorridorEnv 固定行为策略采集且生成 RTG，train 不再调用 env.step。 见 [环境数据清单](../datasets/README.md)。若扩展到真实 benchmark，请核实源许可证与评价预算。

**论文 PDF**：在仓库根目录执行 `python scripts/download_papers.py --chapter 14-decision-transformer` 可选择性下载 arXiv 原文到本地 papers/，第三方论文不会提交 Git 仓库。

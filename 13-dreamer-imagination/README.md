# 13 · Dreamer: Recurrent Stochastic World Model

[交互式 HTML 教材](index.html) · [代码 train.py](train.py) · [论文目录](papers/README.md) · [架构设计](../DESIGN.md)

> MuZero 使用学得的模型去做有限树搜索；Dreamer 则在学习的世界模型内部生成想象轨迹，用这些轨迹更新 Actor-Critic。

## 1. 论文定位和任务

**年份 / 研究路线**：2023–2025。

**数据/环境**：CorridorEnv 行为轨迹数据，想象控制阶段不访问真实环境。

**论文**：[Mastering Diverse Domains through World Models (DreamerV3)](https://arxiv.org/abs/2301.04104)。

重要背景：前一阶段的机制并非被取代。MuZero 使用学得的模型去做有限树搜索；Dreamer 则在学习的世界模型内部生成想象轨迹，用这些轨迹更新 Actor-Critic。

## 2. 数学与变量

```text
KL[q(z_t|h_t,o_t) || p(z_t|h_t)] + L_obs + L_reward + L_done
```

请读出每个运算的参与者：观测、动作、概率分布、value、reward/return、监督目标。不要只背诵算法名；要知道哪些量可以 backprop，哪些是从旧策略或环境采集后冻结的数据。

## 3. 单步更新对应到核心代码

1. 从实际观测与动作序列更新 GRU 确定状态 h
2. 网络分别输出先验 p(z|h) 与观察后验 q(z|h,o)
3. 通过观测重建、奖励/终止预测和 KL 训练世界模型
4. 从真实轨迹后验初态出发，仅通过 prior 想象未来 action/reward/continue
5. 按 λ-return 构建优势，再更新 Actor 和价值 Critic

`train.py` 保持公开可查的更新顺序，没有依赖第三方算法训练器。统一调用 `utils.seed_everything`、`RunLogger`；当需要环境时调用 `utils.CorridorEnv`。训练结果保存在 TensorBoard 的 `runs/` 目录。

## 4. 运行命令与调试

```bash
pip install -r requirements.txt
python 13-dreamer-imagination/train.py --help
python 13-dreamer-imagination/train.py
tensorboard --logdir runs
python -m pytest tests -q
```

建议先在 HTML 中操作公式，再查看 `train.py` 对应函数，最后删掉一个关键项做消融实验。以上是 CPU 小预算教程，并不保证达到论文指标。

## 5. 关键数学与工程不变量

- 预测 reward 使用 prior latent，不偷偷依赖 next_obs 后验
- 想象训练中固定世界模型（stop gradients）；避免真实未来观测泄露
- KL(q||p) 区别于 KL(p||q)；terminal 模型影响继续折扣
- 这不是原论文完整 DreamerV3 数值稳定性与分布配置

## 6. 消融实验与延伸

比较 prior/posterior KL 系数、imagination horizon、λ；监控 prior 采样 reward 和后验 reconstruction 误差。

**简化边界**：使用简单高斯 RSSM，省略 DreamerV3 的 categorical states、symlog 与 actor gradient 细节；政策更新为 imagination REINFORCE。

**数据来源 / 许可**：CorridorEnv 生成可复现小轨迹；不下载 Atari/Minecraft。 见 [环境数据清单](../datasets/README.md)。若扩展到真实 benchmark，请核实源许可证与评价预算。

**论文 PDF**：在仓库根目录执行 `python scripts/download_papers.py --chapter 13-dreamer-imagination` 可选择性下载 arXiv 原文到本地 papers/，第三方论文不会提交 Git 仓库。

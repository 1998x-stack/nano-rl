# 12 · MuZero: Latent Dynamics & Search

[交互式 HTML 教材](index.html) · [代码 train.py](train.py) · [论文目录](papers/README.md) · [架构设计](../DESIGN.md)

> AlphaZero 搜索依赖已知游戏规则；MuZero 则在训练中学习可规划的隐藏动态与奖励，搜索时不能再访问真实环境转移函数。

## 1. 论文定位和任务

**年份 / 研究路线**：2019–2020。

**数据/环境**：CorridorEnv: 5 状态、两动作、明确终止奖励。

**论文**：[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)。

重要背景：前一阶段的机制并非被取代。AlphaZero 搜索依赖已知游戏规则；MuZero 则在训练中学习可规划的隐藏动态与奖励，搜索时不能再访问真实环境转移函数。

## 2. 数学与变量

```text
h = representation(o); (h',r,done)=g(h,a); (P,V)=f(h)
```

请读出每个运算的参与者：观测、动作、概率分布、value、reward/return、监督目标。不要只背诵算法名；要知道哪些量可以 backprop，哪些是从旧策略或环境采集后冻结的数据。

## 3. 单步更新对应到核心代码

1. 生成固定的 CorridorEnv 转移序列作为有监督训练数据
2. representation h(o) 建立隐状态；recurrent g(h,a) 预测新状态、reward、done
3. unroll K 步，预测 policy/value 与实际 transition/reward 一起训练
4. 搜索阶段只调用 h,g,f，使用 PUCT 选择 learned latent tree action
5. 树搜索对 reward + γ·continuation value 做单智能体正号备份

`train.py` 保持公开可查的更新顺序，没有依赖第三方算法训练器。统一调用 `utils.seed_everything`、`RunLogger`；当需要环境时调用 `utils.CorridorEnv`。训练结果保存在 TensorBoard 的 `runs/` 目录。

## 4. 运行命令与调试

```bash
pip install -r requirements.txt
python 12-muzero-latent/train.py --help
python 12-muzero-latent/train.py
tensorboard --logdir runs
python -m pytest tests -q
```

建议先在 HTML 中操作公式，再查看 `train.py` 对应函数，最后删掉一个关键项做消融实验。以上是 CPU 小预算教程，并不保证达到论文指标。

## 5. 关键数学与工程不变量

- latent_mcts 不调用 CorridorEnv.transition；有测试将它替换成直接报错
- 树上即时奖励保存在 parent→child 边，并参与折扣价值备份
- 动作目标由人工最短路径教师提供，是明确的简化，不是原论文自对弈价值训练
- 不可将模型预测误差小直接等同于策略真实回报高

## 6. 消融实验与延伸

比较 unroll 长度 1/3/5、搜索深度 1/3/5，观察模型误差放大与动作分布。

**简化边界**：监督 teacher labels 代替完整 MuZero 自生成 target/reanalyse；无支持分布、分布式 replay。仍具有实际 h/g/f 和 learned-model tree search。

**数据来源 / 许可**：CorridorEnv 自生成观测/动作/奖励序列；无需下载。 见 [环境数据清单](../datasets/README.md)。若扩展到真实 benchmark，请核实源许可证与评价预算。

**论文 PDF**：在仓库根目录执行 `python scripts/download_papers.py --chapter 12-muzero-latent` 可选择性下载 arXiv 原文到本地 papers/，第三方论文不会提交 Git 仓库。

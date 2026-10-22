# 06 · SAC · Soft Actor-Critic

**历史位置**：2018 · **教学环境**：Gymnasium Pendulum-v1 · continuous action · [在线可视化](index.html)

> 一边学习更高回报，一边通过熵正则避免策略过早收缩；双 Q 缓解价值高估。

## 1. 从上一个阶段到本阶段的核心变化

**问题**：Max-Entropy + Twin Q 解决什么困难？

一边学习更高回报，一边通过熵正则避免策略过早收缩；双 Q 缓解价值高估。

从最简单的小问题开始，而不是先复用大型库。所有策略/目标值更新均能直接在 [train.py](train.py) 阅读和修改。

## 2. 数学目标及直觉

```text
y = r + γ [min(Q̄1,Q̄2)(s',a') − α logπ(a'|s')]
```

公式的直觉：用已采集数据估计下一次改进方向。请观察目标中的 reward、bootstrapping、策略概率比或熵项是否依赖于未来状态、行为策略与训练模型。核心函数：`Actor / Critic / soft_target`。

## 3. 代码走读：一次完整更新

1. 随机探索并累积 ReplayBuffer
2. tanh-Gaussian Actor 产生有界连续动作与修正 logπ
3. 通过 soft Bellman backup 更新两个 Critic
4. 重参数化优化 Actor，Polyak 软更新 target Critic

这份简化代码保留算法的教学核心；训练循环使用 loguru 显示关键指标，TensorBoard 记录 reward/loss 等曲线。共享组件位于 `../utils/`，尽量不将本章优化目标抽象掉。

## 4. 运行实验

```bash
pip install -r requirements.txt
python 06-sac/train.py --help
python 06-sac/train.py
tensorboard --logdir runs
```

目录路径相对于仓库根目录。**默认训练预算很小，教学可运行不等于复现论文的 benchmark。**

### 数据 / 环境来源

Gymnasium Pendulum-v1 · continuous action。完整来源、授权提醒及标准 benchmark 扩展建议见 [datasets/README.md](../datasets/README.md)。自包含 GridWorld 的精确定义在 [utils/envs.py](../utils/envs.py)。

### 消融实验

不同 α 对动作多样性、奖励与收敛的影响。

### 常见 bug / 正确性边界

tanh-squashed Normal 必须加入 Jacobian 修正；本教材固定 α，论文还有自动温度调整形式。

## 5. 自检问题

- 若关闭当前模块中最关键的更新项，算法将退化成哪种方法？
- 哪些量在梯度更新时必须固定？为什么？
- 本章的数据由当前策略还是旧行为策略生成？需要新环境交互吗？
- 终止状态的目标值是否仍允许 bootstrap？

## 6. 核心论文

- [Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)
- [本章 papers 目录与获取说明](papers/README.md)。
- **注意**：代码为原创教学重写，非论文作者的官方实现；论文版权与使用条款请遵循原站说明。

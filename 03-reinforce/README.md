# 03 · REINFORCE · Monte Carlo Policy Gradient

**历史位置**：1992 · **教学环境**：Gymnasium CartPole-v1 · [在线可视化](index.html)

> 不再显式学习每个离散状态动作 Q 表，而是直接通过 log-prob 让高回报动作更可能出现。

## 1. 从上一个阶段到本阶段的核心变化

**问题**：Policy Gradient + Baseline 解决什么困难？

不再显式学习每个离散状态动作 Q 表，而是直接通过 log-prob 让高回报动作更可能出现。

从最简单的小问题开始，而不是先复用大型库。所有策略/目标值更新均能直接在 [train.py](train.py) 阅读和修改。

## 2. 数学目标及直觉

```text
∇J ≈ Σ_t ∇log πθ(a_t|s_t) · (G_t − b)
```

公式的直觉：用已采集数据估计下一次改进方向。请观察目标中的 reward、bootstrapping、策略概率比或熵项是否依赖于未来状态、行为策略与训练模型。核心函数：`Policy / reinforce_loss`。

## 3. 代码走读：一次完整更新

1. 用神经网络产生 Categorical 策略 πθ(a|s)
2. 采集完整 episode 的动作 log-probs 与 reward
3. 倒序计算折扣累计回报 G_t，并减去不依赖动作的 baseline
4. 最小化 −Σ logπ(a|s)·advantage

这份简化代码保留算法的教学核心；训练循环使用 loguru 显示关键指标，TensorBoard 记录 reward/loss 等曲线。共享组件位于 `../utils/`，尽量不将本章优化目标抽象掉。

## 4. 运行实验

```bash
pip install -r requirements.txt
python 03-reinforce/train.py --help
python 03-reinforce/train.py
tensorboard --logdir runs
```

目录路径相对于仓库根目录。**默认训练预算很小，教学可运行不等于复现论文的 benchmark。**

### 数据 / 环境来源

Gymnasium CartPole-v1。完整来源、授权提醒及标准 benchmark 扩展建议见 [datasets/README.md](../datasets/README.md)。自包含 GridWorld 的精确定义在 [utils/envs.py](../utils/envs.py)。

### 消融实验

去掉 baseline 或缩短 episode，观察梯度波动；多随机种子。

### 常见 bug / 正确性边界

advantage 和旧轨迹 reward 当作常数；该实现只用同一 episode 中的返回均值作简单 baseline。

## 5. 自检问题

- 若关闭当前模块中最关键的更新项，算法将退化成哪种方法？
- 哪些量在梯度更新时必须固定？为什么？
- 本章的数据由当前策略还是旧行为策略生成？需要新环境交互吗？
- 终止状态的目标值是否仍允许 bootstrap？

## 6. 核心论文

- [Simple Statistical Gradient-Following Algorithms (Williams, 1992)](https://link.springer.com/article/10.1007/BF00992696)
- [本章 papers 目录与获取说明](papers/README.md)。
- **注意**：代码为原创教学重写，非论文作者的官方实现；论文版权与使用条款请遵循原站说明。

### 关于 Baseline 的一个重要数学细节

经典策略梯度要求 baseline 不依赖于当前选择的动作。本章用**之前 episode 的指数滑动回报平均**作为当前 episode 的固定 baseline；不要直接减去当前完整轨迹的平均回报并断言这必定无偏，因为该平均回报自身取决于轨迹内的动作。完整 Actor-Critic 可改用只依赖状态的 (V(s)) 作为 baseline。

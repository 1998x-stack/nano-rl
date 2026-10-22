# 07 · Dyna-Q · Learn a Model, Then Plan

**历史位置**：1990—1991 · **教学环境**：自包含 GridWorld · online learned model · [在线可视化](index.html)

> 每次真实交互后，从已观察到的转移模型中模拟更多 TD 更新；实现经验与规划的桥梁。

## 1. 从上一个阶段到本阶段的核心变化

**问题**：Learned Transition Model + Planning 解决什么困难？

每次真实交互后，从已观察到的转移模型中模拟更多 TD 更新；实现经验与规划的桥梁。

从最简单的小问题开始，而不是先复用大型库。所有策略/目标值更新均能直接在 [train.py](train.py) 阅读和修改。

## 2. 数学目标及直觉

```text
Q ← Q + α [r̂ + γ max Q(s',·) − Q]
```

公式的直觉：用已采集数据估计下一次改进方向。请观察目标中的 reward、bootstrapping、策略概率比或熵项是否依赖于未来状态、行为策略与训练模型。核心函数：`dyna_q / backup`。

## 3. 代码走读：一次完整更新

1. 真实环境交互一次并更新 Q
2. 存储当前观察到的 model[(s,a)]
3. 均匀采样已见过的 (s,a)
4. 用模拟 transition 额外进行 planning_steps 次 TD 更新

这份简化代码保留算法的教学核心；训练循环使用 loguru 显示关键指标，TensorBoard 记录 reward/loss 等曲线。共享组件位于 `../utils/`，尽量不将本章优化目标抽象掉。

## 4. 运行实验

```bash
pip install -r requirements.txt
python 07-dyna-q/train.py --help
python 07-dyna-q/train.py
tensorboard --logdir runs
```

目录路径相对于仓库根目录。**默认训练预算很小，教学可运行不等于复现论文的 benchmark。**

### 数据 / 环境来源

自包含 GridWorld · online learned model。完整来源、授权提醒及标准 benchmark 扩展建议见 [datasets/README.md](../datasets/README.md)。自包含 GridWorld 的精确定义在 [utils/envs.py](../utils/envs.py)。

### 消融实验

planning_steps=0/5/20 在同样真实交互预算下比较奖励。

### 常见 bug / 正确性边界

模型只允许用真实观察到的状态动作对填充；不要把已知环境 oracle transition 当成 learned model。

## 5. 自检问题

- 若关闭当前模块中最关键的更新项，算法将退化成哪种方法？
- 哪些量在梯度更新时必须固定？为什么？
- 本章的数据由当前策略还是旧行为策略生成？需要新环境交互吗？
- 终止状态的目标值是否仍允许 bootstrap？

## 6. 核心论文

- [Dyna, an Integrated Architecture for Learning, Planning, and Reacting (Sutton, 1991)](https://dl.acm.org/doi/10.1145/122344.122377)
- [本章 papers 目录与获取说明](papers/README.md)。
- **注意**：代码为原创教学重写，非论文作者的官方实现；论文版权与使用条款请遵循原站说明。

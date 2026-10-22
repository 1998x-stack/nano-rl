# 01 · MDP · Bellman Planning

**历史位置**：1950s—1980s · **教学环境**：自包含 GridWorld · 16 states × 4 actions · [在线可视化](index.html)

> 知道转移模型 P 时，直接通过 Bellman 最优性方程规划；下一章将不再依赖 P。

## 1. 从上一个阶段到本阶段的核心变化

**问题**：Value Iteration + Policy Iteration 解决什么困难？

知道转移模型 P 时，直接通过 Bellman 最优性方程规划；下一章将不再依赖 P。

从最简单的小问题开始，而不是先复用大型库。所有策略/目标值更新均能直接在 [train.py](train.py) 阅读和修改。

## 2. 数学目标及直觉

```text
V*(s) = max_a [ r(s,a) + γ V*(s') ]
```

公式的直觉：用已采集数据估计下一次改进方向。请观察目标中的 reward、bootstrapping、策略概率比或熵项是否依赖于未来状态、行为策略与训练模型。核心函数：`value_iteration / policy_iteration`。

## 3. 代码走读：一次完整更新

1. 初始化每个状态 V(s)=0
2. 保存上一轮 oldV，枚举每个状态的所有动作
3. 用已知环境转移计算 r+γV(next)，取 max
4. 最大残差小于阈值时停止；抽取 greedy policy

这份简化代码保留算法的教学核心；训练循环使用 loguru 显示关键指标，TensorBoard 记录 reward/loss 等曲线。共享组件位于 `../utils/`，尽量不将本章优化目标抽象掉。

## 4. 运行实验

```bash
pip install -r requirements.txt
python 01-mdp-planning/train.py --help
python 01-mdp-planning/train.py
tensorboard --logdir runs
```

目录路径相对于仓库根目录。**默认训练预算很小，教学可运行不等于复现论文的 benchmark。**

### 数据 / 环境来源

自包含 GridWorld · 16 states × 4 actions。完整来源、授权提醒及标准 benchmark 扩展建议见 [datasets/README.md](../datasets/README.md)。自包含 GridWorld 的精确定义在 [utils/envs.py](../utils/envs.py)。

### 消融实验

改变 gamma 对远离目标状态的价值如何影响？比较 value_iteration 和 policy_iteration 的输出。

### 常见 bug / 正确性边界

不要就地修改 value_iteration 本轮 oldV，否则会变为异步备份；终止目标状态 V(goal)=0。

## 5. 自检问题

- 若关闭当前模块中最关键的更新项，算法将退化成哪种方法？
- 哪些量在梯度更新时必须固定？为什么？
- 本章的数据由当前策略还是旧行为策略生成？需要新环境交互吗？
- 终止状态的目标值是否仍允许 bootstrap？

## 6. 核心论文

- [Dynamic Programming (Bellman, 1957)](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming)
- [本章 papers 目录与获取说明](papers/README.md)。
- **注意**：代码为原创教学重写，非论文作者的官方实现；论文版权与使用条款请遵循原站说明。

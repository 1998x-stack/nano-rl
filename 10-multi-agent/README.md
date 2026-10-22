# 10 · Multi-Agent · Independent Q-Learning

**历史位置**：1990s—2026 · **教学环境**：两智能体 2×2 协作矩阵游戏；扩展 MPE2 · [在线可视化](index.html)

> 两个智能体同时改变策略，单个智能体眼里的环境因此不断变化；最简协调游戏展示共同奖励的作用。

## 1. 从上一个阶段到本阶段的核心变化

**问题**：Coordination + Non-Stationarity 解决什么困难？

两个智能体同时改变策略，单个智能体眼里的环境因此不断变化；最简协调游戏展示共同奖励的作用。

从最简单的小问题开始，而不是先复用大型库。所有策略/目标值更新均能直接在 [train.py](train.py) 阅读和修改。

## 2. 数学目标及直觉

```text
Q_i(a_i) ← Q_i(a_i) + α [r_team − Q_i(a_i)]
```

公式的直觉：用已采集数据估计下一次改进方向。请观察目标中的 reward、bootstrapping、策略概率比或熵项是否依赖于未来状态、行为策略与训练模型。核心函数：`independent_q / coordination_reward`。

## 3. 代码走读：一次完整更新

1. 两个 agent 各自维持一张动作 Q 表
2. 每轮同时采用 ε-greedy 选择动作
3. 只有动作匹配才获得团队奖励 1
4. 各自使用共享奖励做独立的 Q 更新

这份简化代码保留算法的教学核心；训练循环使用 loguru 显示关键指标，TensorBoard 记录 reward/loss 等曲线。共享组件位于 `../utils/`，尽量不将本章优化目标抽象掉。

## 4. 运行实验

```bash
pip install -r requirements.txt
python 10-multi-agent/train.py --help
python 10-multi-agent/train.py
tensorboard --logdir runs
```

目录路径相对于仓库根目录。**默认训练预算很小，教学可运行不等于复现论文的 benchmark。**

### 数据 / 环境来源

两智能体 2×2 协作矩阵游戏；扩展 MPE2。完整来源、授权提醒及标准 benchmark 扩展建议见 [datasets/README.md](../datasets/README.md)。自包含 GridWorld 的精确定义在 [utils/envs.py](../utils/envs.py)。

### 消融实验

分别禁用/提高探索率，观察共同收敛与失配次数。

### 常见 bug / 正确性边界

当前实验不是 MADDPG，也没有 centralized critic；它仅展示最小的协调学习与非平稳性。

## 5. 自检问题

- 若关闭当前模块中最关键的更新项，算法将退化成哪种方法？
- 哪些量在梯度更新时必须固定？为什么？
- 本章的数据由当前策略还是旧行为策略生成？需要新环境交互吗？
- 终止状态的目标值是否仍允许 bootstrap？

## 6. 核心论文

- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (Lowe et al., 2017)](https://arxiv.org/abs/1706.02275)
- [本章 papers 目录与获取说明](papers/README.md)。
- **注意**：代码为原创教学重写，非论文作者的官方实现；论文版权与使用条款请遵循原站说明。

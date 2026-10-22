# 02 · TD Control · Q-Learning / SARSA

**历史位置**：1988—1992 · **教学环境**：自包含 GridWorld；扩展 FrozenLake-v1 / CliffWalking-v1 · [在线可视化](index.html)

> Q-Learning 用 max Q 计算行为策略以外的目标；SARSA 使用真正执行的下一动作，是 On-policy TD 控制。

## 1. 从上一个阶段到本阶段的核心变化

**问题**：Bootstrap + Exploration 解决什么困难？

Q-Learning 用 max Q 计算行为策略以外的目标；SARSA 使用真正执行的下一动作，是 On-policy TD 控制。

从最简单的小问题开始，而不是先复用大型库。所有策略/目标值更新均能直接在 [train.py](train.py) 阅读和修改。

## 2. 数学目标及直觉

```text
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',·) − Q(s,a)]
```

公式的直觉：用已采集数据估计下一次改进方向。请观察目标中的 reward、bootstrapping、策略概率比或熵项是否依赖于未来状态、行为策略与训练模型。核心函数：`update_q / epsilon_greedy`。

## 3. 代码走读：一次完整更新

1. 按 ε-greedy 从行为策略选动作
2. 与未知环境交互，得到 (s,a,r,s')
3. Q-Learning 使用 max_a' Q(s',a')；SARSA 使用 Q(s',a_next)
4. TD error 驱动局部在线更新

这份简化代码保留算法的教学核心；训练循环使用 loguru 显示关键指标，TensorBoard 记录 reward/loss 等曲线。共享组件位于 `../utils/`，尽量不将本章优化目标抽象掉。

## 4. 运行实验

```bash
pip install -r requirements.txt
python 02-td-control/train.py --help
python 02-td-control/train.py
tensorboard --logdir runs
```

目录路径相对于仓库根目录。**默认训练预算很小，教学可运行不等于复现论文的 benchmark。**

### 数据 / 环境来源

自包含 GridWorld；扩展 FrozenLake-v1 / CliffWalking-v1。完整来源、授权提醒及标准 benchmark 扩展建议见 [datasets/README.md](../datasets/README.md)。自包含 GridWorld 的精确定义在 [utils/envs.py](../utils/envs.py)。

### 消融实验

同一个环境使用较大 epsilon，对比 SARSA 与 Q-Learning 最终策略。

### 常见 bug / 正确性边界

SARSA 的 next_action 必须来自当前 ε-greedy；terminal target 没有 bootstrap 项。

## 5. 自检问题

- 若关闭当前模块中最关键的更新项，算法将退化成哪种方法？
- 哪些量在梯度更新时必须固定？为什么？
- 本章的数据由当前策略还是旧行为策略生成？需要新环境交互吗？
- 终止状态的目标值是否仍允许 bootstrap？

## 6. 核心论文

- [Q-learning (Watkins & Dayan, 1992)](https://link.springer.com/article/10.1007/BF00992698)
- [本章 papers 目录与获取说明](papers/README.md)。
- **注意**：代码为原创教学重写，非论文作者的官方实现；论文版权与使用条款请遵循原站说明。

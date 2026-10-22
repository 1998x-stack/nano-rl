# 08 · Offline RL · Fitted Q Iteration

**历史位置**：2000s—2020s · **教学环境**：固定行为策略日志 · 程序生成的 GridWorld transition · [在线可视化](index.html)

> 训练期间不允许采集新数据；日志覆盖不足会造成分布外动作价值估计问题。

## 1. 从上一个阶段到本阶段的核心变化

**问题**：Fixed Dataset + Bellman Regression 解决什么困难？

训练期间不允许采集新数据；日志覆盖不足会造成分布外动作价值估计问题。

从最简单的小问题开始，而不是先复用大型库。所有策略/目标值更新均能直接在 [train.py](train.py) 阅读和修改。

## 2. 数学目标及直觉

```text
Q_{k+1} = argmin_Q Σ_D [Q(s,a) − (r + γ max Q_k(s',·))]²
```

公式的直觉：用已采集数据估计下一次改进方向。请观察目标中的 reward、bootstrapping、策略概率比或熵项是否依赖于未来状态、行为策略与训练模型。核心函数：`collect_dataset / fitted_q_iteration`。

## 3. 代码走读：一次完整更新

1. 先使用固定 seed 的行为策略采集日志 D
2. 冻结数据集，训练阶段完全不调用 env.step
3. 每一轮构建基于旧 Q 的 Bellman target
4. 仅在数据覆盖的动作集合中选择 greedy action

这份简化代码保留算法的教学核心；训练循环使用 loguru 显示关键指标，TensorBoard 记录 reward/loss 等曲线。共享组件位于 `../utils/`，尽量不将本章优化目标抽象掉。

## 4. 运行实验

```bash
pip install -r requirements.txt
python 08-offline-fqi/train.py --help
python 08-offline-fqi/train.py
tensorboard --logdir runs
```

目录路径相对于仓库根目录。**默认训练预算很小，教学可运行不等于复现论文的 benchmark。**

### 数据 / 环境来源

固定行为策略日志 · 程序生成的 GridWorld transition。完整来源、授权提醒及标准 benchmark 扩展建议见 [datasets/README.md](../datasets/README.md)。自包含 GridWorld 的精确定义在 [utils/envs.py](../utils/envs.py)。

### 消融实验

将 behavior ε 从 0 改为 1，比较覆盖率与策略质量。

### 常见 bug / 正确性边界

这里是带数据支持约束的表格 FQI 示范，不是 CQL；未出现动作的 Q=0 不应直接参与 argmax。

## 5. 自检问题

- 若关闭当前模块中最关键的更新项，算法将退化成哪种方法？
- 哪些量在梯度更新时必须固定？为什么？
- 本章的数据由当前策略还是旧行为策略生成？需要新环境交互吗？
- 终止状态的目标值是否仍允许 bootstrap？

## 6. 核心论文

- [Conservative Q-Learning for Offline Reinforcement Learning (Kumar et al., 2020)](https://arxiv.org/abs/2006.04779)
- [本章 papers 目录与获取说明](papers/README.md)。
- **注意**：代码为原创教学重写，非论文作者的官方实现；论文版权与使用条款请遵循原站说明。

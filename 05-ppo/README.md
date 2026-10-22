# 05 · PPO · Proximal Policy Optimization

**历史位置**：2017 · **教学环境**：Gymnasium CartPole-v1 · [在线可视化](index.html)

> 同一批新采样轨迹可以多轮优化，但过大概率比改变不能无限增加目标函数。

## 1. 从上一个阶段到本阶段的核心变化

**问题**：GAE + Clipped Policy Ratio 解决什么困难？

同一批新采样轨迹可以多轮优化，但过大概率比改变不能无限增加目标函数。

从最简单的小问题开始，而不是先复用大型库。所有策略/目标值更新均能直接在 [train.py](train.py) 阅读和修改。

## 2. 数学目标及直觉

```text
Lclip = E[min(rθ A, clip(rθ,1−ε,1+ε) A)]
```

公式的直觉：用已采集数据估计下一次改进方向。请观察目标中的 reward、bootstrapping、策略概率比或熵项是否依赖于未来状态、行为策略与训练模型。核心函数：`ActorCritic / generalized_advantage / ppo_surrogate`。

## 3. 代码走读：一次完整更新

1. 使用当前策略收集 horizon 个 (s,a,logπ_old,r,V)
2. 用 GAE 计算 advantage 与 bootstrap return
3. 把 old log-probs 当作固定数据
4. 4 个 epoch 小批量最大化 clipped surrogate，加 value loss / entropy

这份简化代码保留算法的教学核心；训练循环使用 loguru 显示关键指标，TensorBoard 记录 reward/loss 等曲线。共享组件位于 `../utils/`，尽量不将本章优化目标抽象掉。

## 4. 运行实验

```bash
pip install -r requirements.txt
python 05-ppo/train.py --help
python 05-ppo/train.py
tensorboard --logdir runs
```

目录路径相对于仓库根目录。**默认训练预算很小，教学可运行不等于复现论文的 benchmark。**

### 数据 / 环境来源

Gymnasium CartPole-v1。完整来源、授权提醒及标准 benchmark 扩展建议见 [datasets/README.md](../datasets/README.md)。自包含 GridWorld 的精确定义在 [utils/envs.py](../utils/envs.py)。

### 消融实验

扫 ε=0.05/0.2/0.4，观察 ratio、return、approx KL。

### 常见 bug / 正确性边界

旧 log-probs 不得用更新后的模型重新计算；要区分 terminated 与 truncated，不能跨 episode 重置传播 GAE。

## 5. 自检问题

- 若关闭当前模块中最关键的更新项，算法将退化成哪种方法？
- 哪些量在梯度更新时必须固定？为什么？
- 本章的数据由当前策略还是旧行为策略生成？需要新环境交互吗？
- 终止状态的目标值是否仍允许 bootstrap？

## 6. 核心论文

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [本章 papers 目录与获取说明](papers/README.md)。
- **注意**：代码为原创教学重写，非论文作者的官方实现；论文版权与使用条款请遵循原站说明。

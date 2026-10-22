# 04 · DQN · Deep Q-Network

**历史位置**：2013—2015 · **教学环境**：Gymnasium CartPole-v1 · [在线可视化](index.html)

> 用深度网络代替 Q 表；经验回放降低连续数据的相关性，慢更新 target 网络降低不稳定性。

## 1. 从上一个阶段到本阶段的核心变化

**问题**：Replay + Target Q 解决什么困难？

用深度网络代替 Q 表；经验回放降低连续数据的相关性，慢更新 target 网络降低不稳定性。

从最简单的小问题开始，而不是先复用大型库。所有策略/目标值更新均能直接在 [train.py](train.py) 阅读和修改。

## 2. 数学目标及直觉

```text
y = r + γ (1−terminal) max_a Q_target(s',a)
```

公式的直觉：用已采集数据估计下一次改进方向。请观察目标中的 reward、bootstrapping、策略概率比或熵项是否依赖于未来状态、行为策略与训练模型。核心函数：`QNet / dqn_target / train`。

## 3. 代码走读：一次完整更新

1. ε-greedy 采样 CartPole transition
2. 存入 ReplayBuffer 并随机抽取 minibatch
3. 利用冻结的 target network 计算 y
4. 最小化 Huber(Q_online(s,a),y)，定期同步 target

这份简化代码保留算法的教学核心；训练循环使用 loguru 显示关键指标，TensorBoard 记录 reward/loss 等曲线。共享组件位于 `../utils/`，尽量不将本章优化目标抽象掉。

## 4. 运行实验

```bash
pip install -r requirements.txt
python 04-dqn/train.py --help
python 04-dqn/train.py
tensorboard --logdir runs
```

目录路径相对于仓库根目录。**默认训练预算很小，教学可运行不等于复现论文的 benchmark。**

### 数据 / 环境来源

Gymnasium CartPole-v1。完整来源、授权提醒及标准 benchmark 扩展建议见 [datasets/README.md](../datasets/README.md)。自包含 GridWorld 的精确定义在 [utils/envs.py](../utils/envs.py)。

### 消融实验

关闭 replay 或频繁同步 target 时，比较损失曲线；只支持离散动作。

### 常见 bug / 正确性边界

target 只用于计算监督目标，不允许梯度回传；ReplayBuffer 中存 terminated 而非合并 done。

## 5. 自检问题

- 若关闭当前模块中最关键的更新项，算法将退化成哪种方法？
- 哪些量在梯度更新时必须固定？为什么？
- 本章的数据由当前策略还是旧行为策略生成？需要新环境交互吗？
- 终止状态的目标值是否仍允许 bootstrap？

## 6. 核心论文

- [Human-level Control through Deep Reinforcement Learning (Mnih et al., 2015)](https://doi.org/10.1038/nature14236)
- [本章 papers 目录与获取说明](papers/README.md)。
- **注意**：代码为原创教学重写，非论文作者的官方实现；论文版权与使用条款请遵循原站说明。

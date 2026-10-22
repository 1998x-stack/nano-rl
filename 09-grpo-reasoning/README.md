# 09 · GRPO · Verifiable Reward Reasoning

**历史位置**：2024—2026 · **教学环境**：自动生成 (a+b)%10 算术题 / exact-match reward · [在线可视化](index.html)

> 用同一问题的多组采样结果做相对比较，在这个教学模型中不训练独立 value critic。

## 1. 从上一个阶段到本阶段的核心变化

**问题**：Group-Relative Advantages + PPO Clip 解决什么困难？

用同一问题的多组采样结果做相对比较，在这个教学模型中不训练独立 value critic。

从最简单的小问题开始，而不是先复用大型库。所有策略/目标值更新均能直接在 [train.py](train.py) 阅读和修改。

## 2. 数学目标及直觉

```text
A_i = (r_i − mean(r_group)) / (std(r_group) + δ)
```

公式的直觉：用已采集数据估计下一次改进方向。请观察目标中的 reward、bootstrapping、策略概率比或熵项是否依赖于未来状态、行为策略与训练模型。核心函数：`TinyPolicy / group_advantages / train`。

## 3. 代码走读：一次完整更新

1. 随机生成一个算术问题，采样 G 个离散候选答案
2. 使用程序验证是否正确，形成组内 reward
3. 组内中心化并标准化 Advantage
4. 固定旧策略 logπ，通过 clipped surrogate 和 reference KL 更新

这份简化代码保留算法的教学核心；训练循环使用 loguru 显示关键指标，TensorBoard 记录 reward/loss 等曲线。共享组件位于 `../utils/`，尽量不将本章优化目标抽象掉。

## 4. 运行实验

```bash
pip install -r requirements.txt
python 09-grpo-reasoning/train.py --help
python 09-grpo-reasoning/train.py
tensorboard --logdir runs
```

目录路径相对于仓库根目录。**默认训练预算很小，教学可运行不等于复现论文的 benchmark。**

### 数据 / 环境来源

自动生成 (a+b)%10 算术题 / exact-match reward。完整来源、授权提醒及标准 benchmark 扩展建议见 [datasets/README.md](../datasets/README.md)。自包含 GridWorld 的精确定义在 [utils/envs.py](../utils/envs.py)。

### 消融实验

观察 G=2/8/16 对组内非零方差概率的影响。

### 常见 bug / 正确性边界

这是离散小策略的数学玩具，不是 LLM：没有 token 序列、masked completion 或分布式推理。全同 reward 会得到零组优势。

## 5. 自检问题

- 若关闭当前模块中最关键的更新项，算法将退化成哪种方法？
- 哪些量在梯度更新时必须固定？为什么？
- 本章的数据由当前策略还是旧行为策略生成？需要新环境交互吗？
- 终止状态的目标值是否仍允许 bootstrap？

## 6. 核心论文

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (2024)](https://arxiv.org/abs/2402.03300)
- [本章 papers 目录与获取说明](papers/README.md)。
- **注意**：代码为原创教学重写，非论文作者的官方实现；论文版权与使用条款请遵循原站说明。

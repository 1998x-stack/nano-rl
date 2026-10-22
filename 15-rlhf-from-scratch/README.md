# 15 · RLHF: SFT → Reward Model → PPO

[交互式 HTML 教材](index.html) · [代码 train.py](train.py) · [论文目录](papers/README.md) · [架构设计](../DESIGN.md)

> 策略奖励不一定来自环境自带标量；RLHF 用成对偏好训练奖励模型，再以受约束的强化学习更新 SFT 初始化的语言策略。

## 1. 论文定位和任务

**年份 / 研究路线**：2017–2022。

**数据/环境**：合成个位数加法问答与成对正确性偏好。

**论文**：[Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)。

重要背景：前一阶段的机制并非被取代。策略奖励不一定来自环境自带标量；RLHF 用成对偏好训练奖励模型，再以受约束的强化学习更新 SFT 初始化的语言策略。

## 2. 数学与变量

```text
L_RM = BCEWithLogits(R(y_good)-R(y_bad),label); L_PPO = -L_clip + L_value + β KL
```

请读出每个运算的参与者：观测、动作、概率分布、value、reward/return、监督目标。不要只背诵算法名；要知道哪些量可以 backprop，哪些是从旧策略或环境采集后冻结的数据。

## 3. 单步更新对应到核心代码

1. SFT: 合成指令 (a+b)%10，监督训练单 token Answer Policy
2. Pairwise preference: 对同一问题的两个答案生成正确性偏好标签
3. Bradley–Terry logistic loss 训练奖励模型 R(prompt,completion)
4. 冻结 reward model 与初始 SFT reference policy
5. 采样 PPO 动作，按 RM reward−β(logπ_old−logπ_ref) 优化 clipped 策略目标

`train.py` 保持公开可查的更新顺序，没有依赖第三方算法训练器。统一调用 `utils.seed_everything`、`RunLogger`；当需要环境时调用 `utils.CorridorEnv`。训练结果保存在 TensorBoard 的 `runs/` 目录。

## 4. 运行命令与调试

```bash
pip install -r requirements.txt
python 15-rlhf-from-scratch/train.py --help
python 15-rlhf-from-scratch/train.py
tensorboard --logdir runs
python -m pytest tests -q
```

建议先在 HTML 中操作公式，再查看 `train.py` 对应函数，最后删掉一个关键项做消融实验。以上是 CPU 小预算教程，并不保证达到论文指标。

## 5. 关键数学与工程不变量

- Reward Model 的偏好标签在 PPO 阶段不可被直接访问
- 参考策略和奖励模型都必须 frozen，旧动作 logprob 在多个 PPO epoch 内固定
- KL 使用 logπ_old−logπ_ref 的采样估计，区别于准确批量 KL
- 这是单 token 的 conditional policy，不是可生成自然语言的 GPT

## 6. 消融实验与延伸

奖励模型训练轮数、KL β 0/0.05/0.2、PPO 裁剪 ε 对真实答题准确率及模型奖励的影响。

**简化边界**：用合成偏好取代真正人类标注，用一次答案选择而非 token-level autoregressive PPO；用以解释 pipeline，不是 InstructGPT 重现。

**数据来源 / 许可**：脚本内生成一位整数及正确答案，偏好标签明确标为 synthetic。 见 [环境数据清单](../datasets/README.md)。若扩展到真实 benchmark，请核实源许可证与评价预算。

**论文 PDF**：在仓库根目录执行 `python scripts/download_papers.py --chapter 15-rlhf-from-scratch` 可选择性下载 arXiv 原文到本地 papers/，第三方论文不会提交 Git 仓库。

## 7. Advanced lab: genuine autoregressive token-level GPT RLHF

The default `train.py` exposes the clearest one-token RLHF mathematics. Once you understand its SFT → preference model → clipped PPO updates, continue with **[token_rlhf.py](token_rlhf.py)**, which expands the same approach to an actual causal Transformer:

```bash
python 15-rlhf-from-scratch/token_rlhf.py --steps 200
tensorboard --logdir runs
```

Unlike the introductory categorical model, this new experiment has:
- **A causal Transformer** over prompt tokens `[BOS,a,PLUS,b,EQUAL]`, followed by **two sequentially sampled answer digits** and `EOS` for the reward model.
- **Response-token SFT** at autoregressive positions 4 and 5; the model cannot read its future answer.
- **Pairwise sequence reward model** (GRU + Bradley–Terry), learned from synthetic answer preferences.
- **Token-level KL shaping** against a frozen copy of the SFT policy, old log-probs frozen during PPO epochs, and a value head for per-token returns.
- A restricted arithmetic grammar (tens 0–1, ones 0–9). This avoids untrained special tokens and is **not** a general-purpose LLM.

Code reading sequence: `MiniGPT.forward` → `MiniGPT.evaluate` → `SequenceReward` → `supervised_finetuning` → `learn_preference_model` → `token_ppo_step`. Follow the action-token positions and explicitly verify no same-step token leakage.

The toy data are **not** human preferences, and two generated digits are **not** the full variable-length masked sequence PPO used with modern large LLMs. This experiment is a technically meaningful bridge toward that implementation.

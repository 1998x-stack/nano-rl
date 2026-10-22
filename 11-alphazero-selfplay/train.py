"""11 AlphaZero kernel: TicTacToe self-play + neural PUCT MCTS.

No supervised move labels: policy targets are MCTS visit counts and value targets
are final game outcomes. Rules are used inside tree search (as in AlphaZero).
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import RunLogger, seed_everything


WIN_LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),
             (2,5,8),(0,4,8),(2,4,6))


def status(board):
    """Return (winner in {-1,0,+1}, terminal). 0 means a draw or no winner."""
    b = np.asarray(board)
    for line in WIN_LINES:
        if b[line[0]] != 0 and b[line[0]] == b[line[1]] == b[line[2]]:
            return int(b[line[0]]), True
    return 0, not (b == 0).any()


def apply_move(board, player, action):
    if status(board)[1] or action not in range(9) or board[action] != 0:
        raise ValueError("illegal action or terminal board")
    new_board = np.asarray(board, dtype=np.int8).copy()
    new_board[action] = player
    return new_board, -player


def encode(board, player):
    # 每个状态以当前执手的视角编码；换玩家时符号翻转。
    return torch.as_tensor(np.asarray(board, dtype=np.float32) * player)


class PolicyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(9, 64), nn.ReLU(),
                                   nn.Linear(64, 64), nn.ReLU())
        self.policy = nn.Linear(64, 9)
        self.value = nn.Sequential(nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.trunk(x)
        return self.policy(h), self.value(h).squeeze(-1)


@dataclass
class Node:
    prior: float = 1.
    visits: int = 0
    value_sum: float = 0.
    children: dict = field(default_factory=dict)

    def value(self):
        return self.value_sum / self.visits if self.visits else 0.


@torch.no_grad()
def expand(node, board, player, net):
    logits, value = net(encode(board, player))
    moves = np.flatnonzero(np.asarray(board) == 0)
    if len(moves):
        priors = torch.softmax(logits[moves], dim=0).cpu().numpy()
        node.children = {int(a): Node(float(p)) for a, p in zip(moves, priors)}
    return float(value.item())


def select(node, c_puct=1.5):
    """child.value() is from the NEXT player's view => negate for parent."""
    return max(node.children.items(),
               key=lambda item: -item[1].value() +
               c_puct * item[1].prior * np.sqrt(node.visits + 1) /
               (item[1].visits + 1))


@torch.no_grad()
def mcts(board, player, net, simulations=24):
    assert simulations >= 1 and not status(board)[1]
    root = Node()
    expand(root, board, player, net)
    for _ in range(simulations):
        b, turn, node = np.asarray(board).copy(), player, root
        path = [node]
        while node.children:
            action, child = select(node)
            b, turn = apply_move(b, turn, action)
            node = child
            path.append(node)
            winner, terminal = status(b)
            if terminal:
                break
        winner, terminal = status(b)
        # terminal 的 winner 要从“下一位执手”的视角转换。
        value = float(winner * turn) if terminal else expand(node, b, turn, net)
        for ancestor in reversed(path):
            ancestor.visits += 1
            ancestor.value_sum += value
            value = -value  # zero-sum perspective alternates each ply.
    probs = np.zeros(9, dtype=np.float32)
    for action, child in root.children.items():
        probs[action] = child.visits
    probs /= probs.sum()
    return probs, root


def self_play(net, simulations=24):
    board, player = np.zeros(9, dtype=np.int8), 1
    trajectory = []
    while not status(board)[1]:
        visits, _ = mcts(board, player, net, simulations)
        trajectory.append((encode(board, player), visits.copy(), player))
        action = int(np.random.choice(9, p=visits))
        board, player = apply_move(board, player, action)
    winner, _ = status(board)
    # 最终胜负对每个历史行动玩家的价值目标可能符号相反。
    examples = [(s, pi, float(winner * turn)) for s, pi, turn in trajectory]
    return examples, winner


def masked_policy_log_probs(logits, states):
    """非法落子在策略损失的 softmax 分母中也必须被排除。

    states 的符号取决于当前执手；等于 0 才代表未落子的合法格。
    仅在搜索时 mask、训练时不 mask，会把概率质量浪费在无效动作上。
    """
    legal = states == 0
    if not bool(legal.any(dim=-1).all()):
        raise ValueError("policy target requires at least one legal move")
    return F.log_softmax(logits.masked_fill(~legal, -1e9), dim=-1)


def policy_value_loss(net, examples):
    states = torch.stack([e[0] for e in examples])
    target_pi = torch.tensor(np.stack([e[1] for e in examples]), dtype=torch.float32)
    target_value = torch.tensor([e[2] for e in examples], dtype=torch.float32)
    if bool(((target_pi > 0) & (states != 0)).any()):
        raise ValueError("MCTS target assigns probability to an illegal move")
    logits, value = net(states)
    # 合法动作上的交叉熵；非法 logits 不应被动作策略梯度推动。
    logp = masked_policy_log_probs(logits, states)
    return -(target_pi * logp).sum(-1).mean() + F.mse_loss(value, target_value)


def train(games=40, simulations=16, seed=42, log_enabled=True):
    seed_everything(seed)
    net, optimizer = PolicyValue(), None
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
    log = RunLogger("11-alphazero", log_enabled)
    replay = []
    for game in range(games):
        samples, winner = self_play(net, simulations)
        replay.extend(samples)
        replay = replay[-1024:]
        # Self-play target policy = visit distribution, not raw network priors.
        batch = [replay[i] for i in np.random.choice(len(replay),
                size=min(64, len(replay)), replace=False)]
        loss = policy_value_loss(net, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log.scalar("training/policy_value_loss", loss.item(), game)
        log.scalar("game/non_draw", abs(winner), game)
    log.close()
    print("self-play games:", games, "positions:", len(replay))
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()
    train(args.games, args.simulations, args.seed, not args.no_log)


if __name__ == "__main__":
    main()

"""Tiny deterministic GridWorld so Bellman/TD/Dyna chapters work without downloads."""
from dataclasses import dataclass


@dataclass
class GridWorld:
    size: int = 4
    step_reward: float = -0.02
    goal_reward: float = 1.0

    @property
    def n_states(self):
        return self.size * self.size

    @property
    def n_actions(self):
        return 4  # left, down, right, up

    @property
    def goal(self):
        return self.n_states - 1

    def transition(self, state: int, action: int):
        """Returns (next_state, reward, terminated), no mutation: usable by planners."""
        assert 0 <= state < self.n_states and 0 <= action < 4
        if state == self.goal:
            return state, 0.0, True
        row, col = divmod(state, self.size)
        dr, dc = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        row = max(0, min(self.size - 1, row + dr))
        col = max(0, min(self.size - 1, col + dc))
        nxt = row * self.size + col
        return nxt, self.goal_reward if nxt == self.goal else self.step_reward, nxt == self.goal

    def reset(self):
        self.state = 0
        return self.state, {}

    def step(self, action: int):
        nxt, reward, terminated = self.transition(self.state, action)
        self.state = nxt
        return nxt, reward, terminated, False, {}

    def render_ascii(self, policy=None):
        arrows = ["←", "↓", "→", "↑"]
        rows = []
        for i in range(self.n_states):
            rows.append("G" if i == self.goal else arrows[int(policy[i])] if policy is not None else "·")
        return "\n".join(" ".join(rows[r:r+self.size]) for r in range(0, len(rows), self.size))

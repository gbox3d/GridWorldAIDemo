# ===============================
# File: q_agent.py
# ===============================
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import random
from typing import Tuple

@dataclass
class AgentConfig:
    gamma: float = 0.98
    alpha: float = 0.2
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 300

class QAgent:
    def __init__(self, rows: int, cols: int, cfg: AgentConfig):
        self.cfg = cfg
        self.Q = np.zeros((rows, cols, 4), dtype=np.float32)

    def epsilon(self, ep: int) -> float:
        e0, e1, dec = self.cfg.epsilon_start, self.cfg.epsilon_end, self.cfg.epsilon_decay_episodes
        if ep >= dec:
            return e1
        return e0 + (e1 - e0) * (ep / dec)

    def select_action(self, s: Tuple[int,int], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, 3)
        r, c = s
        return int(np.argmax(self.Q[r, c]))

    def greedy(self, s: Tuple[int,int]) -> int:
        r, c = s
        return int(np.argmax(self.Q[r, c]))

    def update(self, s, a, r, s_next, done):
        r0, c0 = s
        r1, c1 = s_next
        q_sa = self.Q[r0, c0, a]
        td_target = r + (0.0 if done else self.cfg.gamma * np.max(self.Q[r1, c1]))
        self.Q[r0, c0, a] = q_sa + self.cfg.alpha * (td_target - q_sa)
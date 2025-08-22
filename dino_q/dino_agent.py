# ===============================
# File: dino_agent.py
# ===============================
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import random
from typing import Tuple

@dataclass
class AgentConfig:
    gamma: float = 0.99
    alpha: float = 0.2
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500

class QAgent:
    """Q-table agent for DinoRun (state=(dist_bin, vy_bin), action∈{0,1})."""
    def __init__(self, dist_bins: int, vy_bins: int, cfg: AgentConfig):
        self.cfg = cfg
        self.Q = np.zeros((dist_bins, vy_bins, 2), dtype=np.float32)  # action: 0=none,1=jump

    def epsilon(self, ep: int) -> float:
        e0, e1, dec = self.cfg.epsilon_start, self.cfg.epsilon_end, self.cfg.epsilon_decay_episodes
        if ep >= dec:
            return e1
        return e0 + (e1 - e0) * (ep / dec)

    def select_action(self, s: Tuple[int,int], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, 1)
        d, v = s
        return int(np.argmax(self.Q[d, v]))

    def greedy(self, s: Tuple[int,int]) -> int:
        d, v = s
        return int(np.argmax(self.Q[d, v]))

    def update(self, s, a, r, s_next, done):
        d0, v0 = s
        d1, v1 = s_next
        q_sa = self.Q[d0, v0, a]
        td_target = r + (0.0 if done else self.cfg.gamma * np.max(self.Q[d1, v1]))
        self.Q[d0, v0, a] = q_sa + self.cfg.alpha * (td_target - q_sa)

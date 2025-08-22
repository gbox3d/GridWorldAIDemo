# ===============================
# File: dqn_dino_agent.py
# ===============================
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 32
    buffer_size: int = 50_000
    learn_start: int = 1_000
    train_freq: int = 4
    target_update_freq: int = 1_000  # by steps
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000  # linear decay
    frame_stack: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class QNet(nn.Module):
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        # Atari-style CNN
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple[int, int, int]):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        C,H,W = state_shape
        self.s = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2= np.zeros((capacity, C, H, W), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.float32)
    def push(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.s2[i] = s2; self.d[i] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=batch)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx])

class DQNAgent:
    def __init__(self, n_actions: int, cfg: DQNConfig):
        self.cfg = cfg
        self.n_actions = n_actions
        self.net = QNet(cfg.frame_stack, n_actions).to(cfg.device)
        self.tgt = QNet(cfg.frame_stack, n_actions).to(cfg.device)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.step_count = 0

    def epsilon(self):
        c = self.cfg
        t = min(self.step_count, c.epsilon_decay_steps)
        eps = c.epsilon_start + (c.epsilon_end - c.epsilon_start) * (t / c.epsilon_decay_steps)
        return eps

    def act(self, state_stacked: np.ndarray) -> int:
        # state_stacked: (C,H,W) float32
        if random.random() < self.epsilon():
            return random.randint(0, self.n_actions-1)
        with torch.no_grad():
            x = torch.from_numpy(state_stacked).unsqueeze(0).to(self.cfg.device)  # (1,C,H,W)
            q = self.net(x)
            return int(torch.argmax(q, dim=1).item())

    def learn(self, buffer: ReplayBuffer):
        c = self.cfg
        if buffer.size < c.learn_start or (self.step_count % c.train_freq) != 0:
            return None
        s,a,r,s2,d = buffer.sample(c.batch_size)
        s = torch.from_numpy(s).to(c.device)
        a = torch.from_numpy(a).to(c.device)
        r = torch.from_numpy(r).to(c.device)
        s2= torch.from_numpy(s2).to(c.device)
        d = torch.from_numpy(d).to(c.device)

        q = self.net(s).gather(1, a.view(-1,1)).squeeze(1)  # Q(s,a)
        with torch.no_grad():
            q_next = self.tgt(s2).max(1)[0]
            target = r + (1.0 - d) * c.gamma * q_next
        loss = nn.MSELoss()(q, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        if self.step_count % c.target_update_freq == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        return float(loss.item())
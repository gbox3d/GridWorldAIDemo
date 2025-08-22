# ===============================
# File: dqn_dino_env.py
# ===============================
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import random

@dataclass
class DinoConfig:
    # World/physics
    screen_w: int = 800
    screen_h: int = 240
    ground_y: int = 200
    gravity: float = 0.9
    jump_v: float = -12.0

    # Obstacles
    base_speed: float = 6.0
    speed_inc_per_ep: float = 0.02
    spawn_gap_min: int = 220
    spawn_gap_max: int = 380
    cactus_w: int = 18
    cactus_h: int = 35

    # Episode
    max_steps: int = 6000
    alive_reward: float = 1.0
    crash_penalty: float = -100.0

class DinoRun:
    """Same dynamics as low-dim env, but used with pixel observations."""
    def __init__(self, cfg: DinoConfig):
        self.cfg = cfg
        self.rng = random.Random(7)
        self.episode_idx = 0
        self.reset()

    # -------------- Public API --------------
    def reset(self):
        self.x = 60
        self.y = self.cfg.ground_y
        self.vy = 0.0
        self.steps = 0
        self.speed = self.cfg.base_speed + self.episode_idx * self.cfg.speed_inc_per_ep
        self.obstacles = [self._spawn_obstacle(initial=True)]
        return self._obs()

    def step(self, action: int):
        assert action in (0, 1)  # 0=idle, 1=jump
        if action == 1 and self._on_ground():
            self.vy = self.cfg.jump_v

        # physics
        self.vy += self.cfg.gravity
        self.y += self.vy
        if self.y > self.cfg.ground_y:
            self.y = self.cfg.ground_y
            self.vy = 0.0

        # obstacles scroll
        for ob in self.obstacles:
            ob[0] -= self.speed
        self.obstacles = [ob for ob in self.obstacles if ob[0] + ob[2] > 0]
        if len(self.obstacles) == 0 or (self.obstacles[-1][0] < self.cfg.screen_w - self._next_gap()):
            self.obstacles.append(self._spawn_obstacle(initial=False))

        # reward & done
        reward = self.cfg.alive_reward
        done = self._collides()
        if done:
            reward = self.cfg.crash_penalty
        self.steps += 1
        if self.steps >= self.cfg.max_steps:
            done = True
        return self._obs(), reward, done, {}

    # -------------- Helpers --------------
    def _player_rect(self) -> Tuple[float,float,float,float]:
        return (self.x, self.y - 30, 22, 30)

    def _on_ground(self) -> bool:
        return abs(self.y - self.cfg.ground_y) < 1e-6 and abs(self.vy) < 1e-6

    def _spawn_obstacle(self, initial: bool) -> List[float]:
        gap = self._next_gap() if not initial else self.rng.randint(self.cfg.spawn_gap_min, self.cfg.spawn_gap_max)
        x = self.cfg.screen_w + gap
        w = self.cfg.cactus_w
        h = self.cfg.cactus_h
        y = self.cfg.ground_y - h
        return [float(x), float(y), float(w), float(h)]

    def _next_gap(self) -> int:
        return self.rng.randint(self.cfg.spawn_gap_min, self.cfg.spawn_gap_max)

    def _collides(self) -> bool:
        px, py, pw, ph = self._player_rect()
        for ox, oy, ow, oh in self.obstacles:
            if (px < ox + ow and px + pw > ox and py < oy + oh and py + ph > oy):
                return True
        return False

    # obs placeholder (pixels handled by renderer)
    def _obs(self):
        return None


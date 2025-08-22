# gridworld_qlearning_pygame.py
# Python 3.10+, pip install pygame numpy
from __future__ import annotations
import math, random, sys, time
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import pygame

# =======================
# Config
# =======================
@dataclass
class Config:
    grid_size: Tuple[int, int] = (5, 5)     # (rows, cols)
    start: Tuple[int, int] = (4, 0)         # (r, c)
    goal: Tuple[int, int]  = (0, 4)         # (r, c)
    walls: List[Tuple[int, int]] = ((1,1),(1,2),(2,2))  # blocked cells
    step_penalty: float = -1.0
    goal_reward: float = 10.0
    gamma: float = 0.98
    alpha: float = 0.2
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 300       # linear decay
    episodes: int = 500
    max_steps_per_episode: int = 200
    # Rendering
    cell_px: int = 90
    fps: int = 30
    show_training: bool = True              # True: 에피소드 중 이동을 보여줌
    show_every_n_episodes: int = 1          # 렌더링 빈도
    slow_motion: bool = False               # True면 한 스텝씩 또렷하게
    seed: int | None = 7                    # 재현성

CONFIG = Config()

# Actions: Up, Right, Down, Left (URDL)
ACTIONS = [( -1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_NAMES = ["↑","→","↓","←"]

# =======================
# Env
# =======================
class GridWorld:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rows, self.cols = cfg.grid_size
        self.start = cfg.start
        self.goal  = cfg.goal
        self.walls = set(cfg.walls)
        assert self._in_bounds(self.start) and self._in_bounds(self.goal)
        assert self.start not in self.walls and self.goal not in self.walls
        self.state = self.start

    def reset(self) -> Tuple[int,int]:
        self.state = self.start
        return self.state

    def _in_bounds(self, rc: Tuple[int,int]) -> bool:
        r, c = rc
        return 0 <= r < self.rows and 0 <= c < self.cols

    def step(self, action: int) -> Tuple[Tuple[int,int], float, bool, Dict]:
        assert 0 <= action < 4
        r, c = self.state
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc

        # 벽/경계 처리: 이동 불가면 제자리
        if not self._in_bounds((nr, nc)) or (nr, nc) in self.walls:
            nr, nc = r, c

        self.state = (nr, nc)

        if self.state == self.goal:
            return self.state, self.cfg.goal_reward, True, {}

        return self.state, self.cfg.step_penalty, False, {}

# =======================
# Q-Learning Agent (tabular)
# =======================
class QAgent:
    def __init__(self, env: GridWorld, cfg: Config):
        self.env = env
        self.cfg = cfg
        self.Q = np.zeros((env.rows, env.cols, 4), dtype=np.float32)

    def epsilon(self, ep: int) -> float:
        # Linear decay
        e_start, e_end, e_decay = self.cfg.epsilon_start, self.cfg.epsilon_end, self.cfg.epsilon_decay_episodes
        if ep >= e_decay: return e_end
        return e_start + (e_end - e_start) * (ep / e_decay)

    def select_action(self, s: Tuple[int,int], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0,3)
        r, c = s
        return int(np.argmax(self.Q[r, c]))

    def update(self, s, a, r, s_next, done):
        r0, c0 = s
        r1, c1 = s_next
        q_sa = self.Q[r0, c0, a]
        td_target = r + (0.0 if done else self.cfg.gamma * np.max(self.Q[r1, c1]))
        self.Q[r0, c0, a] = q_sa + self.cfg.alpha * (td_target - q_sa)

# =======================
# Renderer (pygame)
# =======================
class Renderer:
    def __init__(self, env: GridWorld, cfg: Config):
        pygame.init()
        self.env = env
        self.cfg = cfg
        w = env.cols * cfg.cell_px
        h = env.rows * cfg.cell_px + 60  # status bar
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("GridWorld – Q-Learning")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("consolas", 20)
        self.big   = pygame.font.SysFont("consolas", 28, bold=True)

    def draw(self, Q: np.ndarray, agent_pos: Tuple[int,int], ep: int, step: int, epsilon: float, reward_sum: float):
        self.screen.fill((245,245,245))
        cp = self.cfg.cell_px
        rows, cols = self.env.rows, self.env.cols

        # Cells
        for r in range(rows):
            for c in range(cols):
                x, y = c*cp, r*cp
                color = (255,255,255)
                if (r,c) in self.env.walls:
                    color = (60,60,60)
                if (r,c) == self.env.goal:
                    color = (200,255,200)
                pygame.draw.rect(self.screen, color, (x, y, cp, cp))
                pygame.draw.rect(self.screen, (180,180,180), (x, y, cp, cp), width=1)

                # Q-value arrows (skip walls)
                if (r,c) not in self.env.walls:
                    qvals = Q[r,c]
                    m = np.max(qvals) if np.any(qvals!=0) else 1.0
                    # Draw tiny arrows with length ∝ normalized Q+
                    for ai, name in enumerate(ACTION_NAMES):
                        v = qvals[ai]
                        norm = max(0.0, (v - 0.0)) / (m if m!=0 else 1.0)
                        length = int((cp//2 - 8) * norm)
                        cx, cy = x + cp//2, y + cp//2
                        if ai == 0:  # up
                            pygame.draw.line(self.screen, (0,0,0), (cx, cy), (cx, cy - length), 3)
                        elif ai == 1: # right
                            pygame.draw.line(self.screen, (0,0,0), (cx, cy), (cx + length, cy), 3)
                        elif ai == 2: # down
                            pygame.draw.line(self.screen, (0,0,0), (cx, cy), (cx, cy + length), 3)
                        else:         # left
                            pygame.draw.line(self.screen, (0,0,0), (cx, cy), (cx - length, cy), 3)

        # Agent
        ar, ac = agent_pos
        ax, ay = ac*cp + cp//2, ar*cp + cp//2
        pygame.draw.circle(self.screen, (30,144,255), (ax, ay), cp//4)

        # Status bar
        bar_y = rows * cp
        pygame.draw.rect(self.screen, (250,250,250), (0, bar_y, cols*cp, 60))
        txt = f"Episode {ep+1}/{self.cfg.episodes}  Step {step}  ε={epsilon:.3f}  Return={reward_sum:.1f}"
        self.screen.blit(self.big.render(txt, True, (20,20,20)), (10, bar_y+12))

        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def pump_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)

# =======================
# Train loop
# =======================
def train():
    cfg = CONFIG
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    env = GridWorld(cfg)
    agent = QAgent(env, cfg)
    renderer = Renderer(env, cfg)

    returns = []
    for ep in range(cfg.episodes):
        s = env.reset()
        done = False
        ep_return = 0.0
        epsilon = agent.epsilon(ep)

        render_this_episode = cfg.show_training and ((ep % cfg.show_every_n_episodes) == 0)

        for step in range(cfg.max_steps_per_episode):
            if render_this_episode:
                renderer.pump_events()
                renderer.draw(agent.Q, env.state, ep, step, epsilon, ep_return)
                if cfg.slow_motion: time.sleep(0.02)

            a = agent.select_action(s, epsilon)
            s_next, r, done, _ = env.step(a)
            agent.update(s, a, r, s_next, done)

            s = s_next
            ep_return += r
            if done:
                break

        returns.append(ep_return)

        # 마지막 프레임 한번 더
        if render_this_episode:
            renderer.pump_events()
            renderer.draw(agent.Q, env.state, ep, step, epsilon, ep_return)

    # 학습 종료 후 정책 시연
    demo(env, agent, renderer)

def greedy_action(qrow: np.ndarray) -> int:
    return int(np.argmax(qrow))

def demo(env: GridWorld, agent: QAgent, renderer: Renderer):
    cfg = agent.cfg
    while True:
        s = env.reset()
        ep_return = 0.0
        for step in range(cfg.max_steps_per_episode):
            renderer.pump_events()
            r, c = s
            a = greedy_action(agent.Q[r,c])
            s, rwd, done, _ = env.step(a)
            ep_return += rwd
            renderer.draw(agent.Q, env.state, ep=9999, step=step, epsilon=0.0, reward_sum=ep_return)
            time.sleep(0.05 if cfg.slow_motion else 0.02)
            if done:
                # 잠깐 멈춰서 도착 확인
                time.sleep(0.4)
                break

if __name__ == "__main__":
    train()

# ===============================
# File: renderer.py
# ===============================
from __future__ import annotations
import pygame
import numpy as np
from typing import Tuple

ACTION_NAMES = ["↑","→","↓","←"]

class Renderer:
    def __init__(self, rows: int, cols: int, cell_px: int = 90, fps: int = 30):
        pygame.init()
        self.rows, self.cols = rows, cols
        self.cell_px = cell_px
        self.fps = fps
        w = cols * cell_px
        h = rows * cell_px + 70  # status bar
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("GridWorld – Human vs AI (Q-Learning)")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("consolas", 20)
        self.big   = pygame.font.SysFont("consolas", 26, bold=True)
        self.hud_font = pygame.font.SysFont("consolas", 18, bold=True)

    def draw(self, grid, walls, goal, agent_pos: Tuple[int,int], Q: np.ndarray|None,
             hud_text: str, show_q: bool = True):
        self.screen.fill((245,245,245))
        cp = self.cell_px
        rows, cols = grid

        # cells
        for r in range(rows):
            for c in range(cols):
                x, y = c*cp, r*cp
                color = (255,255,255)
                if (r,c) in walls:
                    color = (60,60,60)
                if (r,c) == goal:
                    color = (200,255,200)
                pygame.draw.rect(self.screen, color, (x, y, cp, cp))
                pygame.draw.rect(self.screen, (180,180,180), (x, y, cp, cp), width=1)

                if show_q and Q is not None and (r,c) not in walls:
                    qvals = Q[r,c]
                    m = np.max(qvals) if np.any(qvals!=0) else 1.0
                    cx, cy = x + cp//2, y + cp//2
                    for ai in range(4):
                        v = qvals[ai]
                        norm = max(0.0, v) / (m if m!=0 else 1.0)
                        length = int((cp//2 - 8) * norm)
                        if ai == 0:  # up
                            pygame.draw.line(self.screen, (0,0,0), (cx, cy), (cx, cy - length), 3)
                        elif ai == 1:  # right
                            pygame.draw.line(self.screen, (0,0,0), (cx, cy), (cx + length, cy), 3)
                        elif ai == 2:  # down
                            pygame.draw.line(self.screen, (0,0,0), (cx, cy), (cx, cy + length), 3)
                        else:          # left
                            pygame.draw.line(self.screen, (0,0,0), (cx, cy), (cx - length, cy), 3)

        # agent
        ar, ac = agent_pos
        ax, ay = ac*cp + cp//2, ar*cp + cp//2
        pygame.draw.circle(self.screen, (30,144,255), (ax, ay), cp//4)

        # HUD
        bar_y = rows * cp
        pygame.draw.rect(self.screen, (250,250,250), (0, bar_y, cols*cp, 70))
        
        y = bar_y + 8
        for line in hud_text.split("\n"):
            surf = self.hud_font.render(line, True, (20,20,20))
            self.screen.blit(surf, (10, y))
            y += surf.get_height() + 2

        pygame.display.flip()
        self.clock.tick(self.fps)

    def pump_events(self):
        return pygame.event.get()
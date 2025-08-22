# ===============================
# File: dino_renderer.py
# ===============================
from __future__ import annotations
import pygame
import numpy as np
from typing import Tuple, List

class Renderer:
    def __init__(self, w: int, h: int, fps: int = 60):
        pygame.init()
        self.w, self.h = w, h
        self.fps = fps
        self.screen = pygame.display.set_mode((w, h + 80))
        pygame.display.set_caption("Dino Run — Human vs AI (Q-learning)")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("consolas", 20)
        self.big   = pygame.font.SysFont("consolas", 26, bold=True)

    def draw(self, ground_y: int, player_rect: Tuple[float,float,float,float],
             obstacles: List[Tuple[float,float,float,float]],
             hud_text: str, score: float | None = None):
        self.screen.fill((250,250,250))
        # ground
        pygame.draw.line(self.screen, (50,50,50), (0, ground_y), (self.w, ground_y), 3)
        # player
        px, py, pw, ph = player_rect
        pygame.draw.rect(self.screen, (30,144,255), pygame.Rect(px, py, pw, ph))
        # obstacles
        for ox, oy, ow, oh in obstacles:
            pygame.draw.rect(self.screen, (60,60,60), pygame.Rect(ox, oy, ow, oh))
        # HUD
        bar_y = self.h
        y = bar_y + 8
        for line in hud_text.split("\n"):
            surf = self.big.render(line, True, (20,20,20))
            self.screen.blit(surf, (10, y))
            y += surf.get_height() + 2
        if score is not None:
            s_text = self.big.render(f"Score {score:.1f}", True, (20,20,20))
            self.screen.blit(s_text, (self.w - s_text.get_width() - 12, bar_y + 8))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def pump_events(self):
        return pygame.event.get()

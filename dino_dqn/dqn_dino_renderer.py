# ===============================
# File: dqn_dino_renderer.py
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

        # display surface (can be hidden if desired)
        #self.screen = pygame.display.set_mode((w, h + 80))

        # __init__ 에서 set_mode 교체
        flags = pygame.SCALED | pygame.DOUBLEBUF
        try:
            self.screen = pygame.display.set_mode((w, h + 80), flags, vsync=1)
        except TypeError:
            self.screen = pygame.display.set_mode((w, h + 80), flags)


        pygame.display.set_caption("Dino Run — DQN")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("consolas", 20)
        self.bar_font   = pygame.font.SysFont("consolas", 12, bold=True)
        # offscreen surface for pixel capture
        self.offsurf = pygame.Surface((w, h))

    def draw_world(self, surf: pygame.Surface, ground_y: int,
                   player_rect: Tuple[float,float,float,float],
                   obstacles: List[Tuple[float,float,float,float]]):
        surf.fill((255,255,255))
        # ground
        pygame.draw.line(surf, (0,0,0), (0, ground_y), (self.w, ground_y), 3)
        # player
        px, py, pw, ph = player_rect
        pygame.draw.rect(surf, (30,144,255), pygame.Rect(px, py, pw, ph))
        # obstacles
        for ox, oy, ow, oh in obstacles:
            pygame.draw.rect(surf, (60,60,60), pygame.Rect(ox, oy, ow, oh))

    def get_frame84(self) -> np.ndarray:
        """Return 84x84 grayscale float32 in [0,1], shape (84,84)."""
        # scale down using pygame (fast), then to numpy
        small = pygame.transform.smoothscale(self.offsurf, (84, 84))
        arr = pygame.surfarray.array3d(small)  # (w,h,3)
        arr = np.transpose(arr, (1, 0, 2)).astype(np.float32)  # (h,w,3)
        gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])
        gray /= 255.0
        return gray  # (84,84)

    def blit_hud(self, hud: str, score: float | None = None):
        bar_y = self.h
        y = bar_y + 8
        pygame.draw.rect(self.screen, (245,245,245), pygame.Rect(0, bar_y, self.w, 80))
        for line in hud.split("\n"):
            surf = self.bar_font.render(line, True, (20,20,20))
            self.screen.blit(surf, (10, y))
            y += surf.get_height() + 2
        if score is not None:
            s_text = self.bar_font.render(f"Score {score:.1f}", True, (20,20,20))
            self.screen.blit(s_text, (self.w - s_text.get_width() - 12, bar_y + 8))

    def render(self, env, hud: str = "", score: float | None = None, show: bool = True):
        # draw into offscreen
        self.draw_world(self.offsurf, env.cfg.ground_y, env._player_rect(), env.obstacles)
        frame = self.get_frame84()
        # optionally display
        if show:
            self.draw_world(self.screen, env.cfg.ground_y, env._player_rect(), env.obstacles)
            self.blit_hud(hud, score)
            pygame.display.flip()
            self.clock.tick(self.fps)
        return frame

    def pump_events(self):
        return pygame.event.get()


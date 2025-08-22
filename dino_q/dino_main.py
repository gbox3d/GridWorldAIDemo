# ===============================
# File: dino_main.py
# ===============================
from __future__ import annotations
import time, random
import pygame

from dino_env import DinoConfig, DinoRun
from dino_agent import AgentConfig, QAgent
from dino_renderer import Renderer

# Modes
MODE_HUMAN = "HUMAN"
MODE_TRAIN = "AI_TRAIN"
MODE_DEMO  = "AI_DEMO"

SLOW_MOTION = False
DEMO_DELAY = 0.02 if not SLOW_MOTION else 0.06
TRAIN_STEPS_PER_FRAME = 8


def main():
    random.seed(7)

    cfg = DinoConfig()
    env = DinoRun(cfg)
    agent = QAgent(cfg.dist_bins, cfg.vy_bins, AgentConfig())

    renderer = Renderer(cfg.screen_w, cfg.screen_h, fps=60)
    pygame.key.set_repeat(0)  # human: keydown 1회당 1동작만

    mode = MODE_HUMAN
    episode = 0
    step_in_ep = 0
    ep_return = 0.0
    epsilon = agent.epsilon(episode)

    running = True
    while running:
        # --- events ---
        human_action = 0  # 기본: 아무것도 안 함 (do-nothing)
        for event in renderer.pump_events():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    mode = MODE_HUMAN
                elif event.key == pygame.K_F2:
                    mode = MODE_TRAIN
                elif event.key == pygame.K_F3:
                    mode = MODE_DEMO
                elif event.key == pygame.K_r:
                    env.reset(); step_in_ep = 0; ep_return = 0.0
                # Human control
                if mode == MODE_HUMAN and event.key in (pygame.K_SPACE, pygame.K_UP):
                    human_action = 1  # 이 프레임에서 1회 점프 처리
                    
        if mode == MODE_HUMAN:
            _, r, done, _ = env.step(human_action)
            ep_return += r
            step_in_ep += 1
            if done:
                time.sleep(0.3)
                env.episode_idx += 1
                env.reset(); step_in_ep = 0; ep_return = 0.0

        # --- AI train ---
        if mode == MODE_TRAIN:
            for _ in range(TRAIN_STEPS_PER_FRAME):
                if step_in_ep == 0:
                    epsilon = agent.epsilon(episode)
                s = env._get_state()
                a = agent.select_action(s, epsilon)
                s_next, r, done, _ = env.step(a)
                agent.update(s, a, r, s_next, done)
                ep_return += r
                step_in_ep += 1
                if done:
                    episode += 1
                    env.episode_idx += 1
                    env.reset(); step_in_ep = 0; ep_return = 0.0
                    break

        # --- AI demo ---
        if mode == MODE_DEMO:
            s = env._get_state()
            a = agent.greedy(s)
            _, r, done, _ = env.step(a)
            ep_return += r
            step_in_ep += 1
            time.sleep(DEMO_DELAY)
            if done:
                time.sleep(0.35)
                env.episode_idx += 1
                env.reset(); step_in_ep = 0; ep_return = 0.0

        # --- HUD & render ---
        hud = (f"Mode:{mode} | Ep:{episode} Step:{step_in_ep}  ε:{epsilon:.3f}\n"
               f"DistBins:{cfg.dist_bins} VyBins:{cfg.vy_bins}  Speed:{env.speed:.1f}")
        renderer.draw(
            cfg.ground_y,
            env._player_rect(),
            env.obstacles,
            hud_text=hud,
            score=ep_return,
        )

    pygame.quit()


if __name__ == "__main__":
    main()

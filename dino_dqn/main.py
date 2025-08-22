# ===============================
# File: dqn_dino_main.py
# ===============================
from __future__ import annotations
import time, random
import numpy as np
import pygame
from collections import deque

from dqn_dino_env import DinoConfig, DinoRun
from dqn_dino_renderer import Renderer
from dqn_dino_agent import DQNConfig, DQNAgent, ReplayBuffer

MODE_TRAIN = "AI_TRAIN"
MODE_DEMO  = "AI_DEMO"
MODE_HUMAN = "HUMAN"

SHOW_SCREEN = True   # 학습 속도를 원하면 False로
MAX_FRAMES  = 50_000 # 데모 목적의 짧은 러닝


def stack_reset(frame: np.ndarray, k: int) -> np.ndarray:
    # frame: (84,84) -> (k,84,84) 복제 스택
    return np.repeat(frame[None, ...], k, axis=0).astype(np.float32)


def stack_next(stack: np.ndarray, frame: np.ndarray) -> np.ndarray:
    # (C,H,W) 스택에 새 프레임 추가, 가장 오래된 것 제거
    c,h,w = stack.shape
    out = np.empty_like(stack)
    out[:-1] = stack[1:]
    out[-1] = frame
    return out


def main():
    random.seed(7); np.random.seed(7)

    cfg = DinoConfig()
    env = DinoRun(cfg)
    ren = Renderer(cfg.screen_w, cfg.screen_h, fps=60)

    # DQN
    acfg = DQNConfig()
    agent = DQNAgent(n_actions=2, cfg=acfg)
    buffer = ReplayBuffer(acfg.buffer_size, (acfg.frame_stack, 84, 84))

    # Modes
    mode = MODE_TRAIN
    episode = 0
    ep_return = 0.0

    # initial obs (pixel)
    frame = ren.render(env, hud="", score=0.0, show=False)  # after reset
    state = stack_reset(frame, acfg.frame_stack)

    running = True
    while running and agent.step_count < MAX_FRAMES:
        # Handle minimal events (quit)
        for e in ren.pump_events():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_F3:
                    mode = MODE_DEMO
                elif e.key == pygame.K_F2:
                    mode = MODE_TRAIN
                elif e.key == pygame.K_F1:
                    mode = MODE_HUMAN

        if mode == MODE_HUMAN:
            keys = pygame.key.get_pressed()
            action = 1 if (keys[pygame.K_SPACE] or keys[pygame.K_UP]) else 0
        elif mode == MODE_DEMO:
            action = agent.act(state)  # greedy/epsilon mixed (epsilon still decays)
        else:
            action = agent.act(state)

        # env step
        _, reward, done, _ = env.step(action)
                
        # HUD 문자열을 먼저 만든 후, render 한 번만 호출
        hud = (f"Mode:{mode}  Ep:{episode}  Steps:{agent.step_count}  ε:{agent.epsilon():.3f}\n"
            f"Buffer:{buffer.size}/{buffer.capacity}  Return:{ep_return:.1f}")

        frame_next = ren.render(env, hud=hud, score=ep_return, show=SHOW_SCREEN)
        next_state = stack_next(state, frame_next)

        # store & learn (only in TRAIN)
        if mode == MODE_TRAIN:
            buffer.push(state, action, reward, next_state, float(done))
            agent.step_count += 1
            loss = agent.learn(buffer)
        ep_return += reward

        if done:
            # new episode
            episode += 1; env.episode_idx += 1
            env.reset(); ep_return = 0.0
            frame = ren.render(env, hud="", score=0.0, show=False)
            state = stack_reset(frame, acfg.frame_stack)
        else:
            state = next_state

        
        # pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

# ===============================
# File: run_app.py
# ===============================
from __future__ import annotations
import time, random
import pygame
from typing import Optional

from env_gridworld import EnvConfig, GridWorld
from q_agent import AgentConfig, QAgent
from renderer import Renderer

# ---------- App Config ----------
CELL_PX = 90
FPS = 45
SLOW_MOTION = False
SEED = 7
TRAIN_STEPS_PER_FRAME = 10  # AI 학습 속도

# ---------- Modes ----------
MODE_HUMAN = "HUMAN"
MODE_TRAIN = "AI_TRAIN"
MODE_DEMO  = "AI_DEMO"


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def main():
    global TRAIN_STEPS_PER_FRAME
    set_seed(SEED)

    # Environment & Agent
    env_cfg = EnvConfig()
    env = GridWorld(env_cfg)

    agent_cfg = AgentConfig()
    agent = QAgent(env.rows, env.cols, agent_cfg)

    renderer = Renderer(env.rows, env.cols, cell_px=CELL_PX, fps=FPS)

    mode = MODE_HUMAN
    human_act = -1 # human action

    episode = 0
    step_in_ep = 0
    ep_return = 0.0
    epsilon = agent.epsilon(episode)

    # For AI_DEMO gentle pacing
    demo_delay = 0.02 if not SLOW_MOTION else 0.06

    show_q = True

    prev_Score = 0.0 # 이전 점수

    # -------------- Main Loop --------------
    running = True
    

    while running:
        # Events
        for event in renderer.pump_events():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:

                # if isKeyDown == True:  # prevent multiple keydown events
                #     continue
                # isKeyDown = True

                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    mode = MODE_HUMAN
                    human_act = -1
                elif event.key == pygame.K_F2:
                    mode = MODE_TRAIN
                elif event.key == pygame.K_F3:
                    mode = MODE_DEMO
                elif event.key == pygame.K_r:
                    env.reset(); step_in_ep = 0; ep_return = 0.0
                elif event.key == pygame.K_v:
                    show_q = not show_q
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    global TRAIN_STEPS_PER_FRAME
                    TRAIN_STEPS_PER_FRAME = max(1, TRAIN_STEPS_PER_FRAME - 1)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    TRAIN_STEPS_PER_FRAME += 1
                #Human action (one step per keydown)
                if mode == MODE_HUMAN and event.key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT):
                    act = {pygame.K_UP:0, pygame.K_RIGHT:1, pygame.K_DOWN:2, pygame.K_LEFT:3}[event.key]
                    human_act = act

        if mode == MODE_HUMAN:

            if human_act == -1:  # no key input
                continue

            s_next, r, done, _ = env.step(human_act)
            human_act = -1  # reset key input after processing
            ep_return += r
            step_in_ep += 1
            
            print(f"score = {ep_return:.1f} | action={act} | state={s_next} | reward={r:.1f} | done={done}")

            if done:
                prev_Score = ep_return
                time.sleep(0.3)                        
                env.reset(); step_in_ep = 0; ep_return = 0.0
        

        # AI TRAIN mode
        if mode == MODE_TRAIN:
            for _ in range(TRAIN_STEPS_PER_FRAME):
                if step_in_ep == 0:
                    epsilon = agent.epsilon(episode)
                s_prev = env.state
                a = agent.select_action(s_prev, epsilon)
                s_next, r, done, _ = env.step(a)
                agent.update(s_prev, a, r, s_next, done)
                ep_return += r
                step_in_ep += 1
                if done:
                    print(f"Episode {episode} finished: score = {ep_return:.1f}")
                    prev_Score = ep_return
                    episode += 1
                    env.reset(); step_in_ep = 0; ep_return = 0.0
                    break

        # AI DEMO mode (greedy policy rollout)
        if mode == MODE_DEMO:
            a = agent.greedy(env.state)  # greedy action
            s_next, r, done, _ = env.step(a)
            ep_return += r
            step_in_ep += 1
            time.sleep(demo_delay)
            if done:
                time.sleep(0.35)
                env.reset(); step_in_ep = 0; ep_return = 0.0

        # HUD text
        hud = (
            f"Mode:{mode} | Ep:{episode} Step:{step_in_ep}  ε:{epsilon:.3f}\n"
            f"Score :{ep_return:.1f}/ {prev_Score:.1f} "
        )

        # Render
        renderer.draw((env.rows, env.cols), env.walls, env.goal, env.state, agent.Q, hud,
              show_q=show_q)

    pygame.quit()


if __name__ == "__main__":
    main()

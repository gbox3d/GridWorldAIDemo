# ===============================
# File: q_agent.py
# ===============================
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import random
from typing import Tuple

@dataclass
class AgentConfig:
    # Q-learning 하이퍼파라미터
    gamma: float = 0.98            # 할인율 γ: 미래 보상 가중치 (1에 가까울수록 먼 미래 중시)
    alpha: float = 0.2             # 학습률 α: TD 오차를 얼마나 반영할지
    epsilon_start: float = 1.0     # 탐색 확률 시작값 ε0 (초반엔 많이 탐색)
    epsilon_end: float = 0.05      # 탐색 확률 하한 ε_min (후반엔 이용 중심)
    epsilon_decay_episodes: int = 300  # ε를 선형 감쇠할 총 에피소드 수

class QAgent:
    def __init__(self, rows: int, cols: int, cfg: AgentConfig):
        self.cfg = cfg
        # Q 테이블: 상태(행 r, 열 c) × 행동(4방향)
        # Q[r, c, a] = 상태 (r,c)에서 행동 a를 했을 때의 예상 누적 보상
        self.Q = np.zeros((rows, cols, 4), dtype=np.float32)

    def epsilon(self, ep: int) -> float:
        """에피소드 번호 ep에 따른 ε(탐색 확률) 스케줄.
        - 여기선 '선형 감쇠'를 사용.
        - ep가 감쇠 기간을 지나면 ε_end로 고정.
        """
        e0, e1, dec = self.cfg.epsilon_start, self.cfg.epsilon_end, self.cfg.epsilon_decay_episodes
        if ep >= dec:
            return e1
        # 선형 보간: ep=0 -> e0, ep=dec -> e1
        return e0 + (e1 - e0) * (ep / dec)

    def select_action(self, s: Tuple[int,int], epsilon: float) -> int:
        """ε-greedy 행동 선택
        - 확률 ε: 임의의 행동(탐색, exploration)
        - 확률 1-ε: 현재 Q에 따른 최적 행동 argmax_a Q(s,a) (이용, exploitation)
        """
        if random.random() < epsilon:
            # 탐색: 0~3 중 무작위 (상/우/하/좌)
            return random.randint(0, 3)
        # 이용: Q 값이 최대인 행동 선택
        r, c = s
        return int(np.argmax(self.Q[r, c]))
        # 주의: argmax는 동률이면 첫 인덱스를 고릅니다(편향 가능).
        # 필요 시 무작위 타이브레이킹을 도입할 수 있습니다.

    def greedy(self, s: Tuple[int,int]) -> int:
        """탐색 없이 항상 argmax 행동을 고르는 순수 탐욕 정책 (데모/평가용)."""
        r, c = s
        return int(np.argmax(self.Q[r, c]))

    def update(self, s, a, r, s_next, done):
        """Q-learning 갱신식 (오프-폴리시, model-free)
        Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s', a') − Q(s,a) ]
        - done=True(종단 상태)면 미래항을 0으로 처리.
        - max_{a'} Q(s',a')에서 '다음 상태의 최대 행동가치'를 사용하므로 오프-폴리시.
        """
        r0, c0 = s       # 현재 상태 s = (row, col)
        r1, c1 = s_next  # 다음 상태 s'
        q_sa = self.Q[r0, c0, a]  # 현재 Q(s,a)

        # TD target = 즉시보상 r + 할인된 다음 상태의 최대 Q
        # (종단이면 미래 보상 없음)
        td_target = r + (0.0 if done else self.cfg.gamma * np.max(self.Q[r1, c1]))

        # TD 오차(δ) = target - 현재값
        # Q ← Q + α * δ
        self.Q[r0, c0, a] = q_sa + self.cfg.alpha * (td_target - q_sa)

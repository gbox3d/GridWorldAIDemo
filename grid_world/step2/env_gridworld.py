# ===============================
# File: env_gridworld.py
# ===============================
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict

import random

# Actions의 인덱스를 "의미"로 고정하여 외부(에이전트/렌더러)와 일치시킵니다.
# 0: ↑, 1: →, 2: ↓, 3: ←  (상, 우, 하, 좌)
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


@dataclass
class EnvConfig:
    """그리드월드 환경 설정값 모음."""
    grid_size: Tuple[int, int] = (5, 5)     # 격자 크기 (행, 열)
    start: Tuple[int, int] = (4, 0)         # 시작 좌표 (행, 열)
    goal: Tuple[int, int]  = (0, 4)         # 목표 좌표 (행, 열)
    walls: List[Tuple[int, int]] = ((1,1),(1,2),(2,2))  # 통과 불가 칸들의 좌표 집합
    step_penalty: float = -1.0              # 한 스텝 이동 시 기본 패널티(에피소드 짧게 유도)
    goal_reward: float = 10.0               # 목표 도달 보상
    max_steps_per_episode: int = 200        # 타임아웃(과도한 탐색 방지)


class GridWorld:
    """Deterministic(결정론적) GridWorld 환경.
    
    - 상태(state): 에이전트의 현재 위치 (행, 열)
    - 행동(action): 상/우/하/좌 중 하나 (0~3)
    - 보상(reward): 기본 이동은 step_penalty, 목표 도달 시 goal_reward
    - 종료(done): 목표 도달 또는 스텝 제한 초과

    * 중점: 교육용으로 단순/명확. 확률 전이(바람), 트랩, 슬립 등을 손쉽게 추가 가능.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rows, self.cols = cfg.grid_size
        self.start = cfg.start
        self.goal  = cfg.goal
        self.walls = set(cfg.walls)  # 조회 성능을 위해 set으로 보관
        # 유효성 검사: 시작/목표가 격자 안쪽이고 벽이 아니어야 함
        assert self._in_bounds(self.start) and self._in_bounds(self.goal)
        assert self.start not in self.walls and self.goal not in self.walls
        self.reset()

    def reset(self):
        """에피소드를 초기화하고 시작 상태를 반환."""
        self.state = self.start
        self.steps = 0
        return self.state

    def _in_bounds(self, rc: Tuple[int,int]) -> bool:
        """격자 범위 내인지 판정."""
        r, c = rc # rc -> row, column
        return 0 <= r < self.rows and 0 <= c < self.cols

    def step(self, action: int):
        """하나의 행동을 적용하여 (다음상태, 보상, 종료여부, 정보)를 반환.
        
        매 스텝의 처리 순서(결정론적 버전):
          1) 현재 상태 self.state 에서 action(0~3)을 받아 방향 벡터(d r, d c)를 구한다.
          2) 제안된 다음 좌표 (nr, nc) = (r+dr, c+dc)을 계산한다.
          3) 경계 밖이거나 벽(walls)인 경우 "충돌"로 간주하고 **제자리 유지**한다.
             - 교육 효과: "실패 이동도 비용(step_penalty)을 낸다"를 직관적으로 체험.
          4) 상태(state)를 업데이트하고 스텝 수를 증가시킨다.
          5) 목표에 도달했으면 (goal_reward, done=True) 반환.
          6) 아니면 (step_penalty, done=False) 또는 스텝 제한 초과 시 done=True 반환.

        확장 지점(과제용):
          - 바람/미끄러짐(확률 전이): nr, nc 결정 전에 action을 확률적으로 변형.
          - 트랩/보너스 칸: 상태 업데이트 후, 위치별 추가 보상/패널티 적용.
          - 벽 충돌 시 더 큰 패널티: 제자리 유지+추가 패널티를 줘도 됨.
        """
        assert 0 <= action < 4  # 잘못된 액션 방지(디버깅 편의)

        # 1) 현재 위치와 이동 벡터
        r, c = self.state
        dr, dc = ACTIONS[action]

        # -------------------------------
        # (예: 바람/확률전이 추가하려면 여기서 action을 교란)
        if random.random() < 0.1:
            # 10% 확률로 좌/우로 빗나감
            action = 1 if action == 0 else 3  # 단순 예시
            dr, dc = ACTIONS[action]
        # -------------------------------

        # 2) 제안된 다음 위치
        nr, nc = r + dr, c + dc

        # 3) 경계/벽 충돌 시 제자리 유지
        if not self._in_bounds((nr, nc)) or (nr, nc) in self.walls:
            nr, nc = r, c  # bump into wall/border -> stay , 충돌시에는 점수 만 소진하고 제자리

        # 4) 상태 업데이트 및 스텝 카운트
        self.state = (nr, nc)
        self.steps += 1

        # -------------------------------
        # (예: 트랩/보너스 칸을 추가하려면 여기서 위치 기반 보상 조정)
        #   extra = 0.0
        #   if self.state in trap_cells:    extra -= 5.0
        #   if self.state in bonus_cells:   extra += 0.5
        # -------------------------------

        # 5) 목표 도달 시 즉시 종료
        if self.state == self.goal:
            # return (다음상태, 보상, 종료여부, 디버그정보)
            return self.state, self.cfg.goal_reward, True, {}

        # 6) 아직 목표가 아니면 기본 패널티 부여
        done = self.steps >= self.cfg.max_steps_per_episode  # 타임아웃
        return self.state, self.cfg.step_penalty, done, {}

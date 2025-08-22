# ===============================
# File: dino_env.py
# ===============================
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random

@dataclass
class DinoConfig:
    """Dino Run 환경 설정값.
    - 화면, 물리, 장애물, 에피소드/보상, 이산화(버킷화) 파라미터를 한 곳에서 관리합니다.
    - 수업/실험 시 여기 값들만 바꿔도 난이도와 학습 양상이 크게 달라집니다.
    """
    # World/physics (세계/물리)
    screen_w: int = 800
    screen_h: int = 240
    ground_y: int = 200              # 지면의 y좌표(아래로 갈수록 값이 커짐; pygame 좌표계)
    gravity: float = 0.9             # 매 프레임 중력 가속도(아래 방향, +)
    jump_v: float = -12.0            # 점프 초기 속도(위 방향, -)  ※ 음수여야 위로 뜁니다.

    # Obstacles (장애물 관련)
    base_speed: float = 6.0          # 장애물의 기본 왼쪽 이동 속도(px/frame): 배경 스크롤 대용
    speed_inc_per_ep: float = 0.05   # 에피소드가 끝날 때마다 스피드를 얼마나 올릴지(난이도 증가)
    spawn_gap_min: int = 220         # 다음 장애물까지 최소 거리(px)
    spawn_gap_max: int = 380         # 다음 장애물까지 최대 거리(px)
    cactus_w: int = 18               # 선인장(장애물) 가로폭(px)
    cactus_h: int = 35               # 선인장(장애물) 높이(px)

    # Episode (에피소드/보상)
    max_steps: int = 3000            # 타임아웃(이 스텝 수를 넘기면 에피소드 종료)
    alive_reward: float = 1.0        # 살아있는 매 프레임마다 주는 보상(+)
    crash_penalty: float = -100.0    # 충돌 시 즉시 주는 큰 패널티(-)

    # Discretization (Q-table 학습을 위한 상태 이산화)
    #  - 저차원 상태: (다음 장애물까지의 거리 버킷, 수직속도 상태)
    dist_bins: int = 12              # 거리 버킷 개수 (0 ~ dist_bins-1)
    dist_bucket_px: int = 40         # 버킷 1칸의 실제 픽셀 폭 (거리/이 값 => 버킷 인덱스)
    vy_bins: int = 3                 # 수직속도 상태 개수: {0=지면(on_ground), 1=상승(vy<0), 2=하강(vy>0)}


class DinoRun:
    """Dino Run 환경 (저차원 상태 버전, Q-learning용).

    상태(state; 이산형):
      - dist_bin: 플레이어의 앞쪽(오른쪽)에 가장 먼저 다가오는 장애물의 "선두 모서리"까지 거리(px)를 버킷화한 값
      - vy_bin: 0(on_ground), 1(rising; vy<0), 2(falling; vy>0)

    행동(action; 이산형):
      - 0: 아무것도 안 함 (대기)
      - 1: 점프 (지면에 있을 때만 유효; 공중에서는 무시)

    보상(reward):
      - 매 스텝 +alive_reward
      - 충돌 시 crash_penalty를 주고 done=True로 종료

    에피소드 종료(done):
      - 장애물과 충돌하거나
      - 최대 스텝 수에 도달하면 True

    구현 특징:
      - 배경 스크롤을 단순화하기 위해 "장애물이 왼쪽으로 이동"하도록 구현합니다.
      - 플레이어는 (x 고정, y만 중력/점프로 변함).
      - 충돌 판정은 AABB(축 정렬 사각형)으로 간단히 처리합니다.
    """

    def __init__(self, cfg: DinoConfig):
        self.cfg = cfg
        self.rng = random.Random(7)  # 재현성 있는 장애물 간격 생성을 위해 내부 RNG 사용
        self.episode_idx = 0         # 진행된 에피소드 수(난이도 상승에 사용)
        self.reset()

    # -------------- Public API --------------
    def reset(self):
        """에피소드 초기화.
        - 플레이어 위치/속도/스텝 초기화
        - 현재 에피소드 난이도(속도) 반영
        - 첫 장애물 1개를 앞쪽에 생성
        - 초기 상태(이산화 결과)를 반환
        """
        self.x = 60                        # 플레이어 x는 고정(왼쪽에서 60px 지점)
        self.y = self.cfg.ground_y         # 시작은 지면에 붙어 있음
        self.vy = 0.0
        self.steps = 0
        # 난이도: 에피소드가 지날수록 조금씩 빠르게
        self.speed = self.cfg.base_speed + self.episode_idx * self.cfg.speed_inc_per_ep
        # 기본적으로 화면 오른쪽 바깥에 장애물 1개 생성
        self.obstacles = [self._spawn_obstacle(initial=True)]
        return self._get_state()

    def step(self, action: int):
        """한 스텝 진행: (다음 상태, 보상, 종료여부, 디버그정보) 반환.
        처리 순서:
          1) 입력 행동 적용(점프 명령)        -> 지면일 때만 vy를 점프 속도로 설정
          2) 물리 적분(중력/위치/속도 갱신)  -> 기본 1차원 수직 운동
          3) 장애물 이동/재활용              -> 왼쪽으로 이동, 화면 밖은 제거, 필요 시 새로 생성
          4) 충돌 판정                       -> 부딪히면 done=True, 큰 패널티 보상
          5) 스텝 증가/타임아웃 체크         -> 너무 오래 진행되면 done=True
          6) 새로운 상태(이산화)와 보상/종료 반환
        """
        assert action in (0, 1), "action은 {0=대기, 1=점프}만 허용"

        # --- 1) 행동 적용: 점프는 '지면에 있을 때만' 유효 ---
        if action == 1 and self._on_ground():
            # 위쪽으로 초기 속도를 준다(음수 방향)
            self.vy = self.cfg.jump_v

        # --- 2) 물리 적분: 중력 적용 후 위치 갱신 ---
        #   vy <- vy + g
        self.vy += self.cfg.gravity
        #   y  <- y  + vy
        self.y += self.vy

        #   지면을 뚫고 내려가지 않도록 클램핑
        #   (y가 ground_y보다 커졌다는 건 지면 아래로 갔다는 뜻. 좌표계는 아래가 +)
        if self.y > self.cfg.ground_y:
            self.y = self.cfg.ground_y
            self.vy = 0.0  # 지면과 닿으면 수직 속도 0

        # --- 3) 장애물 이동 및 재활용 ---
        #   화면 전체가 왼쪽으로 스크롤되는 효과를 위해
        #   '장애물이 왼쪽으로 speed만큼 이동'한다고 가정합니다.
        for ob in self.obstacles:
            ob[0] -= self.speed  # ob = [x, y, w, h]에서 x만 감소

        #   화면 왼쪽 바깥으로 완전히 나간 장애물은 제거
        self.obstacles = [ob for ob in self.obstacles if ob[0] + ob[2] > 0]

        #   장애물이 너무 떨어졌으면 새로운 장애물을 뒤(오른쪽 바깥)에 추가
        #   - 조건: 장애물 리스트가 비었거나, 마지막 장애물의 x가 충분히 왼쪽으로 갔을 때
        if len(self.obstacles) == 0 or (self.obstacles[-1][0] < self.cfg.screen_w - self._next_gap()):
            self.obstacles.append(self._spawn_obstacle(initial=False))

        # --- 4) 충돌 판정 (AABB; Axis-Aligned Bounding Box) ---
        done = False
        reward = self.cfg.alive_reward  # 기본 보상: 살아있으면 +1
        if self._collides():
            done = True
            reward = self.cfg.crash_penalty  # 충돌 시 큰 패널티

        # --- 5) 스텝 증가 및 타임아웃 ---
        self.steps += 1
        if self.steps >= self.cfg.max_steps:
            done = True  # 너무 길게 끌면 종료(학습 안정화/시간 절약 목적)

        # --- 6) 다음 상태(이산화)와 보상/종료 반환 ---
        return self._get_state(), reward, done, {}

    # -------------- Helpers --------------
    def _on_ground(self) -> bool:
        """지면에 있는지 여부.
        - y가 정확히 ground_y이고 vy도 0인 경우로 판단(부동소수 오차 대비 약간의 허용).
        """
        return abs(self.y - self.cfg.ground_y) < 1e-6 and abs(self.vy) < 1e-6

    def _spawn_obstacle(self, initial: bool) -> List[float]:
        """오른쪽 화면 밖에 새로운 장애물을 하나 생성.
        반환 형식: [x, y, w, h] (float로 저장)
        - initial=True: 리셋 직후 첫 장애물 → 기본 gap 랜덤
        - initial=False: 진행 중 추가되는 장애물 → gap 규칙은 _next_gap() 재사용
        """
        gap = self._next_gap() if not initial else self.rng.randint(self.cfg.spawn_gap_min, self.cfg.spawn_gap_max)
        x = self.cfg.screen_w + gap            # 화면 오른쪽 바깥에서 등장
        w = self.cfg.cactus_w
        h = self.cfg.cactus_h
        y = self.cfg.ground_y - h              # 지면 위에 얹히도록 높이 보정
        return [float(x), float(y), float(w), float(h)]  # [x, y, w, h]

    def _next_gap(self) -> int:
        """다음 장애물까지 간격(px)을 랜덤 샘플링."""
        return self.rng.randint(self.cfg.spawn_gap_min, self.cfg.spawn_gap_max)

    def _player_rect(self) -> Tuple[float, float, float, float]:
        """플레이어를 나타내는 간단한 AABB(22x30).
        - x는 고정, y만 위아래로 변함.
        - 반환: (px, py, pw, ph)
        """
        return (self.x, self.y - 30, 22, 30)

    def _collides(self) -> bool:
        """플레이어와 어떤 장애물이라도 AABB 충돌하면 True."""
        px, py, pw, ph = self._player_rect()
        for ox, oy, ow, oh in self.obstacles:
            # AABB 교차 판정: 한 축이라도 분리되어 있지 않으면 충돌
            if (px < ox + ow and px + pw > ox and
                py < oy + oh and py + ph > oy):
                return True
        return False

    # -------------- State (discrete) --------------
    def _get_state(self) -> Tuple[int, int]:
        """현재 연속 상태를 Q-table 학습용 '저차원 이산 상태'로 변환.
        구성:
          1) dist_bin:
             - 플레이어의 앞쪽(오른쪽)에 있는 '가장 가까운 장애물의 선두 모서리'까지의
               수평 거리(px)를 버킷화한 값.
             - 거리가 음수(이미 겹치거나 넘어섰다)면 0으로 클램핑.
             - dist_bucket_px로 나눈 몫을 사용하고, 최대 dist_bins-1로 클립.
          2) vy_bin:
             - 0: 지면(on_ground), 1: 상승(vy<0), 2: 하강(vy>0)

        반환: (dist_bin:int, vy_bin:int)
        """
        # 1) 다음 장애물까지의 거리(px): 플레이어의 '앞쪽' 선두 모서리 기준
        dist_px = 9999
        px, _, pw, _ = self._player_rect()
        for ox, oy, ow, oh in self.obstacles:
            # ox+ow >= px: 플레이어의 x위치보다 장애물의 오른쪽 끝이 오른쪽에(앞에) 있거나 겹치면 '후보'
            if ox + ow >= px:
                # 장애물 왼쪽 선두 모서리(ox)와 플레이어의 오른쪽 끝(px+pw) 사이의 거리
                dist_px = min(dist_px, int(ox - (px + pw)))
        if dist_px < 0:
            dist_px = 0  # 이미 겹치는 경우 음수가 나올 수 있으니 0으로 처리(가장 근접 버킷)

        # 2) 거리 버킷화: 픽셀 단위를 dist_bucket_px로 쪼개서 정수 버킷 인덱스 생성
        dist_bin = min(dist_px // self.cfg.dist_bucket_px, self.cfg.dist_bins - 1)

        # 3) 수직속도 버킷: 지면/상승/하강의 3상태로 단순화
        vy_bin = 0 if self._on_ground() else (1 if self.vy < 0 else 2)

        return (int(dist_bin), int(vy_bin))

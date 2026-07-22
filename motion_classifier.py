"""J, Z 등 모션(동작) 기반 수화 분류 — DTW로 단일 손가락 끝 궤적을 비교.

J는 새끼 손가락 끝(lm[20]), Z는 검지 손가락 끝(lm[8])의 (x, y) 궤적만 추적한다.
전체 손끝 5개를 쓰던 방식보다 DTW 비교가 훨씬 선별적이다.

[양손 호환]
x좌표를 반전한 미러 버전도 함께 비교해 반대 손도 인식 가능하다.
보정 데이터는 한 손으로 녹화된 것만 있어도 된다.
"""

import numpy as np
import hand_detector as HD

# 프레임 수로 정규화한 DTW 거리의 경험적 임계값.
DTW_DISTANCE_THRESHOLD = 1.1

# 각 모션 심볼의 트리거 손가락 끝 랜드마크 인덱스
# J: 새끼 손가락 끝(20), Z: 검지 손가락 끝(8)
MOTION_TIP = {"J": HD.PINKY_TIP, "Z": HD.INDEX_TIP}

# 손 크기 대비 속도 임계값 — 이 값 이상 움직이면 수집 시작
MOTION_VEL_TRIGGER = 0.15

# 수집 조기 종료 파라미터
MOTION_MIN_FRAMES = 14     # 이 프레임 수 이상 수집해야 조기 종료 허용 (낮으면 D/I 정지 중 미세 떨림이 Z/J로 오인식)
MOTION_STOP_VEL = 0.05     # 이 속도 이하를 MOTION_STOP_COUNT 프레임 연속이면 동작 완료로 판단
MOTION_STOP_COUNT = 3      # 연속 정지 프레임 수

# 정적 분류기가 이 글자를 보여줄 때 해당 모션 심볼 모니터링 시작
# I → J 모니터링,  D → Z 모니터링
MOTION_TRIGGER_LETTER = {"J": "I", "Z": "D"}

# 모니터링 버퍼 시간 — 이 시간 안에 움직임이 없으면 정적 글자(I/D)로 확정
MOTION_MONITOR_SEC = 0.8


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """두 궤적(프레임 수 × MOTION_FRAME_DIM) 간 DTW 거리.

    프레임 단위 유클리드 거리를 비용으로 사용하는 표준 동적 시간 워핑.
    """
    n, m = len(seq_a), len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            d = float(np.linalg.norm(ai - seq_b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])


def _mirror_x(seq: np.ndarray) -> np.ndarray:
    """궤적의 x좌표만 반전 — 반대 손 시뮬레이션.

    extract_tip_frame 출력 layout: [x, y] (2차원)
    열 0이 x이므로 해당 값만 부호 반전.
    """
    mirrored = seq.copy()
    mirrored[:, 0] *= -1
    return mirrored


def classify_motion(window: np.ndarray, motion_cal_data: dict) -> tuple[str, float]:
    """현재 슬라이딩 윈도우(프레임 수 × MOTION_FRAME_DIM) → 가장 가까운 모션 심볼과 거리.

    각 심볼마다 저장된 반복 녹화본 전체와 비교해 최소 거리를 사용.
    원본 윈도우와 x-반전(미러) 윈도우를 모두 비교해 반대 손도 인식 가능하게 한다.
    """
    mirror_window = _mirror_x(window)
    best_letter, best_dist = "?", float("inf")
    for letter, reps in motion_cal_data.items():
        for rep in reps:
            norm = max(len(window), len(rep))
            for query in (window, mirror_window):
                dist = _dtw_distance(query, rep) / norm
                if dist < best_dist:
                    best_dist = dist
                    best_letter = letter
    if best_dist < DTW_DISTANCE_THRESHOLD:
        return best_letter, best_dist
    return "?", best_dist

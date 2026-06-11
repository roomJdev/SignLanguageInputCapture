"""시점 불변 특징 벡터 추출.

좌표(x,y) 대신 관절 굽힘 각도 + 손가락 끝 간 거리 + 손목 거리를 사용.
카메라 각도가 달라져도 이 값들은 거의 변하지 않는다.
"""

import numpy as np

WRIST = 0
MIDDLE_MCP = 9
TIPS = [4, 8, 12, 16, 20]   # 엄지~새끼 끝

_FINGERS = [
    [1, 2, 3, 4],    # 엄지
    [5, 6, 7, 8],    # 검지
    [9, 10, 11, 12], # 중지
    [13, 14, 15, 16],# 약지
    [17, 18, 19, 20],# 새끼
]


def _angle(a, b, c) -> float:
    """b를 꼭짓점으로 하는 a-b-c 사이 각도 (라디안)."""
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    return float(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))


def extract(lm: np.ndarray) -> np.ndarray:
    """랜드마크 배열(21×3) → 시점 불변 특징 벡터(25차원).

    구성:
        10개 — 각 손가락 관절 굽힘 각도 (손가락당 2개)
        10개 — 손가락 끝 간 모든 쌍의 거리 (정규화)
         5개 — 각 손가락 끝 ~ 손목 거리 (정규화)
    """
    pts = lm[:, :3]  # z도 사용해 깊이 정보 포함
    scale = np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]) + 1e-8

    # 1. 관절 굽힘 각도 (10개)
    bend = []
    for finger in _FINGERS:
        for i in range(len(finger) - 2):
            bend.append(_angle(pts[finger[i]], pts[finger[i + 1]], pts[finger[i + 2]]))

    # 2. 손가락 끝 간 거리 — C(5,2)=10개 (정규화)
    tip_pairs = []
    for i in range(len(TIPS)):
        for j in range(i + 1, len(TIPS)):
            tip_pairs.append(np.linalg.norm(pts[TIPS[i]] - pts[TIPS[j]]) / scale)

    # 3. 손가락 끝 ~ 손목 거리 (5개, 정규화)
    wrist_dists = [np.linalg.norm(pts[t] - pts[WRIST]) / scale for t in TIPS]

    return np.array(bend + tip_pairs + wrist_dists, dtype=np.float32)

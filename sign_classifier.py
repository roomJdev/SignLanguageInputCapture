"""ASL 정적 수화 알파벳 분류기.

보정 데이터가 있으면 k-NN으로 분류, 없으면 규칙 기반으로 폴백.
J, Z는 동작(모션)이 필요해 정적 판별 불가 — '?' 반환.
"""

import numpy as np
import hand_detector as HD
from calibration import normalize


# 보정 데이터 기반 최근접 이웃 분류
_DISTANCE_THRESHOLD = 1.2   # 이 값보다 멀면 '?' 반환


def classify_calibrated(lm: np.ndarray, cal_data: dict) -> str:
    """보정 데이터와의 유클리드 거리로 가장 가까운 알파벳 반환."""
    vec = normalize(lm)
    best_letter, best_dist = "?", float("inf")
    for letter, ref_vec in cal_data.items():
        dist = float(np.linalg.norm(vec - ref_vec))
        if dist < best_dist:
            best_dist = dist
            best_letter = letter
    return best_letter if best_dist < _DISTANCE_THRESHOLD else "?"


# ---------------------------------------------------------------------------
# 기본 헬퍼
# ---------------------------------------------------------------------------

def _tip(lm, idx): return lm[idx]


def _dist(a, b):
    return float(np.linalg.norm(a[:2] - b[:2]))


def _extended(lm, tip, pip):
    """손가락 끝이 PIP 관절보다 위(y 작음) → 펴짐."""
    return lm[tip][1] < lm[pip][1]


def _curled(lm, tip, mcp):
    """손가락 끝이 MCP 관절보다 아래 → 완전히 구부림."""
    return lm[tip][1] > lm[mcp][1]


def _thumb_out(lm):
    """엄지 끝이 검지 MCP에서 충분히 멀리 떨어짐 → 엄지 펴짐."""
    return _dist(lm[HD.THUMB_TIP], lm[HD.INDEX_MCP]) > 0.12


def _fingers(lm):
    """(엄지, 검지, 중지, 약지, 새끼) 펴짐 여부 튜플."""
    return (
        _thumb_out(lm),
        _extended(lm, HD.INDEX_TIP, HD.INDEX_PIP),
        _extended(lm, HD.MIDDLE_TIP, HD.MIDDLE_PIP),
        _extended(lm, HD.RING_TIP, HD.RING_PIP),
        _extended(lm, HD.PINKY_TIP, HD.PINKY_PIP),
    )


def _all_curled(lm):
    return all(_curled(lm, t, m) for t, m in [
        (HD.INDEX_TIP, HD.INDEX_MCP),
        (HD.MIDDLE_TIP, HD.MIDDLE_MCP),
        (HD.RING_TIP, HD.RING_MCP),
        (HD.PINKY_TIP, HD.PINKY_MCP),
    ])


def _pinch(lm, tip_a, tip_b, threshold=0.06):
    return _dist(lm[tip_a], lm[tip_b]) < threshold


def _spread(lm, tip_a, tip_b, threshold=0.07):
    """두 손가락 끝이 충분히 벌어짐."""
    return _dist(lm[tip_a], lm[tip_b]) > threshold


# ---------------------------------------------------------------------------
# 분류기
# ---------------------------------------------------------------------------

def classify(lm: np.ndarray) -> str:
    """21×3 랜드마크 배열 → ASL 알파벳 문자 (J·Z는 '?' 반환)."""
    th, ix, mi, rg, pk = _fingers(lm)

    # --- J, Z: 모션 필요 ---
    # (검지만 세운 상태에서 J 동작 / 검지로 Z 그리기 — 정적 불가)

    # --- A: 주먹 + 엄지 옆으로 ---
    if th and _all_curled(lm):
        return "A"

    # --- S: 주먹 + 엄지 위로 감쌈 (엄지 접힘) ---
    if not th and _all_curled(lm) and lm[HD.THUMB_TIP][0] > lm[HD.INDEX_MCP][0]:
        return "S"

    # --- E: 네 손가락 구부려 손바닥에 붙임, 엄지 아래 ---
    if not th and not ix and not mi and not rg and not pk:
        # E: 모든 손가락 끝이 MCP보다 낮음
        if _all_curled(lm):
            return "E"

    # --- O: 모든 손가락 끝이 엄지 끝에 모임 ---
    if not ix and not mi and not rg and not pk:
        tips = [HD.INDEX_TIP, HD.MIDDLE_TIP, HD.RING_TIP, HD.PINKY_TIP]
        avg_tip = np.mean([lm[t] for t in tips], axis=0)
        if _dist(avg_tip, lm[HD.THUMB_TIP]) < 0.1:
            return "O"

    # --- C: 모든 손가락 살짝 구부려 C 모양 ---
    if not ix and not mi and not rg and not pk:
        # C는 손가락 끝이 중간 높이 (완전 펴지도도, 완전 구부리지도 않음)
        mid_y = lm[HD.WRIST][1]
        tips_y = [lm[t][1] for t in [HD.INDEX_TIP, HD.MIDDLE_TIP]]
        if all(lm[HD.INDEX_MCP][1] > y > mid_y - 0.1 for y in tips_y):
            return "C"

    # --- B: 네 손가락 펴고 엄지 접음 ---
    if not th and ix and mi and rg and pk:
        return "B"

    # --- F: 엄지+검지 핀치, 나머지 세 손가락 펴기 ---
    if _pinch(lm, HD.THUMB_TIP, HD.INDEX_TIP) and mi and rg and pk:
        return "F"

    # --- Y: 엄지+새끼 펴기 ---
    if th and not ix and not mi and not rg and pk:
        return "Y"

    # --- L: 엄지+검지 펴기 (L 모양) ---
    if th and ix and not mi and not rg and not pk:
        return "L"

    # --- D: 검지만 펴고 엄지 중지에 접촉 ---
    if not th and ix and not mi and not rg and not pk:
        if _pinch(lm, HD.THUMB_TIP, HD.MIDDLE_TIP, threshold=0.08):
            return "D"
        # 단순 검지만 세움
        return "D"

    # --- I: 새끼만 펴기 ---
    if not th and not ix and not mi and not rg and pk:
        return "I"

    # --- G: 검지 수평 + 엄지 수평 (옆을 가리킴) ---
    if th and ix and not mi and not rg and not pk:
        # G는 L과 유사하지만 검지가 수평에 가까움
        ix_horizontal = abs(lm[HD.INDEX_TIP][1] - lm[HD.INDEX_MCP][1]) < 0.05
        if ix_horizontal:
            return "G"
        return "L"

    # --- X: 검지 갈고리 모양 (PIP 위지만 DIP에서 구부림) ---
    if not th and not mi and not rg and not pk:
        ix_hooked = (lm[HD.INDEX_PIP][1] < lm[HD.INDEX_MCP][1] and
                     lm[HD.INDEX_TIP][1] > lm[HD.INDEX_DIP][1])
        if ix_hooked:
            return "X"

    # --- K: 검지+중지 펴고 엄지 검지 사이에 위치 ---
    if th and ix and mi and not rg and not pk:
        thumb_between = (lm[HD.INDEX_MCP][0] < lm[HD.THUMB_TIP][0] < lm[HD.MIDDLE_MCP][0] or
                         lm[HD.MIDDLE_MCP][0] < lm[HD.THUMB_TIP][0] < lm[HD.INDEX_MCP][0])
        if thumb_between:
            return "K"

    # --- V: 검지+중지 펴고 벌림 ---
    if not th and ix and mi and not rg and not pk:
        if _spread(lm, HD.INDEX_TIP, HD.MIDDLE_TIP):
            return "V"
        # --- U: 검지+중지 붙여서 펴기 ---
        return "U"

    # --- R: 검지+중지 교차 (가까이 붙음) ---
    if not th and ix and mi and not rg and not pk:
        return "R"

    # --- H: 검지+중지 수평으로 펴기 ---
    if not th and ix and mi and not rg and not pk:
        both_horizontal = (abs(lm[HD.INDEX_TIP][1] - lm[HD.INDEX_MCP][1]) < 0.06 and
                           abs(lm[HD.MIDDLE_TIP][1] - lm[HD.MIDDLE_MCP][1]) < 0.06)
        if both_horizontal:
            return "H"

    # --- W: 검지+중지+약지 펴기 ---
    if not th and ix and mi and rg and not pk:
        return "W"

    # --- M: 검지+중지+약지 구부려 엄지 위에 ---
    if not th and not ix and not mi and not rg and not pk:
        three_over_thumb = (lm[HD.INDEX_TIP][1] > lm[HD.THUMB_TIP][1] and
                            lm[HD.MIDDLE_TIP][1] > lm[HD.THUMB_TIP][1] and
                            lm[HD.RING_TIP][1] > lm[HD.THUMB_TIP][1])
        if three_over_thumb:
            return "M"

    # --- N: 검지+중지 구부려 엄지 위에 ---
    if not th and not ix and not mi and not rg and not pk:
        two_over_thumb = (lm[HD.INDEX_TIP][1] > lm[HD.THUMB_TIP][1] and
                          lm[HD.MIDDLE_TIP][1] > lm[HD.THUMB_TIP][1])
        if two_over_thumb:
            return "N"

    # --- T: 엄지를 검지+중지 사이에 끼움 ---
    if not th and not ix and not mi and not rg and not pk:
        thumb_high = lm[HD.THUMB_TIP][1] < lm[HD.INDEX_MCP][1]
        if thumb_high:
            return "T"

    # --- P: K 모양 아래로 향함 ---
    if th and ix and mi and not rg and not pk:
        pointing_down = lm[HD.INDEX_TIP][1] > lm[HD.WRIST][1]
        if pointing_down:
            return "P"

    # --- Q: G 모양 아래로 향함 ---
    if th and ix and not mi and not rg and not pk:
        pointing_down = lm[HD.INDEX_TIP][1] > lm[HD.WRIST][1]
        if pointing_down:
            return "Q"

    return "?"

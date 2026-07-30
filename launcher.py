"""앱 시작 화면 — 모드 선택 런처."""

import cv2
import numpy as np

_WIN = "Sign-to-Type"

_MODES = [
    "Calibrate",
    "Live Mode",
    "Live + Injection",
    "Live + Guess Mode",
    "Live + ED Mode",
    "Live + LLM Mode",
    "Test (Ordered)",
    "Test (Random)",
    "ED vs LLM Study",
    "Test Results",
    "ED vs LLM Results",
]
_KEYS = ["calibrate", "live", "live_inject", "live_guess", "live_ed", "live_llm",
         "test_ordered", "test_random", "study", "sessions", "study_sessions"]

_ROW_TOP = 130
_ROW_H   = 54
_PAD_X   = 48
_BOTTOM_PAD = 74   # 하단 구분선 + 안내 텍스트 공간

_W = 820
_H = _ROW_TOP + len(_MODES) * _ROW_H + _BOTTOM_PAD   # 항목 수에 맞춰 자동 계산

_C_BRIGHT = (80, 255, 120)
_C_MID    = (90, 200, 110)
_C_DIM    = (80, 155, 95)
_C_BG_SEL = (18, 48, 24)
_C_BG_HOV = (12, 30, 15)

# macOS / Windows / Linux 방향키 코드 (waitKeyEx 기준)
_UP_KEYS   = {63232, 2490368, 65362}
_DOWN_KEYS = {63233, 2621440, 65364}


def run_launcher(cal_missing: bool = False) -> str | None:
    """런처 화면을 표시하고 선택된 모드 키를 반환한다. 종료 시 None."""
    selected = 0
    hovered  = -1
    _clicked = [None]   # on_mouse → main loop 통신용

    cv2.namedWindow(_WIN, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        nonlocal hovered
        row = (y - _ROW_TOP) // _ROW_H
        in_range = 0 <= row < len(_MODES)
        hovered = row if in_range else -1
        if in_range and event == cv2.EVENT_LBUTTONDOWN:
            _clicked[0] = row

    cv2.setMouseCallback(_WIN, on_mouse)

    while True:
        frame = np.zeros((_H, _W, 3), dtype=np.uint8)

        # ── 타이틀 ──────────────────────────────────────────────
        cv2.putText(frame, "Sign-to-Type", (_PAD_X, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, _C_BRIGHT, 3)
        cv2.putText(frame, "Webcam-Based Fingerspelling Input  |  M1 Energy / Junsik Bang",
                    (_PAD_X, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _C_DIM, 1)

        if cal_missing:
            cv2.putText(frame, "! No calibration data found -- please run Calibrate first",
                        (_PAD_X, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 100, 255), 1)

        # 구분선
        cv2.line(frame, (_PAD_X, 116), (_W - _PAD_X, 116), _C_DIM, 1)

        # ── 옵션 행 ─────────────────────────────────────────────
        for i, title in enumerate(_MODES):
            ry = _ROW_TOP + i * _ROW_H
            is_sel = (i == selected)
            is_hov = (i == hovered) and not is_sel

            if is_sel:
                cv2.rectangle(frame,
                              (_PAD_X - 10, ry + 2),
                              (_W - _PAD_X + 10, ry + _ROW_H - 4),
                              _C_BG_SEL, -1)
                cv2.rectangle(frame,
                              (_PAD_X - 10, ry + 2),
                              (_W - _PAD_X + 10, ry + _ROW_H - 4),
                              _C_MID, 1)
                prefix    = "> "
                title_col = _C_BRIGHT
            elif is_hov:
                cv2.rectangle(frame,
                              (_PAD_X - 10, ry + 2),
                              (_W - _PAD_X + 10, ry + _ROW_H - 4),
                              _C_BG_HOV, -1)
                prefix    = "  "
                title_col = _C_MID
            else:
                prefix    = "  "
                title_col = _C_MID

            cv2.putText(frame, f"{prefix}{title}",
                        (_PAD_X, ry + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.82, title_col, 2)

        # ── 하단 안내 ───────────────────────────────────────────
        cv2.line(frame, (_PAD_X, _H - 34), (_W - _PAD_X, _H - 34), _C_DIM, 1)
        cv2.putText(frame,
                    "w/s or up/down: navigate    click: select    Enter: confirm    q: quit",
                    (_PAD_X, _H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_DIM, 1)

        cv2.imshow(_WIN, frame)

        # 마우스 클릭 처리
        if _clicked[0] is not None:
            result = _KEYS[_clicked[0]]
            _clicked[0] = None
            cv2.destroyWindow(_WIN)
            return result

        key = cv2.waitKeyEx(1)
        if key == -1:
            continue
        klow = key & 0xFF

        if klow == ord("q"):
            cv2.destroyWindow(_WIN)
            return None
        if klow == ord("w") or key in _UP_KEYS:
            selected = (selected - 1) % len(_MODES)
        if klow == ord("s") or key in _DOWN_KEYS:
            selected = (selected + 1) % len(_MODES)
        if klow in (13, 10):   # Enter
            cv2.destroyWindow(_WIN)
            return _KEYS[selected]

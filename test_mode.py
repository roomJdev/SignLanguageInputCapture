"""인식 정확도 측정용 테스트 모드.

A-Z, 0-9 순서대로(또는 랜덤) 목표 심볼과 손모양 레퍼런스 이미지를 보여주고,
사용자가 SPACE를 누르면 그 시점(정적 심볼) 또는 이후 1초간(J·Z 모션)의
인식 결과를 정답과 비교해 기록한다.

세션 결과는 test_results.json에 누적 저장되어, 회차를 거듭할수록
심볼별/전체 인식률 통계를 추적할 수 있다.
"""

import json
import os
import random
import time

import cv2
import numpy as np

from features import extract, extract_tip_frame, save_video_clip
from sign_classifier import classify, classify_calibrated
from motion_classifier import (classify_motion, MOTION_TIP, MOTION_TRIGGER_LETTER,
                               MOTION_MIN_FRAMES, MOTION_STOP_VEL, MOTION_STOP_COUNT)
from motion_calibration import MOTION_FRAMES
from ml_models import ModelManager
from calibration_profiles import list_profiles, load_profile

# 화면에 표시할 모델 이름 약어
_MODEL_SHORT_NAMES = {
    "kNN (custom)": "kNN",
    "SVM": "SVM",
    "RandomForest": "RF",
    "LogisticRegression": "LR",
    "MLP": "MLP",
    "DTW": "DTW",
}

TEST_SEQUENCE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")
MOTION_SYMBOLS = {"J", "Z"}
RESULTS_PATH = "data_new0731/test_results.json"
# 시도별 캡처 화면 원본 — 정적은 스냅샷(.jpg), 모션은 클립(.mp4)으로 세션별 폴더에 저장
TEST_MEDIA_DIR = "data_new0731/test_media"
FEEDBACK_DISPLAY_SEC = 1.0
CAPTURE_FPS_DELAY = 33   # ms
_VIDEO_FPS = round(1000 / CAPTURE_FPS_DELAY)

# macOS / Windows / Linux 방향키 코드 (waitKeyEx 기준) — 진행자가 skip/back을 조작할 때 사용
_LEFT_KEYS = {63234, 2424832, 65361}
_RIGHT_KEYS = {63235, 2555904, 65363}

_REF_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "resources", "letters")
_REF_DISPLAY_HEIGHT = 360

_GREEN_BRIGHT = (80, 255, 120)
_GREEN_MID = (90, 230, 110)
_GREEN_DIM = (100, 200, 110)
_RED_WRONG = (60, 60, 230)


# ---------------------------------------------------------------------------
# 레퍼런스 이미지
# ---------------------------------------------------------------------------

def _load_reference_images() -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    for symbol in TEST_SEQUENCE:
        path = os.path.join(_REF_IMAGE_DIR, f"{symbol}.png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = _REF_DISPLAY_HEIGHT / h
        refs[symbol] = cv2.resize(img, (int(w * scale), _REF_DISPLAY_HEIGHT))
    return refs


def _overlay_reference(annotated: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    h, w = annotated.shape[:2]
    rh, rw = ref_img.shape[:2]
    margin, pad = 16, 10
    panel_w, panel_h = rw + pad * 2, rh + pad * 2 + 24
    x0 = max(0, w - panel_w - margin)
    y0 = margin
    # 패널이 화면 밖으로 나가면 표시 생략
    if x0 + panel_w > w or y0 + panel_h > h:
        return annotated
    cv2.rectangle(annotated, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), -1)
    cv2.rectangle(annotated, (x0, y0), (x0 + panel_w, y0 + panel_h), (180, 180, 180), 1)
    annotated[y0 + pad:y0 + pad + rh, x0 + pad:x0 + pad + rw] = ref_img
    cv2.putText(annotated, "Reference", (x0 + pad, y0 + panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 140, 50), 2)
    return annotated


def _filter_cal(cal_data: dict | None, is_digit: bool) -> dict | None:
    if not cal_data:
        return cal_data
    if is_digit:
        return {k: v for k, v in cal_data.items() if k.isdigit()}
    return {k: v for k, v in cal_data.items() if not k.isdigit()}


# ---------------------------------------------------------------------------
# 인앱 텍스트 입력
# ---------------------------------------------------------------------------

def run_text_input(prompt: str = "Enter participant name:") -> str:
    """cv2 창 안에서 텍스트를 직접 타이핑해 문자열을 반환하는 간단한 입력 화면.

    - 일반 문자/숫자/공백: 입력 버퍼에 추가
    - Backspace (키코드 8): 마지막 문자 삭제
    - Enter (키코드 13): 확인 후 입력값 반환
    - Esc (키코드 27): 취소, 빈 문자열 반환
    """
    text = ""
    canvas_h, canvas_w = 360, 720
    cursor_visible = True
    cursor_tick = 0

    while True:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 안내 문구
        cv2.putText(canvas, prompt, (24, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, _GREEN_MID, 2)

        # 입력 박스 배경
        cv2.rectangle(canvas, (24, 110), (canvas_w - 24, 175), (40, 40, 40), -1)
        cv2.rectangle(canvas, (24, 110), (canvas_w - 24, 175), _GREEN_DIM, 1)

        # 입력 텍스트 + 커서
        cursor_tick += 1
        if cursor_tick % 20 == 0:
            cursor_visible = not cursor_visible
        display = text + ("|" if cursor_visible else " ")
        cv2.putText(canvas, display, (36, 157),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _GREEN_BRIGHT, 2)

        # 안내
        cv2.putText(canvas, "Enter: confirm   Esc: skip", (24, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN_DIM, 1)

        cv2.imshow("Participant", canvas)
        key = cv2.waitKey(33) & 0xFF

        if key == 13:        # Enter
            break
        elif key == 27:      # Esc
            text = ""
            break
        elif key == 8 or key == 127:   # Backspace (Windows: 8, macOS: 127)
            text = text[:-1]
        elif 32 <= key <= 126:         # 일반 출력 가능 문자
            text += chr(key)

    cv2.destroyWindow("Participant")
    return text


# ---------------------------------------------------------------------------
# 결과 저장/통계
# ---------------------------------------------------------------------------

def _load_results() -> list[dict]:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_session(sessions: list[dict], session: dict) -> None:
    sessions.append(session)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def reset_results() -> None:
    """누적 테스트 결과 전체 초기화 — 연습/디버그용 세션을 모두 지우고 싶을 때 사용."""
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)
        print(f"테스트 결과 초기화 완료: {RESULTS_PATH} 삭제됨")
    else:
        print("초기화할 테스트 결과가 없습니다.")


def list_sessions() -> None:
    """저장된 세션을 인덱스, 참가자, 시각, 점수와 함께 출력 — 삭제할 세션을 고를 때 사용."""
    sessions = _load_results()
    if not sessions:
        print("저장된 테스트 세션이 없습니다.")
        return

    print(f"\n저장된 테스트 세션 ({len(sessions)}개):")
    for i, session in enumerate(sessions):
        results = session["results"]
        correct = sum(1 for r in results.values() if r["correct"])
        total = len(results)
        participant = session.get("participant") or "(미입력)"
        profile = session.get("cal_profile") or "Default"
        order = "랜덤" if session.get("randomize") else "순서대로"
        print(f"  [{i}] {session['timestamp']}  참가자: {participant}  "
              f"보정: {profile}  {order}  {correct}/{total} ({correct / total * 100:.1f}%)")


def delete_session(index: int) -> None:
    """인덱스로 특정 세션 하나만 삭제 — list_sessions()로 인덱스 확인 후 사용."""
    sessions = _load_results()
    if not sessions:
        print("저장된 테스트 세션이 없습니다.")
        return
    if not (0 <= index < len(sessions)):
        print(f"잘못된 인덱스입니다 (0 ~ {len(sessions) - 1} 범위). --list-test-sessions 로 확인하세요.")
        return

    removed = sessions.pop(index)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)
    participant = removed.get("participant") or "(미입력)"
    print(f"세션 삭제 완료: [{index}] {removed['timestamp']}  참가자: {participant}")


def _run_session_detail(session: dict) -> None:
    """선택된 세션의 정답/오답 상세 내역을 화면에 표시.

    3색 기준:
      초록 = All Correct  (모든 모델 정답)
      주황 = Partial       (일부 모델만 정답)
      빨강 = All Wrong     (모든 모델 오답)

    JSON에는 테스트 진행 순서(랜덤 포함)로 저장되지만,
    상세 뷰에서는 항상 A-Z, 0-9 순서로 정렬해 표시한다.
    """
    _ORANGE = (0, 165, 255)   # BGR
    canvas_h, canvas_w = 720, 960
    results = session["results"]
    participant = session.get("participant") or "(no name)"
    is_random = session.get("randomize", False)
    mode_label = "RANDOM" if is_random else "IN ORDER"
    mode_color = _ORANGE if is_random else _GREEN_MID

    # TEST_SEQUENCE 순으로 정렬
    seq_order = {sym: i for i, sym in enumerate(TEST_SEQUENCE)}

    def _category(r: dict) -> str:
        """모든 모델 기준으로 3색 분류."""
        models = r.get("models", {})
        if not models:
            return "all_correct" if r["correct"] else "all_wrong"
        if all(m["correct"] for m in models.values()):
            return "all_correct"
        if all(not m["correct"] for m in models.values()):
            return "all_wrong"
        return "partial"

    sorted_results = sorted(results.items(), key=lambda x: seq_order.get(x[0], 99))
    all_correct = [(s, r) for s, r in sorted_results if _category(r) == "all_correct"]
    partial     = [(s, r) for s, r in sorted_results if _category(r) == "partial"]
    all_wrong   = [(s, r) for s, r in sorted_results if _category(r) == "all_wrong"]

    n_all_c, n_part, n_all_w = len(all_correct), len(partial), len(all_wrong)
    total = len(results)

    while True:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # --- 헤더 (4행 구조, 겹침 없음) ---
        # 행1: 제목
        cv2.putText(canvas, "Session Detail", (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _GREEN_BRIGHT, 2)
        # 행2: 시각 + 참가자 + 보정 프로필
        cal_prof = session.get("cal_profile") or "Default"
        info_line = f"{session['timestamp']}   {participant}   cal:{cal_prof}"
        cv2.putText(canvas, info_line, (24, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN_MID, 1)
        # 행3: 모드 배지 + 3색 요약 통계 (같은 행, x위치로 분리)
        cv2.putText(canvas, f"Mode: {mode_label}", (24, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
        cv2.putText(canvas, f"All correct: {n_all_c}", (240, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN_BRIGHT, 1)
        cv2.putText(canvas, f"Partial: {n_part}", (440, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _ORANGE, 1)
        cv2.putText(canvas, f"All wrong: {n_all_w}", (610, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _RED_WRONG, 1)

        divider_y = 102
        cv2.line(canvas, (24, divider_y), (canvas_w - 24, divider_y), _GREEN_DIM, 1)

        # --- 좌측: All Correct 그리드(초록) + Partial 목록(주황) ---
        col_mid = canvas_w // 2 - 12
        COLS = 8
        cell_w, cell_h = 52, 38

        # All Correct
        cv2.putText(canvas, f"All Correct  ({n_all_c})", (24, 136),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN_BRIGHT, 2)
        for idx, (sym, _) in enumerate(all_correct):
            x = 24 + (idx % COLS) * cell_w
            y = 162 + (idx // COLS) * cell_h
            cv2.putText(canvas, sym, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, _GREEN_BRIGHT, 2)

        grid_end_y = 162 + (max(1, (n_all_c + COLS - 1) // COLS)) * cell_h + 6

        # Partial (주황) — 줄별로 "E:  RF->W  MLP->C"
        if partial:
            cv2.line(canvas, (24, grid_end_y), (col_mid - 12, grid_end_y), _ORANGE, 1)
            cv2.putText(canvas, f"Partial  ({n_part})", (24, grid_end_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _ORANGE, 2)
            py = grid_end_y + 48
            for sym, r in partial:
                if py > canvas_h - 30:
                    break
                wrong_m = [(n, m["predicted"]) for n, m in r.get("models", {}).items()
                           if not m["correct"]]
                line = f"{sym}:  " + "  ".join(
                    f"{_MODEL_SHORT_NAMES.get(n, n)}->{p}" for n, p in wrong_m
                )
                cv2.putText(canvas, line, (24, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _ORANGE, 1)
                py += 26

        # --- 수직 구분선 ---
        cv2.line(canvas, (col_mid, divider_y), (col_mid, canvas_h - 40), _GREEN_DIM, 1)

        # --- 우측: All Wrong 목록(빨강) ---
        rx = col_mid + 20
        cv2.putText(canvas, f"All Wrong  ({n_all_w})", (rx, 136),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _RED_WRONG, 2)

        entry_h = 56
        for idx, (sym, r) in enumerate(all_wrong):
            base_y = 162 + idx * entry_h
            if base_y + entry_h > canvas_h - 40:
                cv2.putText(canvas, f"... and {n_all_w - idx} more",
                            (rx, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN_DIM, 1)
                break
            models = r.get("models", {})
            n_ok = sum(1 for m in models.values() if m["correct"]) if models else 0
            header = f"{sym}   ({n_ok}/{len(models)} correct)" if models else f"{sym} -> {r.get('predicted','?')}"
            cv2.putText(canvas, header, (rx, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _RED_WRONG, 2)
            tx = rx
            for name, m in models.items():
                short = _MODEL_SHORT_NAMES.get(name, name)
                token = f"{short}:{m['predicted']}"
                color = _GREEN_BRIGHT if m["correct"] else _RED_WRONG
                if tx > canvas_w - 80:
                    break
                cv2.putText(canvas, token, (tx, base_y + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                tx += cv2.getTextSize(token, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] + 18

        cv2.putText(canvas, "q: back", (24, canvas_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN_DIM, 1)

        cv2.imshow("Test Session Manager", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break


def run_session_manager() -> None:
    """저장된 테스트 세션을 화면에서 직접 조회/삭제 — CLI 인자 없이 인식 화면에서 바로 관리.

    카메라를 쓰지 않고 정적 캔버스에 목록을 그려 W/S로 탐색, D로 선택 세션 삭제,
    C로 전체 초기화한다.
    """
    selected = 0
    canvas_h, canvas_w = 720, 960
    ROW_TOP = 80
    row_h = 30
    max_rows = (canvas_h - ROW_TOP - 60) // row_h
    hovered = -1
    _clicked = [None]

    cv2.namedWindow("Test Session Manager")

    def _on_mouse_sessions(event, x, y, flags, param):
        nonlocal hovered
        row = (y - ROW_TOP) // row_h
        sessions_now = _load_results()
        n = min(len(sessions_now), max_rows)
        hovered = row if 0 <= row < n else -1
        if 0 <= row < n:
            if event == cv2.EVENT_LBUTTONDOWN:
                _clicked[0] = ("select", row)
            elif event == cv2.EVENT_LBUTTONDBLCLK:
                _clicked[0] = ("detail", row)

    cv2.setMouseCallback("Test Session Manager", _on_mouse_sessions)

    while True:
        sessions = _load_results()
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        cv2.putText(canvas, "Test Session Manager", (24, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _GREEN_BRIGHT, 2)

        if not sessions:
            cv2.putText(canvas, "No saved sessions.", (24, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _GREEN_DIM, 2)
        else:
            selected = max(0, min(selected, len(sessions) - 1))
            visible = sessions[:max_rows]
            for i, session in enumerate(visible):
                y = ROW_TOP + i * row_h
                results = session["results"]
                correct = sum(1 for r in results.values() if r["correct"])
                total = len(results)
                participant = session.get("participant") or "(no name)"
                profile = session.get("cal_profile") or "Default"
                order = "random" if session.get("randomize") else "in order"
                line = (f"[{i}] {session['timestamp']}  {participant}  "
                        f"cal:{profile}  {order}  {correct}/{total} ({correct / total * 100:.1f}%)")
                is_sel = (i == selected)
                is_hov = (i == hovered) and not is_sel
                if is_sel:
                    cv2.rectangle(canvas, (16, y - 20), (canvas_w - 16, y + 8), (18, 48, 24), -1)
                elif is_hov:
                    cv2.rectangle(canvas, (16, y - 20), (canvas_w - 16, y + 8), (12, 30, 15), -1)
                color = _GREEN_BRIGHT if is_sel else (_GREEN_MID if is_hov else _GREEN_DIM)
                prefix = "> " if is_sel else "  "
                cv2.putText(canvas, prefix + line, (24, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            if len(sessions) > max_rows:
                cv2.putText(canvas, f"... and {len(sessions) - max_rows} more (not shown)",
                            (24, ROW_TOP + max_rows * row_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN_DIM, 1)

        cv2.putText(canvas, "w/s/arrows: move   Enter/dblclick: detail   d: delete   c: clear all   q: back",
                    (24, canvas_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN_DIM, 1)

        cv2.imshow("Test Session Manager", canvas)

        # 마우스 클릭 처리
        if _clicked[0] is not None:
            action, row = _clicked[0]
            _clicked[0] = None
            if 0 <= row < len(sessions):
                selected = row
                if action == "detail":
                    _run_session_detail(sessions[selected])
            continue

        key = cv2.waitKeyEx(30)
        klow = key & 0xFF

        if klow == ord("q"):
            break
        elif (klow == ord("w") or key in (63232, 2490368, 65362)) and sessions:
            selected = max(0, selected - 1)
        elif (klow == ord("s") or key in (63233, 2621440, 65364)) and sessions:
            selected = min(len(sessions) - 1, selected + 1)
        elif klow == 13 and sessions:
            _run_session_detail(sessions[selected])
        elif klow == ord("d") and sessions:
            delete_session(selected)
            selected = max(0, selected - 1)
        elif klow == ord("c") and sessions:
            reset_results()
            selected = 0

    cv2.destroyWindow("Test Session Manager")


def run_profile_selector(default_cal_data: dict | None = None,
                          default_motion_cal_data: dict | None = None
                          ) -> tuple[str, dict | None, dict | None] | None:
    """보정 프로필 선택 화면.

    저장된 프로필 목록을 캔버스에 표시하고 w/s로 탐색, Enter로 선택한다.
    'Default'를 선택하면 현재 메모리에 로드된 보정 데이터(기본 파일)를 사용한다.
    반환값: (profile_name, cal_data, motion_cal_data) 또는 None(취소)
    """
    canvas_h, canvas_w = 480, 720
    ROW_TOP = 72
    row_h = 36
    selected = 0
    hovered = -1
    _clicked = [None]

    cv2.namedWindow("Select Profile")

    def _on_mouse_profile(event, x, y, flags, param):
        nonlocal hovered
        row = (y - ROW_TOP) // row_h
        profiles_now = list_profiles()
        n_rows = len(profiles_now) + 1
        hovered = row if 0 <= row < n_rows else -1
        if 0 <= row < n_rows and event == cv2.EVENT_LBUTTONDOWN:
            _clicked[0] = row

    cv2.setMouseCallback("Select Profile", _on_mouse_profile)

    while True:
        profiles = list_profiles()
        display = [{"name": "Default (current calibration)", "_is_default": True}] + profiles
        selected = max(0, min(selected, len(display) - 1))

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        cv2.putText(canvas, "Select Calibration Profile", (24, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _GREEN_BRIGHT, 2)
        cv2.line(canvas, (24, 58), (canvas_w - 24, 58), _GREEN_DIM, 1)

        for i, p in enumerate(display):
            y = ROW_TOP + i * row_h
            is_sel = (i == selected)
            is_hov = (i == hovered) and not is_sel
            if is_sel:
                cv2.rectangle(canvas, (16, y - 22), (canvas_w - 16, y + 10), (18, 48, 24), -1)
            elif is_hov:
                cv2.rectangle(canvas, (16, y - 22), (canvas_w - 16, y + 10), (12, 30, 15), -1)
            prefix = "> " if is_sel else "  "
            color = _GREEN_BRIGHT if is_sel else (_GREEN_MID if is_hov else _GREEN_DIM)
            if p.get("_is_default"):
                label = f"{prefix}[Default]  (data/calibration_data.npy)"
            else:
                label = f"{prefix}[{i}] {p['name']}   ({p.get('created_at', '?')[:10]})"
            cv2.putText(canvas, label, (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1 if not is_sel else 2)

        cv2.putText(canvas, "w/s: move   click/Enter: select   q: cancel",
                    (24, canvas_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN_DIM, 1)

        cv2.imshow("Select Profile", canvas)

        # 마우스 클릭 처리
        if _clicked[0] is not None:
            chosen_idx = _clicked[0]
            _clicked[0] = None
            if 0 <= chosen_idx < len(display):
                cv2.destroyWindow("Select Profile")
                chosen = display[chosen_idx]
                if chosen.get("_is_default"):
                    return "Default", default_cal_data, default_motion_cal_data
                cal, motion_cal = load_profile(chosen["name"])
                return chosen["name"], cal, motion_cal

        key = cv2.waitKeyEx(30)
        klow = key & 0xFF

        if klow == ord("q") or klow == 27:
            cv2.destroyWindow("Select Profile")
            return None
        elif klow == ord("w") or key in (63232, 2490368, 65362):
            selected = max(0, selected - 1)
        elif klow == ord("s") or key in (63233, 2621440, 65364):
            selected = min(len(display) - 1, selected + 1)
        elif klow == 13:
            cv2.destroyWindow("Select Profile")
            chosen = display[selected]
            if chosen.get("_is_default"):
                return "Default", default_cal_data, default_motion_cal_data
            cal, motion_cal = load_profile(chosen["name"])
            return chosen["name"], cal, motion_cal


def run_test_confirm(participant: str, profile_name: str) -> bool:
    """테스트 시작 전 참가자/보정 프로필 확인 화면.

    SPACE 또는 Enter로 확인, q/Esc로 취소.
    """
    _ORANGE = (0, 165, 255)
    canvas_h, canvas_w = 300, 640
    while True:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        cv2.putText(canvas, "Confirm Test Settings", (24, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, _GREEN_BRIGHT, 2)
        cv2.line(canvas, (24, 58), (canvas_w - 24, 58), _GREEN_DIM, 1)
        cv2.putText(canvas, f"Participant:   {participant or '(unnamed)'}",
                    (24, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _GREEN_MID, 2)
        cv2.putText(canvas, f"Calibration:   {profile_name}",
                    (24, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _ORANGE, 2)
        cv2.putText(canvas, "SPACE / Enter: start   q / Esc: back",
                    (24, canvas_h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN_DIM, 1)
        cv2.imshow("Confirm Test", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord(" "), 13):
            cv2.destroyWindow("Confirm Test")
            return True
        elif key in (ord("q"), 27):
            cv2.destroyWindow("Confirm Test")
            return False


def _cumulative_stats(sessions: list[dict]) -> dict[str, tuple[int, int]]:
    """심볼별 (정답 수, 시도 수) 누적."""
    stats: dict[str, list[int]] = {}
    for session in sessions:
        for symbol, entry in session["results"].items():
            counts = stats.setdefault(symbol, [0, 0])
            counts[1] += 1
            if entry["correct"]:
                counts[0] += 1
    return {k: (v[0], v[1]) for k, v in stats.items()}


def _cumulative_model_stats(sessions: list[dict]) -> dict[str, tuple[int, int]]:
    """모델별 (정답 수, 시도 수) 누적 — 모델별 인식 성능 비교용.

    구버전 세션(models 필드 없음)은 건너뛴다.
    """
    stats: dict[str, list[int]] = {}
    for session in sessions:
        for entry in session["results"].values():
            for name, m in entry.get("models", {}).items():
                counts = stats.setdefault(name, [0, 0])
                counts[1] += 1
                if m["correct"]:
                    counts[0] += 1
    return {k: (v[0], v[1]) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# 테스트 모드 본체
# ---------------------------------------------------------------------------

def run_test_mode(detector, cal_data: dict | None, motion_cal_data: dict | None,
                   camera_index: int = 0, randomize: bool = False,
                   participant: str = "", cal_profile: str = "Default") -> None:
    """테스트 모드 실행 — 심볼을 순서대로(또는 랜덤) 제시하고 인식 결과를 기록.

    participant: 세션을 구분하기 위한 참가자 이름/번호. 나중에 list_sessions() /
    delete_session()으로 특정 회차만 골라 지울 때 timestamp와 함께 식별자로 쓰인다.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 [{camera_index}]를 열 수 없습니다.")

    reference_images = _load_reference_images()
    sequence = TEST_SEQUENCE.copy()
    if randomize:
        random.shuffle(sequence)

    # 보정 데이터로 비교 모델들을 한 번만 학습 — 모델마다 다시 동작을 입력할 필요 없음
    model_manager = ModelManager()
    model_manager.fit(cal_data)
    print(f"비교 모델: {', '.join(model_manager.model_names())}")

    sessions = _load_results()
    session_results: dict[str, dict] = {}
    # B/←로 되돌아가 재시도하면서 덮어써지는 이전 시도들을 보존 —
    # 최종 결과 기준 정확도와 별개로 "최초 시도 기준" 정확도도 나중에 계산 가능하게
    discarded_results: dict[str, list[dict]] = {}

    # 이번 세션의 캡처 화면 원본을 모아둘 폴더 (참가자명 + 세션 시작 시각)
    media_dir = os.path.join(TEST_MEDIA_DIR, f"{participant or 'unnamed'}_{time.strftime('%Y%m%dT%H%M%S')}")

    idx = 0
    state = "waiting"   # waiting -> recording(모션만) -> feedback -> waiting
    motion_buffer: list[np.ndarray] = []
    motion_landmarks_buffer: list[np.ndarray] = []   # feature 가공 전 21관절 원본 좌표(프레임별)
    motion_frame_buffer: list[np.ndarray] = []   # 캡처 화면 원본(비디오 저장용)
    motion_tip_prev = None
    motion_slow_count = 0
    last_predicted = "?"
    last_correct = False
    last_model_results: dict[str, dict] = {}   # {model_name: {"predicted":..., "correct":...}}
    feedback_until = 0.0
    space_held = False

    participant_label = participant or "(미입력)"
    print(f"\n테스트 모드 시작 — 참가자: {participant_label}  "
          f"{len(sequence)}개 심볼 ({'랜덤' if randomize else '순서대로'})")
    print("심볼을 따라한 뒤 SPACE를 누르세요. S/→: 건너뛰기  B/←: 뒤로  Q: 중단 및 결과 저장\n")

    while idx < len(sequence):
        target = sequence[idx]
        is_digit = target.isdigit()
        is_motion = target in MOTION_SYMBOLS

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks_list, annotated = detector.process(frame)
        h, w = annotated.shape[:2]

        if state == "recording":
            if landmarks_list:
                lm_motion = landmarks_list[0]
                tip_idx = MOTION_TIP[target]
                wrist = lm_motion[0, :2]
                hand_scale = float(np.linalg.norm(lm_motion[9, :2] - wrist)) + 1e-8
                tip_pos = (lm_motion[tip_idx, :2] - wrist) / hand_scale
                motion_buffer.append(extract_tip_frame(lm_motion, tip_idx))
                motion_landmarks_buffer.append(lm_motion.copy())
                motion_frame_buffer.append(frame.copy())
                # 조기 종료 판정
                cur_vel = float(np.linalg.norm(tip_pos - motion_tip_prev)) if motion_tip_prev is not None else 1.0
                if len(motion_buffer) >= MOTION_MIN_FRAMES:
                    motion_slow_count = motion_slow_count + 1 if cur_vel < MOTION_STOP_VEL else 0
                motion_tip_prev = tip_pos
            done = (motion_slow_count >= MOTION_STOP_COUNT or len(motion_buffer) >= MOTION_FRAMES)
            if done and motion_buffer:
                window = np.stack(motion_buffer)
                raw_landmarks_window = np.stack(motion_landmarks_buffer)
                video_path = os.path.join(media_dir, f"{target}.mp4")
                save_video_clip(motion_frame_buffer, video_path, _VIDEO_FPS)
                predicted, _dist = classify_motion(window, motion_cal_data or {})
                motion_buffer = []
                motion_landmarks_buffer = []
                motion_frame_buffer = []
                motion_tip_prev = None
                motion_slow_count = 0
                last_predicted = predicted
                last_correct = (predicted == target)
                last_model_results = {"DTW": {"predicted": predicted, "correct": last_correct}}
                if target in session_results:
                    discarded_results.setdefault(target, []).append(session_results[target])
                session_results[target] = {
                    "predicted": predicted,
                    "correct": last_correct,
                    "models": last_model_results,
                    "raw_window": window.tolist(),
                    "raw_landmarks_window": raw_landmarks_window.tolist(),
                    "video_path": video_path,
                }
                print(f"  [{target}] -> {predicted}  {'OK' if last_correct else 'X'}")
                state = "feedback"
                feedback_until = time.time() + FEEDBACK_DISPLAY_SEC
        elif state == "feedback" and time.time() >= feedback_until:
            idx += 1
            state = "waiting"

        # --- HUD ---
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

        cv2.putText(annotated, target, (w // 2 - 60, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, _GREEN_BRIGHT, 10)

        progress_text = f"{idx + 1} / {len(sequence)}"
        cv2.putText(annotated, progress_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, _GREEN_MID, 2)

        if state == "recording":
            bar_w = int(w * len(motion_buffer) / MOTION_FRAMES)
            cv2.rectangle(annotated, (0, h - 12), (bar_w, h), _GREEN_BRIGHT, -1)
            cv2.putText(annotated, "Recording motion...", (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, _GREEN_MID, 2)
        elif state == "feedback":
            # 모델별 예측 결과를 한 줄에 토큰으로 나열 (예: kNN:D OK   SVM:D OK   RF:T X)
            x = 20
            row_y = h - 24
            if not last_model_results:
                cv2.putText(annotated, "No calibration data for this symbol", (x, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED_WRONG, 2)
            for name, r in last_model_results.items():
                short = _MODEL_SHORT_NAMES.get(name, name)
                token = f"{short}:{r['predicted']}"
                color = _GREEN_BRIGHT if r["correct"] else _RED_WRONG
                if x > w - 100:   # 화면 폭을 넘으면 다음 줄로
                    x = 20
                    row_y -= 30
                cv2.putText(annotated, token, (x, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                token_w = cv2.getTextSize(token, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
                x += token_w + 28
        else:
            if is_motion:
                instruction = f"Get ready, press SPACE, then perform the '{target}' motion (1s)"
            else:
                instruction = f"Perform '{target}', then press SPACE"
            scale = 0.75
            text_w = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
            if text_w > w - 40:
                scale *= (w - 40) / text_w
            cv2.putText(annotated, instruction, (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, _GREEN_DIM, 2)

        cv2.putText(annotated, "SPACE: test  /  S or ->: skip  /  B or <-: back  /  Q: stop & save",
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN_DIM, 1)

        ref_img = reference_images.get(target)
        if ref_img is not None:
            annotated = _overlay_reference(annotated, ref_img)

        cv2.imshow("Test Mode", annotated)

        # waitKeyEx로 원본 키코드를 받아 화살표 키(좌/우)까지 인식 —
        # 스플릿 키보드 등에서 참가자는 SPACE만, 진행자는 화살표로 skip/back을 조작할 수 있게.
        key_raw = cv2.waitKeyEx(CAPTURE_FPS_DELAY)
        key = key_raw & 0xFF
        space_down = (key == ord(" "))

        if space_down and not space_held and state == "waiting":
            if is_motion:
                if not motion_cal_data:
                    print(f"  [{target}] 모션 보정 데이터 없음 — 건너뜀")
                    idx += 1
                elif not landmarks_list:
                    print(f"  [{target}] 손이 감지되지 않음 — 다시 시도하세요")
                else:
                    # 라이브 인식과 동일하게, 시작 트리거 자세(J: I, Z: D)인지 확인 후에만
                    # 녹화 시작 — 엉뚱한 손가락으로 해도 통과되는 것을 방지
                    trigger = MOTION_TRIGGER_LETTER[target]
                    letter_cal = _filter_cal(cal_data, False)
                    current_static = classify_calibrated(landmarks_list[0], letter_cal) if letter_cal else classify(landmarks_list[0])
                    if current_static != trigger:
                        print(f"  [{target}] 시작 자세가 '{trigger}'가 아님(현재: {current_static}) — "
                              f"올바른 손모양으로 준비 후 다시 SPACE")
                    else:
                        state = "recording"
                        motion_buffer = []
                        motion_landmarks_buffer = []
                        motion_frame_buffer = []
                        motion_tip_prev = None
                        motion_slow_count = 0
            elif landmarks_list:
                lm = landmarks_list[0]
                active_cal = _filter_cal(cal_data, is_digit)
                raw_vector = None
                if active_cal:
                    vec = extract(lm)
                    model_preds = model_manager.predict(vec, is_digit)
                    raw_vector = vec.tolist()
                else:
                    model_preds = {"Rule-based": classify(lm)} if not is_digit else {}

                last_model_results = {
                    name: {"predicted": pred, "correct": (pred == target)}
                    for name, pred in model_preds.items()
                }
                # kNN(custom)을 대표 결과로 사용 (없으면 Rule-based, 그것도 없으면 '?')
                predicted = model_preds.get("kNN (custom)", model_preds.get("Rule-based", "?"))
                last_predicted = predicted
                last_correct = (predicted == target)
                frame_path = os.path.join(media_dir, f"{target}.jpg")
                os.makedirs(media_dir, exist_ok=True)
                cv2.imwrite(frame_path, frame)
                if target in session_results:
                    discarded_results.setdefault(target, []).append(session_results[target])
                session_results[target] = {
                    "predicted": predicted,
                    "raw_vector": raw_vector,
                    "raw_landmarks": lm.tolist(),
                    "correct": last_correct,
                    "models": last_model_results,
                    "frame_path": frame_path,
                }
                summary = "  ".join(f"{n}:{p}{'OK' if p == target else 'X'}" for n, p in model_preds.items())
                print(f"  [{target}] -> {summary or '?'}")
                state = "feedback"
                feedback_until = time.time() + FEEDBACK_DISPLAY_SEC
        elif key == ord("s") or key_raw in _RIGHT_KEYS:
            print(f"  [{target}] 건너뜀")
            idx += 1
            state = "waiting"
            motion_buffer = []
            motion_landmarks_buffer = []
            motion_frame_buffer = []
        elif (key == ord("b") or key_raw in _LEFT_KEYS) and idx > 0:
            # 스킵과 동일하게 제한 없이 되돌아감(연속으로 누르면 여러 단계 이동) —
            # 진행자가 화살표 키로 조작(스플릿 키보드에서 참가자는 SPACE만 사용)
            idx -= 1
            state = "waiting"
            motion_buffer = []
            motion_landmarks_buffer = []
            motion_frame_buffer = []
            print(f"  [{sequence[idx]}] 다시 시도")
        elif key == ord("q"):
            print("테스트 중단 — 현재까지 결과를 저장합니다.")
            break

        space_held = space_down

    cap.release()
    cv2.destroyWindow("Test Mode")

    if not session_results:
        print("기록된 테스트 결과 없음.")
        return

    session = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "participant": participant,
        "cal_profile": cal_profile,
        "randomize": randomize,
        "results": session_results,
        "discarded_attempts": discarded_results,
    }
    _save_session(sessions, session)

    if discarded_results:
        n_redone = sum(len(v) for v in discarded_results.values())
        print(f"(재시도로 덮어써진 이전 시도 {n_redone}건은 discarded_attempts에 별도 보존됨: "
              f"{sorted(discarded_results.keys())})")

    correct_count = sum(1 for r in session_results.values() if r["correct"])
    total = len(session_results)
    print(f"\n이번 세션: {correct_count}/{total} 정답 ({correct_count / total * 100:.1f}%)")

    cumulative = _cumulative_stats(sessions)
    overall_correct = sum(c for c, _ in cumulative.values())
    overall_total = sum(t for _, t in cumulative.values())
    print(f"누적 ({len(sessions)}개 세션): {overall_correct}/{overall_total} "
          f"({overall_correct / overall_total * 100:.1f}%)")
    print("\n심볼별 누적 인식률:")
    for symbol in TEST_SEQUENCE:
        if symbol in cumulative:
            c, t = cumulative[symbol]
            print(f"  {symbol}: {c}/{t} ({c / t * 100:.1f}%)")

    model_stats = _cumulative_model_stats(sessions)
    if model_stats:
        print("\n모델별 누적 인식률:")
        ranked = sorted(model_stats.items(), key=lambda kv: -(kv[1][0] / kv[1][1]))
        for name, (c, t) in ranked:
            print(f"  {name}: {c}/{t} ({c / t * 100:.1f}%)")

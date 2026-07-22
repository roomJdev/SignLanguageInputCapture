"""수화 알파벳 실시간 감지 — 메인 실행 파일.

실행:
    python main.py                # 카메라 자동 선택
    python main.py --camera 1     # 카메라 인덱스 직접 지정
    python main.py --list         # 사용 가능한 카메라 목록 출력
    python main.py --calibrate    # 강제 재보정
    python main.py --test         # 테스트 모드 (A-Z, 0-9 순서대로 인식률 측정)
    python main.py --test --test-random --participant "Junsik"   # 테스트 모드, 랜덤 순서, 참가자 지정
    python main.py --list-test-sessions       # 저장된 테스트 세션 목록 (인덱스, 참가자, 시각, 점수)
    python main.py --delete-test-session 2    # 인덱스 2번 세션만 삭제
    python main.py --reset-test-results       # 누적 테스트 결과 전체 초기화 후 종료

단축키 (인식 화면) — E R T Y U 기능 키 + Q 종료:
    e   철자(LETTER) / 숫자(NUMBER) 모드 전환
    r   재보정 세션 시작 (정적 심볼 + J/Z 모션)
    t   테스트 모드 시작 (순서대로)
    y   테스트 모드 시작 (랜덤 순서)
    u   테스트 세션 관리 (조회/삭제)
    q   종료
"""

import argparse
import os
import sys
import platform
import time

# ── 의존성 자동 설치 ────────────────────────────────────────────────────────
def _ensure_deps() -> None:
    import importlib
    import subprocess

    packages = [
        ("cv2",        "opencv-python"),
        ("mediapipe",  "mediapipe"),
        ("numpy",      "numpy"),
        ("sklearn",    "scikit-learn"),
        ("pyautogui",  "pyautogui"),
        ("pynput",     "pynput"),
        ("wordfreq",   "wordfreq"),
    ]
    if platform.system() == "Darwin":
        packages.append(("AVFoundation", "pyobjc-framework-AVFoundation"))

    missing = []
    for module, package in packages:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)

    if not missing:
        return

    print("=" * 56)
    print("[setup] Missing packages detected:")
    for p in missing:
        print(f"         - {p}")
    print("[setup] Installing now...")
    print("=" * 56)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install"] + missing,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    print("=" * 56)
    print("[setup] Done. Restarting...")
    print("=" * 56)
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_deps()
# ────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np

from calibration import CALIBRATION_PATH, load_calibration, run_calibration
from motion_calibration import (
    MOTION_CALIBRATION_PATH,
    MOTION_FRAMES,
    load_motion_calibration,
    run_motion_calibration,
)
from features import extract_tip_frame
from hand_detector import HandDetector
from sign_classifier import classify, classify_calibrated
from motion_classifier import (classify_motion, DTW_DISTANCE_THRESHOLD,
                               MOTION_TIP, MOTION_VEL_TRIGGER,
                               MOTION_MIN_FRAMES, MOTION_STOP_VEL, MOTION_STOP_COUNT,
                               MOTION_TRIGGER_LETTER, MOTION_MONITOR_SEC)
from test_mode import (
    run_test_mode,
    run_profile_selector,
    run_test_confirm,
    reset_results,
    list_sessions,
    delete_session,
    run_session_manager,
    run_text_input,
)
from calibration_profiles import copy_default_as_profile
from keystroke_injector import KeystrokeInjector
from launcher import run_launcher
import word_suggester

_MOTION_COOLDOWN_SEC = 1.5   # 한 번 인식 후 다음 모션 인식까지 최소 간격
_MOTION_DISPLAY_SEC = 1.0    # 인식된 모션 글자를 화면에 유지하는 시간
_MOTION_LETTERS = ["J", "Z"] # 시작 포즈로 감지하는 모션 심볼 목록

_GUESS_STABLE_FRAMES = 10    # Guess 모드에서 글자가 이 프레임 수만큼 유지돼야 버퍼에 추가
_WI_STABLE_FRAMES    = 10    # Word Injection 모드 동일 기준


# ---------------------------------------------------------------------------
# 카메라 감지
# ---------------------------------------------------------------------------

def _find_camera_macos() -> int:
    """macOS: AVFoundation으로 내장 카메라 이름을 찾아 인덱스 반환."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
        for i, device in enumerate(devices):
            name = device.localizedName() or ""
            if "macbook" in name.lower() or "facetime" in name.lower():
                print(f"내장 카메라 감지: [{i}] {name}")
                return i
    except Exception:
        pass
    return 0


def _find_camera_default() -> int:
    """Windows / 외장 웹캠: 인덱스 0~4 중 처음 열리는 카메라 반환."""
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return 0


def auto_camera_index() -> int:
    if platform.system() == "Darwin":
        return _find_camera_macos()
    return _find_camera_default()


def list_cameras() -> None:
    print("사용 가능한 카메라:")
    found = False
    for i in range(8):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w, h = int(cap.get(3)), int(cap.get(4))
            print(f"  [{i}] {w}x{h}")
            cap.release()
            found = True
    if not found:
        print("  감지된 카메라 없음")

    if platform.system() == "Darwin":
        try:
            from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
            devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
            print("\nAVFoundation 카메라 이름:")
            for i, d in enumerate(devices):
                print(f"  [{i}] {d.localizedName()}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 인식 루프
# ---------------------------------------------------------------------------

def _filter_by_symbol_mode(cal_data: dict | None, symbol_mode: str) -> dict | None:
    """보정 데이터를 현재 심볼 모드(letter/number)에 맞게 필터링."""
    if not cal_data:
        return cal_data
    if symbol_mode == "number":
        return {k: v for k, v in cal_data.items() if k.isdigit()}
    return {k: v for k, v in cal_data.items() if not k.isdigit()}


def run(detector: HandDetector, cal_data: dict | None, motion_cal_data: dict | None,
        camera_index: int, inject: bool = False, start_guess: bool = False,
        start_ed: bool = False, start_llm: bool = False) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"카메라 [{camera_index}]를 열 수 없습니다. --list 로 가용 카메라를 확인하세요.")
        sys.exit(1)

    current_letter = "?"
    symbol_mode = "letter"   # "letter" 또는 "number" — e 키로 전환
    cal_mode = "CALIBRATED" if cal_data else "RULE-BASED"
    print(f"수화 감지 시작 [{cal_mode}] — e: 철자/숫자 모드 전환  r: 재보정  t/y: 테스트  q: 종료")

    # 심볼별 속도 기반 모션 트래커
    # collecting: 수집 중 여부 / window: 수집된 프레임 / tip_prev: 직전 프레임 손가락 끝 위치
    motion_trackers = {
        letter: {
            "monitoring": False,    # 정적 trigger 글자 감지 후 대기 중
            "monitor_start": 0.0,
            "timed_out": False,     # 타임아웃 후 trigger 글자 재진입 전까지 재모니터링 금지
            "collecting": False,    # 실제 궤적 수집 중
            "window": [],
            "tip_prev": None,
            "slow_count": 0,
        }
        for letter in _MOTION_LETTERS
    }
    motion_cooldown_until = 0.0
    motion_display_letter = ""
    motion_display_until = 0.0
    # 모션 인식 직후 static 분류가 덮어쓰지 않도록 동결하는 타이머
    motion_override_until = 0.0

    injector = KeystrokeInjector()
    if inject:
        injector.enabled = True

    # 마우스 클릭 통신 — [action, arg]
    # action: "candidate"(idx), "space", "backspace"
    _mouse_click: list = [None, None]

    _WIN_NAME = "Sign Language Input"
    cv2.namedWindow(_WIN_NAME, cv2.WINDOW_AUTOSIZE)

    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # 후보 버튼 / Space / Backspace 영역은 draw 단계에서 _btn_rects에 저장
        for action, arg, (x0, y0, x1, y1) in _btn_rects:
            if x0 <= x <= x1 and y0 <= y <= y1:
                _mouse_click[0] = action
                _mouse_click[1] = arg

    cv2.setMouseCallback(_WIN_NAME, _on_mouse)
    _btn_rects: list = []   # [(action, arg, (x0,y0,x1,y1))]
    print("[injector] keystroke injection ON (--inject flag)" if inject else "")
    text_buffer = ""

    # Guess 모드 상태
    guess_mode = start_guess
    guess_buffer = ""
    guess_candidates: list[str] = []
    guess_matched_prefix = ""   # 실제 매칭된 prefix (fallback 시 잘림)
    _guess_stable_letter = ""
    _guess_stable_count = 0
    _guess_last_added = ""   # 연속 재추가 방지용 — 글자가 바뀌어야 다시 추가
    # Guess 모드 수화 1/2/3 후보 선택 트래커
    _guess_num_letter = ""   # 현재 안정화 중인 숫자 수화 ("1"/"2"/"3")
    _guess_num_count = 0
    _guess_num_triggered = ""  # 마지막으로 선택을 발동한 숫자 (재발동 방지)
    _guess_num_cooldown_until = 0.0  # 선택 직후 재발동 방지 타이머
    _guess_candidates_frozen: list[str] = []  # 숫자 수화 감지 시작 시 freeze된 후보

    # ── Word Injection 모드 상태 ─────────────────────────────────────
    wi_mode = False
    wi_buffer = ""
    wi_candidates: list[str] = []
    wi_matched_prefix = ""
    _wi_stable_letter = ""
    _wi_stable_count = 0
    _wi_last_added = ""

    # ── ED 모드 상태 ─────────────────────────────────────────────────
    ed_mode = start_ed   # X키로 토글 — guess/wi 모드와 독립적으로 동작
    ed_buffer = ""
    ed_candidates: list[str] = []
    ed_matched_prefix = ""
    _ed_stable_letter = ""
    _ed_stable_count = 0
    _ed_last_added = ""
    _ED_STABLE_FRAMES = 10

    # ── LLM 모드 상태 ────────────────────────────────────────────────
    llm_mode = start_llm   # L키로 토글
    llm_context = ""       # 확정된 단어들 ("I was waiting for a ")
    llm_buffer = ""        # 현재 입력 중인 prefix
    llm_candidates: list[str] = []
    llm_source = ""        # "llm" or "ed~..." (fallback 표시용)
    _llm_stable_letter = ""
    _llm_stable_count = 0
    _llm_last_added = ""
    _LLM_STABLE_FRAMES = 10
    _key_ignore_until = 0.0   # 단어 선택 직후 주입된 키가 CV2로 돌아와 단축키를 오트리거하는 것 방지


    # 백그라운드 로드 (첫 suggest 호출 전 미리 준비)
    word_suggester.preload()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks_list, annotated = detector.process(frame)
        active_cal = _filter_by_symbol_mode(cal_data, symbol_mode)
        now = time.time()

        if landmarks_list:
            lm = landmarks_list[0]

            # 정적 분류
            if now >= motion_override_until:
                if active_cal:
                    current_letter = classify_calibrated(lm, active_cal)
                elif symbol_mode == "letter":
                    current_letter = classify(lm)
                else:
                    current_letter = "?"

            # --- Guess 모드: 수화 1/2/3으로 후보 선택 ---
            # number 분류기를 병렬 실행해 "1"/"2"/"3"이 안정되면 후보 선택
            if False and guess_mode and cal_data and now >= _guess_num_cooldown_until:  # 수화 숫자 선택 비활성화
                num_cal = _filter_by_symbol_mode(cal_data, "number")
                if num_cal:
                    num_sign = classify_calibrated(lm, num_cal)
                    if num_sign in ("1", "2", "3"):
                        if num_sign == _guess_num_letter:
                            _guess_num_count += 1
                        else:
                            _guess_num_letter = num_sign
                            _guess_num_count = 1
                            _guess_num_triggered = ""
                            # 숫자 수화 감지 시작 시 현재 후보 freeze
                            _guess_candidates_frozen = list(guess_candidates)
                        if (_guess_num_count >= _GUESS_STABLE_FRAMES
                                and _guess_num_triggered != num_sign):
                            idx_sel = int(num_sign) - 1
                            # freeze된 후보 기준으로 선택 (감지 중 candidates 변경 무시)
                            frozen = _guess_candidates_frozen if _guess_candidates_frozen else guess_candidates
                            if idx_sel < len(frozen):
                                word = frozen[idx_sel]
                                # guess_buffer는 외부 앱에 주입된 적 없으므로 단어 전체 주입
                                injector.inject_string(word + " ")
                                text_buffer += word + " "
                                guess_buffer = ""
                                guess_candidates = []
                                _guess_candidates_frozen = []
                                _guess_stable_letter = ""
                                _guess_stable_count = 0
                                _guess_last_added = ""
                                _guess_num_triggered = num_sign
                                _guess_num_count = 0
                                _guess_num_cooldown_until = now + 1.5  # 1.5초 재발동 방지
                                print(f"[guess] sign {num_sign} -> {word}")
                    else:
                        if num_sign != _guess_num_letter:
                            _guess_num_letter = ""
                            _guess_num_count = 0
                            _guess_num_triggered = ""
                            _guess_candidates_frozen = []

            # --- J, Z 모션 인식 ---
            # 정적 분류가 trigger 글자(I→J, D→Z)를 보여줄 때 모니터링 시작
            # 모니터링 중 움직임 감지 → 수집 → DTW → 맞으면 J/Z 주입
            # 버퍼 시간 초과 + 움직임 없음 → I/D 확정, 정적 injector 정상 동작
            any_monitoring = False
            if motion_cal_data and symbol_mode == "letter":
                wrist = lm[0, :2]
                hand_scale = np.linalg.norm(lm[9, :2] - wrist) + 1e-8
                for ml, tracker in motion_trackers.items():
                    if now < motion_cooldown_until:
                        tracker["monitoring"] = False
                        tracker["collecting"] = False
                        tracker["window"] = []
                        tracker["tip_prev"] = None
                        tracker["slow_count"] = 0
                        continue

                    tip_pos = (lm[MOTION_TIP[ml], :2] - wrist) / hand_scale
                    trigger = MOTION_TRIGGER_LETTER[ml]

                    if not tracker["collecting"] and not tracker["monitoring"]:
                        if tracker["timed_out"]:
                            # 타임아웃 후 trigger 글자가 사라져야 재모니터링 허용
                            if current_letter != trigger:
                                tracker["timed_out"] = False
                        elif current_letter == trigger:
                            tracker["monitoring"] = True
                            tracker["monitor_start"] = now
                            tracker["tip_prev"] = tip_pos
                            print(f"[motion] {ml} 모니터링 시작 (trigger={trigger})")

                    elif tracker["monitoring"]:
                        any_monitoring = True
                        if tracker["tip_prev"] is not None:
                            vel = float(np.linalg.norm(tip_pos - tracker["tip_prev"]))
                            if vel > MOTION_VEL_TRIGGER:
                                tracker["collecting"] = True
                                tracker["monitoring"] = False
                                tracker["window"] = [extract_tip_frame(lm, MOTION_TIP[ml])]
                                tracker["slow_count"] = 0
                                print(f"[motion] {ml} 수집 시작 (vel={vel:.3f})")
                        tracker["tip_prev"] = tip_pos
                        if now - tracker["monitor_start"] > MOTION_MONITOR_SEC:
                            tracker["monitoring"] = False
                            tracker["timed_out"] = True
                            tracker["tip_prev"] = None
                            print(f"[motion] {ml} 모니터링 타임아웃 → {trigger} 확정")

                    elif tracker["collecting"]:
                        any_monitoring = True
                        tracker["window"].append(extract_tip_frame(lm, MOTION_TIP[ml]))
                        cur_vel = float(np.linalg.norm(tip_pos - tracker["tip_prev"])) if tracker["tip_prev"] is not None else 1.0
                        if len(tracker["window"]) >= MOTION_MIN_FRAMES:
                            tracker["slow_count"] = tracker["slow_count"] + 1 if cur_vel < MOTION_STOP_VEL else 0
                        tracker["tip_prev"] = tip_pos
                        done = (tracker["slow_count"] >= MOTION_STOP_COUNT or
                                len(tracker["window"]) >= MOTION_FRAMES)
                        if done:
                            reps = motion_cal_data.get(ml, [])
                            if reps:
                                window_arr = np.stack(tracker["window"])
                                _, dist = classify_motion(window_arr, {ml: reps})
                                print(f"[motion] {ml} DTW dist={dist:.4f}  frames={len(tracker['window'])}  threshold={DTW_DISTANCE_THRESHOLD}")
                                if dist < DTW_DISTANCE_THRESHOLD:
                                    current_letter = ml
                                    motion_display_letter = ml
                                    motion_display_until = now + _MOTION_DISPLAY_SEC
                                    motion_cooldown_until = now + _MOTION_COOLDOWN_SEC
                                    motion_override_until = now + _MOTION_DISPLAY_SEC
                                    if guess_mode:
                                        guess_buffer += ml
                                        guess_candidates, guess_matched_prefix = word_suggester.suggest(guess_buffer)
                                    elif wi_mode:
                                        wi_buffer += ml
                                        wi_candidates, wi_matched_prefix = word_suggester.suggest(wi_buffer)
                                    elif ed_mode:
                                        ed_buffer = word_suggester.clean_motion_artifacts(ed_buffer + ml)
                                        ed_candidates, ed_matched_prefix = word_suggester.suggest_ed(ed_buffer)
                                    elif llm_mode:
                                        llm_buffer = word_suggester.clean_motion_artifacts(llm_buffer + ml)
                                        llm_candidates, llm_source = word_suggester.suggest_llm(llm_context, llm_buffer)
                                    else:
                                        injected = injector.inject_now(ml)
                                        if injected:
                                            text_buffer += injected
                            tracker["collecting"] = False
                            tracker["window"] = []
                            tracker["tip_prev"] = None
                            tracker["slow_count"] = 0

            # 모니터링/수집 중에는 I/D 정적 주입 억제
            if not any_monitoring:
                if guess_mode:
                    # Guess 모드: 주입 없이 안정화 프레임 수만 추적 → guess_buffer에 누적
                    if current_letter not in ("?", ""):
                        if current_letter == _guess_stable_letter:
                            _guess_stable_count += 1
                        else:
                            _guess_stable_letter = current_letter
                            _guess_stable_count = 1
                            _guess_last_added = ""
                        if (_guess_stable_count >= _GUESS_STABLE_FRAMES
                                and _guess_last_added != current_letter):
                            _guess_last_added = current_letter
                            guess_buffer += current_letter
                            guess_candidates, guess_matched_prefix = word_suggester.suggest(guess_buffer)
                            _guess_num_letter = ""
                            _guess_num_count = 0
                            _guess_num_triggered = ""
                            _guess_candidates_frozen = []
                            _guess_num_cooldown_until = now + 0.8
                    else:
                        _guess_stable_letter = ""
                        _guess_stable_count = 0
                elif wi_mode:
                    # Word Injection 모드: guess_buffer와 동일한 누적 방식
                    if current_letter not in ("?", ""):
                        if current_letter == _wi_stable_letter:
                            _wi_stable_count += 1
                        else:
                            _wi_stable_letter = current_letter
                            _wi_stable_count = 1
                            _wi_last_added = ""
                        if (_wi_stable_count >= _WI_STABLE_FRAMES
                                and _wi_last_added != current_letter):
                            _wi_last_added = current_letter
                            wi_buffer += current_letter
                            wi_candidates, wi_matched_prefix = word_suggester.suggest(wi_buffer)
                    else:
                        _wi_stable_letter = ""
                        _wi_stable_count = 0
                elif ed_mode:
                    # ED 모드: suggest_ed 기반 단어 추천
                    if current_letter not in ("?", ""):
                        if current_letter == _ed_stable_letter:
                            _ed_stable_count += 1
                        else:
                            _ed_stable_letter = current_letter
                            _ed_stable_count = 1
                            _ed_last_added = ""
                        if (_ed_stable_count >= _ED_STABLE_FRAMES
                                and _ed_last_added != current_letter):
                            _ed_last_added = current_letter
                            ed_buffer = word_suggester.clean_motion_artifacts(ed_buffer + current_letter)
                            ed_candidates, ed_matched_prefix = word_suggester.suggest_ed(ed_buffer)
                    else:
                        _ed_stable_letter = ""
                elif llm_mode:
                    # LLM 모드: Ollama 컨텍스트 기반 단어 추천
                    if current_letter not in ("?", ""):
                        if current_letter == _llm_stable_letter:
                            _llm_stable_count += 1
                        else:
                            _llm_stable_letter = current_letter
                            _llm_stable_count = 1
                            _llm_last_added = ""
                        if (_llm_stable_count >= _LLM_STABLE_FRAMES
                                and _llm_last_added != current_letter):
                            _llm_last_added = current_letter
                            llm_buffer = word_suggester.clean_motion_artifacts(llm_buffer + current_letter)
                            llm_candidates, llm_source = word_suggester.suggest_llm(llm_context, llm_buffer)
                    else:
                        _llm_stable_letter = ""
                        _ed_stable_count = 0
                else:
                    injected = injector.update(current_letter)
                    if injected == "SPACE":
                        text_buffer += " "
                    elif injected:
                        text_buffer += injected
        else:
            # 손이 사라지면 수집 중인 윈도우 초기화
            for tracker in motion_trackers.values():
                tracker["monitoring"] = False
                tracker["timed_out"] = False
                tracker["collecting"] = False
                tracker["window"] = []
                tracker["tip_prev"] = None
                tracker["slow_count"] = 0

        h, w = annotated.shape[:2]

        # BGR 연두색 팔레트 — 밝기로 정보 위계 구분
        GREEN_BRIGHT = (80, 255, 120)   # 큰 글자(인식 결과)
        GREEN_MID = (90, 230, 110)      # 모드/상태 표시
        GREEN_DIM = (100, 200, 110)     # 하단 안내 텍스트

        # 좌상단 — 인식된 글자만 단독 표시
        cv2.rectangle(annotated, (10, 10), (110, 90), (0, 0, 0), -1)
        cv2.putText(annotated, current_letter, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, GREEN_BRIGHT, 4)

        # Guess 모드 패널 — 화면 중앙 하단 클릭 버튼 UI
        _btn_rects.clear()
        if guess_mode:
            # ── 후보 버튼 3개 가로 배치 ──────────────────────────
            BTN_W, BTN_H = 220, 68
            BTN_GAP = 16
            total_w = 3 * BTN_W + 2 * BTN_GAP
            bx_start = w // 2 - total_w // 2
            # 화면 중앙보다 아래 (60% 지점)
            by = int(h * 0.60)

            # 버퍼 표시 (버튼 위)
            g_cursor = "_" if int(now * 2) % 2 == 0 else " "
            if not guess_buffer:
                buf_label = "[ GUESS MODE ]"
                buf_col = GREEN_BRIGHT
            elif guess_matched_prefix == guess_buffer.lower():
                buf_label = f"[ {guess_buffer}{g_cursor} ]"
                buf_col = GREEN_BRIGHT
            else:
                # fallback: 실제 매칭 prefix 강조 (노란빛)
                matched_upper = guess_matched_prefix.upper().lstrip("~")
                tail = guess_buffer[len(matched_upper):]
                buf_label = f"[ {matched_upper} +{tail}{g_cursor} ]"
                buf_col = (80, 200, 255)
            buf_tw = cv2.getTextSize(buf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
            cv2.putText(annotated, buf_label, (w // 2 - buf_tw // 2, by - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, buf_col, 2)

            for ci in range(3):
                bx = bx_start + ci * (BTN_W + BTN_GAP)
                has_cand = ci < len(guess_candidates)
                label = guess_candidates[ci] if has_cand else "--"
                num_label = f"{ci + 1}"
                bg_col = (15, 50, 22) if has_cand else (15, 15, 15)
                border_col = GREEN_MID if has_cand else (40, 40, 40)
                txt_col = GREEN_BRIGHT if has_cand else (50, 50, 50)
                num_col = GREEN_DIM if has_cand else (40, 40, 40)
                cv2.rectangle(annotated, (bx, by), (bx + BTN_W, by + BTN_H), bg_col, -1)
                cv2.rectangle(annotated, (bx, by), (bx + BTN_W, by + BTN_H), border_col, 2)
                # 번호 (좌상단 작게)
                cv2.putText(annotated, num_label, (bx + 10, by + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, num_col, 1)
                # 단어 (중앙)
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0]
                cv2.putText(annotated, label, (bx + BTN_W // 2 - tw // 2, by + BTN_H - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, txt_col, 2)
                if has_cand:
                    _btn_rects.append(("candidate", ci, (bx, by, bx + BTN_W, by + BTN_H)))

            # ── Space / Backspace 버튼 (후보 버튼 아래) ──────────
            sb_y = by + BTN_H + 18
            SB_W, SB_H = 160, 52
            sp_x = w // 2 - SB_W - 10
            bs_x = w // 2 + 10
            for label, action, rx in [("SPACE", "space", sp_x), ("BKSP", "backspace", bs_x)]:
                cv2.rectangle(annotated, (rx, sb_y), (rx + SB_W, sb_y + SB_H), (18, 18, 45), -1)
                cv2.rectangle(annotated, (rx, sb_y), (rx + SB_W, sb_y + SB_H), (90, 90, 180), 2)
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0][0]
                cv2.putText(annotated, label, (rx + SB_W // 2 - tw // 2, sb_y + 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (160, 160, 255), 2)
                _btn_rects.append((action, None, (rx, sb_y, rx + SB_W, sb_y + SB_H)))

        # Word Injection 모드 패널
        if wi_mode:
            AMBER_BRIGHT = (50, 200, 255)   # 황금빛 (BGR)
            AMBER_MID    = (60, 180, 230)
            AMBER_DIM    = (80, 150, 180)

            BTN_W, BTN_H = 220, 68
            BTN_GAP = 16
            total_w = 3 * BTN_W + 2 * BTN_GAP
            bx_start = w // 2 - total_w // 2
            by = int(h * 0.60)

            # 버퍼 표시
            wi_cursor = "_" if int(now * 2) % 2 == 0 else " "
            if not wi_buffer:
                wi_buf_label = "[ WORD INJECT ]"
                wi_buf_col = AMBER_BRIGHT
            elif wi_matched_prefix == wi_buffer.lower():
                wi_buf_label = f"[ {wi_buffer}{wi_cursor} ]"
                wi_buf_col = AMBER_BRIGHT
            else:
                matched_upper = wi_matched_prefix.upper().lstrip("~")
                tail = wi_buffer[len(matched_upper):]
                wi_buf_label = f"[ {matched_upper} +{tail}{wi_cursor} ]"
                wi_buf_col = (80, 200, 255)
            buf_tw = cv2.getTextSize(wi_buf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
            cv2.putText(annotated, wi_buf_label, (w // 2 - buf_tw // 2, by - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, wi_buf_col, 2)

            for ci in range(3):
                bx = bx_start + ci * (BTN_W + BTN_GAP)
                has_cand = ci < len(wi_candidates)
                label = wi_candidates[ci] if has_cand else "--"
                num_label = f"{ci + 1}"
                bg_col    = (10, 35, 55) if has_cand else (15, 15, 15)
                border_col = AMBER_MID if has_cand else (40, 40, 40)
                txt_col    = AMBER_BRIGHT if has_cand else (50, 50, 50)
                num_col    = AMBER_DIM if has_cand else (40, 40, 40)
                cv2.rectangle(annotated, (bx, by), (bx + BTN_W, by + BTN_H), bg_col, -1)
                cv2.rectangle(annotated, (bx, by), (bx + BTN_W, by + BTN_H), border_col, 2)
                cv2.putText(annotated, num_label, (bx + 10, by + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, num_col, 1)
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0]
                cv2.putText(annotated, label, (bx + BTN_W // 2 - tw // 2, by + BTN_H - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, txt_col, 2)

            # BKSP 버튼만 (Space는 pynput 선택 후 자동)
            sb_y = by + BTN_H + 18
            SB_W, SB_H = 160, 52
            bs_x = w // 2 - SB_W // 2
            cv2.rectangle(annotated, (bs_x, sb_y), (bs_x + SB_W, sb_y + SB_H), (18, 18, 45), -1)
            cv2.rectangle(annotated, (bs_x, sb_y), (bs_x + SB_W, sb_y + SB_H), (90, 90, 180), 2)
            tw = cv2.getTextSize("BKSP", cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0][0]
            cv2.putText(annotated, "BKSP", (bs_x + SB_W // 2 - tw // 2, sb_y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (160, 160, 255), 2)
            _btn_rects.append(("wi_backspace", None, (bs_x, sb_y, bs_x + SB_W, sb_y + SB_H)))

            hint = "switch here, press 1/2/3  ->  auto-returns to target app"
            ht = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            cv2.putText(annotated, hint, (w // 2 - ht // 2, sb_y + SB_H + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER_DIM, 1)

        # ED 모드 패널 — 청록색 계열
        if ed_mode:
            ED_BRIGHT = (220, 210, 80)   # 청록 (BGR)
            ED_MID    = (180, 175, 60)
            ED_DIM    = (130, 125, 50)

            BTN_W, BTN_H = 220, 68
            BTN_GAP = 16
            total_w = 3 * BTN_W + 2 * BTN_GAP
            bx_start = w // 2 - total_w // 2
            by = int(h * 0.60)

            # 버퍼 표시
            ed_cursor = "_" if int(now * 2) % 2 == 0 else " "
            if not ed_buffer:
                ed_buf_label = "[ ED MODE ]"
                ed_buf_col = ED_BRIGHT
            elif ed_matched_prefix.startswith("ed~"):
                ed_buf_label = f"[ {ed_buffer}{ed_cursor} ] (ed)"
                ed_buf_col = (80, 210, 255)
            else:
                ed_buf_label = f"[ {ed_buffer}{ed_cursor} ]"
                ed_buf_col = ED_BRIGHT
            buf_tw = cv2.getTextSize(ed_buf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
            cv2.putText(annotated, ed_buf_label, (w // 2 - buf_tw // 2, by - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, ed_buf_col, 2)

            for ci in range(3):
                bx = bx_start + ci * (BTN_W + BTN_GAP)
                has_cand = ci < len(ed_candidates)
                label = ed_candidates[ci] if has_cand else "--"
                num_label = f"{ci + 1}"
                bg_col     = (10, 40, 40) if has_cand else (15, 15, 15)
                border_col = ED_MID if has_cand else (40, 40, 40)
                txt_col    = ED_BRIGHT if has_cand else (50, 50, 50)
                num_col    = ED_DIM if has_cand else (40, 40, 40)
                cv2.rectangle(annotated, (bx, by), (bx + BTN_W, by + BTN_H), bg_col, -1)
                cv2.rectangle(annotated, (bx, by), (bx + BTN_W, by + BTN_H), border_col, 2)
                cv2.putText(annotated, num_label, (bx + 10, by + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, num_col, 1)
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0]
                cv2.putText(annotated, label, (bx + BTN_W // 2 - tw // 2, by + BTN_H - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, txt_col, 2)
                if has_cand:
                    _btn_rects.append(("candidate", ci, (bx, by, bx + BTN_W, by + BTN_H)))

            # BKSP 버튼
            sb_y = by + BTN_H + 18
            SB_W, SB_H = 160, 52
            bs_x = w // 2 - SB_W // 2
            cv2.rectangle(annotated, (bs_x, sb_y), (bs_x + SB_W, sb_y + SB_H), (18, 35, 35), -1)
            cv2.rectangle(annotated, (bs_x, sb_y), (bs_x + SB_W, sb_y + SB_H), ED_MID, 2)
            tw = cv2.getTextSize("BKSP", cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0][0]
            cv2.putText(annotated, "BKSP", (bs_x + SB_W // 2 - tw // 2, sb_y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, ED_BRIGHT, 2)
            _btn_rects.append(("backspace", None, (bs_x, sb_y, bs_x + SB_W, sb_y + SB_H)))

            hint = "sign prefix  ->  1/2/3 to select  (edit distance fallback)"
            ht = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            cv2.putText(annotated, hint, (w // 2 - ht // 2, sb_y + SB_H + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ED_DIM, 1)

        # LLM 모드 패널 — 보라색 계열
        if llm_mode:
            LLM_BRIGHT = (220, 130, 255)   # 보라 (BGR)
            LLM_MID    = (175, 100, 200)
            LLM_DIM    = (120, 70,  140)

            BTN_W, BTN_H = 220, 68
            BTN_GAP = 16
            total_w = 3 * BTN_W + 2 * BTN_GAP
            bx_start = w // 2 - total_w // 2
            by = int(h * 0.60)

            # 컨텍스트 표시 (상단)
            ctx_short = llm_context[-50:] if len(llm_context) > 50 else llm_context
            ctx_label = f"ctx: \"{ctx_short}\"" if llm_context else "ctx: (empty — ED fallback)"
            ctx_tw = cv2.getTextSize(ctx_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
            cv2.putText(annotated, ctx_label, (w // 2 - ctx_tw // 2, by - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, LLM_DIM, 1)

            # 버퍼 표시
            llm_cursor = "_" if int(now * 2) % 2 == 0 else " "
            is_fallback = llm_source.startswith("ed~")
            if not llm_buffer:
                llm_buf_label = "[ LLM MODE ]"
                llm_buf_col = LLM_BRIGHT
            elif is_fallback:
                llm_buf_label = f"[ {llm_buffer}{llm_cursor} ] (ed fallback)"
                llm_buf_col = (80, 210, 255)
            else:
                llm_buf_label = f"[ {llm_buffer}{llm_cursor} ]"
                llm_buf_col = LLM_BRIGHT
            buf_tw = cv2.getTextSize(llm_buf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
            cv2.putText(annotated, llm_buf_label, (w // 2 - buf_tw // 2, by - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, llm_buf_col, 2)

            for ci in range(3):
                bx = bx_start + ci * (BTN_W + BTN_GAP)
                has_cand = ci < len(llm_candidates)
                label = llm_candidates[ci] if has_cand else "--"
                num_label = f"{ci + 1}"
                bg_col     = (30, 10, 40) if has_cand else (15, 15, 15)
                border_col = LLM_MID if has_cand else (40, 40, 40)
                txt_col    = LLM_BRIGHT if has_cand else (50, 50, 50)
                num_col    = LLM_DIM if has_cand else (40, 40, 40)
                cv2.rectangle(annotated, (bx, by), (bx + BTN_W, by + BTN_H), bg_col, -1)
                cv2.rectangle(annotated, (bx, by), (bx + BTN_W, by + BTN_H), border_col, 2)
                cv2.putText(annotated, num_label, (bx + 10, by + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, num_col, 1)
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0]
                cv2.putText(annotated, label, (bx + BTN_W // 2 - tw // 2, by + BTN_H - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, txt_col, 2)
                if has_cand:
                    _btn_rects.append(("llm_candidate", ci, (bx, by, bx + BTN_W, by + BTN_H)))

            # BKSP 버튼
            sb_y = by + BTN_H + 18
            SB_W, SB_H = 160, 52
            bs_x = w // 2 - SB_W // 2
            cv2.rectangle(annotated, (bs_x, sb_y), (bs_x + SB_W, sb_y + SB_H), (30, 10, 40), -1)
            cv2.rectangle(annotated, (bs_x, sb_y), (bs_x + SB_W, sb_y + SB_H), LLM_MID, 2)
            tw = cv2.getTextSize("BKSP", cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0][0]
            cv2.putText(annotated, "BKSP", (bs_x + SB_W // 2 - tw // 2, sb_y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, LLM_BRIGHT, 2)
            _btn_rects.append(("llm_backspace", None, (bs_x, sb_y, bs_x + SB_W, sb_y + SB_H)))

            src_label = "LLM" if llm_source == "llm" else "ED fallback"
            hint = f"sign prefix  ->  1/2/3 to select  [{src_label}  model: {word_suggester.get_llm_model()}  m: switch]"
            ht = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            cv2.putText(annotated, hint, (w // 2 - ht // 2, sb_y + SB_H + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LLM_DIM, 1)

        # 텍스트 버퍼 영역 — 모드 바 바로 위
        cv2.rectangle(annotated, (0, h - 128), (w, h - 96), (20, 20, 20), -1)
        # 너무 길면 뒤에서 잘라서 표시
        max_chars = 55
        display_buf = text_buffer[-max_chars:] if len(text_buffer) > max_chars else text_buffer
        cursor = "_" if int(now * 2) % 2 == 0 else " "   # 깜빡이는 커서
        cv2.putText(annotated, f"> {display_buf}{cursor}", (10, h - 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN_BRIGHT, 2)

        # 좌하단 — 모드/보정 상태 + 단축키 안내 (별도 배경)
        mode_label = "NUMBER" if symbol_mode == "number" else "LETTER"
        extra_label = ("  GUESS:ON" if guess_mode else "  WORD-INJECT:ON" if wi_mode else "  ED:ON" if ed_mode else "  LLM:ON" if llm_mode else "")
        cv2.rectangle(annotated, (0, h - 96), (w, h), (0, 0, 0), -1)
        cv2.putText(annotated, f"Mode: {mode_label}  /  {cal_mode}{extra_label}", (10, h - 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN_MID, 2)
        inject_label = "INJECT:ON" if injector.enabled else "INJECT:OFF"
        wi_label     = "WI:ON"     if wi_mode         else "WI:OFF"
        ed_label     = "ED:ON"     if ed_mode         else "ED:OFF"
        llm_label    = "LLM:ON"   if llm_mode        else "LLM:OFF"
        cv2.putText(annotated, f"e: mode  r: recalib  i: {inject_label}  g: guess  w: {wi_label}  x: {ed_label}  l: {llm_label}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, GREEN_DIM, 2)
        cv2.putText(annotated, "t/y: test (random)  u: sessions  p: profile  q: quit", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN_DIM, 2)

        # 우상단 — J/Z 모션 인식 결과를 잠시 표시(flash)
        if motion_display_letter and now < motion_display_until:
            badge_text = f"Motion: {motion_display_letter}"
            text_w = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
            bx0 = w - text_w - 36
            cv2.rectangle(annotated, (bx0, 10), (w - 10, 56), (0, 0, 0), -1)
            cv2.putText(annotated, badge_text, (bx0 + 12, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN_BRIGHT, 2)

        cv2.imshow(_WIN_NAME, annotated)

        # ── 마우스 클릭 처리 ──────────────────────────────────────
        if _mouse_click[0] is not None:
            action, arg = _mouse_click
            _mouse_click[0] = None
            if action == "candidate" and guess_mode:
                if arg < len(guess_candidates):
                    word = guess_candidates[arg]
                    injector.inject_string(word + " ")
                    text_buffer += word + " "
                    guess_buffer = ""
                    guess_candidates = []
                    _guess_candidates_frozen = []
                    _guess_stable_letter = ""
                    _guess_stable_count = 0
                    _guess_last_added = ""
                    print(f"[guess] 클릭 선택: {word}")
            elif action == "space" and guess_mode:
                if guess_buffer:
                    injector.inject_string(guess_buffer + " ")
                    text_buffer += guess_buffer + " "
                else:
                    injector.inject_string(" ")
                    text_buffer += " "
                guess_buffer = ""
                guess_candidates = []
                _guess_stable_letter = ""
                _guess_stable_count = 0
                _guess_last_added = ""
            elif action == "backspace" and guess_mode:
                guess_buffer = guess_buffer[:-1]
                guess_candidates, guess_matched_prefix = word_suggester.suggest(guess_buffer) if guess_buffer else ([], "")
                _guess_stable_letter = ""
                _guess_stable_count = 0
                _guess_last_added = ""
            elif action == "wi_backspace" and wi_mode:
                wi_buffer = wi_buffer[:-1]
                wi_candidates, wi_matched_prefix = word_suggester.suggest(wi_buffer) if wi_buffer else ([], "")
                _wi_stable_letter = ""
                _wi_stable_count = 0
                _wi_last_added = ""
            elif action == "backspace" and ed_mode:
                ed_buffer = ed_buffer[:-1]
                ed_candidates, ed_matched_prefix = word_suggester.suggest_ed(ed_buffer) if ed_buffer else ([], "")
                _ed_stable_letter = ""
                _ed_stable_count = 0
                _ed_last_added = ""
            elif action == "llm_candidate" and llm_mode:
                if arg < len(llm_candidates):
                    word = llm_candidates[arg]
                    injector.inject_string(word + " ")
                    text_buffer += word + " "
                    llm_context += word + " "
                    llm_buffer = ""
                    llm_candidates = []
                    llm_source = ""
                    _llm_stable_letter = ""
                    _llm_stable_count = 0
                    _llm_last_added = ""
                    print(f"[llm] 클릭 선택: {word}")
            elif action == "llm_backspace" and llm_mode:
                llm_buffer = llm_buffer[:-1]
                # BKSP는 빠른 응답이 중요하므로 LLM 대신 ED로 즉시 갱신
                llm_candidates, llm_source = word_suggester.suggest_ed(llm_buffer) if llm_buffer else ([], "")
                _llm_stable_letter = ""
                _llm_stable_count = 0
                _llm_last_added = ""

        key = cv2.waitKey(1) & 0xFF
        if now < _key_ignore_until:
            key = 0xFF   # 쿨다운 중 — 주입된 키가 단축키를 오트리거하는 것 방지
        if key == ord("q"):
            break
        if key == ord("i"):
            injector.toggle()
        if key == ord("w"):
            wi_mode = not wi_mode
            if wi_mode:
                guess_mode = False
                ed_mode = False
                llm_mode = False
                guess_buffer = ""; guess_candidates = []; _guess_stable_letter = ""; _guess_stable_count = 0; _guess_last_added = ""
                ed_buffer = ""; ed_candidates = []; _ed_stable_letter = ""; _ed_stable_count = 0; _ed_last_added = ""
                llm_buffer = ""; llm_candidates = []; llm_source = ""; _llm_stable_letter = ""; _llm_stable_count = 0; _llm_last_added = ""
            else:
                wi_buffer = ""; wi_candidates = []; _wi_stable_letter = ""; _wi_stable_count = 0; _wi_last_added = ""
            print(f"[wi] Word Injection 모드 {'ON' if wi_mode else 'OFF'}")

        # ED 모드 토글 (X키)
        if key == ord("x"):
            ed_mode = not ed_mode
            if ed_mode:
                guess_mode = False
                wi_mode = False
                llm_mode = False
                guess_buffer = ""; guess_candidates = []; _guess_stable_letter = ""; _guess_stable_count = 0; _guess_last_added = ""
                wi_buffer = ""; wi_candidates = []; _wi_stable_letter = ""; _wi_stable_count = 0; _wi_last_added = ""
                llm_buffer = ""; llm_candidates = []; llm_source = ""; _llm_stable_letter = ""; _llm_stable_count = 0; _llm_last_added = ""
            else:
                ed_buffer = ""; ed_candidates = []; _ed_stable_letter = ""; _ed_stable_count = 0; _ed_last_added = ""
            print(f"[ed] ED 모드 {'ON' if ed_mode else 'OFF'}")

        # LLM 모드 토글 (L키)
        if key == ord("l"):
            llm_mode = not llm_mode
            if llm_mode:
                guess_mode = False
                wi_mode = False
                ed_mode = False
                guess_buffer = ""; guess_candidates = []; _guess_stable_letter = ""; _guess_stable_count = 0; _guess_last_added = ""
                wi_buffer = ""; wi_candidates = []; _wi_stable_letter = ""; _wi_stable_count = 0; _wi_last_added = ""
                ed_buffer = ""; ed_candidates = []; _ed_stable_letter = ""; _ed_stable_count = 0; _ed_last_added = ""
            else:
                llm_buffer = ""; llm_candidates = []; llm_source = ""; _llm_stable_letter = ""; _llm_stable_count = 0; _llm_last_added = ""
            print(f"[llm] LLM 모드 {'ON' if llm_mode else 'OFF'}")

        # LLM 모드 — 키보드 1/2/3 후보 선택
        if llm_mode and key in (ord("1"), ord("2"), ord("3")):
            idx = key - ord("1")
            if idx < len(llm_candidates):
                word = llm_candidates[idx]
                injector.inject_string(word + " ")
                text_buffer += word + " "
                llm_context += word + " "
                llm_buffer = ""
                llm_candidates = []
                llm_source = ""
                _llm_stable_letter = ""
                _llm_stable_count = 0
                _llm_last_added = ""
                _key_ignore_until = now + 0.4   # 주입된 글자가 CV2로 돌아오는 것 방지
                print(f"[llm] 선택: {word}")

        # LLM 모드 — M키로 모델 순환
        if llm_mode and key == ord("m"):
            new_model = word_suggester.cycle_llm_model()
            llm_candidates = []
            llm_source = ""
            print(f"[llm] 모델 전환: {new_model}")

        # ED 모드 — 키보드 1/2/3 후보 선택
        if ed_mode and key in (ord("1"), ord("2"), ord("3")):
            idx = key - ord("1")
            if idx < len(ed_candidates):
                word = ed_candidates[idx]
                injector.inject_string(word + " ")
                text_buffer += word + " "
                ed_buffer = ""
                ed_candidates = []
                ed_matched_prefix = ""
                _ed_stable_letter = ""
                _ed_stable_count = 0
                _ed_last_added = ""
                _key_ignore_until = now + 0.4
                print(f"[ed] 선택: {word}")

        # Word Injection 모드 — 키보드 1/2/3 후보 선택 후 Cmd+Tab으로 이전 앱 복귀
        if wi_mode and key in (ord("1"), ord("2"), ord("3")):
            idx = key - ord("1")
            if idx < len(wi_candidates):
                word = wi_candidates[idx]
                import pyautogui as _pag
                import time as _time
                _pag.hotkey("command", "tab")   # 타겟 앱으로 포커스 복귀
                _time.sleep(0.2)
                _pag.typewrite(word + " ", interval=0.05)
                _time.sleep(0.1)
                _pag.hotkey("command", "tab")   # OpenCV 창으로 복귀
                text_buffer += word + " "
                wi_buffer = ""
                wi_candidates = []
                wi_matched_prefix = ""
                _wi_stable_letter = ""
                _wi_stable_count = 0
                _wi_last_added = ""
                print(f"[wi] 주입: {word}")

        # Guess 모드 — 키보드 1/2/3 후보 선택
        if guess_mode and key in (ord("1"), ord("2"), ord("3")):
            idx = key - ord("1")
            if idx < len(guess_candidates):
                word = guess_candidates[idx]
                injector.inject_string(word + " ")
                text_buffer += word + " "
                guess_buffer = ""
                guess_candidates = []
                _guess_candidates_frozen = []
                _guess_stable_letter = ""
                _guess_stable_count = 0
                _guess_last_added = ""
                _key_ignore_until = now + 0.4
                print(f"[guess] 키 선택: {word}")

        # SPACE
        if key == ord(" "):
            if guess_mode:
                if guess_buffer:
                    text_buffer += guess_buffer + " "
                else:
                    text_buffer += " "
                guess_buffer = ""
                guess_candidates = []
                _guess_stable_letter = ""
                _guess_stable_count = 0
                _guess_last_added = ""
            elif injector.enabled:
                text_buffer += " "

        # Backspace (macOS: 127)
        if key == 127:
            if guess_mode:
                guess_buffer = guess_buffer[:-1]
                guess_candidates, guess_matched_prefix = word_suggester.suggest(guess_buffer) if guess_buffer else ([], "")
                _guess_stable_letter = ""
                _guess_stable_count = 0
                _guess_last_added = ""
            elif injector.enabled:
                text_buffer = text_buffer[:-1]

        # injection ON 중에는 주입된 키가 cv2로 돌아와 단축키를 오트리거할 수 있으므로 억제
        if injector.enabled:
            continue
        if key == ord("g"):
            guess_mode = not guess_mode
            if guess_mode:
                wi_mode = False
                ed_mode = False
                llm_mode = False
                wi_buffer = ""; wi_candidates = []; _wi_stable_letter = ""; _wi_stable_count = 0; _wi_last_added = ""
                ed_buffer = ""; ed_candidates = []; _ed_stable_letter = ""; _ed_stable_count = 0; _ed_last_added = ""
                llm_buffer = ""; llm_candidates = []; llm_source = ""; _llm_stable_letter = ""; _llm_stable_count = 0; _llm_last_added = ""
            else:
                guess_buffer = ""; guess_candidates = []; _guess_stable_letter = ""; _guess_stable_count = 0; _guess_last_added = ""
            print(f"[guess] 모드 {'ON' if guess_mode else 'OFF'}")
        if key == ord("e"):
            symbol_mode = "number" if symbol_mode == "letter" else "letter"
            current_letter = "?"
            print(f"모드 전환: {symbol_mode}")
        if key == ord("r"):
            cap.release()
            cv2.destroyAllWindows()
            cal_data = run_calibration(detector, camera_index)
            motion_cal_data = run_motion_calibration(detector, camera_index)
            cal_mode = "CALIBRATED" if cal_data else "RULE-BASED"
            # 보정 완료 후 프로필 이름 저장
            profile_name = run_text_input("Save calibration as (Enter to skip):")
            if profile_name:
                copy_default_as_profile(profile_name)
            current_letter = "?"
            for tracker in motion_trackers.values():
                tracker["monitoring"] = False; tracker["timed_out"] = False; tracker["collecting"] = False; tracker["window"] = []; tracker["tip_prev"] = None; tracker["slow_count"] = 0
            motion_display_letter = ""
            cap = cv2.VideoCapture(camera_index)
            print(f"재보정 완료 — 인식 재개 [{cal_mode}]")
        if key in (ord("t"), ord("y")):
            randomize = (key == ord("y"))
            cap.release()
            cv2.destroyAllWindows()
            # 1) 참가자 이름 입력
            participant = run_text_input("Enter participant name / number:")
            # 2) 보정 프로필 선택
            profile_result = run_profile_selector(cal_data, motion_cal_data)
            if profile_result is None:
                # 취소 → 테스트 모드 진입 안 함
                cap = cv2.VideoCapture(camera_index)
                continue
            profile_name, test_cal_data, test_motion_cal_data = profile_result
            # 3) 확인 화면
            if not run_test_confirm(participant, profile_name):
                cap = cv2.VideoCapture(camera_index)
                continue
            run_test_mode(detector, test_cal_data, test_motion_cal_data, camera_index,
                          randomize=randomize, participant=participant,
                          cal_profile=profile_name)
            current_letter = "?"
            for tracker in motion_trackers.values():
                tracker["monitoring"] = False; tracker["timed_out"] = False; tracker["collecting"] = False; tracker["window"] = []; tracker["tip_prev"] = None; tracker["slow_count"] = 0
            motion_display_letter = ""
            cap = cv2.VideoCapture(camera_index)
            print(f"테스트 모드 종료 — 인식 재개 [{cal_mode}]")
        if key == ord("u"):
            cap.release()
            cv2.destroyAllWindows()
            run_session_manager()
            cap = cv2.VideoCapture(camera_index)
            print(f"세션 관리 종료 — 인식 재개 [{cal_mode}]")
        if key == ord("p"):
            cap.release()
            cv2.destroyAllWindows()
            profile_result = run_profile_selector(cal_data, motion_cal_data)
            if profile_result is not None:
                _, cal_data, motion_cal_data = profile_result
                cal_mode = "CALIBRATED" if cal_data else "RULE-BASED"
                print(f"프로필 전환 완료 — [{cal_mode}]")
            current_letter = "?"
            for tracker in motion_trackers.values():
                tracker["monitoring"] = False; tracker["timed_out"] = False; tracker["collecting"] = False; tracker["window"] = []; tracker["tip_prev"] = None; tracker["slow_count"] = 0
            motion_display_letter = ""
            cap = cv2.VideoCapture(camera_index)

    injector.stop()
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ASL 수화 알파벳 실시간 감지")
    parser.add_argument("--camera", type=int, default=None,
                        help="카메라 인덱스 직접 지정 (기본: 자동 감지)")
    parser.add_argument("--list", action="store_true",
                        help="사용 가능한 카메라 목록 출력 후 종료")
    parser.add_argument("--reset-test-results", action="store_true",
                        help="누적된 테스트 결과를 전부 초기화하고 종료")
    parser.add_argument("--list-test-sessions", action="store_true",
                        help="저장된 테스트 세션 목록(인덱스, 참가자, 시각, 점수) 출력 후 종료")
    parser.add_argument("--delete-test-session", type=int, default=None, metavar="INDEX",
                        help="--list-test-sessions로 확인한 인덱스의 세션 하나만 삭제 후 종료")
    parser.add_argument("--inject", action="store_true",
                        help="시작할 때부터 키스트로크 주입 ON")
    parser.add_argument("--participant", type=str, default=None,
                        help="테스트 모드 세션에 기록할 참가자 이름/번호")
    args = parser.parse_args()

    # CLI 유틸리티 플래그 — 런처 없이 바로 처리
    if args.list:
        list_cameras()
        return
    if args.reset_test_results:
        reset_results()
        return
    if args.list_test_sessions:
        list_sessions()
        return
    if args.delete_test_session is not None:
        delete_session(args.delete_test_session)
        return

    camera_index = args.camera if args.camera is not None else auto_camera_index()
    print(f"카메라 인덱스: {camera_index}  (변경하려면 --camera N)")

    # 캘리브레이션 데이터 로드 (없으면 None — 런처에서 Calibrate 선택 유도)
    cal_missing = not os.path.exists(CALIBRATION_PATH)
    cal_data = None if cal_missing else load_calibration()
    motion_missing = not os.path.exists(MOTION_CALIBRATION_PATH)
    motion_cal_data = None if motion_missing else load_motion_calibration()
    if cal_data:
        print(f"보정 데이터 로드 완료: {sorted(cal_data.keys())}")
    if motion_cal_data:
        print(f"모션 보정 데이터 로드 완료: {sorted(motion_cal_data.keys())}")

    detector = HandDetector()

    # ── 런처 루프 ────────────────────────────────────────────────────────────
    # 모드를 마치고 나면 다시 런처로 돌아온다 (세션 관리 등 반복 사용 편의)
    while True:
        mode = run_launcher(cal_missing=(cal_missing or motion_missing))
        if mode is None:
            break

        if mode == "calibrate":
            cal_data = run_calibration(detector, camera_index)
            motion_cal_data = run_motion_calibration(detector, camera_index)
            cal_missing = (cal_data is None)
            motion_missing = (motion_cal_data is None)
            profile_name = run_text_input("Save calibration as (Enter to skip):")
            if profile_name:
                copy_default_as_profile(profile_name)
            continue   # 런처로 복귀

        if mode == "sessions":
            run_session_manager()
            continue

        # live / test_ordered / test_random — 프로필 선택 필요
        profile_result = run_profile_selector(cal_data, motion_cal_data)
        if profile_result is None:
            continue   # 취소 → 런처로 복귀
        profile_name, active_cal, active_motion_cal = profile_result

        if mode == "live":
            run(detector, active_cal, active_motion_cal, camera_index,
                inject=False, start_guess=False)
        elif mode == "live_inject":
            run(detector, active_cal, active_motion_cal, camera_index,
                inject=True, start_guess=False)
        elif mode == "live_guess":
            run(detector, active_cal, active_motion_cal, camera_index,
                inject=True, start_guess=True)
        elif mode == "live_ed":
            run(detector, active_cal, active_motion_cal, camera_index,
                inject=True, start_ed=True)
        elif mode == "live_llm":
            run(detector, active_cal, active_motion_cal, camera_index,
                inject=True, start_llm=True)

        elif mode in ("test_ordered", "test_random"):
            randomize = (mode == "test_random")
            participant = args.participant or run_text_input("Enter participant name / number:")
            if not run_test_confirm(participant, profile_name):
                continue
            run_test_mode(detector, active_cal, active_motion_cal, camera_index,
                          randomize=randomize, participant=participant,
                          cal_profile=profile_name)

        # 모드 종료 후 런처로 복귀

    detector.close()


if __name__ == "__main__":
    main()

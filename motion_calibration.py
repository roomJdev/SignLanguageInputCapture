"""J, Z 등 모션(동작) 보정 세션 — 일정 길이의 손동작 궤적을 반복 녹화.

정적 보정(calibration.py)과 달리 한 번의 캡처가 '연속된 프레임 시퀀스' 하나다.
SPACE를 누르면 그 순간부터 MOTION_FRAMES 프레임(약 1초) 동안 자동으로 녹화된다.
"""

import json
import os
import numpy as np
import cv2

from features import extract_tip_frame
from motion_classifier import MOTION_TIP

# extract_motion_frame의 손끝 순서: [thumb=4, index=8, middle=12, ring=16, pinky=20]
# 각 손끝은 10차원 벡터에서 2개 열을 차지 (x, y)
_TIPS_ORDER = [4, 8, 12, 16, 20]


def _migrate_10d_to_2d(data: dict) -> dict:
    """10차원(손끝 5개) 구형 데이터를 심볼별 관련 손가락 끝 2차원으로 변환."""
    migrated = {}
    for letter, reps in data.items():
        tip_idx = MOTION_TIP.get(letter)
        if tip_idx is None or tip_idx not in _TIPS_ORDER:
            migrated[letter] = reps
            continue
        col = _TIPS_ORDER.index(tip_idx) * 2
        migrated[letter] = [rep[:, col:col + 2] for rep in reps]
    return migrated

MOTION_LETTERS = ["J", "Z"]
MOTION_CALIBRATION_PATH = "data_new0731/motion_calibration_data.npy"
# MediaPipe 원본 21관절 궤적(프레임별) — DTW 비교용 손끝 2차원 궤적으로 가공되기 전 데이터
MOTION_CALIBRATION_LANDMARKS_PATH = "data_new0731/motion_calibration_landmarks.npy"
MOTION_FRAMES = 30          # ~1초 (30fps 기준) 동안의 프레임 수
MOTION_REPS = 3             # 심볼당 반복 녹화 횟수 (Wobbrock et al. 2007: DTW는 템플릿 3개면 9개 대비 99.5% 정확도)
CAPTURE_FPS_DELAY = 33      # ms

_GREEN_BRIGHT = (80, 255, 120)
_GREEN_MID = (90, 230, 110)
_GREEN_DIM = (100, 200, 110)
_RED_REC = (60, 60, 230)

_REF_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "resources", "letters")
_REF_DISPLAY_HEIGHT = 360


def _load_reference_images() -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    for letter in MOTION_LETTERS:
        path = os.path.join(_REF_IMAGE_DIR, f"{letter}.png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = _REF_DISPLAY_HEIGHT / h
        refs[letter] = cv2.resize(img, (int(w * scale), _REF_DISPLAY_HEIGHT))
    return refs


def _overlay_reference(annotated: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    h, w = annotated.shape[:2]
    rh, rw = ref_img.shape[:2]
    margin, pad = 16, 10
    panel_w, panel_h = rw + pad * 2, rh + pad * 2 + 24
    x0, y0 = max(0, w - panel_w - margin), margin
    if x0 + panel_w > w or y0 + panel_h > h:
        return annotated
    cv2.rectangle(annotated, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), -1)
    cv2.rectangle(annotated, (x0, y0), (x0 + panel_w, y0 + panel_h), (180, 180, 180), 1)
    annotated[y0 + pad:y0 + pad + rh, x0 + pad:x0 + pad + rw] = ref_img
    cv2.putText(annotated, "Reference", (x0 + pad, y0 + panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 140, 50), 2)
    return annotated


def load_motion_calibration() -> dict | None:
    if not os.path.exists(MOTION_CALIBRATION_PATH):
        return None
    data = np.load(MOTION_CALIBRATION_PATH, allow_pickle=True).item()
    # 10차원 구형 데이터면 자동으로 심볼별 2차원으로 변환
    sample = next(iter(data.values()), [None])[0]
    if sample is not None and sample.shape[-1] == 10:
        print("[motion] 구형 10차원 데이터 감지 — 심볼별 손가락 끝 2차원으로 자동 변환")
        data = _migrate_10d_to_2d(data)
    return data


def save_motion_calibration(data: dict) -> None:
    os.makedirs(os.path.dirname(MOTION_CALIBRATION_PATH), exist_ok=True)
    np.save(MOTION_CALIBRATION_PATH, data)
    json_path = os.path.splitext(MOTION_CALIBRATION_PATH)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({k: [rep.tolist() for rep in reps] for k, reps in data.items()}, f, indent=2)
    print(f"모션 보정 데이터 저장 완료: {MOTION_CALIBRATION_PATH} (+ {json_path})")


def save_motion_calibration_landmarks(data: dict) -> None:
    os.makedirs(os.path.dirname(MOTION_CALIBRATION_LANDMARKS_PATH), exist_ok=True)
    np.save(MOTION_CALIBRATION_LANDMARKS_PATH, data)
    json_path = os.path.splitext(MOTION_CALIBRATION_LANDMARKS_PATH)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({k: [rep.tolist() for rep in reps] for k, reps in data.items()}, f, indent=2)
    print(f"모션 원본 랜드마크 저장 완료: {MOTION_CALIBRATION_LANDMARKS_PATH} (+ {json_path})")


def run_motion_calibration(detector, camera_index: int = 0) -> dict:
    """J, Z 동작 보정 세션 실행. 완료된 데이터 dict 반환.

    SPACE 1회 입력 → 그 순간부터 MOTION_FRAMES 프레임 자동 녹화.
    심볼당 MOTION_REPS회 반복 후 다음 심볼로 자동 진행.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 [{camera_index}]를 열 수 없습니다.")

    # 기존 누적 데이터를 베이스로 삼지 않음 — 이번 세션에서 새로 찍은 반복만 저장
    # (예전엔 load_motion_calibration()으로 기존 데이터를 이어받아서, 여러 사람이
    #  캘리브레이션할수록 J/Z 템플릿이 끝없이 누적되는 문제가 있었음)
    motion_data: dict[str, list[np.ndarray]] = {}
    motion_landmarks: dict[str, list[np.ndarray]] = {}
    reference_images = _load_reference_images()
    letter_index = 0
    state = "waiting"   # waiting -> recording -> waiting
    rep_buffer: list[np.ndarray] = []
    raw_rep_buffer: list[np.ndarray] = []   # feature 가공 전 21관절 원본 좌표(프레임별)
    space_held = False

    print(f"\n모션 보정 시작 — {len(MOTION_LETTERS)}개 심볼 "
          f"({', '.join(MOTION_LETTERS)}), 심볼당 {MOTION_REPS}회 반복")
    print("동작을 취한 뒤 SPACE 키를 누르면 1초간 자동 녹화됩니다.\n")
    print("J: 새끼 손가락만 편 상태에서 새끼 손가락 끝으로 J 모양을 그리세요.")
    print("Z: 검지 손가락만 편 상태에서 검지 손가락 끝으로 Z 모양을 그리세요.\n")

    while letter_index < len(MOTION_LETTERS):
        letter = MOTION_LETTERS[letter_index]
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks_list, annotated = detector.process(frame)
        h, w = annotated.shape[:2]

        # --- 상태 처리 ---
        if state == "recording":
            if landmarks_list:
                rep_buffer.append(extract_tip_frame(landmarks_list[0], MOTION_TIP[letter]))
                raw_rep_buffer.append(landmarks_list[0].copy())
            if len(rep_buffer) >= MOTION_FRAMES:
                reps = motion_data.setdefault(letter, [])
                reps.append(np.stack(rep_buffer))
                raw_reps = motion_landmarks.setdefault(letter, [])
                raw_reps.append(np.stack(raw_rep_buffer))
                print(f"  [{letter}] {len(reps)}/{MOTION_REPS} 회 녹화 완료")
                rep_buffer = []
                raw_rep_buffer = []
                state = "waiting"
                if len(reps) >= MOTION_REPS:
                    letter_index += 1

        # --- HUD ---
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

        cv2.putText(annotated, letter, (w // 2 - 60, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, _GREEN_BRIGHT, 10)

        rep_count = len(motion_data.get(letter, []))
        progress_text = f"Symbol {letter_index + 1}/{len(MOTION_LETTERS)}   Rep {rep_count}/{MOTION_REPS}"
        cv2.putText(annotated, progress_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _GREEN_MID, 2)

        if state == "recording":
            bar_w = int(w * len(rep_buffer) / MOTION_FRAMES)
            cv2.rectangle(annotated, (0, h - 12), (bar_w, h), _RED_REC, -1)
            cv2.putText(annotated, "Recording motion...", (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, _RED_REC, 2)
        else:
            instruction = f"Get ready, press SPACE, then draw '{letter}' with fingertip (1s)"
            inst_scale = 0.7
            text_w = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, 2)[0][0]
            if text_w > w - 40:
                inst_scale *= (w - 40) / text_w
            cv2.putText(annotated, instruction, (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, inst_scale, _GREEN_DIM, 2)

        cv2.putText(annotated, "SPACE: record  /  S: skip symbol  /  Q: save & quit",
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN_DIM, 1)

        ref_img = reference_images.get(letter)
        if ref_img is not None:
            annotated = _overlay_reference(annotated, ref_img)

        cv2.imshow("Motion Calibration", annotated)

        key = cv2.waitKey(CAPTURE_FPS_DELAY) & 0xFF
        space_down = (key == ord(" "))

        if space_down and not space_held and state == "waiting":
            if landmarks_list:
                state = "recording"
                rep_buffer = []
                raw_rep_buffer = []
        elif key == ord("s"):
            print(f"  [{letter}] 건너뜀 (현재까지 {rep_count}회 저장됨)")
            letter_index += 1
            state = "waiting"
            rep_buffer = []
            raw_rep_buffer = []
        elif key == ord("q"):
            print("모션 보정 중단 — 현재까지 수집된 데이터로 저장합니다.")
            break

        # 스페이스바를 떼기 전까지는 새 녹화 트리거 무시
        space_held = space_down

    cap.release()
    cv2.destroyWindow("Motion Calibration")

    if motion_data:
        save_motion_calibration(motion_data)
        save_motion_calibration_landmarks(motion_landmarks)
        print(f"\n모션 보정된 심볼: {sorted(motion_data.keys())}")
    else:
        print("모션 보정 데이터 없음.")

    return motion_data

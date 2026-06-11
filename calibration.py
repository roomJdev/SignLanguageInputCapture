"""보정 세션 — 사용자 손 제스처를 직접 수집해 분류기를 개인화."""

import os
import numpy as np
import cv2

from features import extract

# J, Z는 모션 필요로 제외
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
CALIBRATION_PATH = "calibration_data.npy"
SAMPLES_PER_LETTER = 30
CAPTURE_FPS_DELAY = 33   # ms


def load_calibration() -> dict | None:
    if os.path.exists(CALIBRATION_PATH):
        data = np.load(CALIBRATION_PATH, allow_pickle=True).item()
        return data
    return None


def save_calibration(data: dict) -> None:
    np.save(CALIBRATION_PATH, data)
    print(f"보정 데이터 저장 완료: {CALIBRATION_PATH}")


def run_calibration(detector, camera_index: int = 0) -> dict:
    """보정 세션 실행. 완료된 데이터 dict 반환.

    각 알파벳마다 30개 샘플을 저장 (평균 아님).
    k-NN에서 모든 샘플과 비교해 최소 거리로 분류.
    스킵한 알파벳은 기존 보정 데이터를 유지.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 [{camera_index}]를 열 수 없습니다.")

    # 기존 데이터를 베이스로 시작 — 새로 캡처한 것만 덮어씀
    calibration_data: dict[str, np.ndarray] = load_calibration() or {}
    letter_index = 0
    state = "waiting"
    buffer: list[np.ndarray] = []
    status_msg = "SPACE: 캡처 시작  /  S: 건너뛰기  /  Q: 저장 후 종료"

    print(f"\n보정 시작 — {len(LETTERS)}개 알파벳 ({', '.join(LETTERS)})")
    print("각 알파벳 제스처를 취한 뒤 SPACE 키를 누르세요.\n")

    while letter_index < len(LETTERS):
        letter = LETTERS[letter_index]
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks_list, annotated = detector.process(frame)
        h, w = annotated.shape[:2]

        # --- 상태 처리 ---
        if state == "capturing":
            if landmarks_list:
                buffer.append(extract(landmarks_list[0]))
            if len(buffer) >= SAMPLES_PER_LETTER:
                calibration_data[letter] = np.stack(buffer)   # (30, 25)
                print(f"  [{letter}] 보정 완료 ({len(buffer)} 샘플)")
                buffer = []
                letter_index += 1
                state = "waiting"
                status_msg = "SPACE: 캡처 시작  /  S: 건너뛰기  /  Q: 저장 후 종료"

        # --- HUD ---
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

        cv2.putText(annotated, letter, (w // 2 - 60, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 100), 10)

        progress_text = f"{letter_index + 1} / {len(LETTERS)}"
        cv2.putText(annotated, progress_text, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        if state == "capturing":
            bar_w = int(w * len(buffer) / SAMPLES_PER_LETTER)
            cv2.rectangle(annotated, (0, h - 12), (bar_w, h), (0, 220, 80), -1)
            cv2.putText(annotated, "캡처 중...", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)
        else:
            hand_ok = bool(landmarks_list)
            color = (0, 200, 80) if hand_ok else (0, 80, 200)
            label = "손 감지됨" if hand_ok else "손을 화면에 위치시키세요"
            cv2.putText(annotated, label, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(annotated, status_msg, (20, h - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow("보정 세션", annotated)

        key = cv2.waitKey(CAPTURE_FPS_DELAY) & 0xFF
        if key == ord(" ") and state == "waiting":
            if landmarks_list:
                state = "capturing"
                buffer = []
            else:
                status_msg = "⚠ 손이 감지되지 않습니다. 카메라 앞에 손을 위치시키세요."
        elif key == ord("s"):
            print(f"  [{letter}] 건너뜀")
            letter_index += 1
            state = "waiting"
            buffer = []
            status_msg = "SPACE: 캡처 시작  /  S: 건너뛰기  /  Q: 저장 후 종료"
        elif key == ord("q"):
            print("보정 중단 — 현재까지 수집된 데이터로 저장합니다.")
            break

    cap.release()
    cv2.destroyWindow("보정 세션")

    if calibration_data:
        save_calibration(calibration_data)
        print(f"\n보정된 알파벳: {sorted(calibration_data.keys())}")
    else:
        print("보정 데이터 없음.")

    return calibration_data

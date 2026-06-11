"""보정 세션 — 사용자 손 제스처를 직접 수집해 분류기를 개인화."""

import os
import numpy as np
import cv2

# J, Z는 모션 필요로 제외
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
CALIBRATION_PATH = "calibration_data.npy"
SAMPLES_PER_LETTER = 30   # 프레임 수
CAPTURE_FPS_DELAY = 33    # ms


def normalize(lm: np.ndarray) -> np.ndarray:
    """손목 기준으로 이동·스케일 정규화 후 1D 벡터 반환."""
    lm = lm[:, :2].copy()          # x, y만 사용 (z 제외)
    lm -= lm[0]                    # 손목을 원점으로
    scale = np.linalg.norm(lm[9])  # 중지 MCP까지 거리로 스케일
    if scale > 1e-6:
        lm /= scale
    return lm.flatten()            # (42,)


def load_calibration() -> dict | None:
    if os.path.exists(CALIBRATION_PATH):
        data = np.load(CALIBRATION_PATH, allow_pickle=True).item()
        return data
    return None


def save_calibration(data: dict) -> None:
    np.save(CALIBRATION_PATH, data)
    print(f"보정 데이터 저장 완료: {CALIBRATION_PATH}")


def run_calibration(detector) -> dict:
    """보정 세션 실행. 완료된 데이터 dict 반환."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    calibration_data = {}
    letter_index = 0
    state = "waiting"       # waiting | capturing | done
    buffer = []
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
                buffer.append(normalize(landmarks_list[0]))
            if len(buffer) >= SAMPLES_PER_LETTER:
                calibration_data[letter] = np.mean(buffer, axis=0)
                print(f"  [{letter}] 보정 완료 ({len(buffer)} 프레임)")
                buffer = []
                letter_index += 1
                state = "waiting"
                status_msg = "SPACE: 캡처 시작  /  S: 건너뛰기  /  Q: 저장 후 종료"

        # --- HUD ---
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

        # 현재 알파벳
        cv2.putText(annotated, letter, (w // 2 - 60, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 100), 10)

        # 진행 상황
        progress_text = f"{letter_index + 1} / {len(LETTERS)}"
        cv2.putText(annotated, progress_text, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        # 캡처 진행 바
        if state == "capturing":
            bar_w = int(w * len(buffer) / SAMPLES_PER_LETTER)
            cv2.rectangle(annotated, (0, h - 12), (bar_w, h), (0, 220, 80), -1)
            cv2.putText(annotated, "캡처 중...", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)
        else:
            # 손 감지 여부 표시
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

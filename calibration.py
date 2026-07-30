"""보정 세션 — 사용자 손 제스처를 직접 수집해 분류기를 개인화."""

import os
import numpy as np
import cv2

from features import extract

# J, Z는 모션 필요로 제외
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY") + list("0123456789")
CALIBRATION_PATH = "data/calibration_data.npy"
SAMPLES_PER_LETTER = 30
CAPTURE_FPS_DELAY = 33   # ms

_REF_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "resources", "letters")
_REF_DISPLAY_HEIGHT = 360   # 화면에 표시할 레퍼런스 이미지 높이(px)


def load_calibration() -> dict | None:
    if os.path.exists(CALIBRATION_PATH):
        data = np.load(CALIBRATION_PATH, allow_pickle=True).item()
        data.pop("SPACE", None)
        return data
    return None


def _load_reference_images() -> dict[str, np.ndarray]:
    """resources/letters/<SYMBOL>.png 를 모두 로드해 표시용 크기로 리사이즈.

    초보자가 보정 중 각 철자/숫자의 손모양을 참고할 수 있게 화면에 띄움.
    """
    refs: dict[str, np.ndarray] = {}
    for letter in LETTERS:
        path = os.path.join(_REF_IMAGE_DIR, f"{letter}.png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = _REF_DISPLAY_HEIGHT / h
        resized = cv2.resize(img, (int(w * scale), _REF_DISPLAY_HEIGHT))
        refs[letter] = resized
    return refs


def _overlay_reference(annotated: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    """레퍼런스 이미지를 화면 우측 상단에 흰 배경 패널과 함께 오버레이."""
    h, w = annotated.shape[:2]
    rh, rw = ref_img.shape[:2]
    margin = 16
    pad = 10
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


def run_space_calibration(detector, camera_index: int = 0) -> dict | None:
    """SPACE 제스처만 단독으로 보정. 기존 데이터에 SPACE 항목만 덮어씀."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 [{camera_index}]를 열 수 없습니다.")

    calibration_data: dict = load_calibration() or {}
    state = "waiting"
    buffer: list[np.ndarray] = []
    space_held = False
    status_msg = "Open your hand flat, then press SPACE to capture"

    print("\nSPACE 제스처 보정 시작 — open hand(손 펼침)을 취한 뒤 SPACE를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks_list, annotated = detector.process(frame)
        h, w = annotated.shape[:2]

        if state == "capturing":
            if landmarks_list:
                buffer.append(extract(landmarks_list[0]))
            if len(buffer) >= SAMPLES_PER_LETTER:
                calibration_data["SPACE"] = np.stack(buffer)
                print(f"  [SPACE] 보정 완료 ({len(buffer)} 샘플)")
                state = "done"

        GREEN_BRIGHT = (80, 255, 120)
        GREEN_MID = (90, 230, 110)
        GREEN_DIM = (100, 200, 110)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

        title = "SPACE" if state != "done" else "Done!"
        cv2.putText(annotated, title, (w // 2 - 100, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, GREEN_BRIGHT, 8)

        if state == "capturing":
            bar_w = int(w * len(buffer) / SAMPLES_PER_LETTER)
            cv2.rectangle(annotated, (0, h - 12), (bar_w, h), GREEN_BRIGHT, -1)
            cv2.putText(annotated, "Capturing...", (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN_MID, 2)
        elif state == "done":
            cv2.putText(annotated, "SPACE gesture saved. Press Q to finish.",
                        (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN_BRIGHT, 2)
        else:
            hand_ok = bool(landmarks_list)
            color = GREEN_MID if hand_ok else GREEN_DIM
            label = "Hand detected" if hand_ok else "Place your hand in view"
            cv2.putText(annotated, label, (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(annotated, status_msg, (20, h - 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN_DIM, 2)

        cv2.imshow("Space Calibration", annotated)

        key = cv2.waitKey(CAPTURE_FPS_DELAY) & 0xFF
        space_down = (key == ord(" "))

        if state == "waiting" and space_down and not space_held:
            if landmarks_list:
                state = "capturing"
                buffer = []
            else:
                status_msg = "Warning: no hand detected."
        elif key == ord("q"):
            break

        space_held = space_down

    cap.release()
    cv2.destroyWindow("Space Calibration")

    if "SPACE" in calibration_data:
        save_calibration(calibration_data)
    return calibration_data if "SPACE" in calibration_data else None


def save_calibration(data: dict) -> None:
    os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
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

    reference_images = _load_reference_images()

    # 기존 데이터를 베이스로 시작 — 새로 캡처한 것만 덮어씀
    calibration_data: dict[str, np.ndarray] = load_calibration() or {}
    letter_index = 0
    state = "waiting"
    buffer: list[np.ndarray] = []
    status_msg = "SPACE: start capture  /  S: skip  /  Q: save & quit"
    space_held = False   # 스페이스바 연타/홀드 시 캡처 완료 직후 재트리거 방지용 엣지 감지

    print(f"\n보정 시작 — {len(LETTERS)}개 심볼 ({', '.join(LETTERS)})")
    print("각 심볼 제스처를 취한 뒤 SPACE 키를 누르세요.\n")

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
                status_msg = "SPACE: start capture  /  S: skip  /  Q: save & quit"

        # --- HUD ---
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

        # BGR 연두색 팔레트 — 밝기로 정보 위계 구분
        GREEN_BRIGHT = (80, 255, 120)   # 큰 글자(타깃 심볼)
        GREEN_MID = (90, 230, 110)      # 진행률 / 캡처 상태
        GREEN_DIM = (100, 200, 110)     # 안내 / 경고 텍스트

        cv2.putText(annotated, letter, (w // 2 - 60, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, GREEN_BRIGHT, 10)

        if state == "waiting":
            instruction = "Make the hand shape, then press SPACE once"
            inst_scale = 0.8
            text_w = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, 2)[0][0]
            if text_w > w - 40:
                inst_scale *= (w - 40) / text_w
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, 2)[0]
            cv2.putText(annotated, instruction, ((w - text_size[0]) // 2, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, inst_scale, GREEN_BRIGHT, 2)

        progress_text = f"{letter_index + 1} / {len(LETTERS)}"
        cv2.putText(annotated, progress_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN_MID, 2)

        if state == "capturing":
            bar_w = int(w * len(buffer) / SAMPLES_PER_LETTER)
            cv2.rectangle(annotated, (0, h - 12), (bar_w, h), GREEN_BRIGHT, -1)
            cv2.putText(annotated, "Capturing...", (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN_MID, 2)
        else:
            hand_ok = bool(landmarks_list)
            color = GREEN_MID if hand_ok else GREEN_DIM
            label = "Hand detected" if hand_ok else "Place your hand in view"
            cv2.putText(annotated, label, (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.putText(annotated, status_msg, (20, h - 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN_DIM, 2)

        ref_img = reference_images.get(letter)
        if ref_img is not None:
            annotated = _overlay_reference(annotated, ref_img)

        cv2.imshow("Calibration Session", annotated)

        key = cv2.waitKey(CAPTURE_FPS_DELAY) & 0xFF
        space_down = (key == ord(" "))

        if space_down and not space_held and state == "waiting":
            if landmarks_list:
                state = "capturing"
                buffer = []
            else:
                status_msg = "Warning: no hand detected. Place your hand in front of the camera."
        elif key == ord("s"):
            print(f"  [{letter}] 건너뜀")
            letter_index += 1
            state = "waiting"
            buffer = []
            status_msg = "SPACE: start capture  /  S: skip  /  Q: save & quit"
        elif key == ord("q"):
            print("보정 중단 — 현재까지 수집된 데이터로 저장합니다.")
            break

        # 스페이스바를 떼기 전까지는 새 캡처 트리거 무시 — 연타/홀드로 인한 즉시 재시작 방지
        space_held = space_down

    cap.release()
    cv2.destroyWindow("Calibration Session")

    if calibration_data:
        save_calibration(calibration_data)
        print(f"\n보정된 심볼: {sorted(calibration_data.keys())}")
    else:
        print("보정 데이터 없음.")

    return calibration_data

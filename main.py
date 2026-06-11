"""수화 알파벳 실시간 감지 — 메인 실행 파일.

실행:
    python main.py                # 카메라 자동 선택
    python main.py --camera 1     # 카메라 인덱스 직접 지정
    python main.py --list         # 사용 가능한 카메라 목록 출력
    python main.py --calibrate    # 강제 재보정

단축키 (인식 화면):
    r   재보정 세션 시작
    q   종료
"""

import argparse
import os
import sys
import platform

import cv2

from calibration import CALIBRATION_PATH, load_calibration, run_calibration
from hand_detector import HandDetector
from sign_classifier import classify, classify_calibrated


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

def run(detector: HandDetector, cal_data: dict | None, camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"카메라 [{camera_index}]를 열 수 없습니다. --list 로 가용 카메라를 확인하세요.")
        sys.exit(1)

    current_letter = "?"
    mode = "보정됨" if cal_data else "규칙 기반"
    print(f"수화 감지 시작 [{mode}] — r: 재보정  q: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks_list, annotated = detector.process(frame)

        if landmarks_list:
            lm = landmarks_list[0]
            current_letter = classify_calibrated(lm, cal_data) if cal_data else classify(lm)

        h, w = annotated.shape[:2]

        cv2.rectangle(annotated, (10, 10), (110, 90), (0, 0, 0), -1)
        cv2.putText(annotated, current_letter, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 100), 4)
        cv2.putText(annotated, mode, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        cv2.putText(annotated, "r: 재보정  q: 종료", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Sign Language Input", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            cap.release()
            cv2.destroyAllWindows()
            cal_data = run_calibration(detector, camera_index)
            mode = "보정됨" if cal_data else "규칙 기반"
            current_letter = "?"
            cap = cv2.VideoCapture(camera_index)
            print(f"재보정 완료 — 인식 재개 [{mode}]")

    detector.close()
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
    parser.add_argument("--calibrate", action="store_true",
                        help="보정 세션을 새로 시작")
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    camera_index = args.camera if args.camera is not None else auto_camera_index()
    print(f"카메라 인덱스: {camera_index}  (변경하려면 --camera N)")

    detector = HandDetector()

    if args.calibrate or not os.path.exists(CALIBRATION_PATH):
        if not args.calibrate:
            print("보정 데이터가 없습니다. 보정 세션을 시작합니다.")
        cal_data = run_calibration(detector, camera_index)
    else:
        cal_data = load_calibration()
        print(f"보정 데이터 로드 완료: {sorted(cal_data.keys())}")

    run(detector, cal_data, camera_index)


if __name__ == "__main__":
    main()

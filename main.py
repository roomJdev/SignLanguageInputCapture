"""수화 알파벳 실시간 감지 — 메인 실행 파일.

실행:
    python main.py              # 보정 데이터 있으면 바로 시작, 없으면 보정 먼저
    python main.py --calibrate  # 강제로 보정 세션부터 시작

단축키 (인식 화면):
    r   재보정 세션 시작
    q   종료
"""

import argparse
import os
import sys

import cv2

from calibration import CALIBRATION_PATH, load_calibration, run_calibration
from hand_detector import HandDetector
from sign_classifier import classify, classify_calibrated


def _macbook_camera_index() -> int:
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
        for i, device in enumerate(devices):
            name = device.localizedName() or ""
            if "macbook" in name.lower() or "facetime" in name.lower():
                print(f"맥북 카메라 감지: [{i}] {name}")
                return i
    except Exception:
        pass
    return 0


def run(detector: HandDetector, cal_data: dict | None) -> None:
    index = _macbook_camera_index()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print("맥북 내장 카메라를 열 수 없습니다.")
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
            cal_data = run_calibration(detector)
            mode = "보정됨" if cal_data else "규칙 기반"
            current_letter = "?"
            cap = cv2.VideoCapture(index)
            print(f"재보정 완료 — 인식 재개 [{mode}]")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="ASL 수화 알파벳 실시간 감지")
    parser.add_argument("--calibrate", action="store_true",
                        help="보정 세션을 새로 시작")
    args = parser.parse_args()

    detector = HandDetector()

    if args.calibrate or not os.path.exists(CALIBRATION_PATH):
        if not args.calibrate:
            print("보정 데이터가 없습니다. 보정 세션을 시작합니다.")
        cal_data = run_calibration(detector)
    else:
        cal_data = load_calibration()
        print(f"보정 데이터 로드 완료: {sorted(cal_data.keys())}")

    run(detector, cal_data)


if __name__ == "__main__":
    main()

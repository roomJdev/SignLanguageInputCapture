import mediapipe as mp
import numpy as np
import cv2

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark


# mediapipe 0.10+ Tasks API 기반 손 랜드마크 인덱스 상수
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# hand_landmarker.task 파일 다운로드 경로
_MODEL_PATH = "resources/hand_landmarker.task"

# 연결선 쌍 (시각화용)
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def _ensure_model():
    """모델 파일이 없으면 curl로 자동 다운로드 (macOS SSL 인증서 문제 우회)."""
    import os, subprocess
    if not os.path.exists(_MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        print(f"모델 다운로드 중: {_MODEL_PATH} ...")
        result = subprocess.run(["curl", "-L", "-o", _MODEL_PATH, url], check=True)
        print("다운로드 완료.")


class HandDetector:
    """mediapipe 0.10+ Tasks API 기반 손 랜드마크 감지기."""

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7):
        _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_ts = 0

    def process(self, bgr_frame):
        """BGR 프레임 → (landmarks_list, annotated_frame).

        landmarks_list: list[np.ndarray shape=(21,3)]  — 0~1 정규화 (x, y, z)
        annotated_frame: 랜드마크가 그려진 BGR 프레임
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33  # ~30fps 기준 ms 단위 타임스탬프
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        annotated = bgr_frame.copy()
        landmarks_list = []
        h, w = bgr_frame.shape[:2]

        for hand_lm in result.hand_landmarks:
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)
            landmarks_list.append(coords)

            # 랜드마크 점 그리기
            for lm in hand_lm:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (px, py), 5, (0, 255, 0), -1)

            # 연결선 그리기
            for a, b in _CONNECTIONS:
                ax = int(hand_lm[a].x * w); ay = int(hand_lm[a].y * h)
                bx = int(hand_lm[b].x * w); by = int(hand_lm[b].y * h)
                cv2.line(annotated, (ax, ay), (bx, by), (200, 200, 200), 2)

        return landmarks_list, annotated

    def close(self):
        self._landmarker.close()

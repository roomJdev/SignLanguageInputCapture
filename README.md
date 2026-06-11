# Sign Language Input Capture

실시간 카메라로 ASL(American Sign Language) 수화 알파벳을 인식하는 Python 프로젝트.  
MediaPipe Hand Landmarker 기반으로 손 관절 21개 좌표를 추출하고, 개인 보정 세션을 통해 사용자 손에 맞게 분류기를 최적화합니다.

추후 게임 엔진(Unreal Engine 등)과의 입력 연동을 목표로 합니다.

## 기능

- 실시간 손 랜드마크 감지 (MediaPipe Hand Landmarker)
- ASL 알파벳 A–Y 인식 (J, Z는 모션 필요로 미지원)
- **개인 보정 세션**: 사용자 손 형태에 맞게 k-NN 분류기 최적화
- 보정 데이터 없을 시 규칙 기반 분류 폴백

## 요구 사항

- Python 3.11+
- 웹캠 (macOS 내장 카메라, Continuity Camera, 외장 USB 웹캠 모두 지원)
- Windows / macOS 지원

## 설치

**macOS**
```bash
git clone https://github.com/<your-username>/SignLanguageInputCapture.git
cd SignLanguageInputCapture

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows**
```bash
git clone https://github.com/<your-username>/SignLanguageInputCapture.git
cd SignLanguageInputCapture

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> `pyobjc-framework-AVFoundation`은 macOS에서만 자동 설치됩니다. Windows에서는 설치되지 않습니다.

## 실행

```bash
# 첫 실행 — 카메라 자동 감지, 보정 세션 후 인식 시작
python main.py

# 카메라가 여러 개일 때 직접 지정
python main.py --camera 1

# 연결된 카메라 목록 확인
python main.py --list

# 보정 재수행
python main.py --calibrate
```

### 보정 세션 키

| 키 | 동작 |
|---|---|
| `SPACE` | 현재 알파벳 제스처 수집 시작 |
| `S` | 현재 알파벳 건너뛰기 |
| `Q` | 현재까지 저장 후 종료 |

### 인식 화면 키

| 키 | 동작 |
|---|---|
| `Q` | 종료 |

## 프로젝트 구조

```
SignLanguageInputCapture/
├── main.py              # 진입점 — 보정 후 본 인식 실행
├── hand_detector.py     # MediaPipe Hand Landmarker 래퍼
├── sign_classifier.py   # 알파벳 분류 (규칙 기반 / k-NN)
├── calibration.py       # 보정 세션 UI 및 데이터 수집
├── requirements.txt
└── resources/
    └── reference.html   # 알파벳별 제스처 레퍼런스 (브라우저에서 열기)
```

## 인식 가능 알파벳

| 지원 | 알파벳 |
|---|---|
| ✅ | A B C D E F G H I K L M N O P Q R S T U V W X Y |
| ❌ (모션 필요) | J Z |

## 사용 라이브러리

| 라이브러리 | 용도 | 라이선스 |
|---|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | 손 랜드마크 감지 | Apache 2.0 |
| [OpenCV](https://github.com/opencv/opencv) | 카메라 입력 및 화면 출력 | Apache 2.0 |
| [NumPy](https://numpy.org) | 수치 연산 | BSD |
| [pyobjc-framework-AVFoundation](https://pyobjc.readthedocs.io) | macOS 카메라 감지 | MIT |

> MediaPipe 모델 파일(`hand_landmarker.task`)은 첫 실행 시 자동으로 다운로드됩니다.

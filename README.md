# Sign Language Input Capture

실시간 카메라로 ASL(American Sign Language) 수화 알파벳과 숫자를 인식하는 Python 프로젝트.  
MediaPipe Hand Landmarker 기반으로 손 관절 21개 좌표를 추출하고, 개인 보정 세션을 통해 사용자 손에 맞게 분류기를 최적화합니다.

## 기능

- 실시간 손 랜드마크 감지 (MediaPipe Hand Landmarker)
- **ASL 알파벳 A–Z 전체 + 숫자 0–9 인식** — 정적 24글자(A-Y 중 J 제외)는 k-NN, J·Z는 모션(DTW) 기반
- **시점 불변 25차원 feature vector**(굽힘 각도 10 + 손끝 간 거리 10 + 손목 거리 5)로 정적 심볼 분류 — 카메라 각도뿐 아니라 **좌/우 손 교차 인식도 별도 처리 없이 지원**
- **J, Z 모션 인식**: 정적 분류기가 'I'(J 트리거) 또는 'D'(Z 트리거)를 감지하면 0.8초 모니터링 버퍼 시작 → 움직임 감지 시 단일 손가락 끝 궤적 수집 → DTW 비교
- **개인 보정 세션**: 사용자 손 형태에 맞게 k-NN/모션 템플릿 최적화, 철자별 손모양 레퍼런스 이미지 오버레이 제공
- **철자 모드 / 숫자 모드 전환**(`E` 키): 0/O, 2/V, 9/F처럼 손모양이 겹치는 심볼을 모드별로 분리해 오분류 감소
- **OS-level 키스트로크 주입** (`I` 키 토글): 인식된 글자를 외부 앱에 직접 입력
- **Guess 모드** (`G` 키 또는 런처에서 선택): 수화로 prefix를 입력하면 wordfreq 기반 영어 단어 후보 3개를 화면 중앙에 표시, 마우스 클릭으로 단어 선택
- **단어 추천 fuzzy fallback**: prefix 정확 매칭 실패 시 뒤에서 한 글자씩 제거해 최장 매칭 prefix 검색, 그래도 없으면 같은 첫 글자 단어에서 유사도 순 검색
- **런처 화면**: 앱 시작 시 7가지 모드 선택 — 방향키/w·s 키 또는 마우스 클릭으로 선택
- **다중 모델 비교 테스트**: 보정 데이터 한 번으로 k-NN·SVM·RF·LR·MLP 5개 모델을 동시에 학습해 모델별 예측 결과를 나란히 표시
- **테스트 모드**: A-Z, 0-9를 순서대로(또는 랜덤으로) 제시 → 보정 프로필 선택 → 인식 결과 기록. 세션별/모델별 인식률 통계 제공
- **보정 프로필 관리**: 보정 완료 후 이름을 붙여 저장. 테스트 또는 라이브 모드 시작 전 누구의 보정 데이터를 쓸지 선택 가능

## 요구 사항

- Python 3.11+
- 웹캠 (macOS 내장 카메라, Continuity Camera, 외장 USB 웹캠 모두 지원)
- macOS (키스트로크 주입은 macOS 전용, pyautogui 사용)

## 설치

```bash
git clone https://github.com/<your-username>/SignLanguageInputCapture.git
cd SignLanguageInputCapture

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **macOS 접근성 권한**: 키스트로크 주입 기능을 사용하려면 시스템 설정 → 개인 정보 보호 및 보안 → 손쉬운 사용에서 실행 중인 앱(Terminal 등)을 허용해야 합니다.

## 실행

```bash
# 런처 화면에서 모드 선택 후 시작
python main.py

# 카메라가 여러 개일 때 직접 지정
python main.py --camera 1

# 연결된 카메라 목록 확인
python main.py --list

# 테스트 세션 목록 / 특정 세션 삭제 / 전체 초기화
python main.py --list-test-sessions
python main.py --delete-test-session 2
python main.py --reset-test-results
```

## 런처 모드

| 모드 | 설명 |
|---|---|
| **Calibrate** | 정적 심볼 + J/Z 모션 보정 세션 실행 |
| **Live Mode** | 수화 인식만 (외부 주입 없음) |
| **Live + Injection** | 인식된 글자를 외부 앱에 즉시 주입 |
| **Live + Guess Mode** | Injection ON + Guess 모드 ON으로 시작 |
| **Test (Ordered)** | 알파벳/숫자 순서대로 인식률 테스트 |
| **Test (Random)** | 랜덤 순서 인식률 테스트 |
| **Test Results** | 저장된 테스트 세션 조회/삭제 |

## 인식 화면 키

| 키 | 동작 | 주입 OFF | 주입 ON |
|---|---|---|---|
| `E` | LETTER ↔ NUMBER 모드 전환 | ✓ | — |
| `R` | 재보정 세션 시작 | ✓ | — |
| `T` | 테스트 모드 시작 (순서대로) | ✓ | — |
| `Y` | 테스트 모드 시작 (랜덤 순서) | ✓ | — |
| `U` | 테스트 세션 관리 | ✓ | — |
| `P` | 보정 프로필 전환 | ✓ | — |
| `I` | 키스트로크 주입 ON/OFF 토글 | ✓ | ✓ |
| `G` | Guess 모드 ON/OFF 토글 | ✓ | ✓ |
| `SPACE` | (Guess 모드) 현재 버퍼 확정 / 스페이스 입력 | — | ✓ |
| `Backspace` | (Guess 모드) 버퍼 마지막 글자 삭제 | — | ✓ |
| `Q` | 종료 | ✓ | ✓ |

> 주입 ON 상태에서는 주입된 키가 cv2 창으로 돌아와 단축키를 오트리거하는 것을 방지하기 위해 E·R·T·Y·U·P 단축키가 비활성화됩니다.

## Guess 모드 사용법

1. 런처에서 **Live + Guess Mode** 선택, 또는 라이브 화면에서 `G` 키 토글
2. 수화로 단어의 앞 글자(들)를 입력 → 화면 중앙에 후보 3개 버튼 표시
3. 원하는 단어 버튼을 **마우스 클릭** 또는 키보드 `1`/`2`/`3`으로 선택
4. **SPACE** 버튼(또는 키): 현재 버퍼를 그대로 확정하고 다음 단어로
5. **BKSP** 버튼(또는 Backspace): 마지막 서명 글자 취소

**fuzzy fallback 표시**: 버퍼가 하늘색(`[ VOL +VTEEUVR ]` 형태)으로 표시되면 뒷부분에 오타가 있어 짧은 prefix로 fallback 매칭 중임을 뜻합니다. Backspace로 오타 부분을 지우거나 현재 후보에서 선택하세요.

### 보정 세션 키

| 키 | 동작 |
|---|---|
| `SPACE` | 현재 심볼 제스처 수집 시작 |
| `S` | 현재 심볼 건너뛰기 |
| `Q` | 현재까지 저장 후 종료 |

## 프로젝트 구조

```
SignLanguageInputCapture/
├── main.py                    # 진입점 — 런처, 인식 루프, 모드 전환, 키/마우스 처리
├── launcher.py                # 앱 시작 화면 — 7가지 모드 선택 UI
├── word_suggester.py          # wordfreq 기반 단어 추천 (prefix + fuzzy fallback)
├── hand_detector.py           # MediaPipe Hand Landmarker 래퍼
├── features.py                # 21×3 랜드마크 → 25차원 feature vector + 모션 프레임 추출
├── sign_classifier.py         # 정적 심볼 분류 (규칙 기반 / k-NN, 거리 임계값 1.5)
├── motion_classifier.py       # J, Z 모션 분류 — 트리거 글자 모니터링 + DTW + 좌우 미러
├── ml_models.py               # 다중 ML 모델 관리 (kNN·SVM·RF·LR·MLP)
├── calibration.py             # 정적 심볼 보정 UI, 데이터 수집, 레퍼런스 이미지 오버레이
├── motion_calibration.py      # J, Z 모션 보정 UI
├── calibration_profiles.py    # 보정 프로필 저장/로드/목록 관리
├── keystroke_injector.py      # OS-level 키스트로크 주입 (pyautogui), 안정화 로직
├── test_mode.py               # 테스트 모드, 다중 모델 비교, 세션 저장/통계/관리 UI
├── requirements.txt
├── data/                      # 보정 데이터 및 세션 기록 (gitignore)
│   ├── calibration_data.npy
│   ├── motion_calibration_data.npy
│   ├── cal_<name>.npy / motion_cal_<name>.npy
│   ├── calibration_profiles.json
│   └── test_results.json
└── resources/
    ├── hand_landmarker.task        # MediaPipe 모델
    ├── asl_alphabet_ref_wikipedia.png
    └── letters/                   # 심볼별 크롭 레퍼런스 이미지 (A.png … Z.png, 0.png … 9.png)
```

## 인식 가능 심볼

| 모드 | 분류 방식 | 심볼 |
|---|---|---|
| LETTER (정적) | k-NN (거리 임계값 `1.5`) / 규칙 기반 폴백 | A B C D E F G H I K L M N O P Q R S T U V W X Y |
| LETTER (모션) | DTW 궤적 비교 (임계값 `1.1`, 보정 필수) | J Z |
| NUMBER | k-NN (거리 임계값 `1.5`, 보정 필수) | 0 1 2 3 4 5 6 7 8 9 |

## 사용 라이브러리

| 라이브러리 | 용도 |
|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | 손 랜드마크 감지 |
| [OpenCV](https://github.com/opencv/opencv) | 카메라 입력 및 화면 출력 |
| [NumPy](https://numpy.org) | 수치 연산 |
| [scikit-learn](https://scikit-learn.org) | ML 모델 (SVM, RF, LR, MLP) |
| [pyautogui](https://pyautogui.readthedocs.io) | OS-level 키스트로크 주입 (macOS) |
| [wordfreq](https://github.com/rspeer/wordfreq) | 영어 단어 빈도 기반 추천 |
| [pynput](https://pynput.readthedocs.io) | 키보드 이벤트 처리 |
| [pyobjc-framework-AVFoundation](https://pyobjc.readthedocs.io) | macOS 카메라 감지 |

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
- **Guess 모드** (`G` 키 또는 런처에서 선택): 수화로 prefix를 입력하면 wordfreq 기반 영어 단어 후보 3개를 화면 중앙에 표시, 마우스 클릭 또는 키보드 `1`/`2`/`3`으로 단어 선택 후 외부 앱에 주입
- **Word Injection 모드** (`W` 키): Guess 모드와 동일한 단어 추천 UI. 후보 선택(`1`/`2`/`3`)하면 Cmd+Tab으로 이전 앱에 단어를 주입하고 자동으로 OpenCV 창으로 복귀 — 포커스 전환 없이 연속 입력 가능
- **단어 추천 — Classic fallback**: prefix 정확 매칭 실패 시 뒤에서 한 글자씩 제거해 최장 매칭 prefix 검색, 그래도 없으면 같은 첫 글자 단어에서 유사도 순 검색
- **단어 추천 — Edit Distance 모드** (`Live + ED Mode`): rapidfuzz `fuzz.ratio` 기반 전체 문자열 edit distance 스캔 후 wordfreq 빈도순 재정렬 — 오타 위치에 무관하게 robust한 추천
- **런처 화면**: 앱 시작 시 8가지 모드 선택 — 방향키/w·s 키 또는 마우스 클릭으로 선택
- **다중 모델 비교 테스트**: 보정 데이터 한 번으로 k-NN·SVM·RF·LR·MLP 5개 모델을 동시에 학습해 모델별 예측 결과를 나란히 표시
- **테스트 모드**: A-Z, 0-9를 순서대로(또는 랜덤으로) 제시 → 보정 프로필 선택 → 인식 결과 기록. 세션별/모델별 인식률 통계 제공
- **보정 프로필 관리**: 보정 완료 후 이름을 붙여 저장. 테스트 또는 라이브 모드 시작 전 누구의 보정 데이터를 쓸지 선택 가능

## 요구 사항

- Python 3.11+
- 웹캠 (macOS 내장 카메라, Continuity Camera, 외장 USB 웹캠 모두 지원)
- macOS (키스트로크 주입은 macOS 전용, pyautogui 사용)

> **Center Stage (M4 MacBook Pro)**: 이 프로젝트의 기본 실험 설정은 Center Stage **OFF**입니다. Center Stage가 활성화되면 피사체 추적을 위한 실시간 크롭·줌이 발생해 손 랜드마크의 픽셀 크기와 위치가 지속적으로 변동되며, 특히 J·Z 모션 인식의 궤적 정확도에 영향을 줄 수 있습니다. ON 상태에서도 동작하지만 인식 성능은 보장되지 않습니다.
> 설정 경로: 시스템 설정 → 카메라 → Center Stage

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
| **Live + ED Mode** | Injection ON + Edit Distance 단어 추천 모드로 시작 |
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
| `W` | Word Injection 모드 ON/OFF 토글 | ✓ | ✓ |
| `X` | Edit Distance 모드 ON/OFF 토글 | ✓ | ✓ |
| `SPACE` | (Guess 모드) 현재 버퍼 확정 / 스페이스 입력 | — | ✓ |
| `Backspace` | (Guess/WI 모드) 버퍼 마지막 글자 삭제 | — | ✓ |
| `Q` | 종료 | ✓ | ✓ |

> 주입 ON 상태에서는 주입된 키가 cv2 창으로 돌아와 단축키를 오트리거하는 것을 방지하기 위해 E·R·T·Y·U·P 단축키가 비활성화됩니다.

## Guess 모드 사용법

1. 런처에서 **Live + Guess Mode** 선택, 또는 라이브 화면에서 `G` 키 토글
2. 수화로 단어의 앞 글자(들)를 입력 → 화면 중앙에 후보 3개 버튼 표시
3. 원하는 단어 버튼을 **마우스 클릭** 또는 키보드 `1`/`2`/`3`으로 선택 → 외부 앱에 주입
4. **SPACE** 버튼(또는 키): 현재 버퍼를 그대로 확정하고 다음 단어로
5. **BKSP** 버튼(또는 Backspace): 마지막 서명 글자 취소

**fuzzy fallback 표시**: 버퍼가 하늘색(`[ VOL +VTEEUVR ]` 형태)으로 표시되면 뒷부분에 오타가 있어 짧은 prefix로 fallback 매칭 중임을 뜻합니다. Backspace로 오타 부분을 지우거나 현재 후보에서 선택하세요.

## Edit Distance (ED) 모드 사용법

Classic Guess 모드보다 오타에 강한 단어 추천 모드입니다. 수화 인식 과정에서 1–2개의 오인식 글자가 섞여도 올바른 단어를 추천할 수 있도록 설계되었습니다.

1. 런처에서 **Live + ED Mode** 선택, 또는 라이브 화면에서 `X` 키 토글
2. 하단 상태바에 `ED:ON` 확인
3. 수화로 단어의 앞 글자(들)를 입력 → 화면 중앙에 청록색(cyan) 후보 3개 표시
4. 키보드 `1`/`2`/`3`으로 단어 선택 → 외부 앱에 단어 + 공백 주입
5. **BKSP** 버튼: 마지막 수화 글자 취소

> `G`, `W` 키와 상호 배타적으로 동작합니다 — ED 모드 ON 시 Guess/WI 모드는 자동 OFF.

---

## 단어 추천 알고리즘 상세

`word_suggester.py`에 두 가지 독립적인 추천 전략이 구현되어 있습니다.

### Classic 전략 (`suggest`)

Guess 모드 및 Word Injection 모드에서 사용합니다.

```
입력 prefix p
  1. 정확한 prefix 매칭
     → _en_words(빈도 내림차순)를 순차 스캔, p로 시작하는 단어 n개 반환
  2. suffix trim fallback
     → p의 마지막 글자부터 하나씩 제거하며 재시도 (최소 2자 유지)
     → 수화 인식에서 단어 끝부분에 오인식이 발생하는 경우 대응
  3. fuzzy fallback (같은 첫 글자)
     → p[0]로 시작하는 단어들에 대해 SequenceMatcher 유사도 계산 후 상위 n개 반환
```

**한계**: suffix trim은 끝부분 오타에만 대응하고, fuzzy fallback은 같은 첫 글자로 범위가 제한됩니다. 중간 글자 오타나 전치(transposition) 오류에는 취약합니다.

---

### Edit Distance 전략 (`suggest_ed`)

Live + ED 모드에서 사용합니다. rapidfuzz 라이브러리 기반입니다.

```
입력 prefix p
  1. 정확한 prefix 매칭 (Classic과 동일한 빠른 경로)
     → 정확 매칭이 있으면 즉시 반환 (ED 스캔 생략)

  2. Edit Distance fallback
     2-a. 길이 필터로 후보 압축
          → _en_words 전체(321,180개)에서 len(p)-2 ~ len(p)+4 범위의 단어만 추출
          → 예: p="wrold"(5자) → 3~9자 단어만 스캔 (~26만 단어)
     2-b. rapidfuzz fuzz.ratio 스코어링
          → ratio(p, w) = 2 × |LCS(p,w)| / (|p| + |w|)
          → limit=n×5, score_cutoff=50 으로 상위 15개 후보 추출
     2-c. wordfreq 빈도순 재정렬
          → _en_words 내 인덱스(낮을수록 빈도 높음)를 기준으로 재정렬
          → 상위 n개 반환
```

**핵심 설계 원칙**:
- `fuzz.ratio`는 문자열 전체의 edit distance를 반영하므로 오타 위치(앞/중간/끝)에 무관하게 동작
- 빈도 재정렬이 없으면 edit distance가 작은 희귀 단어(`wold`, `wrld`)가 상위에 오름 → 빈도 재정렬로 `world` 같은 일반 단어를 우선 노출
- 정확 prefix 매칭을 빠른 경로로 두어 오타 없는 입력의 응답속도 보존

**Classic 대비 개선 사례**:

| 오타 입력 | Classic 결과 | ED 결과 | 오류 유형 |
|---------|------------|--------|----------|
| `wrold` | wrong, wrote | **would, world**, old | 중간 전치 |
| `windwo` | (없음) | wind, **window**, windows | 후반 전치 |
| `tpye` | (없음) | **type**, tape, tyre | 전치 |
| `hppy` | (없음) | **happy**, hippy, choppy | 중간 삭제 |
| `wtaer` | (없음) | **water**, wager, taper | 전치 |
| `voiec` | (없음) | **voice**, vic, vie | 후반 전치 |

**현재 한계**:
- 첫 글자 오인식(`kproj` → project)은 ED 스코어가 낮아 미탐지
- wordfreq에 등재된 오타 단어(`leter`, `recieve`)가 있으면 오타 자체가 상위 노출될 수 있음
- 향후 컨텍스트(이전 입력 단어들) 기반 LLM 추천으로 보완 예정

---

## Word Injection 모드 사용법

Guess 모드와 달리 타겟 앱의 포커스를 유지한 채 연속으로 단어를 입력할 수 있습니다.

1. 타겟 앱(메모장, 브라우저 등)에 커서를 위치시키고 **OpenCV 창으로 포커스 이동**
2. `W` 키 → 하단 상태바에 `WI:ON` 표시 확인
3. 수화로 단어의 앞 글자(들)를 입력 → 화면 중앙에 황금색 후보 3개 표시
4. 키보드 `1`/`2`/`3`으로 단어 선택
   - Cmd+Tab → 타겟 앱으로 전환 → 단어 + 공백 주입 → Cmd+Tab → OpenCV 창 복귀
5. 다음 단어 수화 입력 바로 이어서 가능
6. **BKSP** 버튼: 마지막 수화 글자 취소

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

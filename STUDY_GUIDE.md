# ED vs LLM Study 실행 가이드

SignLanguageInputCapture의 "ED vs LLM Study" 모드로 Phase 2(ED vs LLM 사용성 비교) 데이터를 수집하는 절차입니다. 문장 선정 방법론/실험 설계 근거는 `PHRASE_SAMPLING.md` 참고.

Phase 1(분류기 비교, `DATA_COLLECTION_GUIDE.md`)에 참여했던 6명은 Phase 2에는 참여하지 않습니다 — 완전히 새로운 참가자 풀입니다.

---

## 사전 요구사항

`DATA_COLLECTION_GUIDE.md`와 동일합니다.

- macOS (M1 이상), Python 3.11+, 웹캠
- M4/M5 MacBook Pro는 Center Stage OFF 권장

## 환경 설정

```bash
git clone https://github.com/roomJdev/SignLanguageInputCapture.git
cd SignLanguageInputCapture
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ollama가 필요합니다 (LLM 모드용) — 설치 안 돼있으면 앱이 처음 LLM 블록 시작 시 자동으로 실행을 시도합니다. 문제 생기면 [Ollama 공식 설치](https://ollama.com) 참고.

---

## 1. 보정 (Calibration) — 반드시 먼저 진행

`DATA_COLLECTION_GUIDE.md`의 "보정" 섹션과 동일합니다. 본인 이름으로 프로필을 저장하세요 (예: `gildong`).

---

## 2. 참가자 번호 배정받기 — 시작 전 필수

**연구자(정준식)에게 본인 참가자 번호를 미리 배정받으세요.** 이 번호가 ED/LLM 어느 모드가 먼저 나올지, 문장 순서가 어떻게 될지를 결정합니다.

| 참가자 번호 | ED/LLM 순서 | 문장 순서 |
|---|---|---|
| 1, 5, 9, 13, ... | ED 먼저 | ph1 먼저 |
| 2, 6, 10, 14, ... | LLM 먼저 | ph1 먼저 |
| 3, 7, 11, 15, ... | ED 먼저 | ph2 먼저 |
| 4, 8, 12, 16, ... | LLM 먼저 | ph2 먼저 |

번호는 몇 명이 참여하든 4명 단위로 계속 반복됩니다. **본인 컴퓨터의 이전 세션 개수와는 무관하게, 미리 배정받은 번호를 그대로 입력**하면 됩니다 — 여러 명이 각자 다른 노트북에서 진행해도 이 방식이면 항상 의도한 순서로 정확히 배정됩니다.

---

## 3. Study 모드 실행

```bash
python main.py
```

런처에서 **ED vs LLM Study** 선택 후:

1. **보정 프로필 선택** — 본인이 방금 보정한 프로필 선택
2. **참가자 이름/세션명 입력** — 영문 소문자 (예: `gildong`)
3. **참가자 번호 입력** — 위 2번에서 배정받은 번호 (예: `3`)
4. 화면에 "Trial 1/4 — 준비되면 SPACE" 안내가 뜨면 시작
5. SPACE → 목표 문장을 수화로 입력 (ED 또는 LLM 모드, 화면에 표시된 대로) → ENTER로 제출
6. 자동으로 다음 trial로 넘어감. **2개 trial(같은 모드 블록)이 끝나면** 화면이 "◯◯ 블록 완료 — NASA-TLX 작성 후 SPACE"로 바뀝니다
   - 이 시점에 **연구자가 준비한 NASA-TLX 설문(별도 CSV/구글폼)에 응답**한 뒤 SPACE로 계속 진행 — 앱이 자동으로 설문을 띄워주지 않으니 직접 챙겨야 합니다
7. 4개 trial(ED 2개 + LLM 2개) 모두 끝나면 마지막 NASA-TLX 작성 후 자동으로 런처로 복귀
8. **전체 세션이 끝나면 Preference survey(선호도 설문)도 별도로 진행** — 이것도 앱에 내장되어 있지 않으니 연구자가 별도로 안내합니다

결과는 자동으로 `data_phase2/study_results.json`에 저장됩니다 (참가자 이름, 배정된 순서, 각 trial의 소요 시간·backspace 횟수·목표 문장과의 유사도 포함). Phase 1의 `data_new0803/`와는 별도 폴더입니다 — 보정(calibration) 데이터만 기존처럼 `data_new0803/`에 저장됩니다.

---

## 4. NASA-TLX / Preference survey 기록

앱이 자동화하지 않는 부분이라 `data_phase2/survey_responses.csv`에 **직접** 기록합니다. 참가자 1명당 **2행**(ED 블록 끝났을 때 1행, LLM 블록 끝났을 때 1행)을 채웁니다.

| 컬럼 | 설명 |
|---|---|
| `participant_number` | 배정받은 참가자 번호 |
| `name` | 참가자 이름 (study 세션명과 동일하게) |
| `date` | 응답 날짜 |
| `mode` | 이 행이 어느 블록 직후 응답인지 (`ED` / `LLM`) |
| `nasa_tlx_q1` | Mental Demand |
| `nasa_tlx_q2` | Physical Demand |
| `nasa_tlx_q3` | Temporal Demand |
| `nasa_tlx_q4` | Performance |
| `nasa_tlx_q5` | Effort |
| `nasa_tlx_q6` | Frustration |
| `custom_survey_q1`~`q5` | Preference survey (아래 참고) — **양쪽 블록이 다 끝난 뒤 1회만 답하는 설문이라, 두 번째(마지막) 블록 행에만 채우고 첫 번째 블록 행은 비워둡니다** |

NASA-TLX 6개 항목은 표준 방식대로 각각 0~100(또는 1~21) 척도로 응답받으면 됩니다. Preference survey 문항:

1. Which mode did you prefer overall? (ED / LLM / No preference)
2. Preference strength — 7-point scale: Strongly prefer ED ↔ Strongly prefer LLM
3. Which mode's suggestions felt more relevant or natural? (ED / LLM / No difference)
4. Which mode was easier to use overall? (ED / LLM / No difference)
5. [Open-ended] Why did you prefer that mode?

`survey_responses.csv`에는 예시 행(`participant_number=0`)이 들어있습니다 — 실제 데이터 입력 전에 지우고 사용하세요.

---

## 5. 결과 파일 전달

세션 완료 후 아래 파일들을 압축해서 전달해주세요.

```bash
zip -r study_gildong.zip data_phase2/study_results.json data_phase2/survey_responses.csv data_new0803/cal_gildong* data_new0803/calibration_profiles.json
```

- `data_phase2/study_results.json` — study 세션 결과 (여러 명이 한 컴퓨터에서 진행했다면 세션이 리스트로 누적되어 있습니다)
- `data_phase2/survey_responses.csv` — NASA-TLX / Preference survey 응답 (위 4번에서 작성한 파일)
- `data_new0803/cal_<이름>*` — 본인 보정 데이터 (재현/검증용, Phase 1과 같은 폴더에 저장되는 게 맞습니다)

**여러 컴퓨터에서 나눠 진행한 경우**: 각자의 `study_results.json`이 따로 생기므로, 연구자가 나중에 전달받은 파일들을 하나로 합쳐서 분석합니다. 참가자 이름/번호가 서로 겹치지 않게 미리 정해서 배정해주세요.

---

## 참고: 세션 확인/관리 (연구자용)

```bash
python main.py --list-study-sessions           # 저장된 study 세션 목록 확인
python main.py --delete-study-session <인덱스>  # 잘못되거나 중단된 세션 삭제
```

런처의 **"ED vs LLM Results"** 메뉴에서도 세션별 상세 내역(trial별 소요시간/backspace/유사도, 배정된 순서)을 카메라 없이 확인할 수 있습니다.

---

## 문제 해결

카메라/설치 관련 문제는 `DATA_COLLECTION_GUIDE.md`의 "문제 해결" 섹션 참고.

**Ollama/LLM 모드가 응답하지 않는 경우**: ED 모드로 자동 폴백되도록 설계돼 있어 실험 자체는 중단되지 않지만, 정상적으로 LLM 응답을 받으려면 Ollama가 로컬에서 실행 중이어야 합니다.

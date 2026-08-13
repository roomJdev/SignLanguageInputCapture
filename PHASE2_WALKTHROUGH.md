# Phase 2 데이터 수집 — 단계별 실행/검증 워크스루

`Calibration → ED vs LLM Study → NASA-TLX & Custom preference survey` 전체 흐름을, 어떤 화면에서 어떤 키를 눌러 진입하는지 + 각 단계 후 어느 폴더에 어떤 파일이 생겼는지 확인하는 절차까지 정리했습니다. 참가자에게 배포하는 실행 안내는 `STUDY_GUIDE.md`, 실험 설계 근거는 `PHRASE_SAMPLING.md` 참고.

```bash
python main.py
```
로 앱을 켜면 아래 순서가 전부 이 하나의 런처 화면(방향키 또는 w/s로 이동, Enter로 선택)에서 시작됩니다.

---

## 0. 실행 전 상태 (기준점)

미팅에서 "확보되는 파일"을 눈으로 보여주려면, 시작 전에 아래 상태를 먼저 확인해두면 비교가 쉽습니다.

```bash
cat data_new0803/calibration_profiles.json   # 등록된 보정 프로필 목록
ls data_phase2/                               # study_results.json, survey_responses.csv
```

---

## 1단계. Calibration

**런처에서**: `Calibrate` 선택 (목록 맨 위)

**진행 화면**:
1. 정적 심볼(A~Y 제외 J, 0~9) 34개를 화면에 뜨는 대로 1초씩 유지 — 총 30샘플/심볼
2. 이어서 동적 심볼(J, Z) 각 3회 반복 캡처
3. 완료 후 `Save calibration as (Enter to skip):` 입력창 → **참가자 영문 이름 입력** (예: `gildong`)

**확인할 파일** (`data_new0803/`):

```bash
ls data_new0803/cal_gildong*
ls data_new0803/motion_cal_gildong*
cat data_new0803/calibration_profiles.json | python3 -m json.tool
```

| 파일 | 의미 |
|---|---|
| `cal_gildong.npy` / `.json` | 정적 심볼 보정 데이터 (사람이 읽는 사본은 `.json`) |
| `cal_gildong_landmarks.npy` / `.json` | 위 보정 데이터의 가공 전 21관절 원본 좌표 |
| `cal_gildong_videos/<symbol>.mp4` | 심볼별 캡처 영상 원본 |
| `motion_cal_gildong.npy` / `.json` | J/Z 모션 보정 데이터 |
| `motion_cal_gildong_landmarks.npy` / `.json` | 위 모션 데이터의 프레임별 원본 좌표 |
| `motion_cal_gildong_videos/<symbol>_rep<n>.mp4` | J/Z 반복별 캡처 영상 |
| `calibration_profiles.json` | 등록된 프로필 인덱스 — `name: "gildong"` 항목이 새로 추가됐는지 확인 |

**여기서 체크할 것**: `calibration_profiles.json`에 `gildong` 항목이 `created_at` 타임스탬프와 함께 새로 생겼는지 — 이게 없으면 "Save calibration as"에서 이름을 안 넣고 Enter(skip)를 눌렀다는 뜻이라, 다음 단계에서 프로필 선택 화면에 안 뜹니다.

---

## 2단계. ED vs LLM Study

**런처에서**: `ED vs LLM Study` 선택

**진행 화면 (순서대로)**:

1. **`Select Calibration Profile`** 화면 — w/s로 이동, Enter로 선택. 방금 만든 `gildong` 프로필을 선택 (맨 위 `Default (current calibration)`이 아니라, 이름 붙은 프로필을 선택해야 결과 파일에 `cal_profile: "gildong"`으로 정확히 기록됨)
2. **`Enter participant name / number:`** 입력창 — 세션명 입력 (보정 프로필명과 동일하게, 예: `gildong`)
3. **`Enter participant NUMBER (1, 2, 3, ... - cycles ED/LLM-first x phrase order every 4):`** 입력창 — **사전에 배정받은 참가자 번호**를 입력 (예: `3`). 이 숫자가 홀/짝 + 4명 주기로 ED/LLM 순서와 문장 순서를 동시에 결정함 (`STUDY_GUIDE.md` 2번 표 참고)
4. `Trial 1/4 — 준비되면 SPACE` 대기 화면
5. SPACE → 목표 문장을 수화로 입력 → ENTER로 제출 → 자동으로 다음 trial
6. **같은 모드 2개 trial(블록)이 끝나면** 화면이 `◯◯ BLOCK COMPLETE -- fill out NASA-TLX now, ...`로 바뀜 → **여기서 3단계(NASA-TLX)를 진행**하고 SPACE로 계속
7. 4개 trial(ED 2개 + LLM 2개) 모두 끝나면 마지막 NASA-TLX 안내 후 자동으로 런처 복귀

**확인할 파일** (`data_phase2/`):

```bash
python main.py --list-study-sessions
cat data_phase2/study_results.json | python3 -m json.tool
```

| 필드 | 확인 포인트 |
|---|---|
| `participant` | 2번에서 입력한 세션명과 일치하는지 |
| `cal_profile` | 1번에서 선택한 프로필명과 일치하는지 (`Default`로 나오면 프로필 선택을 건너뛴 것) |
| `block_order` / `phrase_order` | 입력한 참가자 번호에 맞는 조합인지 (`STUDY_GUIDE.md` 표와 대조) |
| `trials` | 4개 항목, 각각 `mode`(`ed`/`llm`), `target_phrase`, `typed_text`, `elapsed_sec`, `backspace_count`, `similarity_ratio` 포함 |

세션 상세는 런처의 `ED vs LLM Results`에서 카메라 없이 GUI로도 볼 수 있습니다.

---

## 3단계. NASA-TLX & Custom Preference Survey

**앱이 자동화하지 않는 부분** — 2단계 6번의 "블록 완료" 화면이 뜰 때마다(참가자당 2번, ED 블록 후 1번 + LLM 블록 후 1번) 아래 CSV에 직접 기록합니다.

```bash
open data_phase2/survey_responses.csv   # 또는 Numbers/Excel로 열기
```

**작성 규칙** (참가자 1명당 2행):

- **ED 블록 완료 시점**: `mode=ED` 행에 `nasa_tlx_q1~q6`만 채움 (`custom_survey_*`는 비워둠 — 아직 두 모드 다 안 써봤으므로)
- **LLM 블록 완료 시점(마지막)**: `mode=LLM` 행에 `nasa_tlx_q1~q6` + `custom_survey_q1~q5` 전부 채움 (Preference survey는 양쪽 다 끝난 뒤 1회만 진행)

컬럼 의미는 `STUDY_GUIDE.md`의 "4. NASA-TLX / Preference survey 기록" 섹션 참고 (NASA-TLX 6개 서브스케일명, Preference 5개 문항 원문).

**확인할 파일**: `data_phase2/survey_responses.csv`에 해당 참가자 번호의 행 2개가 채워졌는지 (예시 행 `participant_number=0`은 실사용 전 삭제 대상이므로 제외하고 확인).

---

## 완료 후 최종 체크리스트 (참가자 1명 기준)

```bash
# 1. 보정 데이터
ls data_new0803/cal_gildong* data_new0803/motion_cal_gildong*

# 2. study 세션 결과
python main.py --list-study-sessions

# 3. 설문 응답
grep gildong data_phase2/survey_responses.csv    # 혹은 이름 대신 참가자 번호로 확인
```

세 가지가 전부 확인되면 그 참가자의 데이터 수집은 완료된 것입니다.

---

## 여러 컴퓨터에서 나눠 진행할 때

각 컴퓨터마다 `data_new0803/`(보정)과 `data_phase2/`(study 결과 + survey CSV)가 독립적으로 생성됩니다. 참가자 번호가 서로 겹치지 않도록 미리 배정하고, 수집 완료 후 두 폴더를 함께 전달받아야 합니다 (`STUDY_GUIDE.md` "5. 결과 파일 전달" 참고).

# 데이터 수집 가이드

SignLanguageInputCapture의 테스트 모드를 이용해 ASL 알파벳 인식 정확도 데이터를 수집하는 절차입니다.

---

## 사전 요구사항

- macOS (M1 이상)
- Python 3.11+
- 웹캠 (내장 카메라 가능)
- M4 / M5 MacBook Pro 사용 시: **Center Stage OFF** 권장
  - 시스템 설정 → 카메라 → Center Stage 비활성화
  - Center Stage가 활성화되면 실시간 크롭/줌으로 손 랜드마크 위치가 변동되어 인식 정확도에 영향을 줄 수 있습니다

---

## 1. 환경 설정

```bash
git clone https://github.com/roomJdev/SignLanguageInputCapture.git
cd SignLanguageInputCapture

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

설치 완료 확인:

```bash
python main.py --list
```

카메라 목록이 출력되면 정상입니다.

---

## 2. 보정 (Calibration) — 반드시 먼저 진행

테스트 결과는 보정 데이터 기반으로 분류됩니다. **각 피험자가 직접 본인의 손으로 보정을 완료해야 합니다.**

```bash
python main.py
```

런처에서 **Calibrate** 선택 후 화면 안내에 따라 진행합니다.

- 각 심볼마다 30개 샘플 수집 (스페이스바로 시작, 화면 오른쪽 레퍼런스 이미지 참고)
- J, Z는 정적 보정 완료 후 모션 보정이 이어집니다
- 보정 완료 후 프로필 이름 저장 시 **영문 이름 소문자** 사용
  - 예: `gildong` → `data/cal_gildong.npy`, `data/motion_cal_gildong.npy` 생성

> **주의**: 보정을 건너뛰면 인식률이 현저히 낮게 측정되어 데이터로 사용할 수 없습니다.

---

## 3. 테스트 모드 실행

```bash
python main.py
```

런처에서 **Test (Ordered)** 또는 **Test (Random)** 선택합니다.

| 모드 | 설명 |
|---|---|
| Test (Ordered) | A → Z → 0 → 9 순서로 제시 |
| Test (Random) | 랜덤 순서로 제시 |

**테스트 진행 순서**:

1. 테스트 모드 선택 후 **보정 프로필 선택 화면**이 나옵니다
   - 방향키 또는 w/s 키로 이동, Enter로 선택
   - 본인이 방금 보정한 프로필(`gildong` 등)을 선택하세요
2. 참여자 이름 입력 (영문 소문자, 보정 프로필명과 동일하게)
3. 확인 화면에서 Enter 또는 y 입력 후 시작
4. 화면에 표시되는 심볼을 수화로 표현하면 자동으로 다음으로 넘어갑니다
5. 세션이 끝나면 결과가 자동으로 `data/test_results.json`에 저장됩니다

**권장**: Test (Ordered)와 Test (Random) **두 모드 모두 1회씩** 실행해주세요.

---

## 4. 결과 파일 전달

테스트 완료 후 `data/` 폴더 전체를 압축해서 전달해주세요.

```
data/
├── cal_<이름>.npy               # 보정 데이터 (프로필명으로 저장)
├── motion_cal_<이름>.npy        # J/Z 모션 보정 데이터
├── calibration_profiles.json
└── test_results.json            # 테스트 결과
```

```bash
# 프로젝트 루트에서 실행 — 파일명은 본인 영문 이름으로
zip -r data_gildong.zip data/
```

압축 파일과 함께 아래 **실험 조건 CSV**도 작성해서 보내주세요.

---

## 5. 실험 조건 CSV

아래 양식으로 CSV 파일을 작성해주세요. 파일명은 `info_<이름>.csv` 형식으로 저장하세요.

```
participant,device,asl_experience,calibration,center_stage,notes
gildong,MacBook Pro 14" M2 Pro,없음,완료,OFF,
```

| 컬럼 | 설명 | 입력 예시 |
|---|---|---|
| `participant` | 영문 이름 소문자 (보정 프로필명과 동일) | `gildong` |
| `device` | 기기 모델명 | `MacBook Pro 14" M2 Pro` |
| `asl_experience` | ASL 사전 경험 | `없음` / `조금 있음` / `능숙` |
| `calibration` | 보정 완료 여부 | `완료` / `미완료` |
| `center_stage` | Center Stage 활성화 여부 | `ON` / `OFF` / `해당없음` |
| `notes` | 특이사항 | `조명 어두움`, `외장 웹캠 사용` 등 |

---

## 문제 해결

**카메라가 인식되지 않는 경우**
```bash
python main.py --list        # 연결된 카메라 목록 확인
python main.py --camera 1    # 다른 카메라 인덱스 시도
```

**보정 데이터가 없다고 나오는 경우**
- 런처에서 Calibrate를 먼저 실행하고, 보정 완료 후 프로필 이름을 저장했는지 확인하세요
- `data/` 폴더에 `cal_<이름>.npy` 파일이 생성되어 있어야 합니다

**테스트 세션 초기화가 필요한 경우**
```bash
python main.py --list-test-sessions      # 저장된 세션 목록 확인
python main.py --delete-test-session 2   # 특정 세션 삭제
python main.py --reset-test-results      # 전체 초기화
```

# 데이터 수집 가이드

SignLanguageInputCapture의 테스트 모드를 이용해 ASL 알파벳 인식 정확도 데이터를 수집하는 절차입니다.

---

## 사전 요구사항

- macOS (M1 이상 권장)
- Python 3.11+
- 웹캠 (내장 카메라 가능)
- M4 / M5 MacBook Pro 사용 시: **Center Stage OFF** 필수
  - 시스템 설정 → 카메라 → Center Stage 비활성화

---

## 1. 환경 설정

```bash
git clone https://github.com/roomJdev/SignLanguageInputCapture.git
cd SignLanguageInputCapture

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. macOS 접근성 권한 허용

키스트로크 주입 기능이 내부적으로 접근성 API를 사용하므로 권한을 미리 허용해야 합니다.

> 시스템 설정 → 개인 정보 보호 및 보안 → 손쉬운 사용 → Terminal (또는 iTerm) 허용

---

## 3. 보정 (Calibration) — 반드시 먼저 진행

테스트 결과는 보정 데이터 기반으로 분류됩니다. **각 피험자가 직접 본인의 손으로 보정을 완료해야 합니다.**

```bash
python main.py
```

런처에서 **Calibrate** 선택 후 화면 안내에 따라 진행합니다.

- 각 심볼마다 30개 샘플 수집
- J, Z는 모션 보정 별도 진행
- 완료 후 프로필 이름 저장 시 **영문 이름(소문자)** 사용
  - 예: `gildong` → `cal_gildong.npy`, `motion_cal_gildong.npy` 생성

> 보정을 건너뛰면 인식률이 현저히 낮게 측정되어 데이터로 사용할 수 없습니다.

---

## 4. 테스트 모드 실행

```bash
python main.py
```

런처에서 **Test (Ordered)** 또는 **Test (Random)** 선택합니다.

| 모드 | 설명 |
|---|---|
| Test (Ordered) | A → Z → 0 → 9 순서로 제시 |
| Test (Random) | 랜덤 순서로 제시 |

- 화면에 표시되는 심볼을 수화로 표현하면 자동으로 다음으로 넘어갑니다.
- 세션이 끝나면 결과가 자동 저장됩니다.
- **두 모드 모두 1회씩** 실행하는 것을 권장합니다.

---

## 5. 결과 파일 전달

테스트 완료 후 아래 `data/` 폴더 전체를 압축해서 전달해주세요.

```
data/
├── calibration_data.npy          # 기본 보정 데이터
├── motion_calibration_data.npy   # J/Z 모션 보정 데이터
├── cal_<이름>.npy                 # 프로필로 저장한 경우
├── motion_cal_<이름>.npy
├── calibration_profiles.json
└── test_results.json             # 테스트 결과
```

```bash
# data 폴더 압축 — 파일명은 영문 이름으로 통일 (프로젝트 루트에서 실행)
zip -r data_gildong.zip data/
```

---

### 실험 조건 CSV

아래 양식으로 CSV 파일을 작성해서 압축 파일과 함께 전달해주세요.

```
participant,device,asl_experience,calibration,center_stage,notes
홍길동,MacBook Pro 14" M2 Pro,없음,완료,OFF,
```

| 컬럼 | 입력 예시 |
|---|---|
| `participant` | 영문 이름 소문자 (보정 프로필명과 동일하게) |
| `device` | MacBook Pro 14" M2 Pro |
| `asl_experience` | 없음 / 조금 있음 / 능숙 |
| `calibration` | 완료 / 미완료 |
| `center_stage` | ON / OFF / 해당없음 |
| `notes` | 조명 어두움, 외장 웹캠 사용 등 특이사항 |

---

## 참고: 테스트 세션 관리

```bash
# 저장된 세션 목록 확인
python main.py --list-test-sessions

# 특정 세션 삭제
python main.py --delete-test-session <번호>

# 전체 초기화
python main.py --reset-test-results
```

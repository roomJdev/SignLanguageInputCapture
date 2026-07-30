# 문장 샘플링 방법론 (ED vs LLM 모드 비교 실험)

ED mode와 LLM mode 간 사용성(소요 시간·오타·NASA-TLX)을 비교하는 실험에 사용할 문장을 어떻게 선정했는지 정리한 문서입니다.
(초기에는 Guess mode까지 3-way 비교를 고려했으나, W5 리포트의 핵심 기여가 ED/LLM 비교인 점을 감안해 2-way로 축소함.)

---

## 1. 출처: MacKenzie & Soukoreff Phrase Set (2003)

- HCI 텍스트 입력 평가에서 표준으로 쓰이는 **500개 짧은 영어 문장** 세트
  - MacKenzie, I. S., & Soukoreff, R. W. (2003). *Phrase sets for evaluating text entry techniques.* CHI '03 Extended Abstracts.
- 구두점 없이 소문자 알파벳 + 공백으로만 구성 (평균 28.6자, 16~43자 범위)
- 영어 문자 빈도와의 상관계수 r = .954로 실제 영어를 잘 대표
- ASL fingerspelling text-entry 평가 연구(예: **SpellRing**, CHI 2025)에서도 동일 세트를 사용한 선례가 있어, 본 프로젝트(ASL 기반 텍스트 입력)에도 적합하다고 판단
- 원본 파일: `phrases2.txt` (http://www.yorku.ca/mack/PhraseSets.zip)

### SpellRing과의 차이
SpellRing은 1,164개 단어에서 200개 문장(연습 100 + 테스트 100)을 무작위 생성해 11~20명 참가자에게 5세션에 걸쳐 배포했다. 특정 phrase 리스트를 고정 공개하지 않았고 규모도 본 프로젝트(참가자 1인, 단일 세션, pilot)와 맞지 않아 **동일 문장을 그대로 가져올 수는 없음**. 대신 **동일 출처(MacKenzie-Soukoreff set)에서 원칙적인 방법으로 샘플링**하는 방식을 택함.

---

## 2. 샘플링 절차: 상대 엔트로피(KL Divergence) 기반 Bigram Matching

> **출처 정정**: 이 절차의 원 저자는 **Paek & Hsu (2011)**이며, Vertanen & Kristensson (2011)은 이 절차를 자신들의 EnronMobile phrase set에 "적용"한 것뿐이다 (Vertanen & Kristensson 논문 본문: *"Paek and Hsu [5] describe a procedure for creating phrase sets by randomly sampling sets of n-grams and choosing the set whose character bigram distribution is closest to the distribution over the entire dataset. **We modified their procedure**..."*).

> Paek, T., & Hsu, B. (2011). *Sampling representative phrase sets for text entry experiments: A procedure and public resource.* CHI '11, 2477–2480.
> ([원문 PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SamplingRepresentativePhraseSets.pdf))
>
> Vertanen, K., & Kristensson, P. O. (2011). *A versatile dataset for text entry evaluations based on genuine mobile emails.* MobileHCI '11, 295–298.
> ([원문 PDF](https://www.keithv.com/pub/enronmobile/mobile_email_dataset.pdf)) — 위 절차를 MacKenzie-Soukoreff류 phrase set에 적용한 선례.

### 수식 (Paek & Hsu, 2011 원문 그대로)

문자 집합 χ에 대한 확률분포 p(x)(후보 부분집합의 bigram 분포)와 q(x)(전체 코퍼스의 bigram 분포) 사이의 **상대 엔트로피(relative entropy) = KL divergence**:

  D(p‖q) = Σ_x p(x) · log₂( p(x) / q(x) )

- 로그 밑은 **2**(bits 단위) — 원 논문 표기 그대로
- D(p‖q) ≥ 0이며, 두 분포가 동일할 때만 0
- 후보 집합 S_i 여러 개 중 다음을 만족하는 것을 채택:

  S* = argmin_{S_i} D(S_i ‖ C)   (C = 전체 코퍼스)

### 알고리즘
1. 전체 500개 문장에서 문자 bigram 빈도 분포 계산 (기준 분포 q, 500개 전체 기준으로 고정)
2. 후보 풀을 두 가지로 필터링:
   - **내용 필터**: murder/kill/prison 등 리포트·데모에 부적절한 단어가 포함된 문장 제외
   - **길이 필터**: 원본 평균 길이(28.6자) 근처인 **29자 이하**로 제한 — 세션당 트라이얼 소요 시간을 줄이기 위함 (아래 3번 참고)
3. 필터링된 풀에서 크기 2인 부분집합 p를 무작위로 20만 번 샘플링
4. 각 후보 집합의 D(p‖q) 계산, 가장 작은 집합을 최종 채택 (S*)

### 결과 (KL divergence, bits 단위 — log₂ 기준으로 재계산)

| 방식 | 문장 수 | KL divergence (bits) |
|---|---|---|
| 길이만 맞춘 단순 계층 샘플링 (5분위) | 5 | 1.211 |
| Bigram 매칭 (필터 전) | 5 | 0.830 |
| Bigram 매칭 (내용 필터 적용) | 4 | 0.947 |
| **Bigram 매칭 (내용+길이 필터, 최종 채택)** | **2** | **1.572** |

문장 수를 5→2로 줄이면서 KL divergence는 다소 높아졌지만(대표성이 약간 낮아짐), 실측 결과 참가자 1인이 ED/LLM 각 모드로 문장 1개를 입력하는 데 1분 내외가 걸려 **전체 세션을 20분 예산 안에 맞추는 것을 더 우선**했다.

**최종 선정 문장 (2개)**

| 길이 | 문장 |
|---|---|
| 28자 | all together in one big pile |
| 27자 | the food at this restaurant |

---

## 3. 실험 설계: 모드 블록 + 참가자 간 순서 교대 (ED vs LLM, 2문장 x 2모드 = 4 trial)

NASA-TLX를 트라이얼마다(4회)가 아니라 **모드당 1회(2회)**로 진행하기로 하면서, 같은 모드의 두 문장을 연달아 진행하는 "블록" 구조로 바꿨다. 대신 어느 모드 블록이 먼저 오는지는 참가자마다 교대시켜(between-participant counterbalancing) 순서 효과를 상쇄한다 — 참가자 수(N)가 1명을 넘어가는 시점부터는 문장 단위 ABBA보다 이 방식이 더 적합하다.

**참가자 1 (짝수번째, ED-first)**

| Trial | 문장 | 모드 |
|---|---|---|
| 1 | P1: all together in one big pile | ED |
| 2 | P2: the food at this restaurant | ED |
| — | (ED 블록 완료 → NASA-TLX #1) | |
| 3 | P1: all together in one big pile | LLM |
| 4 | P2: the food at this restaurant | LLM |
| — | (LLM 블록 완료 → NASA-TLX #2) | |

**참가자 2 (홀수번째, LLM-first)** — 위 표에서 ED/LLM 블록 순서만 반대.

구현:
- `study_mode.py`의 `SCHEDULE_ED_FIRST` / `SCHEDULE_LLM_FIRST`, `next_schedule()`이 기존 저장된 세션 수의 짝/홀에 따라 자동으로 교대 배정
- `main.py`의 `run()`에 반자동 실험 모드로 통합(런처 → "ED vs LLM Study"). SPACE로 트라이얼 시작, ENTER로 제출 시 소요 시간·backspace 횟수·목표 문장과의 유사도(difflib)가 자동 계산되어 `data/study_results.json`에 세션 단위로 저장됨(참가자별 배정된 `block_order`도 함께 기록)
- 모드 블록이 끝나는 시점(다음 트라이얼의 모드가 바뀌는 시점)에는 화면 배너가 "◯◯ 블록 완료 — NASA-TLX 작성 후 SPACE"로 바뀌어, 그때 별도 설문(종이/구글폼)으로 응답 후 계속 진행

---

## 4. 리포트 방법론 문구 (참고용)

> Phrases were selected via the relative-entropy (KL divergence) minimization procedure of Paek & Hsu (2011), applied to the character-bigram distribution — the same information-theoretic method later used by Vertanen & Kristensson (2011) when curating their EnronMobile phrase set. We sampled a 2-phrase subset from the MacKenzie-Soukoreff (2003) phrase set (length-capped near the corpus mean of 28.6 characters to keep session duration manageable) whose bigram distribution best approximates the full 500-phrase corpus (KL divergence = 1.572 bits). The same MacKenzie-Soukoreff phrase set has been used in prior ASL fingerspelling text-entry evaluations (e.g., SpellRing, CHI 2025). Each participant typed both phrases in one mode (a block), then both phrases in the other mode, with NASA-TLX administered once per mode block. Block order (ED-first vs. LLM-first) was alternated across participants to counterbalance order effects.

---

## 5. 재현 방법

```python
import random, math
from collections import Counter

with open("phrases2.txt", encoding="utf-8") as f:
    phrases = [l.strip() for l in f if l.strip()]

BLOCK = {"murder", "kill", "killed", "killing", "die", "died", "dead", "death",
         "blood", "gun", "rape", "suicide", "drug", "drugs", "prison", "jail",
         "attack", "war", "bomb", "terrorist", "weapon", "hate", "stupid",
         "hell", "damn"}

def is_clean(p):
    return set(p.lower().split()).isdisjoint(BLOCK)

# 내용 필터 + 길이 필터(원본 평균 28.6자 근처, 세션 시간 단축용)
pool = [p for p in phrases if is_clean(p) and len(p) <= 29]

def bigrams(text):
    text = text.lower()
    return [text[i:i+2] for i in range(len(text) - 1)]

def bigram_dist(phrase_list):
    c = Counter()
    for p in phrase_list:
        c.update(bigrams(p))
    total = sum(c.values())
    return {k: v / total for k, v in c.items()}, c

full_dist, _ = bigram_dist(phrases)   # 기준 분포는 항상 원본 500개 전체
vocab = set(full_dist.keys())

def kl_divergence(p_dist, q_dist, vocab, eps=1e-6):
    """D(p||q) = sum p(x) * log2(p(x)/q(x)) -- Paek & Hsu (2011) 원 논문과 동일하게 log2(bits) 사용"""
    kl = 0.0
    for k in vocab:
        p = p_dist.get(k, 0) + eps
        q = q_dist.get(k, eps)
        kl += p * math.log2(p / q)
    return kl

random.seed(0)
best_kl, best_sample = float("inf"), None
for _ in range(200_000):
    sample = random.sample(pool, 2)
    s_dist, _ = bigram_dist(sample)
    kl = kl_divergence(s_dist, full_dist, vocab)
    if kl < best_kl:
        best_kl, best_sample = kl, sample
```

- `phrases2.txt`는 `http://www.yorku.ca/mack/PhraseSets.zip`에서 받은 원본 파일을 그대로 사용 (레포에는 포함하지 않음, 필요 시 재다운로드)

---

## 6. 참고문헌 원문 링크

- MacKenzie, I. S., & Soukoreff, R. W. (2003). *Phrase sets for evaluating text entry techniques.* CHI '03 Extended Abstracts. — https://www.yorku.ca/mack/chi03b.html
- **Paek, T., & Hsu, B. (2011). *Sampling representative phrase sets for text entry experiments: A procedure and public resource.* CHI '11.** (KL divergence 수식 원 출처) — https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SamplingRepresentativePhraseSets.pdf
- Vertanen, K., & Kristensson, P. O. (2011). *A versatile dataset for text entry evaluations based on genuine mobile emails.* MobileHCI '11. — https://www.keithv.com/pub/enronmobile/mobile_email_dataset.pdf
- Python 공식 문서 — `difflib.SequenceMatcher.ratio()` (`ratio = 2.0*M/T` 정의) — https://docs.python.org/3/library/difflib.html

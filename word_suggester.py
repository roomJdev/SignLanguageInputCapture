"""단어 prefix → 빈도순 추천 단어 목록 제공 (wordfreq 기반).

Classic 전략 (순서대로 시도):
  1. 정확한 prefix 매칭 (빈도 상위 n개)
  2. 뒤에서 한 글자씩 제거해가며 재시도 (끝부분 오타 허용)
  3. 같은 첫 글자 단어 중 유사도 순 fuzzy 매칭 (중간 오타 허용)

ED 전략 (suggest_ed):
  1. 정확한 prefix 매칭 (동일)
  2. rapidfuzz edit distance — 길이 필터 후 전체 스캔

LLM 전략 (suggest_llm):
  이전에 확정된 단어들(컨텍스트) + 현재 prefix → Ollama 로컬 LLM에 질의
  컨텍스트가 없으면 ED 전략으로 fallback
"""

import json
import urllib.request
import urllib.error
from difflib import SequenceMatcher
from wordfreq import iter_wordlist

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODELS = ["gemma3:4b", "phi4-mini"]
_ollama_model_idx = 0


def get_llm_model() -> str:
    return _OLLAMA_MODELS[_ollama_model_idx]


def cycle_llm_model() -> str:
    global _ollama_model_idx
    _ollama_model_idx = (_ollama_model_idx + 1) % len(_OLLAMA_MODELS)
    return get_llm_model()

try:
    from rapidfuzz import process as _rf_process, fuzz as _rf_fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    print("[word_suggester] rapidfuzz 미설치 — ED 모드 비활성화")

_en_words: list[str] = []


def preload() -> None:
    global _en_words
    if not _en_words:
        _en_words = list(iter_wordlist("en"))
        print(f"[word_suggester] {len(_en_words):,}개 단어 로드 완료")


def _prefix_scan(p: str, n: int) -> list[str]:
    results: list[str] = []
    for w in _en_words:
        if w.startswith(p):
            results.append(w)
            if len(results) >= n:
                break
    return results


def _fuzzy_match(p: str, n: int) -> list[str]:
    """같은 첫 글자 단어 중 SequenceMatcher 유사도 상위 n개."""
    first = p[0]
    pl = len(p)
    candidates = [w for w in _en_words if w and w[0] == first]
    scored = sorted(
        candidates,
        key=lambda w: -SequenceMatcher(None, p, w[:pl]).ratio()
    )
    return scored[:n]


def suggest(prefix: str, n: int = 3) -> tuple[list[str], str]:
    """빈도 상위 n개 단어와 실제 매칭된 prefix를 반환.

    반환: (candidates, matched_prefix)
      matched_prefix: 정확 매칭이면 prefix 그대로, fallback이면 잘린 값, fuzzy면 "~" + 첫글자
    """
    if not prefix:
        return [], ""
    if not _en_words:
        preload()
    p = prefix.lower()

    # 1. 정확한 prefix
    results = _prefix_scan(p, n)
    if results:
        return results, p

    # 2. 뒤에서 한 글자씩 제거 (최소 2자)
    for trim in range(1, len(p) - 1):
        short = p[:-trim]
        results = _prefix_scan(short, n)
        if results:
            return results, short

    # 3. fuzzy (첫 글자 기반)
    if len(p) >= 2:
        return _fuzzy_match(p, n), f"~{p[0]}"

    return [], ""


def suggest_ed(prefix: str, n: int = 3) -> tuple[list[str], str]:
    """Edit Distance 기반 단어 추천 (rapidfuzz).

    전략:
      1. 정확한 prefix 매칭 (classic과 동일)
      2. rapidfuzz WRatio — prefix 길이 ±2 범위로 후보 필터 후 스캔

    반환: (candidates, matched_prefix)
      matched_prefix: 정확 매칭이면 prefix 그대로, ED fallback이면 "ed~" + prefix
    """
    if not prefix:
        return [], ""
    if not _en_words:
        preload()
    if not _RAPIDFUZZ_AVAILABLE:
        return suggest(prefix, n)

    p = prefix.lower()

    # 1. 정확한 prefix 매칭 (빠른 경로)
    results = _prefix_scan(p, n)
    if results:
        return results, p

    # 2. edit distance fallback — 길이 필터 후 스캔, 빈도순 재정렬
    pl = len(p)
    length_filtered = [w for w in _en_words if pl - 2 <= len(w) <= pl + 4]
    matches = _rf_process.extract(
        p,
        length_filtered,
        scorer=_rf_fuzz.ratio,
        limit=n * 5,
        score_cutoff=50,
    )
    if matches:
        _word_rank = {w: i for i, w in enumerate(_en_words)}
        sorted_matches = sorted(matches, key=lambda m: _word_rank.get(m[0], 999999))
        return [m[0] for m in sorted_matches[:n]], f"ed~{p}"

    return [], ""


def suggest_llm(context: str, prefix: str, n: int = 3) -> tuple[list[str], str]:
    """LLM 컨텍스트 기반 단어 추천 (Ollama 로컬 LLM).

    Args:
        context: 이전에 확정된 단어들 (예: "I was waiting for a")
        prefix:  현재 수화로 입력 중인 prefix (예: "b")
        n:       추천 단어 수

    Returns:
        (candidates, source)
          source: "llm" (LLM 성공), "ed~..." (ED fallback), "" (실패)

    Ollama 서버가 없거나 오류 시 suggest_ed()로 자동 fallback.
    컨텍스트가 비어있으면 suggest_ed()로 fallback.
    """
    if not prefix:
        return [], ""
    if not _en_words:
        preload()

    # 컨텍스트 없으면 ED로 fallback
    if not context.strip():
        return suggest_ed(prefix, n)

    prompt = (
        f"Complete the sentence. The sentence so far: \"{context.strip()}\"\n"
        f"Suggest exactly {n} English words that could follow, "
        f"starting with the letters \"{prefix.lower()}\". "
        f"Reply with only the {n} words separated by commas, nothing else. "
        f"Example format: word1, word2, word3"
    )

    payload = json.dumps({
        "model": get_llm_model(),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 30},
    }).encode()

    try:
        req = urllib.request.Request(
            _OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = json.loads(resp.read())
        raw = body.get("response", "").strip()
        # "word1, word2, word3" 파싱
        words = [w.strip().lower().rstrip(".,!?") for w in raw.split(",")]
        # prefix로 시작하는 것만 필터 (LLM이 지시를 무시할 경우 대비)
        filtered = [w for w in words if w.startswith(prefix.lower()) and w.isalpha()]
        if filtered:
            return filtered[:n], "llm"
        # prefix 조건 충족 단어가 없으면 ED로 fallback
        return suggest_ed(prefix, n)
    except Exception:
        return suggest_ed(prefix, n)

"""단어 prefix → 빈도순 추천 단어 목록 제공 (wordfreq 기반).

Classic 전략 (순서대로 시도):
  1. 정확한 prefix 매칭 (빈도 상위 n개)
  2. 뒤에서 한 글자씩 제거해가며 재시도 (끝부분 오타 허용)
  3. 같은 첫 글자 단어 중 유사도 순 fuzzy 매칭 (중간 오타 허용)

ED 전략 (suggest_ed):
  1. 정확한 prefix 매칭 (동일)
  2. rapidfuzz edit distance — 길이 필터 후 전체 스캔
"""

from difflib import SequenceMatcher
from wordfreq import iter_wordlist

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

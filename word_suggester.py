"""단어 prefix → 빈도순 추천 단어 목록 제공 (wordfreq 기반).

검색 전략 (순서대로 시도):
  1. 정확한 prefix 매칭 (빈도 상위 n개)
  2. 뒤에서 한 글자씩 제거해가며 재시도 (끝부분 오타 허용)
  3. 같은 첫 글자 단어 중 유사도 순 fuzzy 매칭 (중간 오타 허용)
"""

from difflib import SequenceMatcher
from wordfreq import iter_wordlist

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

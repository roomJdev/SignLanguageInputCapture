"""ED vs LLM 모드 비교 실험(반자동) — 목표 문장 제시, 자동 순서 전환, 결과 자동 기록.

문장 4개 x 모드 2개(ED/LLM) = 8 trial, ABBA 카운터밸런싱(PHRASE_SAMPLING.md 참고).
같은 문장을 ED/LLM 두 모드로 연속 입력하되, 어느 모드가 먼저 오는지는 문장마다 교대한다.

흐름 (main.py의 run() 루프 내부에서 처리):
    1) "Trial i/8 — 준비되면 SPACE" 대기 화면
    2) SPACE → 타이머 시작, 참가자가 손동작으로 목표 문장 입력 (ED 또는 LLM 모드)
    3) ENTER → 트라이얼 종료, 소요 시간·오타(backspace)·유사도 자동 기록 후 다음 트라이얼로 자동 전환
    4) 8개 모두 끝나면 자동으로 런처 복귀

NASA-TLX 설문은 코드로 자동화하지 않음 — 트라이얼 사이 대기 화면에서 종이/구글폼으로 응답 후 SPACE.

결과 저장 형식은 test_mode.py의 test_results.json과 동일하게 "세션 단위"로 묶는다.
세션 하나 = 참가자 1명이 8개 트라이얼을 도는 것. 트라이얼이 끝날 때마다 세션 안에
누적하고 즉시 파일에 다시 씀 (중간에 꺼져도 그때까지의 트라이얼은 보존됨).
"""

import json
import os
from datetime import datetime
from difflib import SequenceMatcher

import cv2
import numpy as np

_GREEN_BRIGHT = (80, 255, 120)
_GREEN_MID = (90, 230, 110)
_GREEN_DIM = (100, 200, 110)
_ED_COLOR = (220, 210, 80)    # 청록 (BGR) — main.py ED 패널과 동일 계열
_LLM_COLOR = (220, 130, 255)  # 보라 (BGR) — main.py LLM 패널과 동일 계열

# ---------------------------------------------------------------------------
# 문장 세트 — Vertanen & Kristensson (2011) bigram-matching 절차로 선정
# (자세한 선정 방법론은 PHRASE_SAMPLING.md 참고)
# ---------------------------------------------------------------------------

STUDY_PHRASES = [
    "all together in one big pile",
    "the food at this restaurant",
]

# ABBA 카운터밸런싱: 문장마다 ED/LLM 순서를 교대 (모드-순서 confound 제거)
STUDY_SCHEDULE = [
    {"phrase": STUDY_PHRASES[0], "mode": "ed"},
    {"phrase": STUDY_PHRASES[0], "mode": "llm"},
    {"phrase": STUDY_PHRASES[1], "mode": "llm"},
    {"phrase": STUDY_PHRASES[1], "mode": "ed"},
]

RESULTS_PATH = "data/study_results.json"


def similarity(typed: str, target: str) -> float:
    """입력 문장과 목표 문장의 유사도(0~1) — difflib SequenceMatcher 기반."""
    return SequenceMatcher(None, typed.strip().lower(), target.strip().lower()).ratio()


def _load_results() -> list[dict]:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _write_results(sessions: list[dict]) -> None:
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 세션 단위 API — run()에서 사용
# ---------------------------------------------------------------------------

def start_session(participant: str, cal_profile: str = "Default") -> dict:
    """새 study 세션을 시작하고, 결과 파일에 빈 세션을 즉시 등록한다 (세션 시작 시점부터 존재)."""
    session = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "participant": participant,
        "cal_profile": cal_profile,
        "trials": [],
    }
    sessions = _load_results()
    sessions.append(session)
    _write_results(sessions)
    return session


def append_trial(session: dict, trial_index: int, mode: str, target_phrase: str,
                  typed_text: str, elapsed_sec: float, backspace_count: int) -> dict:
    """트라이얼 결과를 세션에 추가하고, 세션 전체(=같은 timestamp)를 파일에 즉시 반영한다."""
    record = {
        "trial_index": trial_index,
        "mode": mode,
        "target_phrase": target_phrase,
        "typed_text": typed_text,
        "elapsed_sec": round(elapsed_sec, 2),
        "backspace_count": backspace_count,
        "similarity_ratio": round(similarity(typed_text, target_phrase), 4),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    session["trials"].append(record)

    sessions = _load_results()
    for i, s in enumerate(sessions):
        if s.get("timestamp") == session["timestamp"] and s.get("participant") == session["participant"]:
            sessions[i] = session
            break
    else:
        sessions.append(session)
    _write_results(sessions)
    return record


# ---------------------------------------------------------------------------
# 세션 조회/삭제 — test_mode.py의 list_sessions/delete_session과 동일 패턴
# ---------------------------------------------------------------------------

def list_study_sessions() -> None:
    """저장된 study 세션을 인덱스, 참가자, 트라이얼 수, 평균 유사도와 함께 출력."""
    sessions = _load_results()
    if not sessions:
        print("저장된 study 세션이 없습니다.")
        return

    print(f"\n저장된 ED vs LLM study 세션 ({len(sessions)}개):")
    for i, session in enumerate(sessions):
        trials = session.get("trials", [])
        participant = session.get("participant") or "(미입력)"
        profile = session.get("cal_profile") or "Default"
        n = len(trials)
        avg_sim = sum(t["similarity_ratio"] for t in trials) / n if n else 0.0
        print(f"  [{i}] {session['timestamp']}  참가자: {participant}  보정: {profile}  "
              f"트라이얼: {n}/{len(STUDY_SCHEDULE)}  평균 유사도: {avg_sim:.2f}")


def delete_study_session(index: int) -> None:
    """인덱스로 특정 study 세션 하나만 삭제 — list_study_sessions()로 인덱스 확인 후 사용."""
    sessions = _load_results()
    if not sessions:
        print("저장된 study 세션이 없습니다.")
        return
    if not (0 <= index < len(sessions)):
        print(f"잘못된 인덱스입니다 (0 ~ {len(sessions) - 1} 범위). list_study_sessions()로 확인하세요.")
        return
    removed = sessions.pop(index)
    _write_results(sessions)
    print(f"삭제됨: [{index}] {removed['timestamp']}  참가자: {removed.get('participant')}")


# ---------------------------------------------------------------------------
# GUI 뷰어 — 런처의 "ED vs LLM Results"에서 카메라 없이 실행
# (test_mode.py의 run_session_manager()/_run_session_detail()와 동일한 스타일)
# ---------------------------------------------------------------------------

def _run_study_detail(session: dict) -> None:
    """선택된 study 세션의 트라이얼별 상세(모드/목표/입력/시간/backspace/유사도)."""
    canvas_h, canvas_w = 640, 960
    trials = session.get("trials", [])
    participant = session.get("participant") or "(미입력)"
    profile = session.get("cal_profile") or "Default"

    cv2.namedWindow("Study Session Detail")

    while True:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        cv2.putText(canvas, "ED vs LLM Study — Session Detail", (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _GREEN_BRIGHT, 2)
        info_line = f"{session['timestamp']}   participant: {participant}   cal: {profile}"
        cv2.putText(canvas, info_line, (24, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN_MID, 1)
        cv2.line(canvas, (24, 78), (canvas_w - 24, 78), _GREEN_DIM, 1)

        y = 116
        for t in trials:
            mode_color = _ED_COLOR if t["mode"] == "ed" else _LLM_COLOR
            header = (f"Trial {t['trial_index'] + 1}  [{t['mode'].upper()}]   "
                      f"{t['elapsed_sec']:.1f}s   backspace={t['backspace_count']}   "
                      f"sim={t['similarity_ratio']:.2f}")
            cv2.putText(canvas, header, (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            y += 28
            cv2.putText(canvas, f"target: {t['target_phrase']}", (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _GREEN_DIM, 1)
            y += 24
            cv2.putText(canvas, f"typed : {t['typed_text']}", (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _GREEN_MID, 1)
            y += 40

        if trials:
            avg_sim = sum(t["similarity_ratio"] for t in trials) / len(trials)
            avg_ed = [t for t in trials if t["mode"] == "ed"]
            avg_llm = [t for t in trials if t["mode"] == "llm"]
            summary = f"avg similarity: {avg_sim:.2f}"
            if avg_ed:
                summary += f"   ED avg time: {sum(t['elapsed_sec'] for t in avg_ed) / len(avg_ed):.1f}s"
            if avg_llm:
                summary += f"   LLM avg time: {sum(t['elapsed_sec'] for t in avg_llm) / len(avg_llm):.1f}s"
            cv2.putText(canvas, summary, (24, canvas_h - 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN_BRIGHT, 1)

        cv2.putText(canvas, "q: back", (24, canvas_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN_DIM, 1)

        cv2.imshow("Study Session Detail", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyWindow("Study Session Detail")


def run_study_session_manager() -> None:
    """저장된 study 세션을 화면에서 조회/삭제 — 카메라 없이 정적 캔버스로 표시."""
    selected = 0
    canvas_h, canvas_w = 640, 960
    ROW_TOP = 80
    row_h = 30
    max_rows = (canvas_h - ROW_TOP - 60) // row_h
    hovered = -1
    _clicked = [None]

    cv2.namedWindow("Study Session Manager")

    def _on_mouse(event, x, y, flags, param):
        nonlocal hovered
        row = (y - ROW_TOP) // row_h
        sessions_now = _load_results()
        n = min(len(sessions_now), max_rows)
        hovered = row if 0 <= row < n else -1
        if 0 <= row < n:
            if event == cv2.EVENT_LBUTTONDOWN:
                _clicked[0] = ("select", row)
            elif event == cv2.EVENT_LBUTTONDBLCLK:
                _clicked[0] = ("detail", row)

    cv2.setMouseCallback("Study Session Manager", _on_mouse)

    while True:
        sessions = _load_results()
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        cv2.putText(canvas, "ED vs LLM Study — Session Manager", (24, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _GREEN_BRIGHT, 2)

        if not sessions:
            cv2.putText(canvas, "저장된 study 세션이 없습니다.", (24, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN_DIM, 2)
        else:
            selected = max(0, min(selected, len(sessions) - 1))
            visible = sessions[:max_rows]
            for i, session in enumerate(visible):
                y = ROW_TOP + i * row_h
                trials = session.get("trials", [])
                n = len(trials)
                avg_sim = sum(t["similarity_ratio"] for t in trials) / n if n else 0.0
                participant = session.get("participant") or "(미입력)"
                profile = session.get("cal_profile") or "Default"
                line = (f"[{i}] {session['timestamp']}  {participant}  cal:{profile}  "
                        f"{n}/{len(STUDY_SCHEDULE)} trials  avg sim {avg_sim:.2f}")
                is_sel = (i == selected)
                is_hov = (i == hovered) and not is_sel
                if is_sel:
                    cv2.rectangle(canvas, (16, y - 20), (canvas_w - 16, y + 8), (18, 48, 24), -1)
                elif is_hov:
                    cv2.rectangle(canvas, (16, y - 20), (canvas_w - 16, y + 8), (12, 30, 15), -1)
                color = _GREEN_BRIGHT if is_sel else (_GREEN_MID if is_hov else _GREEN_DIM)
                prefix = "> " if is_sel else "  "
                cv2.putText(canvas, prefix + line, (24, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            if len(sessions) > max_rows:
                cv2.putText(canvas, f"... and {len(sessions) - max_rows} more (not shown)",
                            (24, ROW_TOP + max_rows * row_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN_DIM, 1)

        cv2.putText(canvas, "w/s/arrows: move   Enter/dblclick: detail   d: delete   q: back",
                    (24, canvas_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN_DIM, 1)

        cv2.imshow("Study Session Manager", canvas)

        if _clicked[0] is not None:
            action, row = _clicked[0]
            _clicked[0] = None
            if 0 <= row < len(sessions):
                selected = row
                if action == "detail":
                    _run_study_detail(sessions[selected])
            continue

        key = cv2.waitKeyEx(30)
        klow = key & 0xFF

        if klow == ord("q"):
            break
        elif (klow == ord("w") or key in (63232, 2490368, 65362)) and sessions:
            selected = max(0, selected - 1)
        elif (klow == ord("s") or key in (63233, 2621440, 65364)) and sessions:
            selected = min(len(sessions) - 1, selected + 1)
        elif klow == 13 and sessions:
            _run_study_detail(sessions[selected])
        elif klow == ord("d") and sessions:
            delete_study_session(selected)
            selected = max(0, selected - 1)

    cv2.destroyWindow("Study Session Manager")

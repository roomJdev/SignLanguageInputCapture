"""OS-level keystroke injection via pynput.

인식된 수화 글자를 실제 키스트로크로 주입한다.
정적 심볼은 N프레임 연속 안정화 후 주입, 모션 심볼(J/Z)은 즉시 주입.

사용:
    injector = KeystrokeInjector()
    injector.enabled = True          # i 키로 토글
    injector.update("A")             # 매 프레임 호출 (정적 심볼)
    injector.inject_now("J")         # 즉시 주입 (모션 심볼)
"""

import time

try:
    from pynput.keyboard import Controller, Key
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False
    print("[injector] pynput 미설치 — 키스트로크 주입 비활성화")

try:
    import pyautogui
    pyautogui.PAUSE = 0
    _PYAUTOGUI_AVAILABLE = True
except ImportError:
    _PYAUTOGUI_AVAILABLE = False

# 같은 글자가 이 프레임 수만큼 연속으로 나와야 주입
_STABLE_FRAMES = 10
# 주입 후 같은 글자를 다시 주입하기까지 최소 대기 시간 (초)
_INJECT_COOLDOWN_SEC = 2.0


class KeystrokeInjector:
    def __init__(self):
        self.enabled: bool = False
        self._keyboard = Controller() if _PYNPUT_AVAILABLE else None
        self._stable_letter: str = ""
        self._stable_count: int = 0
        self._last_injected: str = ""
        self._last_inject_time: float = 0.0

    def stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def update(self, letter: str) -> str:
        """매 프레임 정적 심볼 인식 결과를 전달. 주입 발생 시 해당 글자 반환, 아니면 ''."""
        if not self.enabled:
            return ""
        if letter in ("?", ""):
            self._reset_stable()
            return ""

        if letter == self._stable_letter:
            self._stable_count += 1
        else:
            self._stable_letter = letter
            self._stable_count = 1

        if self._stable_count >= _STABLE_FRAMES:
            if self._can_inject(letter):
                if _PYNPUT_AVAILABLE:
                    self._press(letter.lower())
                self._record_inject(letter)
                self._reset_stable()
                return letter

        return ""

    def inject_string(self, s: str) -> None:
        """문자열을 순서대로 즉시 주입 (guess mode 단어 완성용).

        pyautogui.typewrite 로 글자 간 50ms 간격을 두어 드롭 방지.
        """
        if not self.enabled or not s:
            return
        if _PYAUTOGUI_AVAILABLE:
            pyautogui.typewrite(s, interval=0.05)
            print(f"[injector] inject_string: '{s}'")
        elif _PYNPUT_AVAILABLE:
            for ch in s:
                self._press("space" if ch == " " else ch)

    def inject_now(self, letter: str) -> str:
        """모션 심볼처럼 이미 확정된 글자를 즉시 주입. 주입 발생 시 해당 글자 반환, 아니면 ''."""
        if not self.enabled:
            return ""
        if not self._can_inject(letter):
            return ""
        if _PYNPUT_AVAILABLE:
            self._press(letter.lower())
        self._record_inject(letter)
        self._reset_stable()
        return letter

    def toggle(self) -> bool:
        """on/off 전환 후 현재 상태 반환."""
        self.enabled = not self.enabled
        self._reset_stable()
        print(f"[injector] keystroke injection {'ON' if self.enabled else 'OFF'}")
        return self.enabled

    # ------------------------------------------------------------------
    # 내부
    # ------------------------------------------------------------------

    def _can_inject(self, letter: str) -> bool:
        return time.time() - self._last_inject_time >= _INJECT_COOLDOWN_SEC

    def _press(self, char: str) -> None:
        try:
            if _PYAUTOGUI_AVAILABLE:
                key = "space" if char == "space" else char
                pyautogui.press(key)
            elif _PYNPUT_AVAILABLE:
                if char == "space":
                    self._keyboard.press(Key.space)
                    self._keyboard.release(Key.space)
                else:
                    self._keyboard.press(char)
                    self._keyboard.release(char)
            print(f"[injector] injected: '{char}'")
        except Exception as e:
            print(f"[injector] 주입 실패: {e}")

    def _record_inject(self, letter: str) -> None:
        self._last_injected = letter
        self._last_inject_time = time.time()

    def _reset_stable(self) -> None:
        self._stable_letter = ""
        self._stable_count = 0

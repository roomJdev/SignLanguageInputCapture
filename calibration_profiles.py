"""보정 프로필 관리 — 여러 사용자의 보정 데이터를 이름으로 저장/로드.

기본 파일(calibration_data.npy, motion_calibration_data.npy)은 기존과 동일하게 유지하고,
이름이 붙은 프로필을 cal_<name>.npy / motion_cal_<name>.npy 형태로 별도 저장한다.
calibration_profiles.json이 전체 프로필 목록을 관리한다.
"""

import json
import os
import shutil
import time

import numpy as np

PROFILES_PATH = "data_new0731/calibration_profiles.json"
DEFAULT_STATIC_PATH = "data_new0731/calibration_data.npy"
DEFAULT_MOTION_PATH = "data_new0731/motion_calibration_data.npy"
# 가공 전 원본 21관절 랜드마크 (calibration.py / motion_calibration.py가 생성)
DEFAULT_STATIC_LANDMARKS_PATH = "data_new0731/calibration_landmarks.npy"
DEFAULT_MOTION_LANDMARKS_PATH = "data_new0731/motion_calibration_landmarks.npy"
# 심볼별/반복별 캡처 화면 원본 비디오가 모이는 폴더
DEFAULT_STATIC_VIDEOS_DIR = "data_new0731/calibration_videos"
DEFAULT_MOTION_VIDEOS_DIR = "data_new0731/motion_calibration_videos"


# ---------------------------------------------------------------------------
# 인덱스 읽기/쓰기
# ---------------------------------------------------------------------------

def _load_index() -> list[dict]:
    if os.path.exists(PROFILES_PATH):
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_index(profiles: list[dict]) -> None:
    os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def list_profiles() -> list[dict]:
    """등록된 보정 프로필 목록 반환.

    각 항목: {name, static_path, motion_path, created_at}
    """
    return _load_index()


def copy_default_as_profile(name: str) -> bool:
    """현재 default 보정 파일을 이름 붙여 복사한 뒤 프로필로 등록.

    보정이 완료된 직후(calibration_data.npy가 최신 상태일 때) 호출한다.
    이미 같은 이름 프로필이 있으면 덮어쓴다.
    """
    safe = name.strip()
    if not safe or not os.path.exists(DEFAULT_STATIC_PATH):
        return False

    slug = safe.replace(" ", "_")
    static_dst = f"data_new0731/cal_{slug}.npy"
    motion_dst = f"data_new0731/motion_cal_{slug}.npy" if os.path.exists(DEFAULT_MOTION_PATH) else None

    os.makedirs(os.path.dirname(static_dst), exist_ok=True)
    shutil.copy2(DEFAULT_STATIC_PATH, static_dst)
    if motion_dst:
        shutil.copy2(DEFAULT_MOTION_PATH, motion_dst)

    # 사람이 읽을 수 있는 JSON 사본도 있으면 같이 복사 (save_calibration/save_motion_calibration이 생성)
    default_static_json = os.path.splitext(DEFAULT_STATIC_PATH)[0] + ".json"
    if os.path.exists(default_static_json):
        shutil.copy2(default_static_json, os.path.splitext(static_dst)[0] + ".json")
    if motion_dst:
        default_motion_json = os.path.splitext(DEFAULT_MOTION_PATH)[0] + ".json"
        if os.path.exists(default_motion_json):
            shutil.copy2(default_motion_json, os.path.splitext(motion_dst)[0] + ".json")

    # 가공 전 원본 21관절 랜드마크(.npy + .json)도 있으면 함께 복사
    if os.path.exists(DEFAULT_STATIC_LANDMARKS_PATH):
        static_landmarks_dst = f"data_new0731/cal_{slug}_landmarks.npy"
        shutil.copy2(DEFAULT_STATIC_LANDMARKS_PATH, static_landmarks_dst)
        default_static_landmarks_json = os.path.splitext(DEFAULT_STATIC_LANDMARKS_PATH)[0] + ".json"
        if os.path.exists(default_static_landmarks_json):
            shutil.copy2(default_static_landmarks_json, os.path.splitext(static_landmarks_dst)[0] + ".json")
    if motion_dst and os.path.exists(DEFAULT_MOTION_LANDMARKS_PATH):
        motion_landmarks_dst = f"data_new0731/motion_cal_{slug}_landmarks.npy"
        shutil.copy2(DEFAULT_MOTION_LANDMARKS_PATH, motion_landmarks_dst)
        default_motion_landmarks_json = os.path.splitext(DEFAULT_MOTION_LANDMARKS_PATH)[0] + ".json"
        if os.path.exists(default_motion_landmarks_json):
            shutil.copy2(default_motion_landmarks_json, os.path.splitext(motion_landmarks_dst)[0] + ".json")

    # 캡처 화면 원본 비디오 폴더도 있으면 통째로 복사
    if os.path.isdir(DEFAULT_STATIC_VIDEOS_DIR):
        shutil.copytree(DEFAULT_STATIC_VIDEOS_DIR, f"data_new0731/cal_{slug}_videos", dirs_exist_ok=True)
    if motion_dst and os.path.isdir(DEFAULT_MOTION_VIDEOS_DIR):
        shutil.copytree(DEFAULT_MOTION_VIDEOS_DIR, f"data_new0731/motion_cal_{slug}_videos", dirs_exist_ok=True)

    profiles = [p for p in _load_index() if p["name"] != safe]
    profiles.append({
        "name": safe,
        "static_path": static_dst,
        "motion_path": motion_dst,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _save_index(profiles)
    print(f"[profile] '{safe}' 보정 프로필 저장 완료")
    return True


def load_profile(name: str) -> tuple[dict | None, dict | None]:
    """이름으로 프로필을 로드해 (cal_data, motion_cal_data) 반환.

    없거나 파일이 손상된 경우 None 반환.
    """
    for p in _load_index():
        if p["name"] == name:
            cal_data = None
            motion_cal_data = None
            sp = p.get("static_path")
            if sp and os.path.exists(sp):
                cal_data = np.load(sp, allow_pickle=True).item()
            mp = p.get("motion_path")
            # 프로필 모션 파일이 없으면 default 파일로 폴백
            if not mp or not os.path.exists(mp):
                mp = DEFAULT_MOTION_PATH
            if os.path.exists(mp):
                from motion_calibration import _migrate_10d_to_2d
                motion_cal_data = np.load(mp, allow_pickle=True).item()
                sample = next(iter(motion_cal_data.values()), [None])[0]
                if sample is not None and sample.shape[-1] == 10:
                    motion_cal_data = _migrate_10d_to_2d(motion_cal_data)
            return cal_data, motion_cal_data
    return None, None


def delete_profile(name: str) -> bool:
    """프로필 인덱스에서 제거 (파일은 삭제하지 않음)."""
    profiles = _load_index()
    new = [p for p in profiles if p["name"] != name]
    if len(new) == len(profiles):
        return False
    _save_index(new)
    return True

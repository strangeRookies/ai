from __future__ import annotations

import re
from pathlib import Path
from typing import Final


SAFE_CAMERA_LOGIN_ID_RE: Final = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_safe_camera_login_id(camera_login_id: str) -> bool:
    return SAFE_CAMERA_LOGIN_ID_RE.fullmatch(camera_login_id) is not None


def resolved_video_pool(video_pool: Path, repo_root: Path) -> Path:
    return (video_pool if video_pool.is_absolute() else repo_root / video_pool).resolve()


def is_path_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def assigned_video_candidates(assigned_video_path: str, video_pool: Path, repo_root: Path) -> list[Path]:
    assigned = Path(assigned_video_path)
    if assigned.is_absolute():
        return [assigned]
    return [video_pool / assigned, repo_root / assigned]

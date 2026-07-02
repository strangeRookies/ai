from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from ai.camera_input_safety import assigned_video_candidates, is_path_under, resolved_video_pool
from ai.registered_cameras import RegisteredCamera


def scan_video_directory(directory_path: str) -> list[Path]:
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        print(f"[simulated-rtsp][error] Specified --video-dir is not a directory: {directory_path}", file=sys.stderr)
        sys.exit(1)

    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    video_files.sort(key=lambda x: x.name)
    return video_files


def resolve_assigned_video_path(camera: RegisteredCamera, video_pool: Path, repo_root: Path) -> Path | None:
    if not camera.assigned_video_path:
        return None

    pool_root = resolved_video_pool(video_pool, repo_root)
    for candidate in assigned_video_candidates(camera.assigned_video_path, pool_root, repo_root):
        if candidate.exists() and candidate.is_file() and is_path_under(candidate, pool_root):
            return candidate

    return None


def video_for_camera(camera: RegisteredCamera, video_files: list[Path], index: int, repo_root: Path) -> Path:
    video_pool = video_files[0].parent if video_files else repo_root
    assigned_video = resolve_assigned_video_path(camera, video_pool, repo_root)
    if assigned_video is not None:
        return assigned_video
    return video_files[index % len(video_files)]


def stable_video_index(camera_login_id: str, video_count: int) -> int:
    if video_count <= 0:
        raise ValueError("video_count must be positive")
    digest = hashlib.sha256(camera_login_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % video_count

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Final

from ai.camera_input_safety import assigned_video_candidates, is_path_under, resolved_video_pool
from ai.registered_cameras import RegisteredCamera


DEFAULT_STREAM_DOMAIN: Final = "outside"
CHROMAKEY_PATH_PATTERN: Final = re.compile(
    r"(indoor_chromakey|croki|크로마키|chroma|chromakey|green[_ -]?screen|studio|chm)",
    re.IGNORECASE,
)


def scan_video_directory(directory_path: str) -> list[Path]:
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        print(f"[simulated-rtsp][error] Specified --video-dir is not a directory: {directory_path}", file=sys.stderr)
        sys.exit(1)

    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    video_files.sort(key=lambda x: x.name)
    return video_files


def is_chromakey_video_path(video_path: Path) -> bool:
    return CHROMAKEY_PATH_PATTERN.search(str(video_path)) is not None


def filter_video_files(
    video_files: list[Path],
    domain: str | None,
    label: str | None,
    video_filter: str | None,
) -> tuple[list[Path], list[Path]]:
    matching: list[Path] = []
    excluded: list[Path] = []
    
    for p in video_files:
        path_lower = str(p.absolute()).lower()
        matched = True

        if is_chromakey_video_path(p):
            matched = False
        
        if domain and domain.lower() not in path_lower:
            matched = False
        if label and label.lower() not in path_lower:
            matched = False
        if video_filter and video_filter.lower() not in path_lower:
            matched = False
            
        if matched:
            matching.append(p)
        else:
            excluded.append(p)
            
    if not matching:
        filter_summary = f"domain={domain}, label={label}, video_filter={video_filter}"
        raise ValueError(
            f"No video files matched the filters ({filter_summary}) in the directory. "
            f"Total scanned: {len(video_files)}, Excluded: {len(excluded)}"
        )
        
    return matching, excluded


def estimate_video_metadata(video_path: Path) -> dict[str, str]:
    path_lower = str(video_path.absolute()).lower()
    
    # Domain estimation
    if "inside" in path_lower or "indoor" in path_lower:
        domain = "inside"
    elif "outside" in path_lower or "outdoor" in path_lower:
        domain = "outside"
    else:
        domain = "unknown"
        
    # Label estimation
    if "swoon" in path_lower or "swoom" in path_lower:
        label = "swoon"
    elif "fall" in path_lower:
        label = "fall"
    elif "fight" in path_lower:
        label = "fight"
    elif "assault" in path_lower:
        label = "assault"
    else:
        label = "unknown"
        
    # Source estimation (synthetic vs real)
    if "synthetic" in path_lower or "preview" in path_lower:
        source = "synthetic"
    else:
        source = "real"
        
    return {
        "domain": domain,
        "label": label,
        "source": source,
    }


def resolve_assigned_video_path(camera: RegisteredCamera, video_pool: Path, repo_root: Path) -> Path | None:
    if not camera.assigned_video_path:
        return None

    pool_root = resolved_video_pool(video_pool, repo_root)
    for candidate in assigned_video_candidates(camera.assigned_video_path, pool_root, repo_root):
        if candidate.exists() and candidate.is_file() and is_path_under(candidate, pool_root) and not is_chromakey_video_path(candidate):
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

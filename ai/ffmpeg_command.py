from __future__ import annotations

import os
from pathlib import Path

from ai.simulated_rtsp_publisher import build_ffmpeg_cmd


def build_ffmpeg_command(video_path: Path, rtsp_url: str) -> list[str]:
    ffmpeg_mode = os.environ.get("FFMPEG_MODE", "auto")
    return build_ffmpeg_cmd(video_path, rtsp_url, loop=True, ffmpeg_mode=ffmpeg_mode)

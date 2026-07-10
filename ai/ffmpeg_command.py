from __future__ import annotations

import os
from pathlib import Path

from ai.simulated_rtsp_publisher import build_ffmpeg_cmd


def build_ffmpeg_command(video_path: Path, rtsp_url: str, *, loop: bool = False) -> list[str]:
    """Build ffmpeg publish command for registered-camera SIMULATED_RTSP.

    Infinite ``-stream_loop -1`` is off by default for registration-type simulation
    so a finished clip ends cleanly and the worker lifecycle can respawn both
    ffmpeg + overlay. Pass ``loop=True`` only when an explicit continuous loop is required.
    Folder-based publisher uses ``build_ffmpeg_cmd`` directly and keeps its own loop flag.
    """
    ffmpeg_mode = os.environ.get("FFMPEG_MODE", "auto")
    return build_ffmpeg_cmd(video_path, rtsp_url, loop=loop, ffmpeg_mode=ffmpeg_mode)

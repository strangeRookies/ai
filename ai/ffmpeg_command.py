from __future__ import annotations

import os
from pathlib import Path


def build_ffmpeg_command(video_path: Path, rtsp_url: str) -> list[str]:
    ffmpeg_mode = os.environ.get("FFMPEG_MODE", "cpu").lower()
    command = ["ffmpeg", "-re", "-stream_loop", "-1", "-i", str(video_path), "-an"]
    if ffmpeg_mode == "copy":
        command.extend(["-c:v", "copy"])
    elif ffmpeg_mode == "nvenc":
        command.extend(["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "zerolatency"])
    else:
        command.extend(["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"])
    command.extend(["-f", "rtsp", rtsp_url])
    return command

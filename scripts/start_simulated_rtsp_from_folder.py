#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.simulated_rtsp_publisher import FFMPEG_MODE_CHOICES, build_ffmpeg_cmd
from ai.simulated_rtsp_runtime import run_simulated_rtsp_publisher
from ai.simulated_rtsp_sources import (
    DEFAULT_STREAM_DOMAIN,
    scan_video_directory,
    stable_video_index,
    video_for_camera as _video_for_camera,
)
from ai.registered_cameras import DEFAULT_BACKEND_BASE_URL, RegisteredCamera


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-scan a folder of videos and publish them as simulated RTSP streams based on active cameras."
    )
    parser.add_argument("--video-dir", required=True, help="Directory containing video files (mp4, avi, mov, mkv).")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_BASE_URL, help="Backend base URL.")
    parser.add_argument("--rtsp-host", default="127.0.0.1", help="RTSP server publish host.")
    parser.add_argument("--rtsp-port", type=int, default=8554, help="RTSP server publish port.")
    parser.add_argument("--poll-interval", type=int, default=30, help="Interval in seconds to poll the backend.")
    parser.add_argument("--loop", action="store_true", default=True, help="Indefinitely loop video playback.")
    parser.add_argument("--no-loop", action="store_false", dest="loop", help="Do not loop video playback.")
    parser.add_argument(
        "--ffmpeg-mode",
        choices=FFMPEG_MODE_CHOICES,
        default=os.environ.get("FFMPEG_MODE", "auto"),
        help="FFmpeg encoding mode. auto preflights NVENC, then falls back to copy/cpu on repeated failure.",
    )
    parser.add_argument(
        "--no-ffmpeg-fallback",
        action="store_true",
        help="Keep the requested FFmpeg mode even after repeated failures.",
    )
    parser.add_argument("--domain", default=os.environ.get("VIDEO_DOMAIN", DEFAULT_STREAM_DOMAIN), help="Domain to filter (e.g. inside, outside)")
    parser.add_argument("--label", default=os.environ.get("VIDEO_LABEL"), help="Label to filter (e.g. swoon, assault, fight)")
    parser.add_argument("--video-filter", default=os.environ.get("VIDEO_FILTER"), help="Sub-string to filter filenames (e.g. outside_swoon)")
    return parser.parse_args()


def video_for_camera(camera: RegisteredCamera, video_files: list[Path], index: int) -> Path:
    return _video_for_camera(camera, video_files, index, REPO_ROOT)


def main() -> None:
    run_simulated_rtsp_publisher(parse_arguments(), REPO_ROOT)


if __name__ == "__main__":
    main()

"""Tracker timing policy shared by RTSP and file workers."""
from __future__ import annotations

import math

DEFAULT_TRACKER_FPS = 30.0
MIN_VALID_TRACKER_FPS = 1.0
MAX_VALID_TRACKER_FPS = 240.0


def resolve_tracker_fps(source_fps: float | None, configured_fps: float | None) -> float:
    """Use a valid source rate; otherwise use configured rate then a safe default."""
    for candidate in (source_fps, configured_fps, DEFAULT_TRACKER_FPS):
        value = float(candidate or 0.0)
        if math.isfinite(value) and MIN_VALID_TRACKER_FPS <= value <= MAX_VALID_TRACKER_FPS:
            return value
    return DEFAULT_TRACKER_FPS

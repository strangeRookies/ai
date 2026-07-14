"""Tracker timebase policy shared by RTSP and file workers."""
from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_TRACKER_FPS = 30.0
MIN_VALID_TRACKER_FPS = 1.0
MAX_VALID_TRACKER_FPS = 240.0


def _valid_fps(value: float | None) -> float | None:
    """Return a finite bounded FPS value, or None at the input boundary."""
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric) and MIN_VALID_TRACKER_FPS <= numeric <= MAX_VALID_TRACKER_FPS:
        return numeric
    return None


def resolve_tracker_fps(source_fps: float | None, configured_fps: float | None) -> float:
    """Use a valid source rate; otherwise use configured rate then a safe default."""
    return _valid_fps(source_fps) or _valid_fps(configured_fps) or DEFAULT_TRACKER_FPS


@dataclass
class TrackerUpdateFpsEstimator:
    """Mutable cadence accumulator; updates are applied only after stable audit windows."""

    source_fps: float | None
    configured_fps: float | None
    latest_frame_mode: bool
    min_samples: int = 15
    relative_difference: float = 0.20
    stable_windows_required: int = 3
    ema_alpha: float = 0.35
    sample_count: int = 0
    stable_windows: int = 0
    measured_fps: float | None = None
    _last_timestamp: float | None = None
    _window_candidate: float | None = None

    def __post_init__(self) -> None:
        self.source_fps = _valid_fps(self.source_fps)
        self.configured_fps = _valid_fps(self.configured_fps) or DEFAULT_TRACKER_FPS

    @property
    def fallback_fps(self) -> float:
        return self.source_fps or self.configured_fps or DEFAULT_TRACKER_FPS

    @property
    def state(self) -> str:
        if not self.latest_frame_mode:
            return "source_fps" if self.source_fps is not None else "source_fallback"
        if self.sample_count < self.min_samples or self.measured_fps is None:
            return "warming_up"
        return "measured_tracker_update_fps"

    @property
    def effective_fps(self) -> float:
        if self.state == "measured_tracker_update_fps" and self.measured_fps is not None:
            return self.measured_fps
        return self.fallback_fps

    @property
    def source(self) -> str:
        if self.state == "measured_tracker_update_fps":
            return "measured_tracker_update_fps"
        if self.source_fps is not None:
            return "source_fps"
        return "configured_fallback"

    def observe_update(self, timestamp: float | None) -> None:
        """Observe tracker-update timestamp without mutating tracker configuration."""
        if not self.latest_frame_mode:
            return
        try:
            current = float(timestamp) if timestamp is not None else None
        except (TypeError, ValueError):
            return
        if current is None or not math.isfinite(current):
            return
        previous = self._last_timestamp
        self._last_timestamp = current
        if previous is None:
            return
        elapsed = current - previous
        if elapsed <= 0.0:
            return
        instantaneous = _valid_fps(1.0 / elapsed)
        if instantaneous is None:
            return
        self.sample_count += 1
        if self.measured_fps is None:
            self.measured_fps = instantaneous
        else:
            self.measured_fps = self.ema_alpha * instantaneous + (1.0 - self.ema_alpha) * self.measured_fps

    def record_window(self, tracker_update_fps: float | None) -> None:
        """Require consecutive cadence windows before allowing a reconfiguration."""
        if not self.latest_frame_mode or self.sample_count < self.min_samples:
            return
        observed = _valid_fps(tracker_update_fps)
        if observed is None or self.measured_fps is None:
            self.stable_windows = 0
            self._window_candidate = None
            return
        ratio = abs(observed - self.measured_fps) / max(self.measured_fps, MIN_VALID_TRACKER_FPS)
        if ratio <= self.relative_difference:
            if self._window_candidate is None:
                self._window_candidate = observed
                self.stable_windows = 1
            else:
                candidate_ratio = abs(observed - self._window_candidate) / max(self._window_candidate, MIN_VALID_TRACKER_FPS)
                if candidate_ratio <= self.relative_difference:
                    self.stable_windows += 1
                else:
                    self._window_candidate = observed
                    self.stable_windows = 1
            self.measured_fps = self.ema_alpha * observed + (1.0 - self.ema_alpha) * self.measured_fps
        else:
            self._window_candidate = observed
            self.stable_windows = 1

    def should_apply(self, configured_fps: float | None) -> bool:
        """Return true only for a persistent material difference from configured FPS."""
        current = _valid_fps(configured_fps) or self.fallback_fps
        return (
            self.state == "measured_tracker_update_fps"
            and self.stable_windows >= self.stable_windows_required
            and abs(self.effective_fps - current) / max(current, MIN_VALID_TRACKER_FPS) >= self.relative_difference
        )

    def reset(self, source_fps: float | None = None) -> None:
        """Start a new video session without carrying cadence evidence across reconnects."""
        self.source_fps = _valid_fps(source_fps) or self.source_fps
        self.sample_count = 0
        self.stable_windows = 0
        self.measured_fps = None
        self._last_timestamp = None
        self._window_candidate = None
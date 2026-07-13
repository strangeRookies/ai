"""Separate source/capture/analysis/tracker/mjpeg FPS counters for one camera worker."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


def fps_audit_enabled() -> bool:
    return os.getenv("FPS_AUDIT", "true").lower() in {"1", "true", "yes", "on"}


@dataclass
class FpsAuditWindow:
    camera_login_id: str
    stream_run_id: str = ""
    window_sec: float = 10.0
    source_fps: float | None = None
    tracker_config_frame_rate: float | None = None
    mjpeg_target_fps: float | None = None

    _window_started: float = field(default_factory=time.time)
    captured_frames: int = 0
    analyzed_frames: int = 0
    detector_frames: int = 0
    tracker_update_frames: int = 0
    mjpeg_emitted_frames_start: int = 0
    dropped_before_analysis: int = 0
    queue_replaced_frames: int = 0
    detector_skipped_frames: int = 0

    def note_capture(self, *, queue_dropped: int = 0) -> None:
        self.captured_frames += 1
        self.queue_replaced_frames += max(0, int(queue_dropped))

    def note_analysis(self) -> None:
        self.analyzed_frames += 1

    def note_detector(self) -> None:
        self.detector_frames += 1

    def note_tracker_update(self) -> None:
        self.tracker_update_frames += 1

    def note_detector_skip(self) -> None:
        self.detector_skipped_frames += 1

    def note_drop_before_analysis(self, n: int = 1) -> None:
        self.dropped_before_analysis += max(0, int(n))

    def maybe_emit(self, *, mjpeg_frame_count: int | None = None) -> dict | None:
        now = time.time()
        elapsed = now - self._window_started
        if elapsed < self.window_sec:
            return None
        mjpeg_emitted = 0
        if mjpeg_frame_count is not None:
            mjpeg_emitted = max(0, int(mjpeg_frame_count) - int(self.mjpeg_emitted_frames_start))
        record = {
            "cameraLoginId": self.camera_login_id,
            "streamRunId": self.stream_run_id,
            "windowSec": round(elapsed, 3),
            "sourceFps": self.source_fps,
            "capturedFps": round(self.captured_frames / elapsed, 3),
            "analysisFps": round(self.analyzed_frames / elapsed, 3),
            "detectorFps": round(self.detector_frames / elapsed, 3),
            "trackerUpdateFps": round(self.tracker_update_frames / elapsed, 3),
            "mjpegEmitFps": round(mjpeg_emitted / elapsed, 3) if mjpeg_frame_count is not None else None,
            "mjpegTargetFps": self.mjpeg_target_fps,
            "trackerConfigFrameRate": self.tracker_config_frame_rate,
            "capturedFrames": self.captured_frames,
            "analyzedFrames": self.analyzed_frames,
            "detectorFrames": self.detector_frames,
            "trackerUpdateFrames": self.tracker_update_frames,
            "mjpegEmittedFrames": mjpeg_emitted if mjpeg_frame_count is not None else None,
            "droppedBeforeAnalysis": self.dropped_before_analysis,
            "queueReplacedFrames": self.queue_replaced_frames,
            "detectorSkippedFrames": self.detector_skipped_frames,
        }
        # reset window
        self._window_started = now
        self.captured_frames = 0
        self.analyzed_frames = 0
        self.detector_frames = 0
        self.tracker_update_frames = 0
        self.dropped_before_analysis = 0
        self.queue_replaced_frames = 0
        self.detector_skipped_frames = 0
        if mjpeg_frame_count is not None:
            self.mjpeg_emitted_frames_start = int(mjpeg_frame_count)
        return record


def log_fps_audit(record: dict) -> None:
    if not fps_audit_enabled():
        return
    lines = ["[fps-audit]"]
    for key, value in record.items():
        lines.append(f"{key}={value}")
    print("\n".join(lines), flush=True)

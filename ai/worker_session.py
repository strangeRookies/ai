"""Worker session identity and full state reset on video/source boundaries.

Synthetic/offline-safe: does not require RTSP, GPU, or real video I/O.
When frameId restarts on a new source, uniqueness is preserved via streamRunId+frameId.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable


class SessionResetReason(StrEnum):
    VIDEO_EOF = "video_eof"
    SOURCE_CHANGE = "source_change"
    STREAM_RECONNECT = "stream_reconnect"
    MANUAL = "manual"
    WORKER_START = "worker_start"


@dataclass
class SessionIdentity:
    camera_login_id: str
    stream_run_id: str
    worker_run_id: str
    frame_id: int = 0

    def evidence_frame_key(self) -> str:
        """Unique across cameras and stream restarts (frameId may restart)."""
        return f"{self.camera_login_id}:{self.stream_run_id}:{self.frame_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cameraLoginId": self.camera_login_id,
            "streamRunId": self.stream_run_id,
            "workerRunId": self.worker_run_id,
            "frameId": self.frame_id,
            "evidenceFrameKey": self.evidence_frame_key(),
        }


def new_stream_run_id() -> str:
    return f"stream-{uuid.uuid4().hex[:12]}"


def new_worker_run_id() -> str:
    return f"worker-{uuid.uuid4().hex[:12]}"


@dataclass
class WorkerSession:
    """Holds per-camera runtime state that must not leak across video sources."""

    camera_login_id: str
    worker_run_id: str = field(default_factory=new_worker_run_id)
    stream_run_id: str = field(default_factory=new_stream_run_id)
    frame_id: int = 0
    reset_count: int = 0
    last_reset_reason: str | None = None
    reset_history: list[dict[str, Any]] = field(default_factory=list)

    tracker: Any = None
    sequence_buffers: Any = None
    fall_faint_processor: Any = None
    fall_state_machine: Any = None
    overlay_state: dict[str, Any] = field(default_factory=dict)
    hazard_state: dict[str, Any] = field(default_factory=dict)
    displayed_track_ids: dict[str, Any] = field(default_factory=dict)
    pending_events: list[Any] = field(default_factory=list)
    frame_sync_state: dict[str, Any] = field(default_factory=dict)

    # Optional hooks for components that expose reset()/clear()/reset_all()
    extra_resetters: list[Callable[[], None]] = field(default_factory=list)

    def identity(self) -> SessionIdentity:
        return SessionIdentity(
            camera_login_id=self.camera_login_id,
            stream_run_id=self.stream_run_id,
            worker_run_id=self.worker_run_id,
            frame_id=self.frame_id,
        )

    def next_frame_id(self) -> int:
        self.frame_id += 1
        return self.frame_id

    def bind_components(
        self,
        *,
        tracker: Any = None,
        sequence_buffers: Any = None,
        fall_faint_processor: Any = None,
        fall_state_machine: Any = None,
    ) -> None:
        if tracker is not None:
            self.tracker = tracker
        if sequence_buffers is not None:
            self.sequence_buffers = sequence_buffers
        if fall_faint_processor is not None:
            self.fall_faint_processor = fall_faint_processor
        if fall_state_machine is not None:
            self.fall_state_machine = fall_state_machine

    def reset(self, reason: SessionResetReason | str) -> dict[str, Any]:
        """Full session reset for EOF / source change / reconnect."""
        reason_value = str(reason)
        previous = self.identity().to_dict()
        previous_stream = self.stream_run_id

        tracker_info = None
        if self.tracker is not None and hasattr(self.tracker, "reset"):
            tracker_info = self.tracker.reset(reason=reason_value)
        elif self.tracker is not None and hasattr(self.tracker, "_tracks"):
            self.tracker._tracks = {}
            if hasattr(self.tracker, "_next_track_id"):
                self.tracker._next_track_id = 1
            if hasattr(self.tracker, "_frame_index"):
                self.tracker._frame_index = 0

        if self.sequence_buffers is not None:
            if hasattr(self.sequence_buffers, "clear"):
                self.sequence_buffers.clear()
            elif hasattr(self.sequence_buffers, "_buffers"):
                self.sequence_buffers._buffers = {}

        if self.fall_faint_processor is not None and hasattr(self.fall_faint_processor, "reset"):
            self.fall_faint_processor.reset()

        if self.fall_state_machine is not None and hasattr(self.fall_state_machine, "reset_all"):
            self.fall_state_machine.reset_all()

        for resetter in list(self.extra_resetters):
            try:
                resetter()
            except Exception:
                pass

        self.overlay_state = {}
        self.hazard_state = {}
        self.displayed_track_ids = {}
        self.pending_events = []
        self.frame_sync_state = {}

        self.stream_run_id = new_stream_run_id()
        self.frame_id = 0
        self.reset_count += 1
        self.last_reset_reason = reason_value
        record = {
            "reason": reason_value,
            "resetCount": self.reset_count,
            "previousStreamRunId": previous_stream,
            "newStreamRunId": self.stream_run_id,
            "previousIdentity": previous,
            "tracker": tracker_info,
            "ts": time.time(),
        }
        self.reset_history.append(record)
        return record

    def snapshot(self) -> dict[str, Any]:
        return {
            "identity": self.identity().to_dict(),
            "resetCount": self.reset_count,
            "lastResetReason": self.last_reset_reason,
            "activeTracks": (
                len(getattr(self.tracker, "_tracks", {}) or {})
                if self.tracker is not None
                else 0
            ),
            "pendingEventCount": len(self.pending_events),
            "overlayKeys": sorted(self.overlay_state.keys()),
            "hazardKeys": sorted(self.hazard_state.keys()),
        }


def begin_session(camera_login_id: str, **components: Any) -> WorkerSession:
    session = WorkerSession(camera_login_id=str(camera_login_id))
    session.bind_components(**components)
    session.reset(SessionResetReason.WORKER_START)
    return session

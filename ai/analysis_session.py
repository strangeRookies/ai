"""Unified analysis-session reset, metrics flush, and session identity for live workers.

workerRunId: process/worker lifetime identity (survives RTSP reconnect).
streamRunId: continuous analysis stream identity (new after EOF / source change / reconnect / large gap).
session_generation: monotonic int stamped on FramePacket; inference drops lower generations.

Reset policy (single path — do not recreate parallel stacks):
- VIDEO_EOF (mid-run next file): flush metrics → clear analysis state → new streamRunId
- SOURCE_CHANGED: same, after capture close/open at call site
- RTSP_RECONNECTED (or large frame/time gap proxy): same; keep workerRunId
- WORKER_RESTARTED: new process → new workerRunId via begin_session
- Worker process exit: finalize_for_exit() flushes only (no empty next-session artifact)

Metrics artifacts (session end/reset):
  runs/tracking_metrics/{cameraLoginId}/{streamRunId}/
    session-summary.json
    track-lifecycle.csv
    match-decisions.csv

Save failures warn only; inference continues. Flush can run on a background thread
so the inference loop is not blocked by disk I/O.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ai.tracking_runtime_metrics import TrackingRuntimeMetrics
from ai.worker_session import (
    SessionResetReason,
    WorkerSession,
    begin_session,
)

logger = logging.getLogger(__name__)

DEFAULT_METRICS_ROOT = Path("runs") / "tracking_metrics"

# Gap reasons used by serve_ai_overlay inference loop (legacy strings).
GAP_REASON_MAP = {
    "FRAME_ID_RESET": SessionResetReason.STREAM_RECONNECT,
    "LARGE_TIME_GAP": SessionResetReason.STREAM_RECONNECT,
    "LARGE_FRAME_GAP": SessionResetReason.STREAM_RECONNECT,
    "VIDEO_EOF": SessionResetReason.VIDEO_EOF,
    "STREAM_ENDED": SessionResetReason.VIDEO_EOF,
    "SOURCE_CHANGED": SessionResetReason.SOURCE_CHANGE,
    "SOURCE_CHANGE": SessionResetReason.SOURCE_CHANGE,
    "RTSP_RECONNECTED": SessionResetReason.STREAM_RECONNECT,
    "STREAM_RECONNECT": SessionResetReason.STREAM_RECONNECT,
    "WORKER_RESTARTED": SessionResetReason.WORKER_START,
    "MANUAL_RESET": SessionResetReason.MANUAL,
    "MANUAL": SessionResetReason.MANUAL,
}


def map_boundary_reason(reason: str | SessionResetReason) -> SessionResetReason:
    if isinstance(reason, SessionResetReason):
        return reason
    key = str(reason or "").strip()
    if key in GAP_REASON_MAP:
        return GAP_REASON_MAP[key]
    upper = key.upper()
    if upper in GAP_REASON_MAP:
        return GAP_REASON_MAP[upper]
    lower = key.lower()
    for item in SessionResetReason:
        if item.value == lower or item.name.lower() == lower:
            return item
    return SessionResetReason.MANUAL


def metrics_dir_for(camera_login_id: str, stream_run_id: str, *, root: str | Path = DEFAULT_METRICS_ROOT) -> Path:
    """Per-camera / per-stream isolation — paths do not collide across processes."""
    safe_cam = str(camera_login_id or "unknown").replace("/", "_").replace("\\", "_")
    safe_stream = str(stream_run_id or "unknown").replace("/", "_").replace("\\", "_")
    # Include pid so parallel process runs for same camera (restart) do not clobber mid-write temps.
    return Path(root) / safe_cam / f"pid-{os.getpid()}" / safe_stream


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write via temp file + os.replace (atomic on same filesystem)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


@dataclass
class AnalysisWorkerRuntime:
    """Worker-facing facade: session identity + metrics + unified reset + stale guards."""

    session: WorkerSession
    metrics: TrackingRuntimeMetrics = field(default_factory=TrackingRuntimeMetrics)
    metrics_root: Path = field(default_factory=lambda: Path(DEFAULT_METRICS_ROOT))
    lifecycle_rows: list[dict[str, Any]] = field(default_factory=list)
    decision_rows: list[dict[str, Any]] = field(default_factory=list)
    metrics_save_failures: int = 0
    metrics_flush_failed_total: int = 0
    metrics_flush_duration_ms: float = 0.0
    stale_packet_dropped_total: int = 0
    stale_event_dropped_total: int = 0
    queue_cleared_frame_count: int = 0
    session_reset_failed_total: int = 0
    session_generation: int = 1
    _flushed_stream_ids: set[str] = field(default_factory=set)
    _metrics_ingested_this_frame: bool = False
    _reset_lock: threading.Lock = field(default_factory=threading.Lock)
    _stamp_lock: threading.Lock = field(default_factory=threading.Lock)
    _last_reset_stream_id: str | None = None
    _last_reset_reason: str | None = None
    _last_reset_mono: float = 0.0
    _finalized: bool = False
    _flush_thread: threading.Thread | None = None

    # Optional loosely-held components reset via duck-typing
    display_id_mapper: Any = None
    overlay_publish_state: Any = None
    exit_post_processor: Any = None
    hazard_post_processor: Any = None
    track_selector: Any = None
    crop_sequence_buffers: Any = None
    incident_pipeline: Any = None
    last_incident_closures: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def start(
        cls,
        camera_login_id: str,
        *,
        metrics_root: str | Path = DEFAULT_METRICS_ROOT,
        **components: Any,
    ) -> "AnalysisWorkerRuntime":
        session = begin_session(str(camera_login_id), **{
            k: v
            for k, v in components.items()
            if k in {"tracker", "sequence_buffers", "fall_faint_processor", "fall_state_machine"}
        })
        runtime = cls(session=session, metrics_root=Path(metrics_root))
        runtime.bind_optional(**components)
        return runtime

    def bind_optional(self, **components: Any) -> None:
        for key in (
            "display_id_mapper",
            "overlay_publish_state",
            "exit_post_processor",
            "hazard_post_processor",
            "track_selector",
            "crop_sequence_buffers",
            "incident_pipeline",
        ):
            if key in components and components[key] is not None:
                setattr(self, key, components[key])
        bind_keys = {
            k: components[k]
            for k in ("tracker", "sequence_buffers", "fall_faint_processor", "fall_state_machine")
            if k in components and components[k] is not None
        }
        if bind_keys:
            self.session.bind_components(**bind_keys)

    def stamp_snapshot(self) -> dict[str, Any]:
        """Thread-safe view of generation/stream for reader stamping."""
        with self._stamp_lock:
            return {
                "session_generation": int(self.session_generation),
                "stream_run_id": self.session.stream_run_id,
                "worker_run_id": self.session.worker_run_id,
            }

    def identity_fields(self) -> dict[str, Any]:
        ident = self.session.identity()
        return {
            "workerRunId": ident.worker_run_id,
            "streamRunId": ident.stream_run_id,
            "frameId": ident.frame_id,
            "evidenceFrameKey": ident.evidence_frame_key(),
            "cameraLoginId": ident.camera_login_id,
            "resetCount": self.session.reset_count,
            "lastResetReason": self.session.last_reset_reason,
            "sessionGeneration": self.session_generation,
        }

    def advance_frame(self) -> dict[str, Any]:
        self.session.next_frame_id()
        self._metrics_ingested_this_frame = False
        return self.identity_fields()

    def accept_packet(self, packet: Any) -> bool:
        """Return False if packet belongs to an older session generation / stream."""
        if packet is None:
            return False
        pkt_gen = int(getattr(packet, "session_generation", 0) or 0)
        pkt_stream = getattr(packet, "stream_run_id", None)
        cur_gen = int(self.session_generation)
        cur_stream = self.session.stream_run_id
        # Untagged packets (legacy tests): accept only before any boundary (gen still 1).
        if pkt_gen == 0 and pkt_stream is None:
            if cur_gen <= 1:
                return True
            self.stale_packet_dropped_total += 1
            return False
        if pkt_gen != cur_gen:
            self.stale_packet_dropped_total += 1
            return False
        if pkt_stream is not None and str(pkt_stream) != str(cur_stream):
            self.stale_packet_dropped_total += 1
            return False
        return True

    def should_publish(self, *, capture_generation: int | None, capture_stream_run_id: str | None = None) -> bool:
        """Block late publish if session rolled since the analysis that produced the event."""
        if self._finalized:
            self.stale_event_dropped_total += 1
            return False
        if capture_generation is None or int(capture_generation) <= 0:
            # Legacy untagged path — do not drop solely on generation.
            if capture_stream_run_id is not None and str(capture_stream_run_id) != str(self.session.stream_run_id):
                self.stale_event_dropped_total += 1
                return False
            return True
        if int(capture_generation) != int(self.session_generation):
            self.stale_event_dropped_total += 1
            return False
        if capture_stream_run_id is not None and str(capture_stream_run_id) != str(self.session.stream_run_id):
            self.stale_event_dropped_total += 1
            return False
        return True

    def enrich_payload(
        self,
        payload: dict[str, Any] | None,
        *,
        capture_generation: int | None = None,
        capture_stream_run_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Attach optional session identifiers. Returns None if stale for publish."""
        if capture_generation is not None or capture_stream_run_id is not None:
            if not self.should_publish(
                capture_generation=capture_generation,
                capture_stream_run_id=capture_stream_run_id,
            ):
                return None
        out = dict(payload or {})
        fields = self.identity_fields()
        for key in ("workerRunId", "streamRunId", "evidenceFrameKey", "sessionGeneration"):
            out.setdefault(key, fields[key] if key != "sessionGeneration" else fields["sessionGeneration"])
        if out.get("frameId") is None and out.get("frame_id") is None:
            out["frameId"] = fields["frameId"]
        return out

    def note_tracker_events(self, tracker: Any, *, now: float | None = None) -> None:
        """Single ingest point for tracker last_events (no double-count per frame)."""
        if self._metrics_ingested_this_frame:
            return
        try:
            events = list(getattr(tracker, "last_events", None) or [])
            self.metrics.ingest_tracker_events(events, now=now)
            active = len(getattr(tracker, "_tracks", {}) or {})
            if not active and hasattr(tracker, "diagnostics"):
                try:
                    active = int((tracker.diagnostics() or {}).get("active_tracks") or 0)
                except Exception:
                    active = 0
            self.metrics.observe_active(active)
            for event in events:
                kind = event.get("event")
                row = {
                    "streamRunId": self.session.stream_run_id,
                    "sessionGeneration": self.session_generation,
                    "frameId": self.session.frame_id,
                    "event": kind,
                    "reason": event.get("reason"),
                    "trackId": event.get("trackId"),
                    "iou": event.get("iou"),
                    "centerRatio": event.get("centerRatio"),
                }
                self.decision_rows.append(row)
                if kind in {"new_track", "lost", "match", "id_switch_like", "filter"}:
                    self.lifecycle_rows.append(row)
            self._metrics_ingested_this_frame = True
        except Exception as exc:
            logger.warning("note_tracker_events failed (non-fatal): %s", exc)

    def flush_metrics(self, *, force: bool = False, blocking: bool = True) -> dict[str, Path]:
        """Write session metrics artifacts. Idempotent per streamRunId unless force.

        blocking=False snapshots and writes on a daemon thread (inference-safe).
        """
        stream_id = self.session.stream_run_id
        if self._finalized and not force and stream_id in self._flushed_stream_ids:
            return {}
        if not force and stream_id in self._flushed_stream_ids:
            return {}

        # Snapshot under lock so background write is consistent
        with self._reset_lock:
            if not force and stream_id in self._flushed_stream_ids:
                return {}
            summary = self.metrics.summary()
            summary.update(
                {
                    "cameraLoginId": self.session.camera_login_id,
                    "workerRunId": self.session.worker_run_id,
                    "streamRunId": stream_id,
                    "sessionGeneration": self.session_generation,
                    "resetCount": self.session.reset_count,
                    "lastResetReason": self.session.last_reset_reason,
                    "metricsSaveFailures": self.metrics_save_failures,
                    "metrics_flush_failed_total": self.metrics_flush_failed_total,
                    "stale_packet_dropped_total": self.stale_packet_dropped_total,
                    "stale_event_dropped_total": self.stale_event_dropped_total,
                    "queue_cleared_frame_count": self.queue_cleared_frame_count,
                    "session_reset_failed_total": self.session_reset_failed_total,
                }
            )
            lifecycle = list(self.lifecycle_rows)
            decisions = list(self.decision_rows)
            out_dir = metrics_dir_for(self.session.camera_login_id, stream_id, root=self.metrics_root)
            # Mark flushed early so double-reset does not double-write
            self._flushed_stream_ids.add(stream_id)

        def _write() -> dict[str, Path]:
            paths: dict[str, Path] = {}
            started = time.perf_counter()
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                summary_path = out_dir / "session-summary.json"
                atomic_write_text(
                    summary_path,
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                )
                paths["session-summary"] = summary_path
                paths["track-lifecycle"] = self._write_csv_atomic(out_dir / "track-lifecycle.csv", lifecycle)
                paths["match-decisions"] = self._write_csv_atomic(out_dir / "match-decisions.csv", decisions)
            except Exception as exc:
                self.metrics_save_failures += 1
                self.metrics_flush_failed_total += 1
                # allow retry on force
                self._flushed_stream_ids.discard(stream_id)
                logger.warning(
                    "tracking metrics flush failed camera=%s stream=%s: %s",
                    self.session.camera_login_id,
                    stream_id,
                    exc,
                )
            finally:
                self.metrics_flush_duration_ms = (time.perf_counter() - started) * 1000.0
            return paths

        if blocking:
            return _write()

        thread = threading.Thread(target=_write, name="analysis-metrics-flush", daemon=True)
        self._flush_thread = thread
        thread.start()
        return {}

    def wait_for_flush(self, timeout_sec: float = 5.0) -> bool:
        """Join background metrics flush thread. Returns True if completed or none pending."""
        thread = self._flush_thread
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout=max(0.0, float(timeout_sec)))
        return not thread.is_alive()

    def finalize_for_exit(self, *, blocking: bool = True, flush_timeout_sec: float = 5.0) -> dict[str, Path]:
        """Flush current session metrics without opening a new empty stream session.

        On exit paths (SIGTERM / KeyboardInterrupt handlers), prefer blocking=True or
        wait_for_flush so async writes complete within timeout.
        """
        if blocking:
            paths = self.flush_metrics(blocking=True)
        else:
            paths = self.flush_metrics(blocking=False)
            finished = self.wait_for_flush(flush_timeout_sec)
            if not finished:
                logger.warning(
                    "metrics flush timeout camera=%s stream=%s timeout_sec=%s",
                    self.session.camera_login_id,
                    self.session.stream_run_id,
                    flush_timeout_sec,
                )
                self.metrics_flush_failed_total += 1
        self._finalized = True
        return paths

    def multi_cam_instrumentation(self, *, queue: Any = None, pid: int | None = None) -> dict[str, Any]:
        """Fields for multi-camera ops dashboards (no performance claims)."""
        qstats = queue.stats() if queue is not None and hasattr(queue, "stats") else {}
        return {
            "camera_login_id": self.session.camera_login_id,
            "worker_run_id": self.session.worker_run_id,
            "stream_run_id": self.session.stream_run_id,
            "pid": int(pid if pid is not None else os.getpid()),
            "stale_packet_dropped_total": self.stale_packet_dropped_total,
            "metrics_flush_duration_ms": self.metrics_flush_duration_ms,
            "metrics_flush_failed_total": self.metrics_flush_failed_total,
            "queue_size": qstats.get("queue_size"),
            "queue_capacity": qstats.get("queue_capacity"),
            "queue_overflow_drop_total": qstats.get("queue_overflow_drop_total"),
            "oldest_packet_age_ms": qstats.get("oldest_packet_age_ms"),
            **self.safety_counters(),
        }

    def clear_frame_queue(self, queue: Any) -> int:
        if queue is None:
            return 0
        try:
            if hasattr(queue, "clear"):
                cleared = int(queue.clear())
            else:
                cleared = 0
            self.queue_cleared_frame_count += cleared
            return cleared
        except Exception as exc:
            logger.warning("queue clear failed: %s", exc)
            return 0

    def reset_analysis_session(
        self,
        reason: str | SessionResetReason,
        *,
        tracker_factory: Callable[[], Any] | None = None,
        sequence_factory: Callable[[], Any] | None = None,
        fall_faint_factory: Callable[[], Any] | None = None,
        crop_sequence_factory: Callable[[], Any] | None = None,
        display_id_mapper_factory: Callable[[], Any] | None = None,
        overlay_publish_state_factory: Callable[[], Any] | None = None,
        exit_post_processor_factory: Callable[[], Any] | None = None,
        hazard_post_processor_factory: Callable[[], Any] | None = None,
        skip_flush: bool = False,
        frame_queue: Any = None,
        open_new_stream: bool = True,
        flush_blocking: bool = True,
    ) -> dict[str, Any]:
        """Flush metrics, clear bound state, optionally issue new streamRunId.

        Concurrent/duplicate resets for the same stream are coalesced (idempotent).
        open_new_stream=False is for exit paths that must not create empty artifacts.
        Live workers may pass flush_blocking=False to avoid stalling the loop.
        """
        mapped = map_boundary_reason(reason)
        with self._reset_lock:
            previous_stream = self.session.stream_run_id
            previous_worker = self.session.worker_run_id
            previous_gen = self.session_generation
            now_mono = time.monotonic()
            # Coalesce burst resets (double reconnect / double gap) within 250ms same reason.
            if (
                open_new_stream
                and self._last_reset_reason == str(mapped)
                and (now_mono - self._last_reset_mono) < 0.25
            ):
                cleared = self.clear_frame_queue(frame_queue)
                return {
                    "reason": str(mapped),
                    "coalesced": True,
                    "previousStreamRunId": previous_stream,
                    "newStreamRunId": self.session.stream_run_id,
                    "sessionGeneration": self.session_generation,
                    "queueCleared": cleared,
                    "workerRunId": self.session.worker_run_id,
                    "mappedReason": str(mapped),
                }

        if not skip_flush and not self._finalized:
            self.flush_metrics(blocking=flush_blocking)

        # Close open incidents before rolling stream identity (reachable terminal policy).
        incident_closures: list[dict[str, Any]] = []
        if self.incident_pipeline is not None and hasattr(self.incident_pipeline, "on_stream_reset"):
            try:
                incident_closures = list(
                    self.incident_pipeline.on_stream_reset(
                        reason=str(mapped),
                        camera_login_id=self.session.camera_login_id,
                    )
                    or []
                )
                self.last_incident_closures = incident_closures
            except Exception as exc:
                logger.warning("incident on_stream_reset failed: %s", exc)

        if not open_new_stream:
            self._finalized = True
            cleared = self.clear_frame_queue(frame_queue)
            return {
                "reason": str(mapped),
                "finalized": True,
                "previousStreamRunId": previous_stream,
                "newStreamRunId": previous_stream,
                "sessionGeneration": previous_gen,
                "queueCleared": cleared,
                "mappedReason": str(mapped),
                "workerRunId": previous_worker,
                "incidentClosures": incident_closures,
            }

        tracker_error = None
        sequence_error = None
        reset_failed = False

        try:
            if self.session.tracker is not None:
                if hasattr(self.session.tracker, "reset"):
                    self.session.tracker.reset(reason=str(mapped))
                elif tracker_factory is None and hasattr(self.session.tracker, "_tracks"):
                    self.session.tracker._tracks = {}
        except Exception as exc:
            tracker_error = exc
            reset_failed = True
            logger.error("tracker reset failed reason=%s: %s", mapped, exc)

        try:
            if self.session.sequence_buffers is not None and hasattr(self.session.sequence_buffers, "clear"):
                self.session.sequence_buffers.clear()
            if self.crop_sequence_buffers is not None and hasattr(self.crop_sequence_buffers, "clear"):
                self.crop_sequence_buffers.clear()
        except Exception as exc:
            sequence_error = exc
            reset_failed = True
            logger.error("sequence buffer clear failed reason=%s: %s", mapped, exc)

        for name, obj, method in (
            ("fall_faint_processor", self.session.fall_faint_processor, "reset"),
            ("fall_state_machine", self.session.fall_state_machine, "reset_all"),
            ("display_id_mapper", self.display_id_mapper, "reset"),
            ("exit_post_processor", self.exit_post_processor, "reset"),
            ("hazard_post_processor", self.hazard_post_processor, "reset"),
            ("track_selector", self.track_selector, "reset"),
        ):
            try:
                if obj is not None and hasattr(obj, method):
                    getattr(obj, method)()
                elif obj is not None and name in {"exit_post_processor", "hazard_post_processor"}:
                    for attr in ("_consecutive_by_track", "_last_event_time"):
                        if hasattr(obj, attr):
                            getattr(obj, attr).clear()
            except Exception as exc:
                logger.warning("optional reset %s failed: %s", name, exc)

        if self.overlay_publish_state is not None:
            try:
                if hasattr(self.overlay_publish_state, "signals_by_track"):
                    self.overlay_publish_state.signals_by_track.clear()
            except Exception as exc:
                logger.warning("overlay publish state clear failed: %s", exc)

        # Force-clear core state if reset failed partially
        if reset_failed:
            self.session_reset_failed_total += 1
            try:
                if self.session.tracker is not None and hasattr(self.session.tracker, "_tracks"):
                    self.session.tracker._tracks = {}
                    if hasattr(self.session.tracker, "_next_track_id"):
                        self.session.tracker._next_track_id = 1
                if self.session.sequence_buffers is not None and hasattr(self.session.sequence_buffers, "_buffers"):
                    self.session.sequence_buffers._buffers.clear()
                if self.session.fall_state_machine is not None and hasattr(self.session.fall_state_machine, "_tracks"):
                    self.session.fall_state_machine._tracks.clear()
                if self.session.fall_faint_processor is not None and hasattr(self.session.fall_faint_processor, "reset"):
                    try:
                        self.session.fall_faint_processor.reset()
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("force-clear after partial reset failed: %s", exc)

        saved_tracker = self.session.tracker
        saved_seq = self.session.sequence_buffers
        saved_faint = self.session.fall_faint_processor
        saved_sm = self.session.fall_state_machine
        self.session.tracker = None
        self.session.sequence_buffers = None
        self.session.fall_faint_processor = None
        self.session.fall_state_machine = None
        record = self.session.reset(mapped)
        self.session.tracker = saved_tracker
        self.session.sequence_buffers = saved_seq
        self.session.fall_faint_processor = saved_faint
        self.session.fall_state_machine = saved_sm

        if tracker_factory is not None:
            try:
                self.session.tracker = tracker_factory()
            except Exception as exc:
                tracker_error = tracker_error or exc
                self.session_reset_failed_total += 1
                logger.error("tracker_factory failed: %s", exc)
        if sequence_factory is not None:
            try:
                self.session.sequence_buffers = sequence_factory()
            except Exception as exc:
                sequence_error = sequence_error or exc
                self.session_reset_failed_total += 1
                logger.error("sequence_factory failed: %s", exc)
        if fall_faint_factory is not None:
            try:
                self.session.fall_faint_processor = fall_faint_factory()
            except Exception as exc:
                logger.warning("fall_faint_factory failed: %s", exc)
        if crop_sequence_factory is not None:
            try:
                self.crop_sequence_buffers = crop_sequence_factory()
            except Exception as exc:
                logger.warning("crop_sequence_factory failed: %s", exc)
        if display_id_mapper_factory is not None:
            self.display_id_mapper = display_id_mapper_factory()
        if overlay_publish_state_factory is not None:
            self.overlay_publish_state = overlay_publish_state_factory()
        if exit_post_processor_factory is not None:
            self.exit_post_processor = exit_post_processor_factory()
        if hazard_post_processor_factory is not None:
            self.hazard_post_processor = hazard_post_processor_factory()

        with self._stamp_lock:
            self.session_generation += 1
            new_gen = self.session_generation

        cleared = self.clear_frame_queue(frame_queue)

        self.metrics = TrackingRuntimeMetrics()
        self.lifecycle_rows = []
        self.decision_rows = []
        self._metrics_ingested_this_frame = False
        self._finalized = False
        self._last_reset_stream_id = previous_stream
        self._last_reset_reason = str(mapped)
        self._last_reset_mono = time.monotonic()

        record["workerRunId"] = self.session.worker_run_id
        record["previousWorkerRunId"] = previous_worker
        record["workerRunIdUnchanged"] = previous_worker == self.session.worker_run_id
        record["previousStreamRunId"] = previous_stream
        record["mappedReason"] = str(mapped)
        record["sessionGeneration"] = new_gen
        record["previousSessionGeneration"] = previous_gen
        record["queueCleared"] = cleared
        record["coalesced"] = False
        record["trackerError"] = str(tracker_error) if tracker_error else None
        record["sequenceError"] = str(sequence_error) if sequence_error else None
        record["session_reset_failed_total"] = self.session_reset_failed_total
        record["incidentClosures"] = incident_closures
        logger.info(
            "analysis session reset reason=%s camera=%s stream %s -> %s gen %s -> %s worker=%s incidents_closed=%s",
            mapped,
            self.session.camera_login_id,
            previous_stream,
            self.session.stream_run_id,
            previous_gen,
            new_gen,
            self.session.worker_run_id,
            len(incident_closures),
        )
        print(
            f"[analysis-session] reset reason={mapped} camera={self.session.camera_login_id} "
            f"stream={previous_stream}->{self.session.stream_run_id} "
            f"gen={previous_gen}->{new_gen} "
            f"workerRunId={self.session.worker_run_id} resetCount={self.session.reset_count} "
            f"queueCleared={cleared} incidentClosures={len(incident_closures)}",
            flush=True,
        )
        return record

    def safety_counters(self) -> dict[str, Any]:
        return {
            "stale_packet_dropped_total": self.stale_packet_dropped_total,
            "stale_event_dropped_total": self.stale_event_dropped_total,
            "queue_cleared_frame_count": self.queue_cleared_frame_count,
            "session_reset_failed_total": self.session_reset_failed_total,
            "metrics_flush_failed_total": self.metrics_flush_failed_total,
            "metrics_flush_duration_ms": self.metrics_flush_duration_ms,
            "session_generation": self.session_generation,
        }

    @staticmethod
    def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> Path:
        import io

        buf = io.StringIO()
        if rows:
            keys: list[str] = []
            for row in rows:
                for key in row:
                    if key not in keys:
                        keys.append(key)
            writer = csv.DictWriter(buf, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        atomic_write_text(path, buf.getvalue())
        return path


def backend_start_log_from_detector(
    detector: Any,
    *,
    requested_model: str | None = None,
    device: str | None = None,
    precision: str | None = None,
    worker_run_id: str | None = None,
    stream_run_id: str | None = None,
    camera_login_id: str | None = None,
) -> dict[str, Any]:
    """Build start log from real detector state only — never invent TRT timings."""
    from ai.inference.backend_instrumentation import BackendStartLog
    from ai.inference.tensorrt_runtime import (
        RUNTIME_PYTORCH,
        RUNTIME_PYTORCH_FALLBACK,
        RUNTIME_TENSORRT,
        infer_requested_backend,
        is_tensorrt_engine_path,
    )

    if detector is None:
        log = BackendStartLog(
            requested_backend=str(requested_model or "unknown"),
            actual_backend="unavailable",
            fallback_reason="detector_not_initialized",
            model_path=requested_model,
            engine_path=None,
            device=device or "unknown",
            precision=precision or "n/a",
        )
        record = log.to_dict()
    else:
        requested = str(requested_model or getattr(detector, "requested_model_path", "") or "")
        actual_path = str(getattr(detector, "model_path", None) or getattr(detector, "model_name", requested) or "")
        actual_backend = getattr(detector, "runtime", None)
        if not actual_backend:
            if is_tensorrt_engine_path(actual_path):
                actual_backend = RUNTIME_TENSORRT
            else:
                actual_backend = RUNTIME_PYTORCH
        requested_backend = infer_requested_backend(requested) if requested else infer_requested_backend(actual_path)
        fallback = bool(getattr(detector, "fallback_occurred", False) or actual_backend == RUNTIME_PYTORCH_FALLBACK)
        reason = getattr(detector, "tensorrt_error", None)
        if fallback and not reason:
            reason = "unspecified_tensorrt_fallback"
        engine_path = requested if is_tensorrt_engine_path(requested) else (
            actual_path if is_tensorrt_engine_path(actual_path) else None
        )
        log = BackendStartLog(
            requested_backend=str(requested_backend),
            actual_backend=str(actual_backend),
            fallback_reason=str(reason) if fallback else None,
            model_path=actual_path or requested_model,
            engine_path=engine_path,
            device=str(device if device is not None else getattr(detector, "device", "unknown")),
            precision=precision or getattr(detector, "precision", None) or "fp32",
        )
        record = log.to_dict()

    if worker_run_id:
        record["workerRunId"] = worker_run_id
    if stream_run_id:
        record["streamRunId"] = stream_run_id
    if camera_login_id:
        record["cameraLoginId"] = camera_login_id
    print(f"[worker-backend-session] {record}", flush=True)
    return record

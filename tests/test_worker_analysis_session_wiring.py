"""Regression: AnalysisWorkerRuntime wiring for EOF / reconnect / source change.

Drives the shipped reset/metrics path used by overlay and RTSP workers (no live RTSP).
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ai.action.fall_event_state import FallEventStateMachine, FallState, FallTrackState
from ai.action.faint_post_processing import FaintEventPostProcessor
from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers
from ai.analysis_session import (
    AnalysisWorkerRuntime,
    backend_start_log_from_detector,
    map_boundary_reason,
    metrics_dir_for,
)
from ai.worker_session import SessionResetReason
from tracking.simple_tracker import SimpleTrackAssigner


class AnalysisSessionWiringTest(unittest.TestCase):
    def _seed_runtime(self, metrics_root: Path) -> AnalysisWorkerRuntime:
        tracker = SimpleTrackAssigner(match_thresh=0.2, track_buffer=10, min_box_area=1)
        buffers = PerTrackKeypointSequenceBuffers(sequence_length=4)
        fall_sm = FallEventStateMachine()
        faint = FaintEventPostProcessor()
        # attach SM via construction if supported, else bind separately
        runtime = AnalysisWorkerRuntime.start(
            "cam_wire",
            metrics_root=metrics_root,
            tracker=tracker,
            sequence_buffers=buffers,
            fall_faint_processor=faint,
            fall_state_machine=fall_sm,
        )
        # Simulate video A activity
        tracked = tracker.update([{"bbox": [10, 10, 60, 120], "confidence": 0.9}], now=1.0)
        tid = tracked[0]["track_id"]
        runtime.advance_frame()
        runtime.note_tracker_events(tracker, now=1.0)
        buffers._buffers[str(tid)] = object()
        fall_sm._tracks[f"cam_wire:track:{tid}"] = FallTrackState(state=FallState.POST_FALL_LYING)
        runtime.session.overlay_state["x"] = 1
        runtime.session.pending_events.append({"e": 1})
        return runtime

    def test_file_a_to_b_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self._seed_runtime(root)
            stream_a = runtime.session.stream_run_id
            worker_a = runtime.session.worker_run_id
            self.assertGreater(runtime.metrics.track_created_total, 0)

            record = runtime.reset_analysis_session(SessionResetReason.VIDEO_EOF, flush_blocking=True)
            stream_b = runtime.session.stream_run_id
            self.assertNotEqual(stream_a, stream_b)
            self.assertEqual(runtime.session.worker_run_id, worker_a)
            self.assertEqual(runtime.session.frame_id, 0)
            self.assertEqual(runtime.session.tracker._tracks, {})
            self.assertEqual(len(runtime.session.sequence_buffers._buffers), 0)
            self.assertEqual(runtime.session.fall_state_machine._tracks, {})
            self.assertEqual(runtime.session.overlay_state, {})
            self.assertEqual(runtime.session.pending_events, [])

            # metrics for A flushed
            a_dir = metrics_dir_for("cam_wire", stream_a, root=root)
            self.assertTrue((a_dir / "session-summary.json").is_file())

            # B first detection is new_track
            tracked_b = runtime.session.tracker.update(
                [{"bbox": [10, 10, 60, 120], "confidence": 0.95}],
                now=10.0,
            )
            runtime.advance_frame()
            runtime.note_tracker_events(runtime.session.tracker, now=10.0)
            self.assertEqual(tracked_b[0]["track_id"], 1)
            self.assertTrue(any(e.get("event") == "new_track" for e in runtime.session.tracker.last_events))
            self.assertEqual(record["mappedReason"], SessionResetReason.VIDEO_EOF.value)

    def test_rtsp_reconnect_keeps_worker_changes_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._seed_runtime(Path(tmp))
            worker = runtime.session.worker_run_id
            stream_a = runtime.session.stream_run_id
            metrics_a = runtime.metrics
            runtime.reset_analysis_session("RTSP_RECONNECTED")
            self.assertEqual(runtime.session.worker_run_id, worker)
            self.assertNotEqual(runtime.session.stream_run_id, stream_a)
            self.assertIsNot(runtime.metrics, metrics_a)
            self.assertEqual(runtime.session.tracker._tracks, {})
            self.assertEqual(len(runtime.session.sequence_buffers._buffers), 0)

    def test_source_change_reason_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._seed_runtime(Path(tmp))
            capture = {"closed": 0, "opened": 0}

            def close_open_sim():
                capture["closed"] += 1
                capture["opened"] += 1

            # Call site pattern: close capture → reset → open (open after reset)
            close_open_sim()
            record = runtime.reset_analysis_session(SessionResetReason.SOURCE_CHANGE)
            self.assertEqual(record["mappedReason"], SessionResetReason.SOURCE_CHANGE.value)
            self.assertEqual(runtime.session.last_reset_reason, SessionResetReason.SOURCE_CHANGE.value)
            self.assertEqual(capture["closed"], 1)
            self.assertEqual(capture["opened"], 1)

    def test_metrics_save_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._seed_runtime(Path(tmp))
            original_mkdir = Path.mkdir

            def boom(self, *args, **kwargs):
                raise OSError("injected write failure")

            Path.mkdir = boom  # type: ignore[method-assign]
            try:
                paths = runtime.flush_metrics()
            finally:
                Path.mkdir = original_mkdir  # type: ignore[method-assign]
            self.assertEqual(paths, {})
            self.assertGreaterEqual(runtime.metrics_save_failures, 1)
            # inference path continues after save failure
            runtime.advance_frame()
            runtime.session.tracker.update([{"bbox": [1, 1, 50, 50], "confidence": 0.9}], now=2.0)
            runtime.note_tracker_events(runtime.session.tracker, now=2.0)

    def test_double_reset_idempotent_no_double_flush(self):
        import time

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self._seed_runtime(root)
            stream_a = runtime.session.stream_run_id
            runtime.reset_analysis_session(SessionResetReason.MANUAL, flush_blocking=True)
            stream_b = runtime.session.stream_run_id
            # Immediate second reset coalesces (no extra stream rollover)
            r2 = runtime.reset_analysis_session(SessionResetReason.MANUAL, flush_blocking=True)
            self.assertTrue(r2.get("coalesced"))
            self.assertEqual(runtime.session.stream_run_id, stream_b)
            self.assertNotEqual(stream_a, stream_b)
            # Outside coalesce window: advances again
            time.sleep(0.3)
            runtime.reset_analysis_session(SessionResetReason.MANUAL, flush_blocking=True)
            stream_c = runtime.session.stream_run_id
            self.assertNotEqual(stream_b, stream_c)
            # A flushed once
            a_summary = metrics_dir_for("cam_wire", stream_a, root=root) / "session-summary.json"
            self.assertTrue(a_summary.is_file())
            # force double flush of same stream is no-op
            runtime.session.stream_run_id = stream_a  # simulate
            paths = runtime.flush_metrics()
            self.assertEqual(paths, {})  # already in _flushed_stream_ids

    def test_map_boundary_reasons(self):
        self.assertEqual(map_boundary_reason("FRAME_ID_RESET"), SessionResetReason.STREAM_RECONNECT)
        self.assertEqual(map_boundary_reason("LARGE_TIME_GAP"), SessionResetReason.STREAM_RECONNECT)
        self.assertEqual(map_boundary_reason("SOURCE_CHANGED"), SessionResetReason.SOURCE_CHANGE)
        self.assertEqual(map_boundary_reason("VIDEO_EOF"), SessionResetReason.VIDEO_EOF)

    def test_payload_enrichment_backward_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AnalysisWorkerRuntime.start("cam_p", metrics_root=Path(tmp))
            runtime.advance_frame()
            base = {"schemaVersion": "1.1", "messageType": "overlay", "frameId": 99}
            enriched = runtime.enrich_payload(base)
            self.assertEqual(enriched["frameId"], 99)  # existing frameId preserved
            self.assertIn("workerRunId", enriched)
            self.assertIn("streamRunId", enriched)
            self.assertIn("evidenceFrameKey", enriched)
            self.assertIn(runtime.session.stream_run_id, str(enriched["evidenceFrameKey"]))
            self.assertTrue(str(enriched["evidenceFrameKey"]).startswith(runtime.session.camera_login_id))

    def test_note_tracker_events_single_ingest_per_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._seed_runtime(Path(tmp))
            tracker = runtime.session.tracker
            before = runtime.metrics.track_created_total
            runtime.advance_frame()
            tracker.update([{"bbox": [200, 10, 250, 120], "confidence": 0.9}], now=3.0)
            runtime.note_tracker_events(tracker, now=3.0)
            mid = runtime.metrics.track_created_total
            runtime.note_tracker_events(tracker, now=3.0)  # duplicate same frame
            self.assertEqual(runtime.metrics.track_created_total, mid)
            self.assertGreaterEqual(mid, before)

    def test_backend_start_unavailable_without_fake_perf(self):
        record = backend_start_log_from_detector(
            None,
            requested_model="models/x.engine",
            device="cpu",
            worker_run_id="worker-1",
            stream_run_id="stream-1",
            camera_login_id="cam",
        )
        self.assertEqual(record["actual_backend"], "unavailable")
        self.assertEqual(record["workerRunId"], "worker-1")
        self.assertNotIn("avg_inference_ms", record)

    def test_overlay_worker_has_analysis_session_hooks(self):
        """Structural: shipped OverlayWorker wires AnalysisWorkerRuntime + reconnect gen."""
        # Avoid importing serve_ai_overlay (pulls cv2); read source on disk.
        root = Path(__file__).resolve().parents[1]
        src = (root / "scripts" / "serve_ai_overlay.py").read_text(encoding="utf-8")
        self.assertIn("AnalysisWorkerRuntime", src)
        self.assertIn("reset_analysis_session", src)
        self.assertIn("_bump_reconnect_generation", src)
        self.assertIn("RTSP_RECONNECTED", src)
        self.assertIn("analysis_runtime=analysis_runtime", src)
        self.assertIn("def process_frame", src)
        self.assertIn("analysis_runtime=None", src)

    def test_rtsp_inference_run_wires_analysis_runtime(self):
        root = Path(__file__).resolve().parents[1]
        src = (root / "scripts" / "run_rtsp_inference.py").read_text(encoding="utf-8")
        self.assertIn("AnalysisWorkerRuntime", src)
        self.assertIn("note_tracker_events", src)
        self.assertIn("VIDEO_EOF", src)
        self.assertIn("enrich_payload", src)


if __name__ == "__main__":
    unittest.main()

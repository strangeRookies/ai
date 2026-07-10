"""Stale packet/event, double-reset, EOF finalize, payload contract regression tests."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers
from ai.analysis_session import AnalysisWorkerRuntime, metrics_dir_for
from ai.frame_sync import CameraFrameQueue, FramePacket
from ai.publishers.mqtt_payloads import build_frame_sync_payload, build_overlay_payload
from ai.worker_session import SessionResetReason
from tracking.simple_tracker import SimpleTrackAssigner


def _packet(
    *,
    frame_id: int = 1,
    gen: int = 1,
    stream: str | None = "stream-a",
    camera: str = "cam_stale",
) -> FramePacket:
    return FramePacket(
        camera_login_id=camera,
        frame_id=frame_id,
        captured_at_ms=1000 + frame_id,
        frame=None,
        width=64,
        height=64,
        frame_idx=frame_id,
        timestamp=float(frame_id),
        fps=10.0,
        stream_run_id=stream,
        session_generation=gen,
    )


class SessionStaleGuardsTest(unittest.TestCase):
    def test_queue_clears_and_stale_packet_dropped_after_session_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracker = SimpleTrackAssigner(match_thresh=0.2, min_box_area=1)
            buf = PerTrackKeypointSequenceBuffers(sequence_length=4)
            rt = AnalysisWorkerRuntime.start("cam_stale", metrics_root=root, tracker=tracker, sequence_buffers=buf)
            q = CameraFrameQueue("cam_stale", maxsize=8)
            stream_a = rt.session.stream_run_id
            gen_a = rt.session_generation

            # Simulate video A packets still in queue
            for i in range(3):
                q.put_latest(_packet(frame_id=i + 1, gen=gen_a, stream=stream_a))
            self.assertEqual(q.size(), 3)

            rt.reset_analysis_session(SessionResetReason.VIDEO_EOF, frame_queue=q, flush_blocking=True)
            self.assertEqual(q.size(), 0)
            self.assertGreater(rt.queue_cleared_frame_count, 0)

            # Late A packet arrives after B session start
            late_a = _packet(frame_id=99, gen=gen_a, stream=stream_a)
            self.assertFalse(rt.accept_packet(late_a))
            self.assertGreaterEqual(rt.stale_packet_dropped_total, 1)

            # B packet accepted
            stamp = rt.stamp_snapshot()
            b = _packet(frame_id=1, gen=int(stamp["session_generation"]), stream=stamp["stream_run_id"])
            self.assertTrue(rt.accept_packet(b))

    def test_stale_event_publish_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = AnalysisWorkerRuntime.start("cam_ev", metrics_root=Path(tmp))
            gen_a = rt.session_generation
            stream_a = rt.session.stream_run_id
            payload = {"schemaVersion": "1.1", "messageType": "event", "summary": "A"}
            ok = rt.enrich_payload(payload, capture_generation=gen_a, capture_stream_run_id=stream_a)
            self.assertIsNotNone(ok)
            rt.reset_analysis_session(SessionResetReason.STREAM_RECONNECT, flush_blocking=True)
            blocked = rt.enrich_payload(payload, capture_generation=gen_a, capture_stream_run_id=stream_a)
            self.assertIsNone(blocked)
            self.assertGreaterEqual(rt.stale_event_dropped_total, 1)

    def test_double_reconnect_coalesces_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = AnalysisWorkerRuntime.start("cam_rc", metrics_root=Path(tmp))
            gen0 = rt.session_generation
            r1 = rt.reset_analysis_session("RTSP_RECONNECTED", flush_blocking=True)
            r2 = rt.reset_analysis_session("RTSP_RECONNECTED", flush_blocking=True)
            self.assertFalse(r1.get("coalesced"))
            self.assertTrue(r2.get("coalesced"))
            # Only one generation step for the burst
            self.assertEqual(rt.session_generation, gen0 + 1)

    def test_reset_idempotency_double_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = AnalysisWorkerRuntime.start("cam_id", metrics_root=Path(tmp))
            rt.reset_analysis_session(SessionResetReason.MANUAL, flush_blocking=True)
            # outside coalesce window
            time.sleep(0.3)
            before = rt.session_generation
            rt.reset_analysis_session(SessionResetReason.MANUAL, flush_blocking=True)
            self.assertEqual(rt.session_generation, before + 1)
            # immediate double is coalesced
            rt.reset_analysis_session(SessionResetReason.MANUAL, flush_blocking=True)
            self.assertEqual(rt.session_generation, before + 1)

    def test_eof_finalize_no_empty_new_session_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = AnalysisWorkerRuntime.start("cam_eof", metrics_root=root)
            tracker = SimpleTrackAssigner(min_box_area=1)
            rt.bind_optional(tracker=tracker)
            tracker.update([{"bbox": [0, 0, 40, 80], "confidence": 0.9}], now=1.0)
            rt.note_tracker_events(tracker, now=1.0)
            stream = rt.session.stream_run_id
            streams_before = {p.name for p in (root / "cam_eof").iterdir()} if (root / "cam_eof").exists() else set()
            rt.finalize_for_exit(blocking=True)
            cam_dir = root / "cam_eof"
            self.assertTrue(cam_dir.is_dir())
            # Path layout: cam/pid-*/streamRunId/
            stream_dirs = list(cam_dir.glob("pid-*/" + stream))
            self.assertEqual(len(stream_dirs), 1)
            self.assertTrue((stream_dirs[0] / "session-summary.json").is_file())
            # No sibling stream dirs under same pid
            pid_dir = stream_dirs[0].parent
            self.assertEqual([p.name for p in pid_dir.iterdir() if p.is_dir()], [stream])

    def test_metrics_flush_failure_worker_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = AnalysisWorkerRuntime.start("cam_mf", metrics_root=Path(tmp))
            original = Path.mkdir

            def boom(self, *a, **k):
                raise OSError("disk full")

            Path.mkdir = boom  # type: ignore[method-assign]
            try:
                paths = rt.flush_metrics(blocking=True)
            finally:
                Path.mkdir = original  # type: ignore[method-assign]
            self.assertEqual(paths, {})
            self.assertGreaterEqual(rt.metrics_flush_failed_total, 1)
            # continues
            self.assertTrue(rt.accept_packet(_packet(gen=rt.session_generation, stream=rt.session.stream_run_id)))
            rt.advance_frame()

    def test_same_frame_id_different_stream_evidence_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = AnalysisWorkerRuntime.start("cam_fk", metrics_root=Path(tmp))
            rt.session.frame_id = 0
            rt.advance_frame()  # frame 1
            key_a = rt.session.identity().evidence_frame_key()
            stream_a = rt.session.stream_run_id
            rt.reset_analysis_session(SessionResetReason.SOURCE_CHANGE, flush_blocking=True)
            rt.session.frame_id = 0
            rt.advance_frame()  # frame 1 again
            key_b = rt.session.identity().evidence_frame_key()
            self.assertNotEqual(stream_a, rt.session.stream_run_id)
            self.assertNotEqual(key_a, key_b)
            self.assertTrue(key_a.endswith(":1"))
            self.assertTrue(key_b.endswith(":1"))

    def test_payload_session_fields_consistent_across_message_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = AnalysisWorkerRuntime.start("cam_pl", metrics_root=Path(tmp))
            rt.advance_frame()
            fields = rt.identity_fields()
            overlay = build_overlay_payload("cam_pl", 640, 480, [], frame_id=7)
            overlay = rt.enrich_payload(overlay)
            fs = build_frame_sync_payload("cam_pl", 7, 1000, 1010, 5, 0)
            fs = rt.enrich_payload(fs)
            event = rt.enrich_payload({"schemaVersion": "1.1", "messageType": "event", "frameId": 7})
            assert overlay and fs and event
            for payload in (overlay, fs, event):
                self.assertEqual(payload["workerRunId"], fields["workerRunId"])
                self.assertEqual(payload["streamRunId"], fields["streamRunId"])
                self.assertEqual(payload["evidenceFrameKey"], fields["evidenceFrameKey"])
                # optional sessionGeneration present
                self.assertEqual(payload.get("sessionGeneration"), fields["sessionGeneration"])

    def test_backend_dto_ignores_unknown_optional_session_fields(self):
        """Mirror OverlayMessage @JsonIgnoreProperties(ignoreUnknown=true) contract in Python."""
        known = {
            "schemaVersion",
            "messageType",
            "timestampMs",
            "streamId",
            "cameraLoginId",
            "frameWidth",
            "frameHeight",
            "events",
            "frameId",
            "capturedAtMs",
            "processedAtMs",
            "publishedAtMs",
            "queueLagMs",
            "droppedFrameCount",
        }
        with tempfile.TemporaryDirectory() as tmp:
            rt = AnalysisWorkerRuntime.start("cam_dto", metrics_root=Path(tmp))
            rt.advance_frame()
            payload = build_overlay_payload("cam_dto", 640, 480, [], frame_id=3, timestamp_ms=123)
            payload = rt.enrich_payload(payload)
            assert payload is not None
            # Unknown optional fields must not break required keys
            for key in ("schemaVersion", "messageType", "streamId", "cameraLoginId", "events"):
                self.assertIn(key, payload)
            extras = set(payload.keys()) - known
            # session fields are extras that backend ignores
            self.assertTrue({"workerRunId", "streamRunId", "evidenceFrameKey"}.issubset(extras))
            # JSON round-trip preserves extras (subscriber ignores unknowns)
            roundtrip = json.loads(json.dumps(payload))
            self.assertEqual(roundtrip["workerRunId"], payload["workerRunId"])
            self.assertEqual(roundtrip["schemaVersion"], "1.1")

    def test_reset_failure_force_clears_partial_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            class BrokenTracker:
                def __init__(self):
                    self._tracks = {1: {}}
                    self._next_track_id = 2

                def reset(self, *, reason=None):
                    raise RuntimeError("tracker boom")

            broken = BrokenTracker()
            buf = PerTrackKeypointSequenceBuffers(sequence_length=4)
            buf._buffers["1"] = object()
            rt = AnalysisWorkerRuntime.start("cam_fail", metrics_root=Path(tmp))
            rt.bind_optional(tracker=broken, sequence_buffers=buf)
            rt.reset_analysis_session(SessionResetReason.MANUAL, flush_blocking=True)
            self.assertGreaterEqual(rt.session_reset_failed_total, 1)
            self.assertEqual(broken._tracks, {})
            self.assertEqual(len(buf._buffers), 0)


if __name__ == "__main__":
    unittest.main()

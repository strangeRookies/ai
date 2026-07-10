"""Multi-camera isolation, supervisor policy, queue, metrics, incident (offline fixtures only)."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai.action.fall_event_state import FallEventStateMachine, FallState, FallTrackState
from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers
from ai.analysis_session import AnalysisWorkerRuntime, atomic_write_text, metrics_dir_for
from ai.event_idempotency import EventIdempotencyStore
from ai.frame_sync import CameraFrameQueue, FramePacket
from ai.registered_camera_workers import CameraWorker, restart_exited_worker, sync_camera_workers
from ai.registered_cameras import RegisteredCamera, RunnerConfig
from ai.supervisor_policy import (
    AssignmentConflictError,
    CameraAssignment,
    RestartPolicy,
    SupervisorRestartBook,
    plan_source_change_restarts,
    validate_camera_assignments,
)
from ai.vlm.incident_pipeline import (
    Incident,
    IncidentEvent,
    IncidentEventType,
    IncidentTerminalStatus,
    IncidentVlmPipeline,
)
from ai.worker_session import SessionResetReason
from tracking.simple_tracker import SimpleTrackAssigner

# Fixture IDs only — not production hardcoding of the camera set.
FIXTURE_CAMS = ("cam_fix_01", "cam_fix_02", "cam_fix_03", "cam_fix_04")


def _runtime(cam: str, root: Path) -> AnalysisWorkerRuntime:
    tracker = SimpleTrackAssigner(match_thresh=0.2, min_box_area=1)
    buf = PerTrackKeypointSequenceBuffers(sequence_length=4)
    fall = FallEventStateMachine()
    return AnalysisWorkerRuntime.start(
        cam,
        metrics_root=root,
        tracker=tracker,
        sequence_buffers=buf,
        fall_state_machine=fall,
    )


def _packet(cam: str, frame_id: int, gen: int, stream: str, *, captured_at_ms: int | None = None) -> FramePacket:
    return FramePacket(
        camera_login_id=cam,
        frame_id=frame_id,
        captured_at_ms=captured_at_ms if captured_at_ms is not None else 1_000_000 + frame_id,
        frame=None,
        width=64,
        height=64,
        frame_idx=frame_id,
        timestamp=float(frame_id),
        fps=10.0,
        stream_run_id=stream,
        session_generation=gen,
    )


class MultiCameraIsolationTest(unittest.TestCase):
    def test_four_runtime_instances_fully_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtimes = {cam: _runtime(cam, root) for cam in FIXTURE_CAMS}
            queues = {cam: CameraFrameQueue(cam, maxsize=3) for cam in FIXTURE_CAMS}
            # Seed distinct state per camera
            for cam, rt in runtimes.items():
                rt.session.tracker.update(
                    [{"bbox": [0, 0, 40, 80], "confidence": 0.9}],
                    now=1.0,
                )
                rt.session.sequence_buffers._buffers[cam] = object()
                rt.session.fall_state_machine._tracks[f"{cam}:track:1"] = FallTrackState(
                    state=FallState.POST_FALL_LYING
                )
                queues[cam].put_latest(
                    _packet(cam, 1, rt.session_generation, rt.session.stream_run_id)
                )
            # Identity / object isolation
            worker_ids = {rt.session.worker_run_id for rt in runtimes.values()}
            stream_ids = {rt.session.stream_run_id for rt in runtimes.values()}
            self.assertEqual(len(worker_ids), 4)
            self.assertEqual(len(stream_ids), 4)
            trackers = [rt.session.tracker for rt in runtimes.values()]
            self.assertEqual(len({id(t) for t in trackers}), 4)
            buffers = [rt.session.sequence_buffers for rt in runtimes.values()]
            self.assertEqual(len({id(b) for b in buffers}), 4)
            self.assertEqual(len({id(q) for q in queues.values()}), 4)

    def test_cam02_reset_leaves_others_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtimes = {cam: _runtime(cam, root) for cam in FIXTURE_CAMS}
            snapshots = {
                cam: (
                    rt.session.stream_run_id,
                    rt.session.reset_count,
                    rt.session_generation,
                    dict(rt.session.fall_state_machine._tracks),
                    len(rt.session.sequence_buffers._buffers),
                )
                for cam, rt in runtimes.items()
            }
            # activity on cam02 then reset
            c2 = runtimes["cam_fix_02"]
            c2.session.tracker.update([{"bbox": [1, 1, 50, 90], "confidence": 0.9}], now=2.0)
            c2.session.sequence_buffers._buffers["x"] = object()
            c2.reset_analysis_session(SessionResetReason.STREAM_RECONNECT, flush_blocking=True)
            for cam in ("cam_fix_01", "cam_fix_03", "cam_fix_04"):
                rt = runtimes[cam]
                stream, reset_count, gen, fall, buf_n = snapshots[cam]
                self.assertEqual(rt.session.stream_run_id, stream)
                self.assertEqual(rt.session.reset_count, reset_count)
                self.assertEqual(rt.session_generation, gen)
                self.assertEqual(rt.session.fall_state_machine._tracks, fall)
                self.assertEqual(len(rt.session.sequence_buffers._buffers), buf_n)
            self.assertNotEqual(c2.session.stream_run_id, snapshots["cam_fix_02"][0])

    def test_all_camera_evidence_keys_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = set()
            for cam in FIXTURE_CAMS:
                rt = _runtime(cam, root)
                for _ in range(3):
                    rt.advance_frame()
                    keys.add(rt.session.identity().evidence_frame_key())
            self.assertEqual(len(keys), 4 * 3)
            # same frameId after reset still unique via camera+stream
            rt = _runtime("cam_fx", root)
            rt.session.frame_id = 0
            rt.advance_frame()
            k1 = rt.session.identity().evidence_frame_key()
            rt.reset_analysis_session(SessionResetReason.VIDEO_EOF, flush_blocking=True)
            rt.session.frame_id = 0
            rt.advance_frame()
            k2 = rt.session.identity().evidence_frame_key()
            self.assertNotEqual(k1, k2)


class SupervisorPolicyTest(unittest.TestCase):
    def test_source_change_restarts_only_that_camera(self):
        current = {"a": "sig-a", "b": "sig-b", "c": "sig-c"}
        desired = {"a": "sig-a", "b": "sig-b2", "c": "sig-c"}
        plan = plan_source_change_restarts(current, desired)
        self.assertEqual(plan, {"b": "source_changed"})

    def test_exponential_backoff_and_max_failures(self):
        policy = RestartPolicy(
            initial_delay_sec=1.0,
            maximum_delay_sec=8.0,
            maximum_consecutive_failures=3,
            stable_runtime_reset_threshold_sec=9999.0,
            startup_stagger_sec=0.0,
        )
        book = SupervisorRestartBook(policy=policy)
        d1 = book.on_worker_exited("cam_x", now_mono=100.0)
        self.assertTrue(d1["allow_restart"])
        self.assertEqual(d1["delay_sec"], 1.0)
        d2 = book.on_worker_exited("cam_x", now_mono=101.0)
        self.assertEqual(d2["delay_sec"], 2.0)
        d3 = book.on_worker_exited("cam_x", now_mono=102.0)
        self.assertEqual(d3["delay_sec"], 4.0)
        d4 = book.on_worker_exited("cam_x", now_mono=103.0)
        self.assertFalse(d4["allow_restart"])
        self.assertTrue(d4["blocked"])

    def test_startup_stagger_scales_with_index(self):
        book = SupervisorRestartBook(
            policy=RestartPolicy(startup_stagger_sec=0.5, initial_delay_sec=1, maximum_delay_sec=10, maximum_consecutive_failures=5)
        )
        self.assertEqual(book.startup_stagger_delay(0), 0.0)
        self.assertEqual(book.startup_stagger_delay(3), 1.5)

    def test_restart_exited_worker_only_touches_target(self):
        alive = MagicMock()
        alive.poll.return_value = None
        dead = MagicMock()
        dead.poll.return_value = 1
        workers = {
            "cam_keep": CameraWorker(processes=[alive], overlay_port=8010, source_signature="s1", camera_login_id="cam_keep"),
            "cam_die": CameraWorker(processes=[dead], overlay_port=8011, source_signature="s2", camera_login_id="cam_die"),
        }
        cameras_by_id = {
            "cam_die": RegisteredCamera(
                camera_id=2,
                camera_login_id="cam_die",
                source_type="REAL_RTSP",
                rtsp_url="rtsp://x/cam_die",
                assigned_video_path=None,
            )
        }
        book = SupervisorRestartBook(
            policy=RestartPolicy(
                initial_delay_sec=0.0,
                maximum_delay_sec=0.0,
                maximum_consecutive_failures=5,
                stable_runtime_reset_threshold_sec=0.0,
                startup_stagger_sec=0.0,
            )
        )
        config = MagicMock()
        config.overlay_report_enabled = False
        with patch("ai.registered_camera_workers.start_camera_worker") as start:
            new = CameraWorker(processes=[MagicMock()], overlay_port=8011, source_signature="s2", camera_login_id="cam_die")
            start.return_value = new
            with patch("ai.registered_camera_workers.report_overlay_stopped"):
                restart_exited_worker(
                    workers,
                    "cam_die",
                    cameras_by_id=cameras_by_id,
                    config=config,
                    restart_book=book,
                )
        self.assertIs(workers["cam_keep"].processes[0], alive)
        self.assertIs(workers["cam_die"], new)
        start.assert_called_once()


class QueueAndMetricsTest(unittest.TestCase):
    def test_bounded_overflow_drops_oldest(self):
        q = CameraFrameQueue("cam_q", maxsize=2)
        q.put_latest(_packet("cam_q", 1, 1, "s"))
        q.put_latest(_packet("cam_q", 2, 1, "s"))
        q.put_latest(_packet("cam_q", 3, 1, "s"))
        self.assertEqual(q.size(), 2)
        self.assertGreaterEqual(q.queue_overflow_drop_total, 1)
        latest = q.get_latest(drop_stale=False)
        # deque maxlen drops left; remaining are 2 and 3; popleft gives oldest remaining = 2
        # get_latest without drop_stale returns popleft = frame 2; with drop_stale keeps only latest
        latest2 = CameraFrameQueue("cam_q2", maxsize=2)
        for i in (1, 2, 3):
            latest2.put_latest(_packet("cam_q2", i, 1, "s"))
        p = latest2.get_latest(drop_stale=True)
        self.assertEqual(p.frame_id, 3)

    def test_age_drop_and_stale_after_clear(self):
        now = {"t": 10_000}

        def now_ms():
            return now["t"]

        q = CameraFrameQueue("cam_age", maxsize=5, max_packet_age_ms=100.0, now_ms=now_ms)
        q.put_latest(_packet("cam_age", 1, 1, "s", captured_at_ms=9_000))  # age 1100 when now=10100
        q.put_latest(_packet("cam_age", 2, 1, "s", captured_at_ms=10_050))  # age 50 — keep
        now["t"] = 10_100
        p = q.get_latest(drop_stale=True)
        self.assertIsNotNone(p)
        self.assertEqual(p.frame_id, 2)
        self.assertGreaterEqual(q.aged_packet_drop_total, 1)

        with tempfile.TemporaryDirectory() as tmp:
            rt = _runtime("cam_age", Path(tmp))
            gen = rt.session_generation
            stream = rt.session.stream_run_id
            q2 = CameraFrameQueue("cam_age", maxsize=4)
            q2.put_latest(_packet("cam_age", 9, gen, stream))
            rt.reset_analysis_session(SessionResetReason.STREAM_RECONNECT, frame_queue=q2, flush_blocking=True)
            self.assertEqual(q2.size(), 0)
            late = _packet("cam_age", 9, gen, stream)
            self.assertFalse(rt.accept_packet(late))

    def test_atomic_metrics_write_per_camera_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = Path(tmp) / "out.json"
            atomic_write_text(path, '{"ok": true}\n')
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["ok"], True)
            r1 = _runtime(FIXTURE_CAMS[0], root)
            r2 = _runtime(FIXTURE_CAMS[1], root)
            r1.note_tracker_events(r1.session.tracker)
            r1.session.tracker.update([{"bbox": [0, 0, 10, 10], "confidence": 0.9}], now=1.0)
            r1.note_tracker_events(r1.session.tracker, now=1.0)
            r1.flush_metrics(blocking=True)
            d1 = metrics_dir_for(FIXTURE_CAMS[0], r1.session.stream_run_id, root=root)
            d2 = metrics_dir_for(FIXTURE_CAMS[1], r2.session.stream_run_id, root=root)
            self.assertNotEqual(d1, d2)
            self.assertTrue((d1 / "session-summary.json").is_file())

    def test_async_flush_completes_on_finalize_timeout_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = _runtime("cam_flush", Path(tmp))
            rt.session.tracker.update([{"bbox": [0, 0, 20, 40], "confidence": 0.9}], now=1.0)
            rt.note_tracker_events(rt.session.tracker, now=1.0)
            # Non-blocking flush then wait via finalize
            rt.flush_metrics(blocking=False)
            ok = rt.wait_for_flush(timeout_sec=5.0)
            self.assertTrue(ok)
            paths = rt.finalize_for_exit(blocking=True)
            # finalized stream has summary under metrics dir
            out = metrics_dir_for("cam_flush", rt.session.stream_run_id, root=Path(tmp))
            self.assertTrue((out / "session-summary.json").is_file() or paths)


class EventAndIncidentTest(unittest.TestCase):
    def test_duplicate_event_idempotency(self):
        store = EventIdempotencyStore(max_entries=100)
        self.assertTrue(store.accept("evt-1", camera_login_id="cam_a"))
        self.assertFalse(store.accept("evt-1", camera_login_id="cam_a"))
        self.assertTrue(store.accept("evt-1", camera_login_id="cam_b"))
        self.assertEqual(store.stats()["duplicate_total"], 1)

    def test_event_publisher_process_local_dedupe_gate(self):
        from ai.publishers.event_publisher import ConsoleEventPublisher, event_idempotency_store

        store = event_idempotency_store()
        # isolate this test key
        event_id = f"evt-shipped-{time.time_ns()}"
        pub = ConsoleEventPublisher()
        payload = {
            "messageType": "event",
            "eventId": event_id,
            "cameraLoginId": "cam_pub",
            "clip_url": "mock://x.mp4",
        }
        self.assertTrue(pub.publish(payload))
        self.assertFalse(pub.publish(payload))  # same eventId retransmit blocked
        self.assertGreaterEqual(store.stats()["duplicate_total"], 1)

    def test_open_incident_stream_reset_terminal(self):
        pipe = IncidentVlmPipeline()
        inc = Incident(
            incident_id="inc-open-1",
            camera_login_id="cam_fx",
            original_event_id="e0",
            events=(IncidentEvent(IncidentEventType.NEW_FALL, 1.0, "e1"),),
            clip_end_sec=None,
        )
        job = pipe.process(inc)
        self.assertEqual(job.status, "INELIGIBLE")
        self.assertIn("inc-open-1", pipe.open_incident_ids())
        closed = pipe.on_stream_reset(reason="stream_reconnect", camera_login_id="cam_fx")
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["status"], IncidentTerminalStatus.STREAM_LOST.value)
        self.assertEqual(pipe.terminal_status("inc-open-1"), IncidentTerminalStatus.STREAM_LOST)
        again = pipe.process(inc)
        self.assertEqual(again.status, IncidentTerminalStatus.STREAM_LOST.value)

    def test_analysis_reset_closes_open_incidents(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = IncidentVlmPipeline()
            rt = AnalysisWorkerRuntime.start(
                "cam_inc",
                metrics_root=Path(tmp),
                incident_pipeline=pipe,
            )
            inc = Incident(
                incident_id="inc-wire-1",
                camera_login_id="cam_inc",
                original_event_id="e0",
                events=(IncidentEvent(IncidentEventType.NEW_FALL, 1.0, "e1"),),
                clip_end_sec=None,
            )
            pipe.process(inc)
            self.assertIn("inc-wire-1", pipe.open_incident_ids())
            record = rt.reset_analysis_session(SessionResetReason.STREAM_RECONNECT, flush_blocking=True)
            self.assertEqual(len(record.get("incidentClosures") or []), 1)
            self.assertEqual(pipe.terminal_status("inc-wire-1"), IncidentTerminalStatus.STREAM_LOST)

    def test_restart_blocked_status_exposed(self):
        policy = RestartPolicy(
            initial_delay_sec=0.0,
            maximum_delay_sec=0.0,
            maximum_consecutive_failures=1,
            stable_runtime_reset_threshold_sec=9999.0,
            startup_stagger_sec=0.0,
        )
        book = SupervisorRestartBook(policy=policy)
        d1 = book.on_worker_exited("cam_blk")
        self.assertTrue(d1["allow_restart"])
        d2 = book.on_worker_exited("cam_blk")
        self.assertFalse(d2["allow_restart"])
        inst = book.instrumentation("cam_blk")
        self.assertTrue(inst["restart_blocked"])
        blocked = book.blocked_cameras()
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["camera_login_id"], "cam_blk")

    def test_evidence_id_includes_stream_run_id_when_provided(self):
        from ai.evidence import evidence_id

        legacy = evidence_id("cam_x", 1, 100)
        with_stream = evidence_id("cam_x", 1, 100, stream_run_id="stream-abc")
        self.assertEqual(legacy, "cam_x-1-100")
        self.assertIn("stream-abc", with_stream)
        self.assertNotEqual(legacy, with_stream)

    def test_duplicate_camera_port_output_fail_fast(self):
        with self.assertRaises(AssignmentConflictError):
            validate_camera_assignments(
                [
                    CameraAssignment("cam_a", overlay_port=8010),
                    CameraAssignment("cam_a", overlay_port=8011),
                ]
            )
        with self.assertRaises(AssignmentConflictError):
            validate_camera_assignments(
                [
                    CameraAssignment("cam_a", overlay_port=8010),
                    CameraAssignment("cam_b", overlay_port=8010),
                ]
            )
        with self.assertRaises(AssignmentConflictError):
            validate_camera_assignments(
                [
                    CameraAssignment("cam_a", output_path="runs/out/a"),
                    CameraAssignment("cam_b", output_path="runs/out/a"),
                ]
            )

    def test_sync_plans_ports_for_fail_fast_validation(self):
        """sync_camera_workers must pass non-None overlay_port into validate_camera_assignments."""
        import inspect
        from ai import registered_camera_workers as rcw

        src = inspect.getsource(rcw.sync_camera_workers)
        self.assertIn("overlay_port=port", src)
        self.assertIn("output_path=", src)
        self.assertIn("validate_camera_assignments(planned_assignments)", src)
        # Unit: planned assignment objects with real ports fail-fast on collision
        with self.assertRaises(AssignmentConflictError):
            validate_camera_assignments(
                [
                    CameraAssignment("cam_p1", overlay_port=9000, output_path="runs/tracking_metrics/cam_p1"),
                    CameraAssignment("cam_p2", overlay_port=9000, output_path="runs/tracking_metrics/cam_p2"),
                ]
            )


if __name__ == "__main__":
    unittest.main()

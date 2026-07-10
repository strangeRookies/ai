"""Regression: video A state must not leak into video B after session reset."""

from __future__ import annotations

import unittest

from ai.action.fall_event_state import FallEventStateMachine, FallTrackState, FallState
from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers
from ai.worker_session import SessionResetReason, WorkerSession, begin_session
from tracking.simple_tracker import SimpleTrackAssigner


class WorkerSessionResetTest(unittest.TestCase):
    def _seed_session(self) -> tuple[WorkerSession, SimpleTrackAssigner, PerTrackKeypointSequenceBuffers, FallEventStateMachine]:
        tracker = SimpleTrackAssigner(match_thresh=0.2, track_buffer=10, min_box_area=1)
        buffers = PerTrackKeypointSequenceBuffers(sequence_length=4)
        fall_sm = FallEventStateMachine()
        session = begin_session(
            "cam_01",
            tracker=tracker,
            sequence_buffers=buffers,
            fall_state_machine=fall_sm,
        )
        # Simulate video A activity
        tracked = tracker.update([{"bbox": [10, 10, 60, 120], "confidence": 0.9}], now=1.0)
        track_id = tracked[0]["track_id"]
        session.next_frame_id()
        session.overlay_state["track"] = track_id
        session.hazard_state["zone"] = "A"
        session.displayed_track_ids["primary"] = track_id
        session.pending_events.append({"event": "NEW_FALL", "track_id": track_id})
        session.frame_sync_state["last_pts"] = 12345
        # seed buffer and fall state for track A (minimal internal stubs)
        buffers._buffers[str(track_id)] = object()
        buffers._last_seen_at[str(track_id)] = 1.0
        fall_sm._tracks[f"cam_01:track:{track_id}"] = FallTrackState(state=FallState.POST_FALL_LYING)
        return session, tracker, buffers, fall_sm

    def test_video_a_track_not_in_video_b(self):
        session, tracker, buffers, fall_sm = self._seed_session()
        track_ids_a = set(tracker._tracks.keys())
        self.assertTrue(track_ids_a)

        stream_a = session.stream_run_id
        record = session.reset(SessionResetReason.SOURCE_CHANGE)

        self.assertEqual(tracker._tracks, {})
        self.assertEqual(tracker._next_track_id, 1)
        self.assertEqual(session.frame_id, 0)
        self.assertNotEqual(session.stream_run_id, stream_a)
        self.assertEqual(record["reason"], SessionResetReason.SOURCE_CHANGE.value)
        self.assertEqual(session.reset_count, 2)  # WORKER_START + SOURCE_CHANGE
        self.assertEqual(session.last_reset_reason, SessionResetReason.SOURCE_CHANGE.value)

        tracked_b = tracker.update([{"bbox": [10, 10, 60, 120], "confidence": 0.95}], now=10.0)
        self.assertEqual(tracked_b[0]["track_id"], 1)
        self.assertNotIn(list(track_ids_a)[0], tracker._tracks) if list(track_ids_a)[0] != 1 else True

    def test_sequence_buffers_not_merged_across_sessions(self):
        session, tracker, buffers, _fall = self._seed_session()
        self.assertGreater(len(getattr(buffers, "_buffers", buffers.__dict__)), 0)
        session.reset(SessionResetReason.VIDEO_EOF)
        # after clear, no residual track sequences
        internal = getattr(buffers, "_buffers", None)
        if internal is not None:
            self.assertEqual(len(internal), 0)

    def test_fall_faint_state_not_carried_over(self):
        session, _tracker, _buffers, fall_sm = self._seed_session()
        # force non-empty fall state
        fall_sm._tracks["cam_01:1"] = object()
        self.assertTrue(fall_sm._tracks)
        session.reset(SessionResetReason.STREAM_RECONNECT)
        self.assertEqual(fall_sm._tracks, {})

    def test_overlay_hazard_pending_frame_sync_cleared(self):
        session, *_ = self._seed_session()
        session.reset(SessionResetReason.SOURCE_CHANGE)
        self.assertEqual(session.overlay_state, {})
        self.assertEqual(session.hazard_state, {})
        self.assertEqual(session.displayed_track_ids, {})
        self.assertEqual(session.pending_events, [])
        self.assertEqual(session.frame_sync_state, {})

    def test_stream_run_id_plus_frame_id_disambiguates_restart(self):
        session = begin_session("cam_x")
        f1 = session.next_frame_id()
        key_a = session.identity().evidence_frame_key()
        stream_a = session.stream_run_id
        session.reset(SessionResetReason.VIDEO_EOF)
        f2 = session.next_frame_id()
        key_b = session.identity().evidence_frame_key()
        self.assertEqual(f1, 1)
        self.assertEqual(f2, 1)
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(stream_a, session.stream_run_id)
        self.assertIn(":", key_b)

    def test_reset_reason_and_count_recorded(self):
        session = begin_session("cam_y")
        self.assertEqual(session.reset_count, 1)
        self.assertEqual(session.last_reset_reason, SessionResetReason.WORKER_START.value)
        r1 = session.reset(SessionResetReason.VIDEO_EOF)
        r2 = session.reset(SessionResetReason.SOURCE_CHANGE)
        self.assertEqual(session.reset_count, 3)
        self.assertEqual(r1["reason"], "video_eof")
        self.assertEqual(r2["reason"], "source_change")
        self.assertEqual(len(session.reset_history), 3)
        snap = session.snapshot()
        self.assertIn("identity", snap)
        self.assertEqual(snap["resetCount"], 3)

    def test_new_session_first_detection_is_new_track(self):
        tracker = SimpleTrackAssigner(match_thresh=0.2, min_box_area=1)
        session = begin_session("cam_z", tracker=tracker)
        a = tracker.update([{"bbox": [0, 0, 50, 100], "confidence": 0.9}], now=1.0)
        session.reset(SessionResetReason.SOURCE_CHANGE)
        b = tracker.update([{"bbox": [0, 0, 50, 100], "confidence": 0.9}], now=2.0)
        self.assertEqual(b[0]["track_id"], 1)
        self.assertTrue(any(e.get("event") == "new_track" for e in tracker.last_events))
        # track id may reuse 1 after reset — identity is new streamRunId
        self.assertNotEqual(session.identity().stream_run_id, "")


if __name__ == "__main__":
    unittest.main()

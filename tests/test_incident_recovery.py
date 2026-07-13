"""Unit tests for Fall/Faint ROI incident recovery (lying-person continuity)."""
from __future__ import annotations

import unittest

from ai.postprocess.incident_recovery import (
    IncidentPhase,
    IncidentRecoveryManager,
    RecoveryConfig,
    candidate_passes_geometry,
    crop_bbox_to_source,
    expand_recovery_roi,
)


class ExpandRoiTest(unittest.TestCase):
    def test_roi_clamp_to_frame(self):
        # bbox near bottom-left edge
        roi = expand_recovery_roi([0, 600, 40, 720], frame_width=1280, frame_height=720)
        self.assertEqual(roi[0], 0)
        self.assertEqual(roi[3], 720)
        self.assertGreater(roi[2], roi[0])
        self.assertGreater(roi[3], roi[1])

    def test_roi_expands_left_right_down(self):
        cfg = RecoveryConfig(expand_left_ratio=0.5, expand_right_ratio=0.5, expand_down_ratio=0.5, expand_up_ratio=0.0)
        roi = expand_recovery_roi([100, 100, 200, 200], 1280, 720, cfg)
        self.assertLessEqual(roi[0], 50)  # left expand ~50
        self.assertGreaterEqual(roi[2], 250)
        self.assertGreaterEqual(roi[3], 250)


class CoordMapTest(unittest.TestCase):
    def test_crop_to_source_offset_only(self):
        src = crop_bbox_to_source([10, 20, 30, 40], [100, 200, 400, 500])
        self.assertEqual(src, [110.0, 220.0, 130.0, 240.0])


class GeometryGateTest(unittest.TestCase):
    def test_area_jump_rejected(self):
        ok, reason = candidate_passes_geometry([0, 0, 50, 100], [0, 0, 400, 400])
        self.assertFalse(ok)
        self.assertEqual(reason, "area_ratio")

    def test_near_candidate_ok(self):
        ok, reason = candidate_passes_geometry([100, 100, 160, 220], [105, 110, 165, 225])
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")


class IncidentRecoveryFlowTest(unittest.TestCase):
    def setUp(self):
        self.cfg = RecoveryConfig(
            miss_frames_to_start=2,
            recovery_interval_frames=1,
            max_recovery_seconds=3.0,
            recovery_conf=0.05,
        )
        self.mgr = IncidentRecoveryManager(self.cfg)

    def test_normal_no_recovery_without_suspected(self):
        calls = []

        def detect(_crop, conf, imgsz):
            calls.append((conf, imgsz))
            return [{"bbox": [0, 0, 10, 10], "confidence": 0.9}]

        out = self.mgr.on_tracked_frame(
            camera_login_id="cam_03",
            tracked=[{"track_id": 1, "bbox": [10, 10, 50, 100], "confidence": 0.8}],
            timestamp=1.0,
            frame_id=1,
            frame_shape=(720, 1280),
            frame_bgr=object(),
            detect_roi_fn=detect,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(calls, [])

    def test_no_recovery_before_two_miss_frames(self):
        self.mgr.note_fall_faint_suspected(
            camera_login_id="cam_03",
            track_id=7,
            bbox=[100, 100, 160, 220],
            timestamp=1.0,
            frame_id=10,
        )
        calls = []

        def detect(_c, conf, imgsz):
            calls.append(1)
            return [{"bbox": [0, 0, 50, 100], "confidence": 0.5}]

        # frame without track 7 — miss_frames=1 < 2
        self.mgr.on_tracked_frame(
            camera_login_id="cam_03",
            tracked=[],
            timestamp=1.1,
            frame_id=11,
            frame_shape=(720, 1280),
            frame_bgr=_FakeFrame(720, 1280),
            detect_roi_fn=detect,
        )
        self.assertEqual(calls, [])
        rec = self.mgr.open_incidents("cam_03")[0]
        self.assertEqual(rec.miss_frames, 1)
        self.assertEqual(rec.phase, IncidentPhase.FALL_FAINT_SUSPECTED)

    def test_single_candidate_relink(self):
        rec = self.mgr.note_fall_faint_suspected(
            camera_login_id="cam_03",
            track_id=7,
            bbox=[100, 100, 160, 220],
            timestamp=1.0,
            frame_id=10,
            incident_id="inc-test-1",
        )
        self.assertEqual(rec.incident_id, "inc-test-1")

        def detect(crop, conf, imgsz):
            # return crop-local box roughly matching last person
            return [{"bbox": [20, 20, 80, 140], "confidence": 0.4, "keypoints": []}]

        # two misses
        self.mgr.on_tracked_frame(
            camera_login_id="cam_03",
            tracked=[],
            timestamp=1.1,
            frame_id=11,
            frame_shape=(720, 1280),
            frame_bgr=_FakeFrame(720, 1280),
            detect_roi_fn=detect,
        )
        out = self.mgr.on_tracked_frame(
            camera_login_id="cam_03",
            tracked=[],
            timestamp=1.2,
            frame_id=12,
            frame_shape=(720, 1280),
            frame_bgr=_FakeFrame(720, 1280),
            detect_roi_fn=detect,
        )
        recovered = [d for d in out if d.get("recovery_relink")]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["incident_id"], "inc-test-1")
        # Must NOT force the lost source track_id; assignment happens in finalize.
        self.assertIsNone(recovered[0].get("track_id"))
        self.assertEqual(recovered[0]["recovered_from_track_id"], 7)
        self.assertTrue(recovered[0]["recovery_relink"])
        # bbox remapped with ROI offset (not still crop-local only)
        self.assertGreater(recovered[0]["bbox"][0], 20)
        self.assertEqual(self.mgr.stats.recovery_successes, 1)

        from ai.postprocess.track_state_migration import finalize_recovery_detections
        from tracking.simple_tracker import SimpleTrackAssigner

        tracker = SimpleTrackAssigner()
        finalized, migrations = finalize_recovery_detections(
            recovered,
            camera_login_id="cam_03",
            incident_recovery=self.mgr,
            tracker=tracker,
            now=1.2,
        )
        self.assertEqual(len(finalized), 1)
        self.assertIsNotNone(finalized[0].get("track_id"))
        self.assertNotEqual(int(finalized[0]["track_id"]), 7)
        self.assertEqual(len(migrations), 1)
        self.assertEqual(migrations[0]["from_track_id"], 7)
        self.assertEqual(self.mgr.incident_for_track("cam_03", int(finalized[0]["track_id"])), "inc-test-1")

    def test_multi_candidate_rejected(self):
        self.mgr.note_fall_faint_suspected(
            camera_login_id="cam_03",
            track_id=7,
            bbox=[100, 100, 160, 220],
            timestamp=1.0,
            frame_id=10,
        )

        def detect(_c, conf, imgsz):
            return [
                {"bbox": [10, 10, 50, 80], "confidence": 0.5},
                {"bbox": [60, 10, 100, 80], "confidence": 0.4},
            ]

        self.mgr.on_tracked_frame(
            camera_login_id="cam_03",
            tracked=[],
            timestamp=1.1,
            frame_id=11,
            frame_shape=(720, 1280),
            frame_bgr=_FakeFrame(720, 1280),
            detect_roi_fn=detect,
        )
        out = self.mgr.on_tracked_frame(
            camera_login_id="cam_03",
            tracked=[],
            timestamp=1.2,
            frame_id=12,
            frame_shape=(720, 1280),
            frame_bgr=_FakeFrame(720, 1280),
            detect_roi_fn=detect,
        )
        self.assertEqual([d for d in out if d.get("recovery_relink")], [])
        self.assertGreaterEqual(self.mgr.stats.recovery_rejects, 1)

    def test_active_track_owner_excluded(self):
        self.mgr.note_fall_faint_suspected(
            camera_login_id="cam_03",
            track_id=7,
            bbox=[100, 100, 160, 220],
            timestamp=1.0,
            frame_id=10,
        )

        def detect(_c, conf, imgsz):
            # will map near [100,100,...] after offset — same region as active track 9
            return [{"bbox": [0, 0, 60, 120], "confidence": 0.6}]

        self.mgr.on_tracked_frame(
            camera_login_id="cam_03",
            tracked=[],
            timestamp=1.1,
            frame_id=11,
            frame_shape=(720, 1280),
            frame_bgr=_FakeFrame(720, 1280),
            detect_roi_fn=detect,
        )
        # other person still tracked in same area
        out = self.mgr.on_tracked_frame(
            camera_login_id="cam_03",
            tracked=[{"track_id": 9, "bbox": [100, 100, 160, 220], "confidence": 0.9}],
            timestamp=1.2,
            frame_id=12,
            frame_shape=(720, 1280),
            frame_bgr=_FakeFrame(720, 1280),
            detect_roi_fn=detect,
        )
        self.assertEqual([d for d in out if d.get("recovery_relink")], [])
        rec = self.mgr.open_incidents("cam_03")[0]
        self.assertEqual(rec.last_reject_reason, "owned_by_active_track")

    def test_timeout_to_fall_unrecovered(self):
        self.mgr.note_fall_faint_suspected(
            camera_login_id="cam_03",
            track_id=7,
            bbox=[100, 100, 160, 220],
            timestamp=1.0,
            frame_id=10,
        )

        def detect(_c, conf, imgsz):
            return []  # always miss

        t = 1.0
        fid = 10
        for _ in range(20):
            t += 0.2
            fid += 1
            self.mgr.on_tracked_frame(
                camera_login_id="cam_03",
                tracked=[],
                timestamp=t,
                frame_id=fid,
                frame_shape=(720, 1280),
                frame_bgr=_FakeFrame(720, 1280),
                detect_roi_fn=detect,
            )
        rec = self.mgr.open_incidents("cam_03")[0]
        self.assertEqual(rec.phase, IncidentPhase.FALL_UNRECOVERED)
        self.assertGreaterEqual(self.mgr.stats.timeouts, 1)

    def test_reset_camera_clears_context(self):
        self.mgr.note_fall_faint_suspected(
            camera_login_id="cam_03",
            track_id=7,
            bbox=[100, 100, 160, 220],
            timestamp=1.0,
            frame_id=10,
        )
        self.mgr.note_fall_faint_suspected(
            camera_login_id="cam_02",
            track_id=1,
            bbox=[10, 10, 50, 80],
            timestamp=1.0,
            frame_id=1,
        )
        self.mgr.reset_camera("cam_03")
        self.assertEqual(self.mgr.open_incidents("cam_03"), [])
        self.assertEqual(len(self.mgr.open_incidents("cam_02")), 1)

    def test_cameras_independent(self):
        self.mgr.note_fall_faint_suspected(
            camera_login_id="cam_03",
            track_id=7,
            bbox=[100, 100, 160, 220],
            timestamp=1.0,
            frame_id=10,
            incident_id="inc-a",
        )
        self.mgr.note_fall_faint_suspected(
            camera_login_id="cam_02",
            track_id=7,
            bbox=[10, 10, 50, 80],
            timestamp=1.0,
            frame_id=10,
            incident_id="inc-b",
        )
        self.assertEqual(self.mgr.incident_for_track("cam_03", 7), "inc-a")
        self.assertEqual(self.mgr.incident_for_track("cam_02", 7), "inc-b")

    def test_reset_all(self):
        self.mgr.note_fall_faint_suspected(
            camera_login_id="cam_03",
            track_id=7,
            bbox=[100, 100, 160, 220],
            timestamp=1.0,
            frame_id=10,
        )
        self.mgr.reset_all()
        self.assertEqual(self.mgr.open_incidents("cam_03"), [])


class TrackStateMigrationTest(unittest.TestCase):
    def test_sequence_buffer_migrate(self):
        from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers

        buf = PerTrackKeypointSequenceBuffers(sequence_length=5, stride=1, max_track_age_seconds=30.0)
        det = {
            "track_id": 3,
            "bbox": [10, 10, 50, 100],
            "keypoints": [{"x": 1, "y": 2, "confidence": 0.9}] * 17,
        }
        for i in range(3):
            buf.add(i, [det], frame_shape=(720, 1280), now=float(i))
        self.assertIn(3, buf._buffers)
        self.assertTrue(buf.migrate_track_id(3, 99))
        self.assertNotIn(3, buf._buffers)
        self.assertIn(99, buf._buffers)

    def test_display_id_transfer(self):
        from tracking.display_id_mapper import DisplayIdMapper

        m = DisplayIdMapper()
        m.update({7})
        disp = m.display_id(7)
        self.assertEqual(disp, 1)
        self.assertTrue(m.transfer_raw_id(7, 42))
        self.assertIsNone(m.display_id(7))
        self.assertEqual(m.display_id(42), 1)

    def test_fall_state_migrate(self):
        from ai.action.fall_event_state import FallEventStateMachine, FallState

        sm = FallEventStateMachine(min_consecutive_faint=1)
        sm.update("cam", 1.0, track_id=5, is_alert=True, posture_label="lying_like")
        self.assertNotEqual(sm.get_state("cam", 5), FallState.NORMAL)
        self.assertTrue(sm.migrate_track("cam", 5, 88))
        self.assertEqual(sm.get_state("cam", 5), FallState.NORMAL)
        self.assertNotEqual(sm.get_state("cam", 88), FallState.NORMAL)

    def test_recovery_does_not_force_source_id(self):
        from ai.postprocess.track_state_migration import finalize_recovery_detections
        from tracking.simple_tracker import SimpleTrackAssigner

        cfg = RecoveryConfig(miss_frames_to_start=1, recovery_interval_frames=1)
        mgr = IncidentRecoveryManager(cfg)
        mgr.note_fall_faint_suspected(
            camera_login_id="cam_x",
            track_id=11,
            bbox=[100, 100, 160, 220],
            timestamp=1.0,
            frame_id=1,
            incident_id="inc-x",
        )

        def detect(_c, conf, imgsz):
            return [{"bbox": [20, 20, 80, 140], "confidence": 0.5}]

        out = mgr.on_tracked_frame(
            camera_login_id="cam_x",
            tracked=[],
            timestamp=1.1,
            frame_id=2,
            frame_shape=(720, 1280),
            frame_bgr=_FakeFrame(720, 1280),
            detect_roi_fn=detect,
        )
        recovered = [d for d in out if d.get("recovery_relink")]
        tracker = SimpleTrackAssigner()
        finalized, migrations = finalize_recovery_detections(
            recovered,
            camera_login_id="cam_x",
            incident_recovery=mgr,
            tracker=tracker,
            now=1.1,
        )
        self.assertEqual(len(finalized), 1)
        new_tid = int(finalized[0]["track_id"])
        self.assertNotEqual(new_tid, 11)
        self.assertEqual(migrations[0]["from_track_id"], 11)
        self.assertEqual(migrations[0]["to_track_id"], new_tid)
        # Second recovery continues under linked id (no force back to 11)
        out2 = mgr.on_tracked_frame(
            camera_login_id="cam_x",
            tracked=[],
            timestamp=1.3,
            frame_id=4,
            frame_shape=(720, 1280),
            frame_bgr=_FakeFrame(720, 1280),
            detect_roi_fn=detect,
        )
        # miss again after linked track absent for miss_frames
        recovered2 = [d for d in out2 if d.get("recovery_relink")]
        if recovered2:
            fin2, mig2 = finalize_recovery_detections(
                recovered2,
                camera_login_id="cam_x",
                incident_recovery=mgr,
                tracker=tracker,
                now=1.3,
            )
            self.assertEqual(int(fin2[0]["track_id"]), new_tid)
            self.assertEqual(mig2, [])


class _FakeCrop:
    size = 100


class _FakeFrame:
    def __init__(self, h, w):
        self.shape = (h, w, 3)

    def __getitem__(self, item):
        return _FakeCrop()


if __name__ == "__main__":
    unittest.main()

"""Regression: keypoint_motion54 velocity must not spike across recovery/large gaps."""
from __future__ import annotations

import unittest

import numpy as np

from ai.action.classifier import keypoint_sequence_to_features, sequence_to_lstm_features
from ai.action.feature_schema import KEYPOINT_MOTION54_SCHEMA_VERSION
from ai.action.motion_features import append_motion_features, build_motion_discontinuity_mask
from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers
from ai.postprocess.incident_recovery import IncidentRecoveryManager, RecoveryConfig
from ai.postprocess.track_state_migration import finalize_recovery_detections
from tracking.simple_tracker import SimpleTrackAssigner


BASELINE_P95 = 0.008375
REPORTED_SPIKE = 0.177131


def _pose(y: float, x: float = 100.0) -> list[dict]:
    points = [{"x": 0.0, "y": 0.0, "confidence": 0.0} for _ in range(17)]
    points[5] = {"x": x - 10, "y": y - 50, "confidence": 0.9}
    points[6] = {"x": x + 10, "y": y - 50, "confidence": 0.9}
    points[11] = {"x": x - 5, "y": y, "confidence": 0.9}
    points[12] = {"x": x + 5, "y": y, "confidence": 0.9}
    return points


def _sequence_from_hip_ys(ys: list[float], *, frame_ids=None, times=None, markers=None, width=640.0, height=480.0):
    detections = []
    shapes = []
    for i, y in enumerate(ys):
        det = {
            "keypoints": _pose(y),
            "bbox": [50.0, y - 80, 150.0, y + 20],
        }
        if markers and i < len(markers) and markers[i]:
            det.update(markers[i])
        detections.append(det)
        shapes.append((int(height), int(width), 3))
    sequence = {
        "detections": detections,
        "frame_shapes": shapes,
        "frame_ids": list(frame_ids) if frame_ids is not None else list(range(len(ys))),
        "frame_idxs": list(frame_ids) if frame_ids is not None else list(range(len(ys))),
        "sample_captured_at_ms": list(times) if times is not None else [i * 33 for i in range(len(ys))],
    }
    return sequence


class ContinuousMotionContractTest(unittest.TestCase):
    def test_first_frame_zero_and_raw_displacement(self):
        # hip y in pixels: 200 then 220 → normalized by height 480
        seq = _sequence_from_hip_ys([200.0, 220.0])
        base = keypoint_sequence_to_features(seq, expected_input_size=51, feature_schema="keypoint51")
        out = append_motion_features(base)  # no mask → first zero, second raw
        self.assertEqual(out.shape, (2, 54))
        self.assertEqual(float(out[0, 51]), 0.0)
        self.assertEqual(float(out[0, 52]), 0.0)
        expected_drop = (220.0 - 200.0) / 480.0
        self.assertAlmostEqual(float(out[1, 51]), expected_drop, places=5)
        self.assertAlmostEqual(float(out[1, 52]), abs(expected_drop), places=5)
        self.assertTrue(0.0 <= float(out[0, 53]) <= 1.0)
        self.assertTrue(0.0 <= float(out[1, 53]) <= 1.0)

    def test_feature_order_dims(self):
        seq = _sequence_from_hip_ys([200.0, 210.0, 230.0])
        feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        self.assertEqual(feats.shape[-1], 54)
        # dim51 center_drop, 52 velocity, 53 torso
        self.assertGreater(float(feats[2, 51]), 0.0)
        self.assertGreater(float(feats[2, 52]), 0.0)


class DiscontinuityMotionTest(unittest.TestCase):
    def test_large_frame_gap_zeros_motion(self):
        # 30 continuous then jump by 67 frames with large spatial jump (recovery-like)
        ys = [200.0 + i * 0.5 for i in range(10)]
        ys.append(200.0 + 0.5 * 10 + 80.0)  # large spatial jump after gap
        frame_ids = list(range(10)) + [10 + 67]
        seq = _sequence_from_hip_ys(ys, frame_ids=frame_ids)
        feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        # post-gap index = 10
        self.assertEqual(float(feats[10, 51]), 0.0)
        self.assertEqual(float(feats[10, 52]), 0.0)
        self.assertLess(float(feats[10, 52]), BASELINE_P95 * 2)
        self.assertNotAlmostEqual(float(feats[10, 52]), REPORTED_SPIKE, places=3)
        self.assertTrue(np.isfinite(feats[10, 53]))
        # prior continuous steps still non-zero raw motion
        self.assertGreater(float(feats[5, 52]), 0.0)

    def test_small_queue_drop_does_not_reset(self):
        # gap of 2 frames (within DEFAULT_MAX_CONTINUOUS_FRAME_STEP=3)
        ys = [200.0, 205.0]
        frame_ids = [0, 2]
        seq = _sequence_from_hip_ys(ys, frame_ids=frame_ids)
        feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        expected = abs((205.0 - 200.0) / 480.0)
        self.assertAlmostEqual(float(feats[1, 52]), expected, places=5)

    def test_large_time_gap_zeros_motion(self):
        # Timestamp fallback only when frame metadata is absent.
        ys = [200.0, 280.0]
        times = [0, 500]  # 500ms >> 150ms
        seq = _sequence_from_hip_ys(ys, frame_ids=None, times=times)
        seq["frame_ids"] = []
        seq["frame_idxs"] = []
        feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        self.assertEqual(float(feats[1, 51]), 0.0)
        self.assertEqual(float(feats[1, 52]), 0.0)

    def test_recovery_relink_marker_zeros_motion(self):
        ys = [200.0, 290.0]
        markers = [None, {"recovery_relink": True, "recovered_from_track_id": 7}]
        seq = _sequence_from_hip_ys(ys, frame_ids=[0, 1], markers=markers)
        # even without frame gap, marker forces zero
        feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        self.assertEqual(float(feats[1, 51]), 0.0)
        self.assertEqual(float(feats[1, 52]), 0.0)

    def test_mask_builder_flags_gap_67(self):
        seq = {
            "frame_ids": [0, 67],
            "detections": [{}, {}],
        }
        mask = build_motion_discontinuity_mask(seq, 2)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])


class RecoveryFreshStartTest(unittest.TestCase):
    def test_recovery_does_not_merge_keypoint_history(self):
        buffers = PerTrackKeypointSequenceBuffers(sequence_length=30, stride=1, max_track_age_seconds=60.0)
        # Pre-miss history on track 7
        for i in range(20):
            det = {
                "track_id": 7,
                "bbox": [10, 10, 50, 100],
                "keypoints": _pose(200.0 + i),
            }
            buffers.add(i, [det], frame_shape=(480, 640), now=float(i), frame_id=i, captured_at_ms=i * 33)

        self.assertEqual(len(buffers._buffers[7]._frames), 20)

        class _Tracker:
            def register_recovery_detection(self, detection, *, now=None):
                out = dict(detection)
                out["track_id"] = 99
                return out

        mgr = IncidentRecoveryManager(RecoveryConfig(miss_frames_to_start=1, recovery_interval_frames=1))
        mgr.note_fall_faint_suspected(
            camera_login_id="cam_01",
            track_id=7,
            bbox=[10, 10, 50, 100],
            timestamp=1.0,
            frame_id=20,
            incident_id="inc-1",
        )
        recovery_det = {
            "recovery_relink": True,
            "incident_id": "inc-1",
            "recovered_from_track_id": 7,
            "bbox": [12, 12, 52, 102],
            "keypoints": _pose(280.0),
            "confidence": 0.8,
        }
        finalized, migrations = finalize_recovery_detections(
            [recovery_det],
            camera_login_id="cam_01",
            incident_recovery=mgr,
            tracker=_Tracker(),
            now=30.0,
            sequence_buffer=buffers,
        )
        self.assertEqual(len(finalized), 1)
        self.assertEqual(int(finalized[0]["track_id"]), 99)
        self.assertTrue(migrations)
        self.assertTrue(migrations[0].get("sequence_fresh_start"))
        # Old history dropped; new track does not inherit pre-gap frames
        self.assertNotIn(7, buffers._buffers)
        self.assertNotIn(99, buffers._buffers)  # fresh until next add
        # After recovery frame is added under new id, buffer starts at 1
        buffers.add(
            87,
            [finalized[0]],
            frame_shape=(480, 640),
            now=30.0,
            frame_id=87,
            captured_at_ms=87 * 33,
        )
        self.assertIn(99, buffers._buffers)
        self.assertEqual(len(buffers._buffers[99]._frames), 1)

    def test_merged_history_would_spike_but_fresh_start_prevents(self):
        """Reproduce measured spike class if merge were used; fresh-start avoids it."""
        # Simulate merged sequence: 5 pre frames then post-gap frame 67 steps later
        pre_ys = [200.0 + i for i in range(5)]
        post_y = 200.0 + 4 + 85.0  # large spatial jump
        ys = pre_ys + [post_y]
        frame_ids = list(range(5)) + [5 + 67]
        seq = _sequence_from_hip_ys(ys, frame_ids=frame_ids)
        # Without discontinuity mask (legacy raw only)
        base = keypoint_sequence_to_features(seq, expected_input_size=51, feature_schema="keypoint51")
        raw = append_motion_features(base)
        raw_spike = float(raw[-1, 52])
        self.assertGreater(raw_spike, BASELINE_P95 * 5)

        # With shipped motion54 path (mask)
        feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        self.assertEqual(float(feats[-1, 52]), 0.0)
        self.assertLess(float(feats[-1, 52]), BASELINE_P95)

    def test_ordinary_migrate_still_merges_when_not_fresh_start(self):
        buffers = PerTrackKeypointSequenceBuffers(sequence_length=5, stride=1, max_track_age_seconds=30.0)
        for track_id, frame_id in [(1, 1), (1, 3), (2, 2), (2, 3)]:
            buffers.add(
                frame_id,
                [
                    {
                        "track_id": track_id,
                        "bbox": ([0, 0, 10, 10] if track_id == 1 else [100, 100, 110, 110]),
                        "keypoints": _pose(200.0),
                    }
                ],
                frame_shape=(10, 10),
                now=float(frame_id),
                frame_id=frame_id,
            )
        self.assertTrue(buffers.migrate_track_id(1, 2, fresh_start_history=False))
        self.assertEqual([item["frame_id"] for item in buffers._buffers[2]._frames], [1, 2, 3])

    def test_new_id_existing_history_is_removed_on_recovery_fresh_start(self):
        from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer

        buffers = PerTrackKeypointSequenceBuffers(sequence_length=10, stride=1, max_track_age_seconds=60.0)
        # Seed both tracks directly to avoid ordinary IoU relink inside add().
        old_buf = KeypointSequenceBuffer(sequence_length=10, stride=1)
        new_buf = KeypointSequenceBuffer(sequence_length=10, stride=1)
        old_buf._frames = [
            {"frame_id": i, "frame_idx": i, "detection": {"track_id": 7, "keypoints": _pose(200.0)}, "captured_at_ms": i * 33}
            for i in range(5)
        ]
        new_buf._frames = [
            {"frame_id": 100 + i, "frame_idx": 100 + i, "detection": {"track_id": 99, "keypoints": _pose(250.0)}, "captured_at_ms": (100 + i) * 33}
            for i in range(3)
        ]
        buffers._buffers = {7: old_buf, 99: new_buf}
        buffers._last_seen_at = {7: 1.0, 99: 2.0}
        buffers._last_detection_by_track = {
            7: {"track_id": 7, "bbox": [0, 0, 10, 10]},
            99: {"track_id": 99, "bbox": [500, 500, 520, 520]},
        }
        buffers.last_sequence_diagnostics = {7: {"track_id": 7}, 99: {"track_id": 99, "buffer_length": 3}}
        buffers.sequences_generated_by_track = {7: 1, 99: 2}

        self.assertTrue(buffers.migrate_track_id(7, 99, fresh_start_history=True))
        self.assertNotIn(7, buffers._buffers)
        self.assertNotIn(99, buffers._buffers)
        self.assertNotIn(7, buffers._last_seen_at)
        self.assertNotIn(99, buffers._last_seen_at)
        self.assertNotIn(7, buffers._last_detection_by_track)
        self.assertNotIn(99, buffers._last_detection_by_track)
        self.assertNotIn(7, buffers.sequences_generated_by_track)
        self.assertNotIn(99, buffers.sequences_generated_by_track)
        self.assertEqual(buffers.last_sequence_diagnostics[99]["buffer_length"], 0)
        self.assertEqual(buffers.last_sequence_diagnostics[99]["reason"], "recovery_sequence_fresh_start")
        self.assertEqual(buffers.last_sequence_diagnostics[99]["migrated_from"], 7)
        self.assertNotIn(7, buffers.last_sequence_diagnostics)

    def test_crop_buffer_new_id_history_is_removed_on_fresh_start(self):
        from ai.action.per_track_sequence_buffer import PerTrackCropSequenceBuffers
        import numpy as np

        crops = PerTrackCropSequenceBuffers(sequence_length=5, stride=1, resize_size=32, max_track_age_seconds=60.0)
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        for i in range(3):
            crops.add(i, frame, [{"track_id": 1, "x1": 1, "y1": 1, "x2": 10, "y2": 10}], now=float(i), frame_id=i)
        for i in range(2):
            crops.add(50 + i, frame, [{"track_id": 2, "x1": 2, "y1": 2, "x2": 12, "y2": 12}], now=float(50 + i), frame_id=50 + i)
        self.assertIn(1, crops._buffers)
        self.assertIn(2, crops._buffers)
        self.assertTrue(crops.migrate_track_id(1, 2, fresh_start_history=True))
        self.assertNotIn(1, crops._buffers)
        self.assertNotIn(2, crops._buffers)
        self.assertNotIn(1, crops._last_seen_at)
        self.assertNotIn(2, crops._last_seen_at)
        self.assertNotIn(1, crops.last_sequence_diagnostics)
        self.assertNotIn(2, crops.last_sequence_diagnostics)
        self.assertNotIn(1, crops.sequences_generated_by_track)
        self.assertNotIn(2, crops.sequences_generated_by_track)

    def test_ordinary_tracker_relink_remains_unchanged(self):
        """Alias for merge path: fresh_start_history=False keeps ordered merge."""
        self.test_ordinary_migrate_still_merges_when_not_fresh_start()


class FrameTimestampPriorityTest(unittest.TestCase):
    def test_frame_step_three_uses_frame_metadata_and_does_not_reset(self):
        ys = [200.0, 220.0]
        # step=3 allowed; timestamp 200ms would reset if time were consulted
        seq = _sequence_from_hip_ys(ys, frame_ids=[0, 3], times=[0, 200])
        feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        expected = abs((220.0 - 200.0) / 480.0)
        self.assertAlmostEqual(float(feats[1, 52]), expected, places=5)
        self.assertGreater(float(feats[1, 52]), 0.0)

    def test_timestamp_fallback_resets_when_frame_metadata_missing(self):
        ys = [200.0, 280.0]
        seq = _sequence_from_hip_ys(ys, frame_ids=None, times=[0, 500])
        # clear frame metadata explicitly
        seq["frame_ids"] = []
        seq["frame_idxs"] = []
        feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        self.assertEqual(float(feats[1, 51]), 0.0)
        self.assertEqual(float(feats[1, 52]), 0.0)

    def test_frame_step_four_resets_even_when_timestamp_small(self):
        ys = [200.0, 280.0]
        seq = _sequence_from_hip_ys(ys, frame_ids=[0, 4], times=[0, 30])
        feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        self.assertEqual(float(feats[1, 51]), 0.0)
        self.assertEqual(float(feats[1, 52]), 0.0)

    def test_non_monotonic_frame_metadata_resets(self):
        ys = [200.0, 220.0]
        for frame_ids in ([5, 5], [5, 4]):
            seq = _sequence_from_hip_ys(ys, frame_ids=frame_ids, times=[0, 33])
            feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
            self.assertEqual(float(feats[1, 51]), 0.0, msg=frame_ids)
            self.assertEqual(float(feats[1, 52]), 0.0, msg=frame_ids)


if __name__ == "__main__":
    unittest.main()

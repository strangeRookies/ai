"""Regression tests for incident-recovery data integrity contracts."""
from __future__ import annotations
import unittest
from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers
from ai.postprocess.incident_recovery import IncidentRecoveryManager, RecoveryConfig, RecoveryStats, make_detect_roi_fn_from_yolo_pose
from ai.postprocess.track_state_migration import finalize_recovery_detections, migrate_sequence_buffer

class _PublicApiOnlyDetector:
    def __init__(self):
        self.calls = []
    def detect(self, frame, *, conf=None, imgsz=None):
        self.calls.append((frame, float(conf or 0.0), int(imgsz or 0)))
        return [{"bbox": [1.0, 2.0, 3.0, 4.0], "confidence": 0.8, "keypoints": []}]

class IncidentRecoveryIntegrityTest(unittest.TestCase):
    def test_roi_detection_uses_public_api(self):
        detector = _PublicApiOnlyDetector()
        crop = object()
        detections = make_detect_roi_fn_from_yolo_pose(detector)(crop, 0.05, 640)
        self.assertEqual(detections[0]["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(detector.calls, [(crop, 0.05, 640)])

    def test_registration_failure_never_emits_synthetic_track(self):
        manager = IncidentRecoveryManager(RecoveryConfig())
        manager.note_fall_faint_suspected(camera_login_id="cam-1", track_id=7, bbox=[0, 0, 50, 100], timestamp=1.0, frame_id=1, incident_id="incident-1")
        active, migrations = finalize_recovery_detections(
            [{"recovery_relink": True, "incident_id": "incident-1", "recovered_from_track_id": 7, "bbox": [0, 0, 50, 100]}],
            camera_login_id="cam-1", incident_recovery=manager, tracker=None, now=1.1)
        self.assertEqual(active, [])
        self.assertEqual(migrations, [])
        record = manager.get_incident("cam-1", "incident-1")
        self.assertIsNotNone(record)
        self.assertEqual(record.last_reject_reason, "tracker_registration_unavailable")
        self.assertEqual(record.active_track_id, 7)
        self.assertFalse(record.recovery_linked)

    def test_sequence_collision_merges_ordered_frames(self):
        buffers = PerTrackKeypointSequenceBuffers(sequence_length=5, stride=1, max_track_age_seconds=30.0)
        for track_id, frame_id in [(1, 1), (1, 3), (2, 2), (2, 3)]:
            buffers.add(frame_id, [{"track_id": track_id, "bbox": ([0, 0, 10, 10] if track_id == 1 else [100, 100, 110, 110]), "keypoints": [{"x": 1.0, "y": 2.0, "confidence": 0.9}] * 17}], frame_shape=(10, 10), now=float(frame_id), frame_id=frame_id)
        self.assertTrue(buffers.migrate_track_id(1, 2))
        self.assertNotIn(1, buffers._buffers)
        self.assertEqual([item["frame_id"] for item in buffers._buffers[2]._frames], [1, 2, 3])

    def test_sequence_collision_preserves_latest_emit_timestamp(self):
        buffers = PerTrackKeypointSequenceBuffers(sequence_length=3, stride=1, max_track_age_seconds=30.0)
        source = KeypointSequenceBuffer(sequence_length=3, stride=1)
        destination = KeypointSequenceBuffer(sequence_length=3, stride=1)
        source._frames = [{"frame_id": 1, "frame_idx": 1, "detection": {"track_id": 1}}]
        destination._frames = [{"frame_id": 2, "frame_idx": 2, "detection": {"track_id": 2}}]
        source._last_emit_at_ms = 1000
        destination._last_emit_at_ms = 1100
        buffers._buffers = {1: source, 2: destination}
        buffers._last_seen_at = {1: 1.0, 2: 2.0}

        self.assertTrue(buffers.migrate_track_id(1, 2))
        self.assertEqual(buffers._buffers[2]._last_emit_at_ms, 1100)

    def test_incompatible_collision_preserves_both_and_records_conflict(self):
        buffers = PerTrackKeypointSequenceBuffers(sequence_length=5, stride=1, max_track_age_seconds=30.0)
        buffers._buffers[1] = KeypointSequenceBuffer(sequence_length=3, stride=1)
        buffers._buffers[2] = KeypointSequenceBuffer(sequence_length=5, stride=1)
        buffers._last_seen_at.update({1: 1.0, 2: 2.0})
        self.assertFalse(buffers.migrate_track_id(1, 2))
        self.assertIn(1, buffers._buffers)
        self.assertIn(2, buffers._buffers)
        self.assertEqual(buffers.migration_conflicts[-1]["reason"], "incompatible_buffer")

    def test_generic_collision_preserves_source(self):
        class FallbackBuffer:
            def __init__(self):
                self._buffers = {1: "source", 2: "destination"}
        buffers = FallbackBuffer()
        self.assertFalse(migrate_sequence_buffer(buffers, 1, 2))
        self.assertEqual(buffers._buffers, {1: "source", 2: "destination"})

    def test_wrong_relink_is_not_evaluated(self):
        diagnostics = RecoveryStats().as_dict()
        self.assertIsNone(diagnostics["wrong_relink"])
        self.assertEqual(diagnostics["wrong_relink_evaluation_status"], "not_evaluated")

if __name__ == "__main__":
    unittest.main()

import unittest

from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers


class TrackingResilienceTest(unittest.TestCase):
    def test_keypoint_sequence_buffer_relinks_changed_track_id_by_iou(self):
        buffer = PerTrackKeypointSequenceBuffers(
            sequence_length=2,
            stride=1,
            max_track_age_seconds=10.0,
            relink_iou_threshold=0.4,
            relink_center_distance_ratio=0.5,
            relink_max_time_gap_seconds=2.0,
        )

        first = {
            "track_id": 10,
            "bbox": [0, 0, 100, 200],
            "keypoints": [{"x": 1, "y": 1, "confidence": 0.9} for _ in range(17)],
        }
        changed = {
            "track_id": 99,
            "bbox": [4, 2, 104, 202],
            "keypoints": [{"x": 2, "y": 2, "confidence": 0.9} for _ in range(17)],
        }

        buffer.add(0, [first], (240, 320, 3), now=1.0)
        sequences = buffer.add(1, [changed], (240, 320, 3), now=1.5)

        self.assertEqual(sequences[0]["track_id"], 10)
        self.assertEqual(buffer.sequence_diagnostics()[10]["reason"], "sequence_ready")
        self.assertEqual(buffer.sequence_diagnostics()[10]["relink"], "success")
        self.assertEqual(buffer.relink_success_count, 1)

    def test_keypoint_sequence_buffer_keeps_missing_track_during_grace_period(self):
        buffer = PerTrackKeypointSequenceBuffers(
            sequence_length=3,
            stride=1,
            max_track_age_seconds=1.0,
            missing_track_grace_seconds=3.0,
        )
        detection = {
            "track_id": 7,
            "bbox": [0, 0, 100, 200],
            "keypoints": [{"x": 1, "y": 1, "confidence": 0.9} for _ in range(17)],
        }

        buffer.add(0, [detection], (240, 320, 3), now=1.0)
        buffer.add(1, [], (240, 320, 3), now=2.5)

        self.assertEqual(buffer.buffer_lengths(), {7: 1})
        self.assertEqual(buffer.sequence_diagnostics()[7]["reason"], "missing_track_grace")
        self.assertEqual(buffer.buffer_retained_count, 1)

    def test_keypoint_sequence_buffer_drops_after_grace_period(self):
        buffer = PerTrackKeypointSequenceBuffers(
            sequence_length=3,
            stride=1,
            max_track_age_seconds=1.0,
            missing_track_grace_seconds=1.0,
        )
        detection = {
            "track_id": 7,
            "bbox": [0, 0, 100, 200],
            "keypoints": [{"x": 1, "y": 1, "confidence": 0.9} for _ in range(17)],
        }

        buffer.add(0, [detection], (240, 320, 3), now=1.0)
        buffer.add(1, [], (240, 320, 3), now=3.5)

        self.assertEqual(buffer.buffer_lengths(), {})
        self.assertEqual(buffer.buffer_deleted_count, 1)


if __name__ == "__main__":
    unittest.main()

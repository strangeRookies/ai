import unittest

from ai.action.cheap_filter import CheapFilterConfig, evaluate_sequence_candidate
from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers


def keypoints(horizontal=False):
    points = [{"x": 10.0, "y": 10.0, "confidence": 0.9} for _ in range(17)]
    if horizontal:
        points[5] = {"x": 10.0, "y": 50.0, "confidence": 0.9}
        points[6] = {"x": 20.0, "y": 50.0, "confidence": 0.9}
        points[11] = {"x": 80.0, "y": 55.0, "confidence": 0.9}
        points[12] = {"x": 90.0, "y": 55.0, "confidence": 0.9}
    else:
        points[5] = {"x": 45.0, "y": 20.0, "confidence": 0.9}
        points[6] = {"x": 55.0, "y": 20.0, "confidence": 0.9}
        points[11] = {"x": 45.0, "y": 80.0, "confidence": 0.9}
        points[12] = {"x": 55.0, "y": 80.0, "confidence": 0.9}
    return points


def detection(horizontal=False):
    return {"bbox": [10.0, 10.0, 90.0, 90.0], "keypoints": keypoints(horizontal), "track_id": 1}


class CheapFilterTest(unittest.TestCase):
    def test_horizontal_slope_1_3_keeps_faint_candidate(self):
        sequence = {"detections": [detection(horizontal=True) for _ in range(8)], "frame_shapes": [(100, 100, 3)] * 8}

        decision = evaluate_sequence_candidate(sequence, CheapFilterConfig())

        self.assertTrue(decision.keep)
        self.assertIn("slope_ratio_1.3", decision.reasons)

    def test_low_risk_upright_candidate_is_skipped_before_lstm(self):
        sequence = {"detections": [detection(horizontal=False) for _ in range(8)], "frame_shapes": [(100, 100, 3)] * 8}

        decision = evaluate_sequence_candidate(sequence, CheapFilterConfig())

        self.assertFalse(decision.keep)
        self.assertEqual(decision.reasons, ("keypoint_confidence", "bbox_size"))

    def test_per_track_buffer_counts_kept_and_skipped_sequences(self):
        buffer = PerTrackKeypointSequenceBuffers(sequence_length=2, stride=1, cheap_filter_config=CheapFilterConfig())

        skipped = buffer.add(0, [detection(horizontal=False)], (100, 100, 3))
        skipped += buffer.add(1, [detection(horizontal=False)], (100, 100, 3))
        kept = buffer.add(2, [detection(horizontal=True)], (100, 100, 3))

        self.assertEqual(skipped, [])
        self.assertEqual(len(kept), 1)
        self.assertEqual(buffer.sequences_skipped_by_filter, 1)
        self.assertEqual(buffer.sequences_kept_by_filter, 1)


if __name__ == "__main__":
    unittest.main()

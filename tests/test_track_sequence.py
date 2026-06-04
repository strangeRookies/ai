import unittest

from rules.track_sequence import PerTrackSequenceBuffer


class TrackSequenceTest(unittest.TestCase):
    def test_marks_sequence_ready_per_track(self):
        buffer = PerTrackSequenceBuffer(sequence_length=2)

        first = buffer.update([{"track_id": 1, "bbox": [0, 0, 1, 1], "keypoint_confidence": 0.8}], now=1.0)
        second = buffer.update([{"track_id": 1, "bbox": [0, 0, 1, 1], "keypoint_confidence": 0.7}], now=1.1)

        self.assertFalse(first[0]["sequence_ready"])
        self.assertTrue(second[0]["sequence_ready"])
        self.assertEqual(second[0]["sequence_length"], 2)

    def test_tracks_keypoint_missing_rate(self):
        buffer = PerTrackSequenceBuffer(sequence_length=2)

        buffer.update([{"track_id": 1, "bbox": [0, 0, 1, 1], "keypoint_confidence": None}], now=1.0)
        output = buffer.update([{"track_id": 1, "bbox": [0, 0, 1, 1], "keypoint_confidence": 0.5}], now=1.1)

        self.assertEqual(output[0]["keypoint_missing_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()

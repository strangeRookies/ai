import unittest

from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer


def detection():
    return {
        "track_id": 7,
        "bbox": [0, 0, 20, 20],
        "keypoints": [{"x": 1.0, "y": 2.0, "confidence": 0.9}] * 17,
    }


class SequenceTimestampTimingTest(unittest.TestCase):
    def test_capture_timestamp_controls_stride_despite_frame_index_jump(self):
        buffer = KeypointSequenceBuffer(sequence_length=3, stride=2)

        self.assertIsNone(buffer.add(1, [detection()], captured_at_ms=1000))
        self.assertIsNone(buffer.add(2, [detection()], captured_at_ms=1033))
        first = buffer.add(3, [detection()], captured_at_ms=1066)

        self.assertEqual(first["sequence_timing_mode"], "captured_at_ms")
        self.assertEqual(first["sequence_duration_ms"], 66)
        self.assertEqual(first["sequence_stride_ms"], 66.0)
        self.assertIsNone(buffer.add(500, [detection()], captured_at_ms=1067))

        second = buffer.add(501, [detection()], captured_at_ms=1132)
        self.assertIsNotNone(second)
        self.assertEqual(second["sequence_end_frame_id"], 501)

    def test_missing_capture_timestamp_keeps_frame_index_fallback(self):
        buffer = KeypointSequenceBuffer(sequence_length=3, stride=2)

        buffer.add(1, [detection()])
        buffer.add(2, [detection()])
        first = buffer.add(3, [detection()])

        self.assertEqual(first["sequence_timing_mode"], "frame_index_fallback")
        self.assertIsNone(buffer.add(4, [detection()]))
        self.assertIsNotNone(buffer.add(5, [detection()]))


if __name__ == "__main__":
    unittest.main()

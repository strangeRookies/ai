import unittest

from ai.action.classifier import MockActionClassifier
from ai.publishers.event_publisher import build_event_payload


class ActionPayloadTest(unittest.TestCase):
    def test_mock_classifier_payload(self):
        prediction = MockActionClassifier(default_label="Fight", score=0.82).predict(sequence=[])
        payload = build_event_payload(
            camera_id="cam_01",
            frame_idx=12,
            timestamp=0.4,
            event_type=prediction["label"],
            score=prediction["score"],
            boxes=[],
        )
        self.assertEqual(payload["event_type"], "Fight")
        self.assertEqual(payload["score"], 0.82)
        self.assertEqual(payload["frame_idx"], 12)


if __name__ == "__main__":
    unittest.main()

import unittest

from ai.inference.rtsp_runtime import snapshot_assist_meta_from_event_payload


class SnapshotAssistMetaTest(unittest.TestCase):
    def test_maps_payload_fields(self):
        payload = {
            "eventId": "evt-1",
            "type": "faint",
            "confidence": 0.91,
            "lifecycleState": "CONFIRMED",
            "memoText": "쓰러짐 의심!",
            "capturedAtMs": 1_700_000_000_000,
            "trackingId": 7,
            "events": [
                {
                    "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
                    "keypoints": [[1, 2, 0.9]],
                }
            ],
        }
        meta = snapshot_assist_meta_from_event_payload(payload, track_id=7)
        self.assertEqual(meta["eventType"], "faint")
        self.assertEqual(meta["trackId"], "7")
        self.assertEqual(meta["confidence"], 0.91)
        self.assertEqual(meta["lifecycleState"], "CONFIRMED")
        self.assertEqual(meta["detectorReason"], "쓰러짐 의심!")
        self.assertIn("capturedAt", meta)
        self.assertIn("bbox", meta)
        self.assertIn("keypoints", meta)


if __name__ == "__main__":
    unittest.main()
import unittest

from messaging.event_schema import build_safety_event


class EventSchemaTest(unittest.TestCase):
    def test_build_safety_event_contains_backend_friendly_fields(self):
        event = build_safety_event(
            event_type="fall_detected",
            camera_id="cam_01",
            severity="HIGH",
            message="쓰러짐 의심 상황이 감지되었습니다.",
            track_id=1,
            metadata={"rule_score": 0.91},
            timestamp="2026-05-26T10:00:00Z",
        )

        self.assertEqual(event["type"], "fall_detected")
        self.assertEqual(event["event_type"], "fall_detected")
        self.assertEqual(event["schema_version"], "1.0")
        self.assertIn("event_id", event)
        self.assertEqual(event["camera_id"], "cam_01")
        self.assertEqual(event["status"], "confirmed")
        self.assertEqual(event["source"], "edge-ai")
        self.assertEqual(event["track_id"], 1)
        self.assertEqual(event["metadata"]["rule_score"], 0.91)

    def test_build_safety_event_accepts_operational_evidence(self):
        event = build_safety_event(
            event_type="fall_detected",
            camera_id="cam_01",
            severity="HIGH",
            message="Fall-like safety event detected.",
            track_id=3,
            confidence=0.91,
            bbox=[10, 20, 100, 80],
            model={"detector": "yolov8n-pose.pt", "classifier": "rule-fusion-v1"},
            evidence={"snapshot_url": "snap.jpg", "clip_url": "clip.mp4"},
            timestamp="2026-05-26T10:00:00Z",
            event_id="event-1",
        )

        self.assertEqual(event["event_id"], "event-1")
        self.assertEqual(event["confidence"], 0.91)
        self.assertEqual(event["bbox"], [10, 20, 100, 80])
        self.assertEqual(event["model"]["classifier"], "rule-fusion-v1")
        self.assertEqual(event["evidence"]["clip_url"], "clip.mp4")


if __name__ == "__main__":
    unittest.main()

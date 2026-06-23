import unittest

from ai.publishers.mqtt_payloads import build_confirmed_event_payload, build_overlay_payload


class MqttPayloadsTest(unittest.TestCase):
    def test_overlay_payload_uses_camera_topic_schema(self):
        payload = build_overlay_payload(
            stream_id="cam_01",
            frame_width=640,
            frame_height=360,
            timestamp_ms=1782180000123,
            boxes=[
                {
                    "x1": 120.2,
                    "y1": 80.4,
                    "x2": 320.2,
                    "y2": 230.4,
                    "score": 0.91,
                    "track_id": 3,
                    "faint_probability": 0.72,
                }
            ],
        )

        self.assertEqual(
            payload,
            {
                "schemaVersion": "1.0",
                "messageType": "overlay",
                "timestampMs": 1782180000123,
                "streamId": "cam_01",
                "frameWidth": 640,
                "frameHeight": 360,
                "events": [
                    {
                        "type": "faint",
                        "confidence": 0.72,
                        "trackingId": 3,
                        "boundingBox": {"x": 120, "y": 80, "width": 200, "height": 150},
                    }
                ],
            },
        )

    def test_overlay_payload_publishes_empty_events_when_no_lstm_signal(self):
        payload = build_overlay_payload(
            stream_id="cam_01",
            frame_width=640,
            frame_height=360,
            timestamp_ms=1782180000123,
            boxes=[{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": 0.9}],
        )

        self.assertEqual(payload["events"], [])

    def test_confirmed_event_payload_uses_event_topic_schema(self):
        payload = build_confirmed_event_payload(
            stream_id="cam_01",
            frame_width=640,
            frame_height=360,
            timestamp_ms=1782180000123,
            event_id="evt-20260623-cam_01-000001",
            prediction={"label": "Faint", "score": 0.92, "probabilities": {"Faint": 0.92}},
            sequence={"bbox": [120, 80, 320, 230], "track_id": 3},
            boxes=[],
        )

        self.assertEqual(
            payload,
            {
                "schemaVersion": "1.0",
                "messageType": "event",
                "eventId": "evt-20260623-cam_01-000001",
                "timestampMs": 1782180000123,
                "streamId": "cam_01",
                "type": "faint",
                "memoText": "쓰러짐 의심!",
                "confidence": 0.92,
                "trackingId": 3,
                "frameWidth": 640,
                "frameHeight": 360,
                "boundingBox": {"x": 120, "y": 80, "width": 200, "height": 150},
            },
        )


if __name__ == "__main__":
    unittest.main()

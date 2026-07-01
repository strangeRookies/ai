import unittest

from ai.publishers.mqtt_payloads import build_confirmed_event_payload, build_frame_sync_payload, build_overlay_payload


class MqttPayloadsTest(unittest.TestCase):
    def test_overlay_payload_uses_camera_topic_schema(self):
        payload = build_overlay_payload(
            stream_id="cam_01",
            frame_width=640,
            frame_height=360,
            timestamp_ms=1782180000123,
            frame_id=123,
            captured_at_ms=1782180000100,
            processed_at_ms=1782180000120,
            published_at_ms=1782180000123,
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
                "schemaVersion": "1.1",
                "messageType": "overlay",
                "timestampMs": 1782180000123,
                "streamId": "cam_01",
                "cameraLoginId": "cam_01",
                "frameId": 123,
                "capturedAtMs": 1782180000100,
                "processedAtMs": 1782180000120,
                "publishedAtMs": 1782180000123,
                "aiLatencyMs": 20,
                "publishLatencyMs": 23,
                "frameWidth": 640,
                "frameHeight": 360,
                "events": [
                    {
                        "type": "faint",
                        "confidence": 0.72,
                        "eventTriggered": False,
                        "trackingId": 3,
                        "trackId": 3,
                        "track_id": 3,
                        "frameId": 123,
                        "bbox": {"x": 120, "y": 80, "width": 200, "height": 150},
                        "boundingBox": {"x": 120, "y": 80, "width": 200, "height": 150},
                        "keypoints": [],
                    }
                ],
            },
        )

    def test_overlay_payload_includes_tracking_type_when_no_lstm_signal(self):
        """Boxes with valid bbox but no LSTM signal (faint_probability=None) must
        now be included in overlay events as type='tracking' so the frontend
        can always draw every detected person."""
        payload = build_overlay_payload(
            stream_id="cam_01",
            frame_width=640,
            frame_height=360,
            timestamp_ms=1782180000123,
            boxes=[{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": 0.9}],
        )

        # One box present → one event in payload with type 'tracking'
        self.assertEqual(len(payload["events"]), 1)
        self.assertEqual(payload["events"][0]["type"], "tracking")
        # No faint signal → confidence derived from box score
        self.assertAlmostEqual(payload["events"][0]["confidence"], 0.9)
        self.assertFalse(payload["events"][0]["eventTriggered"])

    def test_overlay_payload_clamps_bbox_to_frame_bounds(self):
        payload = build_overlay_payload(
            stream_id="cam_01",
            frame_width=640,
            frame_height=360,
            timestamp_ms=1782180000123,
            boxes=[
                {
                    "x1": -10,
                    "y1": 20,
                    "x2": 700,
                    "y2": 400,
                    "track_id": 3,
                    "faint_probability": 0.72,
                }
            ],
        )

        self.assertEqual(payload["events"][0]["boundingBox"], {"x": 0, "y": 20, "width": 640, "height": 340})

    def test_confirmed_event_payload_uses_event_topic_schema(self):
        payload = build_confirmed_event_payload(
            stream_id="cam_01",
            frame_width=640,
            frame_height=360,
            timestamp_ms=1782180000123,
            frame_id=123,
            captured_at_ms=1782180000100,
            processed_at_ms=1782180000120,
            published_at_ms=1782180000123,
            sequence_metadata={
                "sequenceLength": 30,
                "sequenceStride": 15,
                "sequenceStartFrameId": 94,
                "sequenceEndFrameId": 123,
                "sequenceStartAtMs": 1782179999000,
                "sequenceEndAtMs": 1782180000100,
            },
            event_id="evt-20260623-cam_01-000001",
            prediction={"label": "Faint", "score": 0.92, "probabilities": {"Faint": 0.92}},
            sequence={"bbox": [120, 80, 320, 230], "track_id": 3},
            boxes=[],
        )

        self.assertEqual(
            payload,
            {
                "schemaVersion": "1.1",
                "messageType": "event",
                "eventId": "evt-20260623-cam_01-000001",
                "timestampMs": 1782180000123,
                "timestamp": 1782180000.123,
                "streamId": "cam_01",
                "cameraLoginId": "cam_01",
                "frameId": 123,
                "capturedAtMs": 1782180000100,
                "processedAtMs": 1782180000120,
                "publishedAtMs": 1782180000123,
                "aiLatencyMs": 20,
                "publishLatencyMs": 23,
                "camera_id": "cam_01",
                "camera_login_id": "cam_01",
                "type": "faint",
                "event_type": "faint",
                "memoText": "쓰러짐 의심!",
                "message": "쓰러짐 의심!",
                "source": "edge-ai",
                "severity": "HIGH",
                "confidence": 0.92,
                "trackingId": 3,
                "track_id": 3,
                "frameWidth": 640,
                "frameHeight": 360,
                "sequence": {
                    "sequenceLength": 30,
                    "sequenceStride": 15,
                    "sequenceStartFrameId": 94,
                    "sequenceEndFrameId": 123,
                    "sequenceStartAtMs": 1782179999000,
                    "sequenceEndAtMs": 1782180000100,
                },
                "boundingBox": {"x": 120, "y": 80, "width": 200, "height": 150},
                "bbox": [120, 80, 320, 230],
                "events": [
                    {
                        "type": "faint",
                        "confidence": 0.92,
                        "trackingId": 3,
                        "frameId": 123,
                        "bbox": {"x": 120, "y": 80, "width": 200, "height": 150},
                        "keypoints": [],
                    }
                ],
            },
        )

    def test_confirmed_event_payload_includes_backend_dto_aliases(self):
        payload = build_confirmed_event_payload(
            stream_id="cam_10",
            frame_width=1280,
            frame_height=720,
            timestamp_ms=1782180000123,
            prediction={"label": "Faint", "score": 0.87, "probabilities": {"Faint": 0.87}},
            sequence={"bbox": [120, 80, 260, 360], "track_id": 7},
            boxes=[],
        )

        self.assertEqual(payload["camera_id"], "cam_10")
        self.assertEqual(payload["cameraLoginId"], "cam_10")
        self.assertEqual(payload["camera_login_id"], "cam_10")
        self.assertEqual(payload["timestamp"], 1782180000.123)
        self.assertEqual(payload["event_type"], "faint")
        self.assertEqual(payload["bbox"], [120, 80, 260, 360])
        self.assertEqual(payload["track_id"], 7)

    def test_payload_events_preserve_sequence_keypoints(self):
        payload = build_confirmed_event_payload(
            stream_id="cam_11",
            frame_width=640,
            frame_height=360,
            timestamp_ms=1782180000123,
            prediction={"label": "Faint", "score": 0.91, "probabilities": {"Faint": 0.91}},
            sequence={
                "bbox": [20, 30, 100, 140],
                "keypoints": [{"x": 30.0, "y": 40.0, "confidence": 0.9}],
                "track_id": 9,
            },
            boxes=[],
        )

        self.assertEqual(payload["events"][0]["keypoints"], [{"x": 30.0, "y": 40.0, "confidence": 0.9}])

    def test_payloads_include_additive_evidence_id_and_trace_fields(self):
        overlay = build_overlay_payload(
            stream_id="cam_01",
            frame_width=640,
            frame_height=360,
            timestamp_ms=1782180000123,
            frame_id=123,
            captured_at_ms=1782180000100,
            processed_at_ms=1782180000120,
            published_at_ms=1782180000123,
            boxes=[{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": 0.7}],
            dropped_frame_count=2,
        )
        event = build_confirmed_event_payload(
            stream_id="cam_01",
            frame_width=640,
            frame_height=360,
            timestamp_ms=1782180000123,
            frame_id=123,
            captured_at_ms=1782180000100,
            processed_at_ms=1782180000120,
            published_at_ms=1782180000123,
            prediction={"label": "Faint", "score": 0.91, "probabilities": {"Faint": 0.91}},
            sequence={"bbox": [1, 2, 3, 4], "track_id": 5},
            boxes=[],
            dropped_frame_count=2,
            snapshot_path="snapshots/cam_01_123.jpg",
            clip_path="clips/cam_01_123.mp4",
        )

        self.assertEqual(overlay["evidenceId"], "cam_01-123-1782180000100")
        self.assertEqual(event["evidenceId"], overlay["evidenceId"])
        self.assertEqual(event["traceId"], overlay["traceId"])
        self.assertEqual(event["metadata"]["evidenceId"], event["evidenceId"])
        self.assertEqual(event["metadata"]["snapshotPath"], "snapshots/cam_01_123.jpg")
        self.assertEqual(event["metadata"]["clipPath"], "clips/cam_01_123.mp4")
        self.assertEqual(event["evidence"]["latency"]["aiLatencyMs"], 20)
        self.assertEqual(event["evidence"]["latency"]["publishLatencyMs"], 23)
        self.assertEqual(event["evidence"]["droppedFrameCount"], 2)
        self.assertTrue(event["evidence"]["latencyOrderValid"])

    def test_frame_sync_payload_uses_same_evidence_key(self):
        payload = build_frame_sync_payload(
            camera_login_id="cam_01",
            frame_id=123,
            captured_at_ms=1782180000100,
            processed_at_ms=1782180000120,
            published_at_ms=1782180000123,
            queue_lag_ms=23,
            dropped_frame_count=2,
        )

        self.assertEqual(payload["messageType"], "frame_sync")
        self.assertEqual(payload["type"], "frame_sync")
        self.assertEqual(payload["evidenceId"], "cam_01-123-1782180000100")
        self.assertEqual(payload["traceId"], payload["evidenceId"])
        self.assertEqual(payload["evidence"]["frameId"], 123)
        self.assertEqual(payload["evidence"]["capturedAtMs"], 1782180000100)
        self.assertEqual(payload["evidence"]["droppedFrameCount"], 2)


if __name__ == "__main__":
    unittest.main()

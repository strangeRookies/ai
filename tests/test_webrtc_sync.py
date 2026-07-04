import unittest

from ai.webrtc_sync import build_overlay_frame_message


class WebRtcSyncPayloadTest(unittest.TestCase):
    def test_overlay_frame_message_preserves_sync_fields_and_marks_transport(self):
        raw_overlay = {
            "schemaVersion": "1.1",
            "messageType": "overlay",
            "cameraLoginId": "cam_02",
            "streamId": "cam_02",
            "frameId": 1234,
            "capturedAtMs": 1783030000000,
            "processedAtMs": 1783030000100,
            "publishedAtMs": 1783030000120,
            "frameWidth": 640,
            "frameHeight": 360,
            "events": [],
        }

        message = build_overlay_frame_message(
            raw_overlay,
            video_queued_at_ms=1783030000130,
            metadata_sent_at_ms=1783030000135,
        )

        self.assertEqual(message["messageType"], "overlay_frame")
        self.assertEqual(message["syncTransport"], "webrtc-datachannel")
        self.assertEqual(message["cameraLoginId"], "cam_02")
        self.assertEqual(message["streamId"], "cam_02")
        self.assertEqual(message["frameId"], 1234)
        self.assertEqual(message["capturedAtMs"], 1783030000000)
        self.assertEqual(message["processedAtMs"], 1783030000100)
        self.assertEqual(message["publishedAtMs"], 1783030000120)
        self.assertEqual(message["videoQueuedAtMs"], 1783030000130)
        self.assertEqual(message["metadataSentAtMs"], 1783030000135)
        self.assertEqual(message["events"], [])


if __name__ == "__main__":
    unittest.main()

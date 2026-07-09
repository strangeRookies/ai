import unittest
from types import SimpleNamespace

from ai.inference.tracking_debug import build_frame_tracking_record, build_tracker_startup_record


class TrackingDebugTest(unittest.TestCase):
    def test_tracker_startup_record_includes_backend_and_constructor_settings(self):
        postprocessor = FakePostprocessor(
            {
                "bytetrack_constructor": {
                    "used": {"lost_track_buffer": 90, "minimum_matching_threshold": 0.2},
                    "ignored": {},
                    "supported_parameters": ["lost_track_buffer", "minimum_matching_threshold"],
                }
            }
        )
        args = SimpleNamespace(
            track_buffer=90,
            match_thresh=0.2,
            track_thresh=0.1,
            frame_rate=30,
            detector_conf=0.15,
            yolo_model="yolo26n-pose.pt",
        )

        record = build_tracker_startup_record("cam_05", "supervision", postprocessor, args)

        self.assertEqual(record["cameraLoginId"], "cam_05")
        self.assertEqual(record["trackerBackend"], "supervision")
        self.assertEqual(record["trackBuffer"], 90)
        self.assertEqual(record["matchThreshold"], 0.2)
        self.assertEqual(record["highTrackThreshold"], 0.1)
        self.assertEqual(record["assumedFps"], 30)
        self.assertEqual(record["detectorConfidence"], 0.15)
        self.assertEqual(record["detectorBackend"], "torch")
        self.assertIsInstance(record["trackerObjectId"], int)
        self.assertEqual(record["bytetrackConstructorUsed"]["lost_track_buffer"], 90)

    def test_frame_tracking_record_includes_counts_confidence_and_track_lifetime(self):
        diagnostics = {
            "new_tracks": 1,
            "lost_tracks": 2,
            "removed_track_ids": [9],
            "tracks": {
                "5": {"track_age": 12},
                "6": {"track_age": 3},
            },
        }
        raw = [
            {"confidence": 0.9, "keypoint_confidence": 0.8},
            {"confidence": 0.7, "keypoint_confidence": 0.6},
        ]
        tracked = [
            {"track_id": 5, "confidence": 0.9, "keypoint_confidence": 0.8},
            {"track_id": 6, "confidence": 0.7, "keypoint_confidence": 0.6},
        ]

        record = build_frame_tracking_record(
            camera_login_id="cam_05",
            frame_id=42,
            captured_at_ms=1000,
            processed_at_ms=1033,
            frame_gap=1,
            dropped_frame_count=4,
            raw_detections=raw,
            tracked_detections=tracked,
            tracker_diagnostics=diagnostics,
            tracker_object_id=123,
        )

        self.assertEqual(record["cameraLoginId"], "cam_05")
        self.assertEqual(record["frameId"], 42)
        self.assertEqual(record["capturedAtMs"], 1000)
        self.assertEqual(record["processedAtMs"], 1033)
        self.assertEqual(record["frameGap"], 1)
        self.assertEqual(record["droppedFrameCount"], 4)
        self.assertEqual(record["rawDetectionCount"], 2)
        self.assertEqual(record["detectionCountAfterConfidenceFilter"], 2)
        self.assertEqual(record["avgDetectionConfidence"], 0.8)
        self.assertEqual(record["avgKeypointConfidence"], 0.7)
        self.assertEqual(record["activeTrackIds"], [5, 6])
        self.assertEqual(record["newTrackCount"], 1)
        self.assertEqual(record["lostTrackCount"], 2)
        self.assertEqual(record["removedTrackIds"], [9])
        self.assertEqual(record["trackLifetimeFrames"], {"5": 12, "6": 3})
        self.assertEqual(record["trackerObjectId"], 123)


class FakePostprocessor:
    def __init__(self, diagnostics):
        self._diagnostics = diagnostics

    def diagnostics(self):
        return self._diagnostics


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers
from scripts.run_rtsp_inference import (
    FaintEventPostProcessor,
    is_alert_prediction,
    run,
)
from ai.visualization.draw import draw_overlay

try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class RtspInferenceTest(unittest.TestCase):
    def write_sample_video(self, video):
        import cv2

        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
        for _ in range(4):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        writer.release()

    def test_keypoint_sequence_buffer_emits_sequence(self):
        buffer = KeypointSequenceBuffer(sequence_length=2, stride=1)
        detection = {
            "bbox": [0, 0, 10, 20],
            "keypoints": [{"x": 1, "y": 2, "confidence": 0.9}],
            "track_id": 7,
        }

        self.assertIsNone(buffer.add(0, [detection]))
        sequence = buffer.add(1, [detection])

        self.assertEqual(sequence["start_frame"], 0)
        self.assertEqual(sequence["end_frame"], 1)
        self.assertEqual(sequence["bbox"], [0, 0, 10, 20])
        self.assertEqual(sequence["track_id"], 7)

    @unittest.skipIf(not CV2_AVAILABLE, "cv2 is required for video smoke tests")
    def test_mock_rtsp_inference_reports_required_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "sample.avi"
            self.write_sample_video(video)
            args = Namespace(
                rtsp_url=str(video),
                camera_id="cam_01",
                max_frames=4,
                detector_mode="mock",
                dry_run=True,
                output=None,
                overlay_output=None,
                yolo_model="yolov8n-pose.pt",
                device="auto",
                action_model=None,
                action_device="auto",
                action_threshold=0.3,
                min_consecutive_faint=2,
                camera_cooldown_seconds=10,
                event_severity="HIGH",
                classifier_input="keypoints",
                sequence_length=2,
                sequence_stride=1,
                resize_size=32,
                tracker_iou_threshold=0.3,
                track_max_missing_seconds=2.0,
                event_log_dir=None,
            )

            summary = run(args)

        self.assertEqual(summary["frames_processed"], 4)
        self.assertEqual(summary["bbox_detections"], 4)
        self.assertEqual(summary["keypoints_extracted"], 4)
        self.assertGreater(summary["generated_sequences"], 0)
        self.assertGreater(summary["lstm_predictions"], 0)
        self.assertIsNotNone(summary["sample_event"])
        self.assertIn("runtime_seconds", summary)
        self.assertIn("effective_fps", summary)
        self.assertIn("avg_frame_read_ms", summary)
        self.assertIn("avg_yolo_inference_ms", summary)
        self.assertIn("avg_lstm_inference_ms", summary)
        self.assertIn("avg_total_frame_ms", summary)
        self.assertEqual(summary["bbox_per_frame"], 1.0)
        self.assertEqual(summary["keypoints_per_frame"], 1.0)
        self.assertGreater(summary["prediction_per_frame"], 0.0)
        self.assertIn("gpu_memory", summary)
        self.assertIn("gpu_memory_warning", summary)
        self.assertIn("active_tracks", summary)
        self.assertIn("max_active_tracks", summary)
        self.assertIn("per_track_sequences_generated", summary)
        self.assertGreater(summary["normal_predictions"] + summary["faint_predictions"], 0)

    @unittest.skipIf(not CV2_AVAILABLE, "cv2 is required for overlay drawing")
    def test_overlay_accepts_keypoints(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        boxes = [
            {
                "x1": 4,
                "y1": 4,
                "x2": 28,
                "y2": 28,
                "score": 0.9,
                "keypoints": [{"x": 8, "y": 8, "confidence": 0.9} for _ in range(17)],
            }
        ]

        output = draw_overlay(frame, boxes, {"label": "Faint", "score": 0.8}, 1)

        self.assertEqual(output.shape, frame.shape)

    def test_normal_prediction_is_not_alert_event(self):
        self.assertFalse(is_alert_prediction({"label": "Normal", "score": 0.9}))
        self.assertTrue(is_alert_prediction({"label": "Faint", "score": 0.6}))

    def test_faint_post_processor_requires_consecutive_predictions_and_cooldown(self):
        processor = FaintEventPostProcessor(min_consecutive_faint=2, cooldown_seconds=5)

        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 1.0))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 2.0))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 4.0))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Normal"}, 8.0))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 9.0))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 10.0))

    def test_faint_post_processor_debounces_per_track(self):
        processor = FaintEventPostProcessor(min_consecutive_faint=2, cooldown_seconds=5)

        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 1.0, track_id=1))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 2.0, track_id=1))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 3.0, track_id=1))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 3.0, track_id=2))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 4.0, track_id=2))

    def test_per_track_keypoint_buffer_emits_independent_sequences(self):
        buffer = PerTrackKeypointSequenceBuffers(sequence_length=2, stride=1)
        detections = [
            {"track_id": 1, "bbox": [0, 0, 10, 10], "keypoints": [{"x": 1, "y": 1, "confidence": 0.9}]},
            {"track_id": 2, "bbox": [20, 0, 30, 10], "keypoints": [{"x": 2, "y": 1, "confidence": 0.9}]},
        ]

        self.assertEqual(buffer.add(0, detections), [])
        sequences = buffer.add(1, detections)

        self.assertEqual({item["track_id"] for item in sequences}, {1, 2})
        self.assertEqual(buffer.sequences_generated_by_track[1], 1)
        self.assertEqual(buffer.sequences_generated_by_track[2], 1)

if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from scripts.run_rtsp_inference import FaintEventPostProcessor, build_inference_event_payload, is_alert_prediction, run
from ai.visualization.draw import draw_overlay


class RtspInferenceTest(unittest.TestCase):
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

    def test_mock_rtsp_inference_reports_required_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "sample.avi"
            import cv2

            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
            for _ in range(4):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
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
            )

            summary = run(args)

        self.assertEqual(summary["frames_processed"], 4)
        self.assertEqual(summary["bbox_detections"], 4)
        self.assertEqual(summary["keypoints_extracted"], 4)
        self.assertGreater(summary["generated_sequences"], 0)
        self.assertGreater(summary["lstm_predictions"], 0)
        self.assertIsNotNone(summary["sample_event"])

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

    def test_inference_event_payload_contains_required_runtime_fields(self):
        args = Namespace(
            camera_id="cam_01",
            action_threshold=0.3,
            min_consecutive_faint=2,
            camera_cooldown_seconds=10,
            event_severity="HIGH",
        )
        packet = Namespace(frame_idx=12, timestamp=123.5)
        prediction = {
            "label": "Faint",
            "score": 0.81,
            "probabilities": {"Normal": 0.19, "Faint": 0.81},
        }
        sequence = {"bbox": [1, 2, 3, 4], "track_id": 9, "start_frame": 4, "end_frame": 12}

        payload = build_inference_event_payload(args, packet, prediction, boxes=[], sequence=sequence)

        self.assertEqual(payload["camera_id"], "cam_01")
        self.assertEqual(payload["event_type"], "Faint")
        self.assertEqual(payload["confidence"], 0.81)
        self.assertEqual(payload["threshold"], 0.3)
        self.assertEqual(payload["track_id"], 9)
        self.assertEqual(payload["timestamp"], 123.5)
        self.assertEqual(payload["bbox"], [1, 2, 3, 4])
        self.assertEqual(payload["severity"], "HIGH")


if __name__ == "__main__":
    unittest.main()

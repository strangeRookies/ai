import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from scripts.run_rtsp_inference import run
from ai.visualization.draw import draw_overlay


class RtspInferenceTest(unittest.TestCase):
    def test_keypoint_sequence_buffer_emits_sequence(self):
        buffer = KeypointSequenceBuffer(sequence_length=2, stride=1)
        detection = {
            "bbox": [0, 0, 10, 20],
            "keypoints": [{"x": 1, "y": 2, "confidence": 0.9}],
        }

        self.assertIsNone(buffer.add(0, [detection]))
        sequence = buffer.add(1, [detection])

        self.assertEqual(sequence["start_frame"], 0)
        self.assertEqual(sequence["end_frame"], 1)
        self.assertEqual(sequence["bbox"], [0, 0, 10, 20])

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


if __name__ == "__main__":
    unittest.main()

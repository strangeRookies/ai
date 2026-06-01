import unittest
from argparse import Namespace

import numpy as np

from ai.streams.video_reader import FramePacket
from scripts.run_rtsp_inference import create_classifier, create_detector
from scripts.serve_ai_overlay import initial_summary, process_frame
from ai.action.sequence_buffer import CropSequenceBuffer


class AiOverlayServerTest(unittest.TestCase):
    def test_process_frame_draws_overlay_and_updates_counts(self):
        args = Namespace(
            detector_mode="mock",
            camera_id="cam_01",
            print_events=False,
        )
        detector = create_detector("mock", "yolov8n-pose.pt", "auto")
        classifier, _ = create_classifier(None, "auto")
        buffer = CropSequenceBuffer(sequence_length=2, stride=1, resize_size=32)
        summary = initial_summary()

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        first = FramePacket(frame_idx=0, fps=10.0, timestamp=0.0, frame=frame)
        second = FramePacket(frame_idx=1, fps=10.0, timestamp=0.1, frame=frame)
        process_frame(first, detector, classifier, buffer, summary, args)
        overlay = process_frame(second, detector, classifier, buffer, summary, args)

        self.assertEqual(overlay.shape, frame.shape)
        self.assertEqual(summary["frames_processed"], 2)
        self.assertEqual(summary["bbox_detections"], 2)
        self.assertEqual(summary["keypoints_extracted"], 2)
        self.assertEqual(summary["generated_sequences"], 1)
        self.assertEqual(summary["lstm_predictions"], 1)
        self.assertEqual(summary["events_generated"], 1)
        self.assertIsNotNone(summary["sample_event"])


if __name__ == "__main__":
    unittest.main()

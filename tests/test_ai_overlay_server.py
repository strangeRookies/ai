import unittest
from argparse import Namespace

import numpy as np

from ai.streams.video_reader import FramePacket
from scripts.run_rtsp_inference import create_classifier, create_detector
from scripts.serve_ai_overlay import format_action_overlay_text, initial_summary, process_frame
from ai.action.per_track_sequence_buffer import PerTrackCropSequenceBuffers
from ai.visualization.action_overlay import annotate_boxes_with_action
from tracking.simple_tracker import SimpleTrackAssigner


class AiOverlayServerTest(unittest.TestCase):
    def test_action_overlay_text_shows_faint_probability_threshold_and_alert(self):
        prediction = {"label": "Normal", "score": 0.63, "probabilities": {"Normal": 0.63, "Faint": 0.37}}

        text = format_action_overlay_text(prediction, threshold=0.3, consecutive_faint=0, event_triggered=False)
        alert_text = format_action_overlay_text({"label": "Faint", "score": 0.82, "probabilities": {"Faint": 0.82}}, threshold=0.3, consecutive_faint=2, event_triggered=True)

        self.assertEqual(text, "Faint: 0.37 | pred: Normal | th: 0.30 | seq: 0")
        self.assertEqual(alert_text, "ALERT Faint: 0.82")

    def test_annotate_boxes_adds_action_overlay_to_each_box(self):
        boxes = [{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": 0.91}]
        args = Namespace(action_threshold=0.3)

        annotate_boxes_with_action(boxes, {"label": "Faint", "score": 0.4, "probabilities": {"Faint": 0.4}}, args, consecutive_faint=1, event_triggered=False)

        self.assertIn("Faint: 0.40", boxes[0]["action_overlay"])
        self.assertFalse(boxes[0]["event_triggered"])

    def test_process_frame_draws_overlay_and_updates_counts(self):
        args = Namespace(
            detector_mode="mock",
            camera_id="cam_01",
            camera_login_id="cam_01",
            print_events=False,
            classifier_input="crops",
            mqtt_camera_topic="camera",
            mqtt_event_topic="event",
            mqtt_topic=None,
        )
        detector = create_detector("mock", "yolov8n-pose.pt", "auto")
        classifier, _ = create_classifier(None, "auto")
        buffer = PerTrackCropSequenceBuffers(sequence_length=2, stride=1, resize_size=32)
        tracker = SimpleTrackAssigner()
        summary = initial_summary()

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        first = FramePacket(frame_idx=0, fps=10.0, timestamp=0.0, frame=frame)
        second = FramePacket(frame_idx=1, fps=10.0, timestamp=0.1, frame=frame)
        process_frame(first, detector, classifier, buffer, summary, args, tracker=tracker)
        overlay = process_frame(second, detector, classifier, buffer, summary, args, tracker=tracker)

        self.assertEqual(overlay.shape, frame.shape)
        self.assertEqual(summary["frames_processed"], 2)
        self.assertEqual(summary["bbox_detections"], 2)
        self.assertEqual(summary["keypoints_extracted"], 2)
        self.assertEqual(summary["latest_frame_bbox"], 1)
        self.assertEqual(summary["latest_frame_keypoints"], 1)
        self.assertEqual(summary["generated_sequences"], 1)
        self.assertEqual(summary["lstm_predictions"], 1)
        self.assertEqual(summary["events_generated"], 1)
        self.assertEqual(summary["active_tracks"], 1)
        self.assertEqual(summary["max_active_tracks"], 1)
        self.assertEqual(summary["per_track_sequences_generated"]["1"], 1)
        self.assertIn("effective_fps", summary)
        self.assertGreaterEqual(summary["effective_fps"], 0.0)
        self.assertIsNotNone(summary["sample_event"])

    def test_process_frame_publishes_overlay_and_confirmed_event_metadata(self):
        args = Namespace(
            detector_mode="mock",
            camera_id="legacy_cam",
            camera_login_id="cam_01",
            print_events=False,
            classifier_input="crops",
            mqtt_camera_topic="camera",
            mqtt_event_topic="event",
            mqtt_topic=None,
        )
        detector = create_detector("mock", "yolov8n-pose.pt", "auto")
        classifier, _ = create_classifier(None, "auto")
        buffer = PerTrackCropSequenceBuffers(sequence_length=2, stride=1, resize_size=32)
        tracker = SimpleTrackAssigner()
        summary = initial_summary()
        publisher = FakePublisher()

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        process_frame(FramePacket(frame_idx=0, fps=10.0, timestamp=0.0, frame=frame), detector, classifier, buffer, summary, args, tracker=tracker, publisher=publisher)
        process_frame(FramePacket(frame_idx=1, fps=10.0, timestamp=0.1, frame=frame), detector, classifier, buffer, summary, args, tracker=tracker, publisher=publisher)

        overlay_topic, overlay_payload = publisher.published[-2]
        event_topic, event_payload = publisher.published[-1]
        self.assertEqual(overlay_topic, "camera")
        self.assertEqual(overlay_payload["messageType"], "overlay")
        self.assertEqual(overlay_payload["streamId"], "cam_01")
        self.assertEqual(overlay_payload["frameWidth"], 64)
        self.assertEqual(overlay_payload["frameHeight"], 64)
        self.assertEqual(event_topic, "event")
        self.assertEqual(event_payload["messageType"], "event")
        self.assertEqual(event_payload["streamId"], "cam_01")
        self.assertIn("boundingBox", event_payload)


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, payload, topic=None):
        self.published.append((topic, payload))
        return True


if __name__ == "__main__":
    unittest.main()

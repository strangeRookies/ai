import unittest
from contextlib import redirect_stdout
from argparse import Namespace
from io import StringIO

import numpy as np

from ai.streams.video_reader import FramePacket
from ai.frame_sync import FrameMetadataBuffer
from scripts.run_rtsp_inference import create_classifier, create_detector
from scripts.serve_ai_overlay import OverlayPublishState, format_action_overlay_text, initial_summary, log_frame_sync, process_frame
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
            mjpeg_debug=True,
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
        now_values = iter([1000, 1010, 1015, 1100, 1125, 1130])
        frame_buffer = FrameMetadataBuffer(now_ms=lambda: next(now_values))

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        process_frame(
            FramePacket(frame_idx=0, fps=10.0, timestamp=0.0, frame=frame),
            detector,
            classifier,
            buffer,
            summary,
            args,
            tracker=tracker,
            publisher=publisher,
            frame_buffer=frame_buffer,
        )
        process_frame(
            FramePacket(frame_idx=1, fps=10.0, timestamp=0.1, frame=frame),
            detector,
            classifier,
            buffer,
            summary,
            args,
            tracker=tracker,
            publisher=publisher,
            frame_buffer=frame_buffer,
        )

        overlay_topic, overlay_payload = publisher.published[-2]
        event_topic, event_payload = publisher.published[-1]
        self.assertEqual(overlay_topic, "camera")
        self.assertEqual(overlay_payload["messageType"], "overlay")
        self.assertEqual(overlay_payload["streamId"], "cam_01")
        self.assertEqual(overlay_payload["frameId"], 2)
        self.assertEqual(overlay_payload["capturedAtMs"], 1100)
        self.assertEqual(overlay_payload["processedAtMs"], 1125)
        self.assertEqual(overlay_payload["publishedAtMs"], 1130)
        self.assertEqual(overlay_payload["aiLatencyMs"], 25)
        self.assertEqual(overlay_payload["publishLatencyMs"], 30)
        self.assertEqual(overlay_payload["frameWidth"], 64)
        self.assertEqual(overlay_payload["frameHeight"], 64)
        self.assertEqual(event_topic, "event")
        self.assertEqual(event_payload["messageType"], "event")
        self.assertEqual(event_payload["streamId"], "cam_01")
        self.assertEqual(event_payload["frameId"], 2)
        self.assertEqual(event_payload["sequence"]["sequenceStartFrameId"], 1)
        self.assertEqual(event_payload["sequence"]["sequenceEndFrameId"], 2)
        self.assertIn("boundingBox", event_payload)

    def test_process_frame_keeps_overlay_and_event_on_same_evidence_id(self):
        args = Namespace(
            detector_mode="mock",
            camera_id="legacy_cam",
            camera_login_id="cam_01",
            print_events=False,
            classifier_input="crops",
            mqtt_camera_topic="camera",
            mqtt_event_topic="event",
            mqtt_topic=None,
            frame_queue_maxsize=2,
        )
        detector = create_detector("mock", "yolov8n-pose.pt", "auto")
        classifier, _ = create_classifier(None, "auto")
        buffer = PerTrackCropSequenceBuffers(sequence_length=2, stride=1, resize_size=32)
        tracker = SimpleTrackAssigner()
        summary = initial_summary()
        publisher = FakePublisher()
        now_values = iter([1000, 1010, 1015, 1100, 1125, 1130])
        frame_buffer = FrameMetadataBuffer(now_ms=lambda: next(now_values))

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for frame_idx in range(2):
            source_packet = FramePacket(frame_idx=frame_idx, fps=10.0, timestamp=frame_idx / 10.0, frame=frame)
            metadata = frame_buffer.record_capture("cam_01", source_packet, frame.shape)
            sync_packet = Namespace(
                camera_login_id="cam_01",
                frame_id=metadata.frame_id,
                captured_at_ms=metadata.captured_at_ms,
                frame=frame,
                width=64,
                height=64,
                frame_idx=frame_idx,
                timestamp=frame_idx / 10.0,
                fps=10.0,
            )
            process_frame(
                sync_packet,
                detector,
                classifier,
                buffer,
                summary,
                args,
                tracker=tracker,
                publisher=publisher,
                frame_buffer=frame_buffer,
                dropped_frame_count=4,
            )

        overlay_topic, overlay_payload = publisher.published[-2]
        event_topic, event_payload = publisher.published[-1]
        self.assertEqual(overlay_topic, "camera")
        self.assertEqual(event_topic, "event")
        self.assertEqual(overlay_payload["frameId"], 2)
        self.assertEqual(event_payload["frameId"], 2)
        self.assertEqual(overlay_payload["evidenceId"], "cam_01-2-1100")
        self.assertEqual(event_payload["evidenceId"], overlay_payload["evidenceId"])
        self.assertEqual(event_payload["evidence"]["frameId"], overlay_payload["evidence"]["frameId"])
        self.assertEqual(event_payload["evidence"]["droppedFrameCount"], 4)
        self.assertEqual(summary["latest_evidence_id"], "cam_01-2-1100")
        self.assertEqual(summary["latest_trace_id"], "cam_01-2-1100")
        self.assertEqual(summary["latest_dropped_frame_count"], 4)
        self.assertTrue(summary["latest_latency_order_valid"])

    def test_log_frame_sync_warns_on_invalid_latency_order(self):
        now_values = iter([1000, 900, 950])
        frame_buffer = FrameMetadataBuffer(now_ms=lambda: next(now_values))
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        captured = frame_buffer.record_capture(
            "cam_01",
            FramePacket(frame_idx=0, fps=10.0, timestamp=0.0, frame=frame),
            frame.shape,
        )
        processed = frame_buffer.mark_processed("cam_01", captured.frame_id)
        published = frame_buffer.mark_published("cam_01", processed.frame_id)
        output = StringIO()

        with redirect_stdout(output):
            log_frame_sync(Namespace(debug_every_n=0, frame_sync_delay_warning_ms=0), "cam_01", published, frame_buffer)

        line = output.getvalue()
        self.assertIn("[frame-sync] warning", line)
        self.assertIn("latency_order_valid=false", line)
        self.assertIn("frame_id=1", line)

    def test_overlay_publish_state_reuses_latest_signal_for_active_track(self):
        state = OverlayPublishState()
        boxes = [{"x1": 10, "y1": 20, "x2": 30, "y2": 40, "track_id": 7, "faint_probability": 0.63}]

        state.apply_latest_signals(boxes)
        self.assertEqual(state.signals_by_track[7]["faint_probability"], 0.63)

        next_boxes = [{"x1": 12, "y1": 22, "x2": 32, "y2": 42, "track_id": 7}]
        state.apply_latest_signals(next_boxes)

        self.assertEqual(next_boxes[0]["faint_probability"], 0.63)
        self.assertFalse(next_boxes[0]["event_triggered"])

    def test_overlay_publish_state_removes_signal_when_track_disappears(self):
        state = OverlayPublishState()
        state.apply_latest_signals([{"x1": 10, "y1": 20, "x2": 30, "y2": 40, "track_id": 7, "faint_probability": 0.63}])

        state.apply_latest_signals([])

        self.assertEqual(state.signals_by_track, {})

    def test_overlay_publish_state_timestamp_is_monotonic(self):
        state = OverlayPublishState()
        first = state.next_timestamp_ms()
        state.last_timestamp_ms = first + 100

        second = state.next_timestamp_ms()

        self.assertEqual(second, first + 101)


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, payload, topic=None):
        self.published.append((topic, payload))
        return True


if __name__ == "__main__":
    unittest.main()

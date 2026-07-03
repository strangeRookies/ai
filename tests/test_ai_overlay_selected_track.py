import unittest
from argparse import Namespace

import numpy as np

from ai.streams.video_reader import FramePacket
from scripts.serve_ai_overlay import initial_summary, process_frame
from ai.action.per_track_sequence_buffer import PerTrackCropSequenceBuffers


class AiOverlaySelectedTrackTest(unittest.TestCase):
    def test_process_frame_routes_only_selected_track_to_sequence_and_overlay(self):
        args = Namespace(
            detector_mode="real",
            camera_id="legacy_cam",
            camera_login_id="cam_dynamic_01",
            print_events=False,
            classifier_input="crops",
            mqtt_camera_topic="camera",
            mqtt_event_topic="event",
            mqtt_topic=None,
            selected_track_id=2,
            action_threshold=0.3,
            overlay_debug_tracks=False,
            mjpeg_debug=False,
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        publisher = FakePublisher()
        summary = initial_summary()
        sequence_buffer = PerTrackCropSequenceBuffers(sequence_length=1, stride=1, resize_size=16)

        process_frame(
            FramePacket(frame_idx=0, fps=10.0, timestamp=0.0, frame=frame),
            TwoTrackDetector(),
            FixedClassifier(),
            sequence_buffer,
            summary,
            args,
            publisher=publisher,
        )

        overlay_topic, overlay_payload = next(item for item in publisher.published if item[0] == "camera")
        self.assertEqual(overlay_topic, "camera")
        self.assertEqual(overlay_payload["streamId"], "cam_dynamic_01")
        self.assertEqual([event["track_id"] for event in overlay_payload["events"]], [2])
        self.assertEqual(summary["per_track_sequences_generated"], {"2": 1})
        self.assertEqual(summary["selected_track_id"], 2)
        self.assertEqual(summary["selected_track_skipped"], 1)

    def test_process_frame_strict_mode_when_selected_track_is_missing(self):
        args = Namespace(
            detector_mode="real",
            camera_id="legacy_cam",
            camera_login_id="cam_dynamic_01",
            print_events=False,
            classifier_input="crops",
            mqtt_camera_topic="camera",
            mqtt_event_topic="event",
            mqtt_topic=None,
            selected_track_id=9,
            selected_track_mode="strict",
            selected_track_missing_frames=2,
            action_threshold=0.3,
            overlay_debug_tracks=False,
            mjpeg_debug=False,
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        publisher = FakePublisher()
        summary = initial_summary()
        sequence_buffer = PerTrackCropSequenceBuffers(sequence_length=1, stride=1, resize_size=16)

        # strict 모드에서는 selected_track_id=9가 없을 때 아무것도 전달되지 않음
        process_frame(
            FramePacket(frame_idx=0, fps=10.0, timestamp=0.0, frame=frame),
            TwoTrackDetector(),
            FixedClassifier(),
            sequence_buffer,
            summary,
            args,
            publisher=publisher,
        )

        overlay_topic, overlay_payload = next(item for item in publisher.published if item[0] == "camera")
        self.assertEqual(overlay_topic, "camera")
        # events list should be empty
        self.assertEqual([event["track_id"] for event in overlay_payload["events"]], [])
        # sequence generated should be empty (or not generated for active tracks)
        self.assertEqual(summary.get("per_track_sequences_generated", {}), {})

    def test_process_frame_fallback_mode_when_selected_track_is_missing(self):
        args = Namespace(
            detector_mode="real",
            camera_id="legacy_cam",
            camera_login_id="cam_dynamic_01",
            print_events=False,
            classifier_input="crops",
            mqtt_camera_topic="camera",
            mqtt_event_topic="event",
            mqtt_topic=None,
            selected_track_id=9,
            selected_track_mode="fallback",
            selected_track_missing_frames=2,
            action_threshold=0.3,
            overlay_debug_tracks=False,
            mjpeg_debug=False,
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        publisher = FakePublisher()
        summary = initial_summary()
        sequence_buffer = PerTrackCropSequenceBuffers(sequence_length=1, stride=1, resize_size=16)

        # TrackSelector 상태를 여러 프레임 간 유지해야 하므로 TrackSelector 생성
        from ai.inference.track_selection import TrackSelector
        track_selector = TrackSelector(
            selected_track_id=args.selected_track_id,
            selected_track_mode=args.selected_track_mode,
            missing_frames_threshold=args.selected_track_missing_frames
        )

        # Frame 1: missing count = 1 (threshold 2 미만이므로 여전히 빈 결과)
        process_frame(
            FramePacket(frame_idx=0, fps=10.0, timestamp=0.0, frame=frame),
            TwoTrackDetector(),
            FixedClassifier(),
            sequence_buffer,
            summary,
            args,
            publisher=publisher,
            track_selector=track_selector,
        )
        overlay_payload_1 = next(payload for topic, payload in publisher.published if topic == "camera")
        self.assertEqual([event["track_id"] for event in overlay_payload_1["events"]], [])

        # Frame 2: missing count = 2 (threshold 도달 -> fallback 적용)
        # TwoTrackDetector의 1번 트랙(conf 0.9)이 더 안정적이므로 1번으로 fallback
        process_frame(
            FramePacket(frame_idx=1, fps=10.0, timestamp=0.1, frame=frame),
            TwoTrackDetector(),
            FixedClassifier(),
            sequence_buffer,
            summary,
            args,
            publisher=publisher,
            track_selector=track_selector,
        )
        camera_payloads = [payload for topic, payload in publisher.published if topic == "camera"]
        overlay_payload_2 = camera_payloads[1]
        self.assertEqual([event["track_id"] for event in overlay_payload_2["events"]], [1])
        self.assertEqual(summary["selected_track_fallbacks"], 1)


    def test_process_frame_regression_keeps_all_tracks_when_no_selected_track(self):
        args = Namespace(
            detector_mode="real",
            camera_id="legacy_cam",
            camera_login_id="cam_dynamic_01",
            print_events=False,
            classifier_input="crops",
            mqtt_camera_topic="camera",
            mqtt_event_topic="event",
            mqtt_topic=None,
            selected_track_id=None,
            action_threshold=0.3,
            overlay_debug_tracks=False,
            mjpeg_debug=False,
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        publisher = FakePublisher()
        summary = initial_summary()
        sequence_buffer = PerTrackCropSequenceBuffers(sequence_length=1, stride=1, resize_size=16)

        process_frame(
            FramePacket(frame_idx=0, fps=10.0, timestamp=0.0, frame=frame),
            TwoTrackDetector(),
            FixedClassifier(),
            sequence_buffer,
            summary,
            args,
            publisher=publisher,
        )

        overlay_payload = next(payload for topic, payload in publisher.published if topic == "camera")
        self.assertEqual([event["track_id"] for event in overlay_payload["events"]], [1, 2])




class TwoTrackDetector:
    def detect(self, frame):
        del frame
        return [
            {"bbox": [0, 0, 20, 20], "confidence": 0.9, "track_id": 1},
            {"bbox": [30, 30, 60, 60], "confidence": 0.8, "track_id": 2},
        ]


class FixedClassifier:
    def predict(self, sequence):
        del sequence
        return {"label": "Faint", "score": 0.92, "probabilities": {"Faint": 0.92}}


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, payload, topic=None):
        self.published.append((topic, payload))
        return True


if __name__ == "__main__":
    unittest.main()

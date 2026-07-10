import json
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.per_track_sequence_buffer import PerTrackKeypointSequenceBuffers
from scripts.run_rtsp_inference import (
    FaintEventPostProcessor,
    is_alert_prediction,
    run,
)
from ai.inference.rtsp_runtime import log_classification_stage, log_tracking_stage
from ai.inference.tracking_debug import log_sequence_stage
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
        for _ in range(5):
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

    def test_keypoint_sequence_buffer_stride_controls_overlap_not_sampling(self):
        buffer = KeypointSequenceBuffer(sequence_length=4, stride=2)
        detection = {
            "bbox": [0, 0, 10, 20],
            "keypoints": [{"x": 1, "y": 2, "confidence": 0.9} for _ in range(17)],
            "track_id": 7,
        }

        emitted = [buffer.add(frame_idx, [detection]) for frame_idx in range(6)]

        self.assertIsNotNone(emitted[3])
        self.assertIsNone(emitted[4])
        self.assertIsNotNone(emitted[5])
        assert emitted[3] is not None
        assert emitted[5] is not None
        self.assertEqual(emitted[3]["start_frame"], 0)
        self.assertEqual(emitted[5]["start_frame"], 2)

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
                cheap_filter_enabled=False,
                resize_size=32,
                tracker_iou_threshold=0.3,
                track_max_missing_seconds=2.0,
                event_log_dir=None,
                publisher=None,
                mqtt_host=None,
                mqtt_port=None,
                mqtt_topic=None,
                mqtt_camera_topic=None,
                mqtt_event_topic=None,
                mqtt_client_id=None,
                mqtt_username=None,
                mqtt_password=None,
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
        # Phase B default requires upright→lying; disable for legacy consecutive/cooldown unit test.
        processor = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=5,
            require_upright_to_lying=False,
            use_posture_estimator=False,
        )

        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 1.0))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 2.0))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 4.0))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Normal"}, 8.0))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 9.0))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 10.0))

    def test_faint_post_processor_debounces_events_per_camera_across_tracks(self):
        processor = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=5,
            require_upright_to_lying=False,
            use_posture_estimator=False,
        )

        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 1.0, track_id=1))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 2.0, track_id=1))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 3.0, track_id=1))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 3.0, track_id=2))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 4.0, track_id=2))
        self.assertTrue(processor.cooldown_active("cam_01", 4.0, track_id=2))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 7.0, track_id=2))

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

    def test_per_track_keypoint_buffer_reports_per_track_accumulation(self):
        buffer = PerTrackKeypointSequenceBuffers(sequence_length=3, stride=1)
        detection = {"track_id": 5, "bbox": [0, 0, 10, 10], "keypoints": [{"x": 1, "y": 1, "confidence": 0.9}]}

        buffer.add(0, [detection], (100, 100, 3))
        buffer.add(1, [detection], (100, 100, 3))

        self.assertEqual(buffer.buffer_lengths(), {5: 2})

    def test_sequence_stage_log_includes_buffer_accumulation_and_lstm_status(self):
        previous = os.environ.get("TRACKING_DEBUG")
        os.environ["TRACKING_DEBUG"] = "true"
        output = StringIO()
        try:
            with redirect_stdout(output):
                log_sequence_stage(
                    "cam_05",
                    42,
                    active_track_ids=[7],
                    buffer_lengths={7: 29},
                    sequences_generated=0,
                    sequences_generated_by_track={7: 0},
                    latest_faint_prob=None,
                )
        finally:
            if previous is None:
                os.environ.pop("TRACKING_DEBUG", None)
            else:
                os.environ["TRACKING_DEBUG"] = previous

        record = json.loads(output.getvalue().strip().removeprefix("[stage-log] "))

        self.assertEqual(record["stage"], "sequence")
        self.assertEqual(record["cameraLoginId"], "cam_05")
        self.assertEqual(record["frameId"], 42)
        self.assertEqual(record["activeTrackIds"], [7])
        self.assertEqual(record["bufferLengths"], {"7": 29})
        self.assertEqual(record["sequencesGenerated"], 0)
        self.assertFalse(record["lstmSequenceGenerated"])
        self.assertIsNone(record["latestFaintProbability"])

    def test_tracking_stage_log_includes_track_display_and_keypoint_diagnostics(self):
        previous = os.environ.get("TRACKING_DEBUG")
        os.environ["TRACKING_DEBUG"] = "true"
        output = StringIO()
        try:
            with redirect_stdout(output):
                log_tracking_stage(
                    "cam_05",
                    42,
                    pre_detections=[{"bbox": [0, 0, 10, 10]}],
                    post_detections=[
                        {
                            "bbox": [0, 0, 10, 10],
                            "confidence": 0.81,
                            "track_id": 987654321,
                            "display_id": 1,
                            "keypoints": [
                                {"x": 1, "y": 2, "confidence": 0.9},
                                {"x": 3, "y": 4, "confidence": 0.7},
                            ],
                        },
                        {
                            "bbox": [20, 0, 30, 10],
                            "confidence": 0.72,
                            "keypoints": [{"x": 5, "y": 6, "confidence": 0.6}],
                        },
                    ],
                    diagnostics={"new_tracks": 1, "lost_tracks": 0, "id_switch_like_events": 0},
                )
        finally:
            if previous is None:
                os.environ.pop("TRACKING_DEBUG", None)
            else:
                os.environ["TRACKING_DEBUG"] = previous

        line = output.getvalue().strip()
        record = json.loads(line.removeprefix("[stage-log] "))

        self.assertEqual(record["cameraLoginId"], "cam_05")
        self.assertEqual(record["trackedCount"], 1)
        self.assertEqual(record["missingTrackCount"], 1)
        self.assertEqual(record["avgKeypointConfidence"], 0.7333)
        self.assertEqual(record["trackDetails"][0]["trackId"], 987654321)
        self.assertEqual(record["trackDetails"][0]["displayId"], 1)
        self.assertEqual(record["trackDetails"][1]["fallbackRisk"], True)

    def test_classification_stage_log_includes_lstm_feature_shape_contract(self):
        previous = os.environ.get("TRACKING_DEBUG")
        os.environ["TRACKING_DEBUG"] = "true"
        output = StringIO()
        try:
            with redirect_stdout(output):
                log_classification_stage(
                    "cam_05",
                    42,
                    7,
                    {"label": "Faint", "score": 0.91, "probabilities": {"Normal": 0.09, "Faint": 0.91}},
                    faint_threshold=0.5,
                    consecutive_count=2,
                    event_triggered=True,
                    checkpoint_input_size=54,
                    runtime_feature_dim=54,
                    tensor_shape=(1, 30, 54),
                    feature_schema="keypoint_bbox54",
                    checkpoint_path="models/faint54.pt",
                )
        finally:
            if previous is None:
                os.environ.pop("TRACKING_DEBUG", None)
            else:
                os.environ["TRACKING_DEBUG"] = previous

        record = json.loads(output.getvalue().strip().removeprefix("[stage-log] "))

        self.assertEqual(record["stage"], "classification")
        self.assertEqual(record["checkpointInputSize"], 54)
        self.assertEqual(record["runtimeFeatureDim"], 54)
        self.assertEqual(record["tensorShape"], [1, 30, 54])
        self.assertEqual(record["featureSchema"], "keypoint_bbox54")
        self.assertEqual(record["checkpointPath"], "models/faint54.pt")

if __name__ == "__main__":
    unittest.main()

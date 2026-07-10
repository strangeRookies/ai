"""Regression tests for registered SIMULATED_RTSP loop-off and worker restart."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai.ffmpeg_command import build_ffmpeg_command
from ai.registered_camera_workers import CameraWorker, restart_exited_worker
from ai.registered_cameras import RegisteredCamera, RunnerConfig
from ai.inference.tensorrt_runtime import (
    attach_runtime_summary_fields,
    finalize_runtime_summary_fields,
)


class RegisteredSimFfmpegLoopTest(unittest.TestCase):
    def test_registered_ffmpeg_command_does_not_infinite_loop(self):
        with patch.dict("os.environ", {"FFMPEG_MODE": "cpu"}, clear=False):
            command = build_ffmpeg_command(Path("sample.mp4"), "rtsp://localhost:8554/cam_01")
        self.assertNotIn("-stream_loop", command)
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("sample.mp4", command)

    def test_registered_ffmpeg_command_can_opt_in_loop(self):
        with patch.dict("os.environ", {"FFMPEG_MODE": "cpu"}, clear=False):
            command = build_ffmpeg_command(
                Path("sample.mp4"),
                "rtsp://localhost:8554/cam_01",
                loop=True,
            )
        self.assertIn("-stream_loop", command)
        self.assertEqual(command[command.index("-stream_loop") + 1], "-1")


class WorkerExitRestartTest(unittest.TestCase):
    def _camera(self) -> RegisteredCamera:
        return RegisteredCamera(
            camera_id=1,
            camera_login_id="cam_01",
            source_type="SIMULATED_RTSP",
            rtsp_url="rtsp://127.0.0.1:8554/cam_01",
            assigned_video_path=None,
        )

    def _config(self) -> RunnerConfig:
        return RunnerConfig(
            backend_base_url="http://mock",
            backend_token=None,
            backend_timeout_seconds=5.0,
            rtsp_base_url="rtsp://127.0.0.1:8554",
            video_pool=Path("."),
            overlay_host="0.0.0.0",
            overlay_base_port=8010,
            python_executable="python",
            publisher="mqtt",
            mqtt_host=None,
            mqtt_port=None,
            mqtt_topic=None,
            mqtt_camera_topic="camera",
            mqtt_event_topic="event",
            mqtt_client_id_prefix="prefix",
            mqtt_username=None,
            mqtt_password=None,
            detector_mode="mock",
            yolo_model="yolo.pt",
            device="cpu",
            detector_conf=0.15,
            action_model=None,
            action_device="cpu",
            action_threshold=None,
            classifier_input="keypoints",
            sequence_length=30,
            sequence_stride=15,
            tracking_mode="auto",
            track_thresh=0.10,
            match_thresh=0.20,
            track_buffer=90,
            bbox_smoothing_alpha=0.60,
            print_events=False,
            dry_run=False,
            rtsp_probe_enabled=False,
            refresh_interval_seconds=30.0,
            skip_ffmpeg_spawn=True,
            overlay_public_base_url="http://localhost:8010",
            overlay_report_enabled=False,
        )

    @patch("ai.registered_camera_workers.start_camera_worker")
    @patch("ai.registered_camera_workers.report_overlay_stopped")
    @patch("ai.registered_camera_workers.stop_processes")
    def test_restart_exited_worker_stops_all_and_recreates_on_same_port(
        self,
        mock_stop,
        mock_report_stopped,
        mock_start,
    ):
        camera = self._camera()
        config = self._config()
        ffmpeg_proc = MagicMock()
        overlay_proc = MagicMock()
        workers = {
            "cam_01": CameraWorker(
                processes=[ffmpeg_proc, overlay_proc],
                overlay_port=8013,
                source_signature="sig",
                camera_login_id="cam_01",
                rtsp_url="rtsp://127.0.0.1:8554/cam_01",
            )
        }
        mock_start.return_value = CameraWorker(
            processes=[MagicMock(), MagicMock()],
            overlay_port=8013,
            source_signature="sig2",
            camera_login_id="cam_01",
            rtsp_url="rtsp://127.0.0.1:8554/cam_01",
        )

        restart_exited_worker(
            workers,
            "cam_01",
            cameras_by_id={"cam_01": camera},
            config=config,
        )

        mock_stop.assert_called_once_with([ffmpeg_proc, overlay_proc])
        mock_report_stopped.assert_called_once()
        mock_start.assert_called_once_with(camera, config, 8013)
        self.assertEqual(workers["cam_01"].overlay_port, 8013)

    @patch("ai.registered_camera_workers.start_camera_worker")
    @patch("ai.registered_camera_workers.report_overlay_stopped")
    @patch("ai.registered_camera_workers.stop_processes")
    def test_restart_exited_worker_removes_worker_when_camera_gone(
        self,
        mock_stop,
        mock_report_stopped,
        mock_start,
    ):
        config = self._config()
        workers = {
            "cam_01": CameraWorker(
                processes=[MagicMock()],
                overlay_port=8010,
                source_signature="sig",
                camera_login_id="cam_01",
                rtsp_url="rtsp://x",
            )
        }
        restart_exited_worker(workers, "cam_01", cameras_by_id={}, config=config)
        mock_stop.assert_called_once()
        mock_start.assert_not_called()
        self.assertNotIn("cam_01", workers)


class RuntimeReportFieldsTest(unittest.TestCase):
    def test_finalize_runtime_summary_adds_backend_latency_fps_count(self):
        detector = SimpleNamespace(
            runtime="tensorrt",
            model_path="yolo26n-pose.engine",
            engine_validation={"ok": True, "reason": None},
        )
        summary = {
            "yolo_model": "yolo26n-pose.engine",
            "frames_processed": 60,
        }
        attach_runtime_summary_fields(summary, detector, requested_model="yolo26n-pose.engine")
        summary["avg_yolo_inference_ms"] = 4.5
        summary["effective_fps"] = 48.2
        finalize_runtime_summary_fields(summary)

        self.assertEqual(summary["runtime"], "tensorrt")
        self.assertEqual(summary["backend"], "tensorrt")
        self.assertEqual(summary["model_path"], "yolo26n-pose.engine")
        self.assertEqual(summary["inference_count"], 60)
        self.assertEqual(summary["avg_latency_ms"], 4.5)
        self.assertEqual(summary["fps"], 48.2)
        self.assertTrue(summary["engine_validation"]["ok"])


if __name__ == "__main__":
    unittest.main()

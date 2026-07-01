import unittest
import os
import time
import tempfile
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai.registered_cameras import RegisteredCamera, RunnerConfig
from ai.registered_camera_workers import (
    CameraWorker,
    sync_camera_workers,
    stop_all_workers,
)
from ai.worker_registry import (
    register_worker,
    unregister_worker,
    load_registry,
    force_kill_existing_worker,
    REGISTRY_FILE,
)


class TestWorkerLifecycle(unittest.TestCase):
    def setUp(self):
        # Backup original registry file path
        self.orig_registry_file = REGISTRY_FILE
        self.temp_dir = tempfile.TemporaryDirectory()
        # Set temp registry file
        import ai.worker_registry
        ai.worker_registry.REGISTRY_FILE = Path(self.temp_dir.name) / "camera_worker_registry.json"

    def tearDown(self):
        import ai.worker_registry
        ai.worker_registry.REGISTRY_FILE = self.orig_registry_file
        self.temp_dir.cleanup()

    def test_double_register_same_camera_id_fails(self):
        # Register a camera worker
        register_worker("cam_01", 99999, "rtsp://mock_input")
        
        # Try to register again with an active PID mock
        with patch("ai.worker_registry.check_pid_alive", return_value=True), \
             patch("ai.worker_registry.check_process_signature_matches", return_value=True):
            with self.assertRaises(RuntimeError):
                register_worker("cam_01", 88888, "rtsp://mock_input")

    def test_register_on_dead_pid_succeeds(self):
        # Register with dead pid
        register_worker("cam_01", 99999, "rtsp://mock_input")
        
        with patch("ai.worker_registry.check_pid_alive", return_value=False):
            # Should succeed as the old process is detected as dead
            register_worker("cam_01", 88888, "rtsp://mock_input_new")
            
        registry = load_registry()
        self.assertEqual(registry["cam_01"]["pid"], 88888)
        self.assertEqual(registry["cam_01"]["rtsp_url"], "rtsp://mock_input_new")

    def get_test_config(self):
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
            action_model=None,
            action_device="cpu",
            action_threshold=None,
            classifier_input="keypoints",
            sequence_length=30,
            sequence_stride=15,
            tracking_mode="auto",
            print_events=False,
            dry_run=False,
            rtsp_probe_enabled=False,
            refresh_interval_seconds=30.0,
            skip_ffmpeg_spawn=True,
            overlay_public_base_url="http://localhost:8010",
            overlay_report_enabled=False,
        )

    @patch("ai.registered_camera_workers.start_camera_worker")
    @patch("ai.registered_camera_workers.stop_processes")
    def test_sync_camera_workers_handles_restarts_on_config_change(self, mock_stop, mock_start):
        # Mock initial worker
        workers = {
            "cam_01": CameraWorker(
                processes=[MagicMock()],
                overlay_port=8010,
                source_signature="REAL_RTSP:rtsp://mock_input_old",
                camera_login_id="cam_01",
                rtsp_url="rtsp://mock_input_old"
            )
        }
        
        cameras = [
            RegisteredCamera(
                camera_id=1,
                camera_login_id="cam_01",
                source_type="REAL_RTSP",
                rtsp_url="rtsp://mock_input_new", # Change URL
                assigned_video_path=None
            )
        ]
        
        config = self.get_test_config()

        mock_start.return_value = CameraWorker(
            processes=[MagicMock()],
            overlay_port=8010,
            source_signature="REAL_RTSP:rtsp://mock_input_new",
            camera_login_id="cam_01",
            rtsp_url="rtsp://mock_input_new"
        )

        # Sync
        sync_camera_workers(workers, cameras, config)
        
        # verify stop was called on old worker processes
        mock_stop.assert_called_once()
        # verify start was called for new worker
        mock_start.assert_called_once_with(cameras[0], config, 8010)

    @patch("ai.registered_camera_workers.stop_processes")
    def test_sync_camera_workers_handles_deletions(self, mock_stop):
        workers = {
            "cam_01": CameraWorker(
                processes=[MagicMock()],
                overlay_port=8010,
                source_signature="REAL_RTSP:rtsp://mock_input",
                camera_login_id="cam_01",
                rtsp_url="rtsp://mock_input"
            )
        }
        
        # Empty camera list (representing deletion)
        cameras = []
        config = self.get_test_config()
        
        sync_camera_workers(workers, cameras, config)
        
        # Verify workers list becomes empty
        self.assertEqual(len(workers), 0)
        mock_stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()

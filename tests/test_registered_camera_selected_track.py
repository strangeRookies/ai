import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ai.registered_cameras import RegisteredCamera
from scripts.run_registered_cameras import config_from_args, parse_args
from tests.test_registered_camera_runner import fake_config
from ai.registered_cameras import build_overlay_command


class RegisteredCameraSelectedTrackTest(unittest.TestCase):
    def test_runner_reads_selected_track_id_from_environment(self):
        with patch.dict("os.environ", {"SELECTED_TRACK_ID": "7"}, clear=False):
            config = config_from_args(parse_args([]))

        self.assertEqual(config.selected_track_id, 7)

    def test_overlay_command_passes_selected_track_id_to_worker(self):
        camera = RegisteredCamera(
            camera_id="12",
            camera_login_id="cam_any",
            rtsp_url="rtsp://cctv/cam_any",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )
        config = replace(fake_config(Path("video_pool")), selected_track_id=7)

        command = build_overlay_command(camera, "rtsp://cctv/cam_any", 8010, config)

        self.assertEqual(command[command.index("--selected-track-id") + 1], "7")


    def test_runner_reads_selected_track_mode_and_missing_frames_from_environment(self):
        with patch.dict("os.environ", {
            "SELECTED_TRACK_MODE": "fallback",
            "SELECTED_TRACK_MISSING_FRAMES": "10"
        }, clear=False):
            config = config_from_args(parse_args([]))

        self.assertEqual(config.selected_track_mode, "fallback")
        self.assertEqual(config.selected_track_missing_frames, 10)

    def test_overlay_command_passes_selected_track_mode_and_missing_frames_to_worker(self):
        camera = RegisteredCamera(
            camera_id="12",
            camera_login_id="cam_any",
            rtsp_url="rtsp://cctv/cam_any",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )
        config = replace(
            fake_config(Path("video_pool")),
            selected_track_id=7,
            selected_track_mode="fallback",
            selected_track_missing_frames=10,
        )

        command = build_overlay_command(camera, "rtsp://cctv/cam_any", 8010, config)

        self.assertEqual(command[command.index("--selected-track-mode") + 1], "fallback")
        self.assertEqual(command[command.index("--selected-track-missing-frames") + 1], "10")


if __name__ == "__main__":
    unittest.main()


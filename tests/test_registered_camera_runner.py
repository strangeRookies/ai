import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.registered_cameras import (
    RegisteredCamera,
    RunnerConfig,
    build_overlay_command,
    camera_rtsp_url,
    input_rtsp_for_camera,
    load_active_cameras,
    parse_camera,
)


class RegisteredCameraRunnerTest(unittest.TestCase):
    def test_parse_camera_uses_camera_login_id_as_external_key(self):
        camera = parse_camera(
            {
                "cameraId": 42,
                "cameraLoginId": "lobby_01",
                "rtspUrl": "rtsp://cctv/stream1",
                "sourceType": "REAL_RTSP",
                "aiEnabled": True,
                "status": "ACTIVE",
            }
        )

        self.assertIsNotNone(camera)
        assert camera is not None
        self.assertEqual(camera.camera_id, "42")
        self.assertEqual(camera.camera_login_id, "lobby_01")
        self.assertEqual(camera.rtsp_url, "rtsp://cctv/stream1")

    def test_simulated_rtsp_uses_camera_login_id_path_and_ffmpeg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            video_path.write_bytes(b"fake")
            camera = RegisteredCamera(
                camera_id="7",
                camera_login_id="ward_a",
                rtsp_url=None,
                source_type="SIMULATED_RTSP",
                assigned_video_path=str(video_path),
            )

            rtsp_url, ffmpeg_command = input_rtsp_for_camera(camera, fake_config(Path(temp_dir)))

        self.assertEqual(rtsp_url, "rtsp://gpu-pc:8554/ward_a")
        self.assertIsNotNone(ffmpeg_command)
        assert ffmpeg_command is not None
        self.assertEqual(ffmpeg_command[-1], "rtsp://gpu-pc:8554/ward_a")
        self.assertIn(str(video_path), ffmpeg_command)

    def test_real_rtsp_uses_backend_rtsp_url_without_simulation(self):
        camera = RegisteredCamera(
            camera_id="8",
            camera_login_id="door_01",
            rtsp_url="rtsp://cctv.local/live",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )

        rtsp_url, ffmpeg_command = input_rtsp_for_camera(camera, fake_config(Path("video_pool")))

        self.assertEqual(rtsp_url, "rtsp://cctv.local/live")
        self.assertIsNone(ffmpeg_command)

    def test_camera_rtsp_url_uses_b_option_without_cam_prefix(self):
        self.assertEqual(camera_rtsp_url("rtsp://gpu-pc:8554/", "cam_01"), "rtsp://gpu-pc:8554/cam_01")

    def test_overlay_command_passes_camera_login_id_to_ai_script(self):
        camera = RegisteredCamera(
            camera_id="9",
            camera_login_id="icu_01",
            rtsp_url="rtsp://cctv/icu",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )

        command = build_overlay_command(camera, "rtsp://cctv/icu", 8012, fake_config(Path("video_pool")))

        self.assertIn("scripts/serve_ai_overlay.py", command)
        self.assertEqual(command[command.index("--camera-id") + 1], "icu_01")
        self.assertEqual(command[command.index("--camera-login-id") + 1], "icu_01")
        self.assertEqual(command[command.index("--rtsp-url") + 1], "rtsp://cctv/icu")

    def test_load_active_cameras_reads_backend_success_data_envelope(self):
        response = FakeHttpResponse(
            b'{"success":true,"data":[{"cameraId":4,"cameraLoginId":"cam_02","rtspUrl":"rtsp://cctv/cam_02","sourceType":"REAL_RTSP","aiEnabled":true,"status":"ACTIVE"}]}'
        )

        with patch("urllib.request.urlopen", return_value=response):
            cameras = load_active_cameras("http://backend:8080", None)

        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0].camera_login_id, "cam_02")
        self.assertEqual(cameras[0].rtsp_url, "rtsp://cctv/cam_02")


class FakeHttpResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, unused_exc_type, unused_exc, unused_traceback):
        return False

    def read(self):
        return self._body


def fake_config(video_pool: Path) -> RunnerConfig:
    return RunnerConfig(
        backend_base_url="http://backend:8080",
        backend_token=None,
        backend_timeout_seconds=10.0,
        rtsp_base_url="rtsp://gpu-pc:8554",
        video_pool=video_pool,
        overlay_host="0.0.0.0",
        overlay_base_port=8010,
        python_executable="python",
        publisher="mqtt",
        mqtt_host="emqx",
        mqtt_port=1883,
        mqtt_topic="safety/events",
        mqtt_client_id_prefix="strange-ai",
        mqtt_username=None,
        mqtt_password=None,
        detector_mode="real",
        yolo_model="yolo26n-pose.pt",
        device="cuda:0",
        action_model=None,
        action_device="cuda:0",
        action_threshold=None,
        classifier_input="keypoints",
        sequence_length=8,
        sequence_stride=4,
        tracking_mode="supervision",
        print_events=False,
        dry_run=True,
    )

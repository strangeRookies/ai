import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ai.registered_cameras import RegisteredCamera, RunnerConfig, build_overlay_command
from ai.registered_camera_workers import safe_command_text, start_camera_worker
from scripts.run_registered_cameras import config_from_args, main as run_registered_main, parse_args as parse_registered_args
from scripts.rtsp_inference_args import parse_args as parse_rtsp_args


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
        mqtt_camera_topic="camera",
        mqtt_event_topic="event",
        mqtt_client_id_prefix="strange-ai",
        mqtt_username=None,
        mqtt_password=None,
        detector_mode="real",
        yolo_model="yolo26n-pose.pt",
        device="cuda:0",
        detector_conf=0.15,
        action_model=None,
        action_device="cuda:0",
        action_threshold=None,
        classifier_input="keypoints",
        sequence_length=30,
        sequence_stride=15,
        tracking_mode="supervision",
        track_thresh=0.10,
        match_thresh=0.20,
        track_buffer=90,
        bbox_smoothing_alpha=0.60,
        print_events=False,
        dry_run=True,
        rtsp_probe_enabled=True,
        refresh_interval_seconds=30.0,
    )


class RegisteredCameraDockerConfigTest(unittest.TestCase):
    def test_parse_args_defaults_to_sequence_30_stride_15(self):
        config = config_from_args(parse_registered_args([]))

        self.assertEqual(config.backend_base_url, "http://localhost:18080")
        self.assertEqual(config.sequence_length, 30)
        self.assertEqual(config.sequence_stride, 15)
        self.assertEqual(config.domain, "outside")
        self.assertIsNone(config.label)
        self.assertTrue(config.mjpeg_enabled)

    def test_parse_args_reads_docker_environment_defaults(self):
        env = {
            "BACKEND_BASE_URL": "http://backend:8080",
            "MEDIAMTX_RTSP_BASE_URL": "rtsp://mediamtx:8554",
            "VIDEO_POOL_DIR": "/app/video_pool",
            "MQTT_HOST": "mqtt",
            "MQTT_PORT": "1884",
            "MQTT_TOPIC": "safety/custom",
            "MQTT_CAMERA_TOPIC": "camera/custom",
            "MQTT_EVENT_TOPIC": "event/custom",
            "YOLO_MODEL_PATH": "/models/yolo26n-pose.pt",
            "MODEL_CHECKPOINT_PATH": "/models/lstm.pt",
            "DEVICE": "cpu",
            "SEQUENCE_LENGTH": "12",
            "SEQUENCE_STRIDE": "6",
            "CAMERA_POLL_INTERVAL_SECONDS": "15",
            "MJPEG_ENABLED": "true",
            "MJPEG_PORT": "8020",
            "MJPEG_FPS": "6",
            "MJPEG_WIDTH": "640",
            "MJPEG_HEIGHT": "360",
            "MJPEG_JPEG_QUALITY": "65",
            "MJPEG_BASE_PATH": "/mjpeg",
            "MJPEG_ENABLE_OVERLAY": "false",
        }

        with patch.dict("os.environ", env, clear=False):
            config = config_from_args(parse_registered_args([]))

        actual = (
            config.backend_base_url, config.rtsp_base_url, config.video_pool, config.mqtt_host,
            config.mqtt_port, config.mqtt_topic, config.mqtt_camera_topic, config.mqtt_event_topic,
            config.yolo_model, config.action_model,
            config.device, config.sequence_length, config.sequence_stride, config.refresh_interval_seconds,
            config.mjpeg_enabled, config.overlay_base_port, config.mjpeg_fps, config.mjpeg_width,
            config.mjpeg_height, config.mjpeg_jpeg_quality, config.mjpeg_base_path,
            config.mjpeg_enable_overlay,
        )
        expected = (
            "http://backend:8080", "rtsp://mediamtx:8554", Path("/app/video_pool"), "mqtt",
            1884, "safety/custom", "camera/custom", "event/custom", "/models/yolo26n-pose.pt", "/models/lstm.pt",
            "cpu", 12, 6, 15.0,
            True, 8020, 6.0, 640, 360, 65, "/mjpeg",
            False,
        )
        self.assertEqual(actual, expected)

    def test_dry_run_continues_when_backend_is_unavailable(self):
        with (
            patch("scripts.run_registered_cameras.load_active_cameras", side_effect=RuntimeError("offline")),
            patch("scripts.run_registered_cameras.run_cameras") as run_cameras,
            patch("scripts.run_registered_cameras.time.sleep") as sleep,
        ):
            run_registered_main(["--dry-run", "--skip-rtsp-probe"])

        sleep.assert_not_called()
        run_cameras.assert_called_once()
        self.assertEqual(run_cameras.call_args.args[0], [])

    def test_overlay_command_does_not_pass_mqtt_password_as_argv(self):
        camera = RegisteredCamera(
            camera_id="9",
            camera_login_id="icu_01",
            rtsp_url="rtsp://user:secret@cctv/icu",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )

        command = build_overlay_command(
            camera,
            "rtsp://user:secret@cctv/icu",
            8012,
            replace(fake_config(Path("video_pool")), mqtt_password="secret"),
        )

        self.assertNotIn("--mqtt-password", command)
        self.assertNotIn("--rtsp-url", command)
        self.assertNotIn("secret", safe_command_text(command))

    def test_overlay_worker_receives_rtsp_url_from_environment(self):
        camera = RegisteredCamera(
            camera_id="9",
            camera_login_id="icu_01",
            rtsp_url="rtsp://user:secret@cctv/icu",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )
        config = replace(fake_config(Path("video_pool")), dry_run=False)

        with (
            patch("ai.registered_camera_workers.rtsp_has_readable_frame", return_value=True),
            patch("ai.registered_camera_workers.spawn_process", return_value=object()) as spawn_process,
        ):
            start_camera_worker(camera, config, 8012)

        overlay_call = spawn_process.call_args_list[-1]
        command = overlay_call.args[0]
        env = overlay_call.kwargs["env"]
        self.assertNotIn("--rtsp-url", command)
        self.assertEqual(env["RTSP_URL"], "rtsp://user:secret@cctv/icu")

    def test_overlay_worker_receives_webrtc_sync_token_from_environment(self):
        camera = RegisteredCamera(
            camera_id="9",
            camera_login_id="icu_01",
            rtsp_url="rtsp://cctv/icu",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )
        config = replace(
            fake_config(Path("video_pool")),
            dry_run=False,
            webrtc_sync_enabled=True,
            webrtc_sync_token="sync-secret",
        )

        with (
            patch("ai.registered_camera_workers.rtsp_has_readable_frame", return_value=True),
            patch("ai.registered_camera_workers.spawn_process", return_value=object()) as spawn_process,
        ):
            start_camera_worker(camera, config, 8012)

        overlay_call = spawn_process.call_args_list[-1]
        command = overlay_call.args[0]
        env = overlay_call.kwargs["env"]
        self.assertNotIn("--webrtc-sync-token", command)
        self.assertNotIn("sync-secret", safe_command_text(command))
        self.assertEqual(env["AI_WEBRTC_SYNC_TOKEN"], "sync-secret")

    def test_rtsp_inference_reads_docker_model_environment_aliases(self):
        env = {
            "YOLO_MODEL_PATH": "/models/yolo26n-pose.pt",
            "MODEL_CHECKPOINT_PATH": "/models/lstm.pt",
            "DEVICE": "cpu",
        }

        with patch.dict("os.environ", env, clear=False):
            args = parse_rtsp_args([])

        self.assertEqual(args.yolo_model, "/models/yolo26n-pose.pt")
        self.assertEqual(args.action_model, "/models/lstm.pt")
        self.assertEqual(args.device, "cpu")

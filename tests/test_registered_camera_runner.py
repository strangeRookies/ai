import tempfile
import unittest
import urllib.error
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from argparse import Namespace

from ai.registered_cameras import (
    DEFAULT_BACKEND_BASE_URL,
    RegisteredCamera,
    RunnerConfig,
    build_overlay_command,
    camera_rtsp_url,
    input_rtsp_for_camera,
    load_active_cameras,
    parse_camera,
)
from ai.registered_camera_workers import CameraWorker, next_overlay_port, publish_unavailable_camera_status, sync_camera_workers
from scripts.run_registered_cameras import log_camera_api_config, warn_if_multiple_registered_camera_runners


class RegisteredCameraRunnerTest(unittest.TestCase):
    def test_default_backend_base_url_uses_host_18080(self):
        self.assertEqual(DEFAULT_BACKEND_BASE_URL, "http://localhost:18080")

    def test_camera_api_config_log_includes_effective_endpoint(self):
        config = replace(fake_config(Path("video_pool")), backend_base_url="http://localhost:18080")

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            logged = log_camera_api_config(config)

        self.assertEqual(logged["base_url"], "http://localhost:18080")
        self.assertEqual(logged["endpoint"], "/api/cameras/active")
        self.assertEqual(logged["url"], "http://localhost:18080/api/cameras/active")
        self.assertIn("[camera-api-config]", stdout.getvalue())

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
        self.assertNotIn("--rtsp-url", command)

    def test_overlay_command_passes_tracker_tuning_arguments(self):
        camera = RegisteredCamera(
            camera_id="9",
            camera_login_id="icu_01",
            rtsp_url="rtsp://cctv/icu",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )
        config = replace(
            fake_config(Path("video_pool")),
            detector_conf=0.21,
            track_thresh=0.07,
            match_thresh=0.12,
            track_buffer=150,
            bbox_smoothing_alpha=0.45,
        )

        command = build_overlay_command(camera, "rtsp://cctv/icu", 8012, config)

        self.assertEqual(command[command.index("--detector-conf") + 1], "0.21")
        self.assertEqual(command[command.index("--track-thresh") + 1], "0.07")
        self.assertEqual(command[command.index("--match-thresh") + 1], "0.12")
        self.assertEqual(command[command.index("--track-buffer") + 1], "150")
        self.assertEqual(command[command.index("--bbox-smoothing-alpha") + 1], "0.45")

    def test_overlay_command_passes_tracking_stability_fallback_only_when_enabled(self):
        camera = RegisteredCamera(
            camera_id="9",
            camera_login_id="icu_01",
            rtsp_url="rtsp://cctv/icu",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )

        default_command = build_overlay_command(camera, "rtsp://cctv/icu", 8012, fake_config(Path("video_pool")))
        fallback_command = build_overlay_command(
            camera,
            "rtsp://cctv/icu",
            8012,
            replace(fake_config(Path("video_pool")), tracking_stability_fallback=True),
        )

        self.assertNotIn("--tracking-stability-fallback", default_command)
        self.assertIn("--tracking-stability-fallback", fallback_command)

    def test_overlay_command_can_enable_tracking_stability_fallback_for_one_camera(self):
        cam4 = RegisteredCamera(
            camera_id="4",
            camera_login_id="cam_04",
            rtsp_url="rtsp://cctv/cam_04",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )
        cam5 = RegisteredCamera(
            camera_id="5",
            camera_login_id="cam_05",
            rtsp_url="rtsp://cctv/cam_05",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )
        config = replace(
            fake_config(Path("video_pool")),
            tracking_stability_fallback=False,
            tracking_stability_fallback_camera_ids=("cam_05",),
        )

        cam4_command = build_overlay_command(cam4, "rtsp://cctv/cam_04", 8012, config)
        cam5_command = build_overlay_command(cam5, "rtsp://cctv/cam_05", 8013, config)

        self.assertNotIn("--tracking-stability-fallback", cam4_command)
        self.assertIn("--tracking-stability-fallback", cam5_command)

    def test_overlay_command_passes_mjpeg_debug_only_when_enabled(self):
        camera = RegisteredCamera(
            camera_id="9",
            camera_login_id="icu_01",
            rtsp_url="rtsp://cctv/icu",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )

        default_command = build_overlay_command(camera, "rtsp://cctv/icu", 8012, fake_config(Path("video_pool")))
        debug_command = build_overlay_command(
            camera,
            "rtsp://cctv/icu",
            8012,
            replace(fake_config(Path("video_pool")), mjpeg_debug=True),
        )

        self.assertNotIn("--mjpeg-debug", default_command)
        self.assertIn("--mjpeg-debug", debug_command)

    def test_overlay_command_passes_webrtc_sync_options_only_when_enabled(self):
        camera = RegisteredCamera(
            camera_id="9",
            camera_login_id="icu_01",
            rtsp_url="rtsp://cctv/icu",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )

        default_command = build_overlay_command(camera, "rtsp://cctv/icu", 8012, fake_config(Path("video_pool")))
        sync_command = build_overlay_command(
            camera,
            "rtsp://cctv/icu",
            8012,
            replace(
                fake_config(Path("video_pool")),
                webrtc_sync_enabled=True,
                webrtc_sync_host="127.0.0.1",
                webrtc_sync_base_port=8090,
                webrtc_sync_token="secret-token",
            ),
        )

        self.assertNotIn("--webrtc-sync-enabled", default_command)
        self.assertIn("--webrtc-sync-enabled", sync_command)
        self.assertIn("--webrtc-sync-host", sync_command)
        self.assertIn("127.0.0.1", sync_command)
        self.assertIn("--webrtc-sync-port", sync_command)
        self.assertIn("8092", sync_command)
        self.assertIn("--webrtc-sync-stream-id", sync_command)
        self.assertIn("icu_01_ai", sync_command)
        self.assertNotIn("--webrtc-sync-token", sync_command)
        self.assertNotIn("secret-token", sync_command)

    def test_load_active_cameras_reads_backend_success_data_envelope(self):
        response = FakeHttpResponse(
            b'{"success":true,"data":[{"cameraId":4,"cameraLoginId":"cam_02","rtspUrl":"rtsp://cctv/cam_02","sourceType":"REAL_RTSP","aiEnabled":true,"status":"ACTIVE"}]}'
        )

        with patch("urllib.request.urlopen", return_value=response):
            cameras = load_active_cameras("http://backend:8080", None)

        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0].camera_login_id, "cam_02")
        self.assertEqual(cameras[0].rtsp_url, "rtsp://cctv/cam_02")

    def test_load_active_cameras_reports_http_status_url_timeout_and_body(self):
        error = urllib.error.HTTPError(
            "http://localhost:18080/api/cameras/active",
            503,
            "Service Unavailable",
            {},
            FakeHttpResponse(b'{"error":"backend booting","detail":"db unavailable"}'),
        )

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                load_active_cameras("http://localhost:18080", None, timeout_seconds=3.0)

        with patch("urllib.request.urlopen", side_effect=error):
            try:
                load_active_cameras("http://localhost:18080", None, timeout_seconds=3.0)
            except RuntimeError as exc:
                message = str(exc)
            else:
                self.fail("expected RuntimeError")
        self.assertIn("http://localhost:18080/api/cameras/active", message)
        self.assertIn("timeout=3", message)
        self.assertIn("backend booting", message)

    def test_publish_unavailable_camera_status_uses_camera_login_id(self):
        camera = RegisteredCamera(
            camera_id="10",
            camera_login_id="cam4",
            rtsp_url="rtsp://gpu-pc:8554/cam4",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )

        with patch("ai.registered_camera_workers.create_event_publisher", return_value=(FakePublisher(), "console")) as create_publisher:
            publish_unavailable_camera_status(
                camera,
                "rtsp://gpu-pc:8554/cam4",
                fake_config(Path("video_pool")),
                "ERROR",
                "RTSP_PROBE_FAILED",
            )

        publisher = create_publisher.return_value[0]
        self.assertEqual(len(publisher.client.payloads), 1)
        self.assertEqual(publisher.client.payloads[0]["camera_login_id"], "cam4")
        self.assertEqual(publisher.client.payloads[0]["status"], "ERROR")
        self.assertEqual(publisher.client.payloads[0]["reason"], "RTSP_PROBE_FAILED")

    def test_sync_camera_workers_skips_unreachable_real_rtsp(self):
        camera = RegisteredCamera(
            camera_id="11",
            camera_login_id="cam5",
            rtsp_url="rtsp://gpu-pc:8554/cam5",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )
        config = replace(fake_config(Path("video_pool")), dry_run=False, publisher="console")
        workers: dict[str, CameraWorker] = {}

        with (
            patch("ai.registered_camera_workers.rtsp_has_readable_frame", return_value=False),
            patch("ai.registered_camera_workers.publish_unavailable_camera_status") as publish_status,
            patch("ai.registered_camera_workers.spawn_process") as spawn_process,
        ):
            sync_camera_workers(workers, [camera], config)

        publish_status.assert_called_once()
        spawn_process.assert_not_called()
        self.assertNotIn("cam5", workers)

    def test_sync_camera_workers_starts_new_active_camera(self):
        camera = RegisteredCamera(
            camera_id="12",
            camera_login_id="cam6",
            rtsp_url="rtsp://gpu-pc:8554/cam6",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )
        worker = CameraWorker(processes=[], overlay_port=8010, source_signature="REAL_RTSP:rtsp://gpu-pc:8554/cam6")
        workers: dict[str, CameraWorker] = {}

        with patch("ai.registered_camera_workers.start_camera_worker", return_value=worker) as start_worker:
            sync_camera_workers(workers, [camera], fake_config(Path("video_pool")))

        start_worker.assert_called_once()
        self.assertIs(workers["cam6"], worker)

    def test_sync_camera_workers_stops_removed_camera(self):
        worker = CameraWorker(processes=[], overlay_port=8010, source_signature="REAL_RTSP:rtsp://gpu-pc:8554/cam7")
        workers = {"cam7": worker}

        with patch("ai.registered_camera_workers.stop_processes") as stop_processes:
            sync_camera_workers(workers, [], fake_config(Path("video_pool")))

        stop_processes.assert_called_once_with(worker.processes)
        self.assertNotIn("cam7", workers)

    def test_sync_camera_workers_restarts_camera_when_rtsp_url_changes(self):
        camera = RegisteredCamera(
            camera_id="13",
            camera_login_id="cam8",
            rtsp_url="rtsp://gpu-pc:8554/cam8-new",
            source_type="REAL_RTSP",
            assigned_video_path=None,
        )
        old_worker = CameraWorker(processes=[], overlay_port=8010, source_signature="REAL_RTSP:rtsp://gpu-pc:8554/cam8-old")
        new_worker = CameraWorker(processes=[], overlay_port=8010, source_signature="REAL_RTSP:rtsp://gpu-pc:8554/cam8-new")
        workers = {"cam8": old_worker}

        with (
            patch("ai.registered_camera_workers.stop_processes") as stop_processes,
            patch("ai.registered_camera_workers.start_camera_worker", return_value=new_worker) as start_worker,
        ):
            sync_camera_workers(workers, [camera], fake_config(Path("video_pool")))

        stop_processes.assert_called_once_with(old_worker.processes)
        start_worker.assert_called_once()
        self.assertIs(workers["cam8"], new_worker)

    def test_next_overlay_port_reuses_preferred_port_when_free(self):
        workers = {
            "cam_02": CameraWorker(processes=[], overlay_port=8011, source_signature="REAL_RTSP:rtsp://cctv/cam_02")
        }

        port = next_overlay_port(workers, fake_config(Path("video_pool")), preferred_port=8010)

        self.assertEqual(port, 8010)

    def test_next_overlay_port_skips_orphan_bound_port(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            occupied_port = server.getsockname()[1]
            config = replace(fake_config(Path("video_pool")), overlay_base_port=occupied_port)

            port = next_overlay_port({}, config)

        self.assertEqual(port, occupied_port + 1)

    def test_runner_warns_when_multiple_registered_camera_runners_exist(self):
        completed = Namespace(
            returncode=0,
            stdout=(
                "111 python scripts/run_registered_cameras.py --backend-base-url http://127.0.0.1:8080\n"
                "222 python scripts/run_registered_cameras.py --backend-base-url http://127.0.0.1:18080\n"
            ),
            stderr="",
        )

        with patch("subprocess.run", return_value=completed):
            warning = warn_if_multiple_registered_camera_runners(current_pid=222)

        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertEqual(warning["runnerCount"], 2)
        self.assertIn("pkill -f 'scripts/run_registered_cameras.py'", warning["cleanupCommand"])


class FakeHttpResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, unused_exc_type, unused_exc, unused_traceback):
        return False

    def read(self):
        return self._body

    def close(self):
        return None


class FakePublisher:
    def __init__(self):
        self.client = FakeMqttClient()

    def close(self):
        return None


class FakeMqttClient:
    def __init__(self):
        self.payloads = []

    def publish(self, unused_topic, payload, qos=0):
        import json

        self.payloads.append(json.loads(payload))
        return None


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

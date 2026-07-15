import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ai.worker_registry
from ai.registered_cameras import RegisteredCamera
from ai.ffmpeg_command import build_ffmpeg_command
from ai.simulated_rtsp_publisher import FfmpegRestartPolicy, PublisherLock, select_initial_ffmpeg_mode, tail_text_file
from ai.simulated_rtsp_runtime import (
    load_video_assignments,
    save_video_assignments,
    scan_stream_video_directories,
    select_rotating_video,
    stream_config_changed,
    video_file_fingerprint,
    video_pool_fingerprint,
    video_pool_for_camera_position,
)
from scripts.start_simulated_rtsp_from_folder import build_ffmpeg_cmd, parse_arguments, stable_video_index, video_for_camera
from ai.simulated_rtsp_sources import filter_stream_video_pools, filter_video_files


class SimulatedRtspFolderPublisherTest(unittest.TestCase):
    def test_fallback_video_index_is_stable_for_camera_id(self):
        first = stable_video_index("cam_01", 12)
        second = stable_video_index("cam_01", 12)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 12)

    def test_ffmpeg_command_limits_browser_stream_load(self):
        command = build_ffmpeg_cmd(Path("sample.mp4"), "rtsp://localhost:8554/cam_01", True)

        self.assertIn("scale=1280:720:force_original_aspect_ratio=decrease", " ".join(command))
        self.assertIn("fps=15", " ".join(command))
        self.assertEqual(command[command.index("-b:v") + 1], "1500k")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")

    def test_ffmpeg_command_accepts_libx264_alias(self):
        command = build_ffmpeg_cmd(
            Path("sample.mp4"),
            "rtsp://localhost:8554/cam_01",
            True,
            "libx264",
        )

        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-preset") + 1], "ultrafast")

    def test_legacy_registered_camera_ffmpeg_helper_uses_auto_default(self):
        with patch.dict("os.environ", {}, clear=True):
            command = build_ffmpeg_command(Path("sample.mp4"), "rtsp://localhost:8554/cam_01")

        self.assertEqual(command[command.index("-c:v") + 1], "copy")

    def test_legacy_registered_camera_ffmpeg_helper_honors_env_override(self):
        with patch.dict("os.environ", {"FFMPEG_MODE": "cpu"}):
            command = build_ffmpeg_command(Path("sample.mp4"), "rtsp://localhost:8554/cam_01")

        self.assertEqual(command[command.index("-c:v") + 1], "libx264")

    def test_cli_default_honors_ffmpeg_mode_environment(self):
        argv = ["start_simulated_rtsp_from_folder.py", "--video-dir", "."]
        with patch.dict("os.environ", {"FFMPEG_MODE": "copy"}), patch("sys.argv", argv):
            args = parse_arguments()

        self.assertEqual(args.ffmpeg_mode, "copy")

    def test_cli_defaults_to_empty_domain_for_local_video_pool(self):
        argv = ["start_simulated_rtsp_from_folder.py", "--video-dir", "."]
        with patch.dict("os.environ", {}, clear=True), patch("sys.argv", argv):
            args = parse_arguments()

        # Empty domain = accept all non-chromakey files (close-up fall clips work).
        self.assertEqual(args.domain, "")
        self.assertIsNone(args.label)

    def test_cli_accepts_separate_chromakey_video_directory(self):
        argv = [
            "start_simulated_rtsp_from_folder.py",
            "--video-dir",
            "video_pool",
            "--chromakey-video-dir",
            "chromakey_pool",
        ]
        with patch.dict("os.environ", {}, clear=True), patch("sys.argv", argv):
            args = parse_arguments()

        self.assertEqual(args.chromakey_video_dir, "chromakey_pool")

    def test_stable_start_script_streams_outside_videos_only(self):
        script = Path("scripts/start_ai_stable.sh").read_text(encoding="utf-8")

        self.assertIn("--domain outside", script)
        self.assertNotIn("--domain inside", script)
        self.assertNotIn("--label swoon", script)

    def test_stable_start_script_pins_bbox54_lstm_checkpoint(self):
        script = Path("scripts/start_ai_stable.sh").read_text(encoding="utf-8")

        self.assertIn("DEFAULT_ACTION_MODEL=\"runs/evaluation_manifest_v2_bbox54_balanced/retrained_best.pt\"", script)
        self.assertIn("ACTION_MODEL=\"${3:-${ACTION_MODEL:-$DEFAULT_ACTION_MODEL}}\"", script)
        self.assertIn("ACTION_MODEL checkpoint not found", script)
        self.assertIn("--action-model \"$ACTION_MODEL\"", script)

    def test_stable_start_script_keeps_torch_default_and_tensorrt_opt_in(self):
        script = Path("scripts/start_ai_stable.sh").read_text(encoding="utf-8")

        self.assertIn("DEFAULT_YOLO_MODEL=\"yolo26n-pose.pt\"", script)
        self.assertIn("TENSORRT_YOLO_MODEL=\"yolo26n-pose.engine\"", script)
        self.assertIn("USE_TENSORRT", script)
        self.assertIn("TensorRT requested but engine not found", script)
        self.assertIn("--yolo-model \"$YOLO_MODEL\"", script)

    def test_click_launcher_cleanup_kills_relative_ai_processes(self):
        for script_path in (Path("AI_실행_딸깍.bat"), Path("../AI_실행_딸깍.bat")):
            with self.subTest(script_path=script_path):
                script = script_path.read_text(encoding="utf-8")

                self.assertIn("pkill -f 'scripts/run_registered_cameras.py'", script)
                self.assertIn("pkill -f 'scripts/start_simulated_rtsp_from_folder.py'", script)
                self.assertIn("pkill -f 'scripts/serve_ai_overlay.py'", script)
                self.assertNotIn("pkill -f '%REMOTE_ROOT%/scripts/run_registered_cameras.py'", script)

    def test_auto_restart_policy_falls_back_from_nvenc_after_exit_255(self):
        policy = FfmpegRestartPolicy(requested_mode="auto", initial_mode="nvenc")

        next_mode = policy.record_exit(exit_code=255)

        self.assertEqual(next_mode, "copy")
        self.assertEqual(policy.active_mode, "copy")
        self.assertEqual(policy.restart_count, 1)

    def test_auto_restart_policy_falls_back_from_nvenc_failure_hints(self):
        policy = FfmpegRestartPolicy(requested_mode="auto", initial_mode="nvenc")

        next_mode = policy.record_exit(exit_code=1, stderr_tail="OpenEncodeSessionEx failed")

        self.assertEqual(next_mode, "copy")
        self.assertEqual(policy.active_mode, "copy")

    def test_auto_restart_policy_falls_back_from_repeated_nvenc_failures(self):
        policy = FfmpegRestartPolicy(requested_mode="auto", initial_mode="nvenc", max_restarts_per_mode=1)

        first_mode = policy.record_exit(exit_code=1)
        second_mode = policy.record_exit(exit_code=1)

        self.assertEqual(first_mode, "nvenc")
        self.assertEqual(second_mode, "copy")

    def test_copy_restart_policy_falls_back_to_cpu_after_repeated_failure(self):
        policy = FfmpegRestartPolicy(requested_mode="auto", initial_mode="copy", max_restarts_per_mode=1)

        first_mode = policy.record_exit(exit_code=1)
        second_mode = policy.record_exit(exit_code=1)

        self.assertEqual(first_mode, "copy")
        self.assertEqual(second_mode, "cpu")
        self.assertEqual(policy.active_mode, "cpu")

    def test_auto_initial_mode_prefers_nvenc_only_when_available(self):
        self.assertEqual(select_initial_ffmpeg_mode("auto", nvenc_available=True), "nvenc")
        self.assertEqual(select_initial_ffmpeg_mode("auto", nvenc_available=False), "copy")

    def test_publisher_lock_rejects_live_parent_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "start.lock"
            lock_path.write_text("1234", encoding="utf-8")
            lock = PublisherLock(lock_path)

            with patch("ai.worker_registry.check_pid_alive", return_value=True), patch(
                "ai.worker_registry.check_process_signature_matches",
                return_value=True,
            ):
                with self.assertRaises(RuntimeError):
                    lock.acquire()

    def test_publisher_lock_removes_stale_parent_process_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "start.lock"
            lock_path.write_text("1234", encoding="utf-8")
            lock = PublisherLock(lock_path)

            with patch("ai.worker_registry.check_pid_alive", return_value=False):
                lock.acquire()
                lock.release()

            self.assertFalse(lock_path.exists())

    def test_tail_text_file_returns_recent_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "cam_01-ffmpeg.log"
            log_path.write_text("\n".join(f"line-{idx}" for idx in range(60)), encoding="utf-8")

            tail = tail_text_file(log_path, max_lines=3)

        self.assertEqual(tail, "line-57\nline-58\nline-59")

    def test_video_for_camera_prefers_backend_assigned_video_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            short_video = Path(temp_dir) / "000_short_clip.mp4"
            full_video = Path(temp_dir) / "full_faint_sequence.mp4"
            short_video.write_bytes(b"short")
            full_video.write_bytes(b"full")
            camera = RegisteredCamera(
                camera_id="1",
                camera_login_id="cam_01",
                rtsp_url=None,
                source_type="SIMULATED_RTSP",
                assigned_video_path=str(full_video),
            )

            selected = video_for_camera(camera, [short_video], 0)

        self.assertEqual(selected, full_video)

    def test_video_for_camera_ignores_backend_assigned_chromakey_video(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_pool = Path(temp_dir)
            chromakey_video = video_pool / "indoor_chromakey_faint.mp4"
            normal_video = video_pool / "outdoor_real_faint.mp4"
            chromakey_video.write_bytes(b"chromakey")
            normal_video.write_bytes(b"normal")
            camera = RegisteredCamera(
                camera_id="1",
                camera_login_id="cam_01",
                rtsp_url=None,
                source_type="SIMULATED_RTSP",
                assigned_video_path=str(chromakey_video),
            )

            selected = video_for_camera(camera, [normal_video], 0)

        self.assertEqual(selected, normal_video)

    def test_filter_video_files_excludes_chromakey_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chromakey_video = root / "indoor_chromakey" / "fall.mp4"
            normal_video = root / "outdoor" / "fall.mp4"
            chromakey_video.parent.mkdir()
            normal_video.parent.mkdir()
            chromakey_video.write_bytes(b"chromakey")
            normal_video.write_bytes(b"normal")

            matching, excluded = filter_video_files([chromakey_video, normal_video], None, None, None)

        self.assertEqual(matching, [normal_video])
        self.assertEqual(excluded, [chromakey_video])

    def test_stream_video_pools_separate_outdoor_and_chromakey_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chromakey_video = root / "indoor_chromakey" / "fall.mp4"
            outdoor_video = root / "outside" / "fall.mp4"
            chromakey_video.parent.mkdir()
            outdoor_video.parent.mkdir()
            chromakey_video.write_bytes(b"chromakey")
            outdoor_video.write_bytes(b"outdoor")

            outdoor, chromakey, excluded = filter_stream_video_pools(
                [chromakey_video, outdoor_video], "outside", None, None
            )

        self.assertEqual(outdoor, [outdoor_video])
        self.assertEqual(chromakey, [chromakey_video])
        self.assertEqual(excluded, [])

    def test_stream_video_pools_require_both_kinds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outdoor_video = Path(temp_dir) / "outside_fall.mp4"
            outdoor_video.write_bytes(b"outdoor")

            with self.assertRaisesRegex(ValueError, "chromakey"):
                filter_stream_video_pools([outdoor_video], None, None, None)

    def test_camera_pairs_alternate_outdoor_and_chromakey_pools(self):
        outdoor = [Path("outside.mp4")]
        chromakey = [Path("chromakey.mp4")]

        first_pool, first_is_chromakey = video_pool_for_camera_position(0, outdoor, chromakey)
        second_pool, second_is_chromakey = video_pool_for_camera_position(1, outdoor, chromakey)
        third_pool, third_is_chromakey = video_pool_for_camera_position(2, outdoor, chromakey)
        fourth_pool, fourth_is_chromakey = video_pool_for_camera_position(3, outdoor, chromakey)

        self.assertEqual(first_pool, outdoor)
        self.assertFalse(first_is_chromakey)
        self.assertEqual(second_pool, chromakey)
        self.assertTrue(second_is_chromakey)
        self.assertEqual(third_pool, outdoor)
        self.assertFalse(third_is_chromakey)
        self.assertEqual(fourth_pool, chromakey)
        self.assertTrue(fourth_is_chromakey)

    def test_scan_stream_video_directories_combines_separate_pools(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outdoor_dir = root / "video_pool"
            chromakey_dir = root / "indoor_chromakey" / "videos"
            outdoor_dir.mkdir()
            chromakey_dir.mkdir(parents=True)
            outdoor_video = outdoor_dir / "outside.mp4"
            chromakey_video = chromakey_dir / "cam1.mp4"
            outdoor_video.write_bytes(b"outdoor")
            chromakey_video.write_bytes(b"chromakey")

            scanned = scan_stream_video_directories(str(outdoor_dir), str(chromakey_dir))

        self.assertEqual(set(scanned), {outdoor_video, chromakey_video})

    def test_rotating_selection_avoids_previous_and_in_use_videos(self):
        videos = [Path("outside_01.mp4"), Path("outside_02.mp4"), Path("outside_03.mp4")]

        selected = select_rotating_video(
            videos,
            previous_video=videos[0],
            used_videos={videos[1]},
            rng=random.Random(7),
        )

        self.assertEqual(selected, videos[2])

    def test_rotating_selection_reuses_only_video_when_no_alternative_exists(self):
        only_video = Path("chromakey_01.mp4")

        selected = select_rotating_video(
            [only_video],
            previous_video=only_video,
            used_videos={only_video},
            rng=random.Random(7),
        )

        self.assertEqual(selected, only_video)

    def test_video_assignments_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "video_assignments.json"
            expected = {
                "cam_03": Path(temp_dir) / "outside.mp4",
                "cam_04": Path(temp_dir) / "chromakey.mp4",
            }

            save_video_assignments(state_path, expected)
            loaded = load_video_assignments(state_path)

        self.assertEqual(loaded, {camera_id: path.resolve() for camera_id, path in expected.items()})

    def test_video_fingerprint_detects_same_path_content_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "outside.mp4"
            video_path.write_bytes(b"old")
            before = video_file_fingerprint(video_path)

            video_path.write_bytes(b"new-content")
            after = video_file_fingerprint(video_path)

        self.assertNotEqual(before, after)

    def test_video_pool_fingerprint_detects_added_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first_video = Path(temp_dir) / "outside_01.mp4"
            second_video = Path(temp_dir) / "outside_02.mp4"
            first_video.write_bytes(b"first")
            second_video.write_bytes(b"second")

            before = video_pool_fingerprint([first_video])
            after = video_pool_fingerprint([first_video, second_video])

        self.assertNotEqual(before, after)

    def test_stream_config_stays_unchanged_for_same_file_fingerprint(self):
        video_path = Path("outside.mp4")
        fingerprint = (str(video_path.resolve()), 100, 12345)
        stream_info = {
            "video_path": video_path,
            "video_fingerprint": fingerprint,
            "rtsp_url": "rtsp://localhost:8554/cam_03",
        }

        changed = stream_config_changed(
            stream_info,
            video_path,
            fingerprint,
            "rtsp://localhost:8554/cam_03",
        )

        self.assertFalse(changed)

    def test_stream_config_changes_when_same_path_content_fingerprint_changes(self):
        video_path = Path("outside.mp4")
        stream_info = {
            "video_path": video_path,
            "video_fingerprint": (str(video_path.resolve()), 100, 12345),
            "rtsp_url": "rtsp://localhost:8554/cam_03",
        }

        changed = stream_config_changed(
            stream_info,
            video_path,
            (str(video_path.resolve()), 200, 67890),
            "rtsp://localhost:8554/cam_03",
        )

        self.assertTrue(changed)

    def test_video_for_chromakey_camera_accepts_assigned_chromakey_video(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chromakey_video = Path(temp_dir) / "chromakey_fall.mp4"
            fallback_video = Path(temp_dir) / "green_screen_fall.mp4"
            chromakey_video.write_bytes(b"assigned")
            fallback_video.write_bytes(b"fallback")
            camera = RegisteredCamera(
                camera_id="2",
                camera_login_id="cam_02",
                rtsp_url=None,
                source_type="SIMULATED_RTSP",
                assigned_video_path=str(chromakey_video),
            )

            selected = video_for_camera(camera, [fallback_video], 0, chromakey=True)

        self.assertEqual(selected, chromakey_video)

    def test_video_for_camera_ignores_assigned_video_outside_video_pool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_pool = Path(temp_dir) / "pool"
            outside_pool = Path(temp_dir) / "outside"
            video_pool.mkdir()
            outside_pool.mkdir()
            fallback_video = video_pool / "fallback.mp4"
            outside_video = outside_pool / "outside.mp4"
            fallback_video.write_bytes(b"fallback")
            outside_video.write_bytes(b"outside")
            camera = RegisteredCamera(
                camera_id="1",
                camera_login_id="cam_01",
                rtsp_url=None,
                source_type="SIMULATED_RTSP",
                assigned_video_path=str(outside_video),
            )

            selected = video_for_camera(camera, [fallback_video], 0)

        self.assertEqual(selected, fallback_video)


if __name__ == "__main__":
    unittest.main()

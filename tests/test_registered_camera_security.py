import tempfile
import unittest
from pathlib import Path

from ai.registered_cameras import RegisteredCamera, parse_camera, resolve_simulated_video


class RegisteredCameraSecurityTest(unittest.TestCase):
    def test_parse_camera_rejects_path_like_camera_login_id(self):
        camera = parse_camera(
            {
                "cameraId": 42,
                "cameraLoginId": "../outside",
                "rtspUrl": "rtsp://cctv/stream1",
                "sourceType": "REAL_RTSP",
                "aiEnabled": True,
                "status": "ACTIVE",
            }
        )

        self.assertIsNone(camera)

    def test_resolve_simulated_video_rejects_assigned_path_outside_video_pool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_pool = root / "videos"
            video_pool.mkdir()
            outside_video = root / "outside.mp4"
            outside_video.write_bytes(b"fake")
            camera = RegisteredCamera(
                camera_id="7",
                camera_login_id="ward_a",
                rtsp_url=None,
                source_type="SIMULATED_RTSP",
                assigned_video_path=str(outside_video),
            )

            with self.assertRaisesRegex(RuntimeError, "assignedVideoPath must stay under video_pool"):
                resolve_simulated_video(camera, video_pool)


if __name__ == "__main__":
    unittest.main()

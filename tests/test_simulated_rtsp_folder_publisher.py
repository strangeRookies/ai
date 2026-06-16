import tempfile
import unittest
from pathlib import Path

from ai.registered_cameras import RegisteredCamera
from scripts.start_simulated_rtsp_from_folder import video_for_camera


class SimulatedRtspFolderPublisherTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

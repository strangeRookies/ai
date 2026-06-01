import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_rtsp_demo import load_demo_camera_yaml_without_pyyaml, publisher_command, read_dataset_videos


class RtspDemoTest(unittest.TestCase):
    def test_demo_config_fallback_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.yaml"
            path.write_text(
                "cameras:\n"
                "  - camera_id: cam_01\n"
                "    name: Demo\n"
                "    rtsp_url: rtsp://localhost:8554/cam1\n"
                "    location: Local\n"
                "    enabled: true\n",
                encoding="utf-8",
            )

            config = load_demo_camera_yaml_without_pyyaml(path)

        self.assertEqual(config["cameras"][0]["camera_id"], "cam_01")
        self.assertTrue(config["cameras"][0]["enabled"])

    def test_read_dataset_videos_uses_clip_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "sample.avi"
            video.write_bytes(b"placeholder")
            metadata = root / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["clip_path"])
                writer.writeheader()
                writer.writerow({"clip_path": str(video)})

            videos = read_dataset_videos(metadata, 1)

        self.assertEqual(len(videos), 1)

    def test_publisher_command_uses_local_rtsp_path(self):
        command = publisher_command("sample.mp4", {"rtsp_url": "rtsp://localhost:8554/cam3"})

        self.assertEqual(command, ["bash", "scripts/publish_sample_video.sh", "sample.mp4", "cam3"])


if __name__ == "__main__":
    unittest.main()

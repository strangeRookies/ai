import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from argparse import Namespace

from scripts.run_rtsp_demo import load_demo_camera_yaml_without_pyyaml, process_camera, publisher_command, read_dataset_videos


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

    def test_read_dataset_videos_resolves_paths_from_metadata_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "data" / "clips" / "sample.avi"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"placeholder")
            metadata_dir = root / "data" / "metadata"
            metadata_dir.mkdir(parents=True)
            metadata = metadata_dir / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["clip_path"])
                writer.writeheader()
                writer.writerow({"clip_path": "data/clips/sample.avi"})

            videos = read_dataset_videos(metadata, 1)

        self.assertEqual(len(videos), 1)

    def test_publisher_command_uses_local_rtsp_path(self):
        command = publisher_command("sample.mp4", {"rtsp_url": "rtsp://localhost:8554/cam3"})

        self.assertEqual(command, ["bash", "scripts/publish_sample_video.sh", "sample.mp4", "cam3"])

    def test_process_camera_records_local_input_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "sample.avi"
            import cv2

            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
            for _ in range(4):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            args = Namespace(
                detector_mode="mock",
                yolo_model="yolov8n.pt",
                yolo_conf=0.25,
                yolo_iou=0.5,
                imgsz=640,
                sequence_length=2,
                sequence_stride=1,
                resize_size=32,
                dry_run=True,
                max_frames=3,
                read_from_rtsp=False,
            )

            summary = process_camera({"camera_id": "cam_01", "name": "Demo", "rtsp_url": "rtsp://localhost:8554/cam1"}, str(video), args)

        self.assertEqual(summary["input_mode"], "local_file")
        self.assertEqual(summary["frames_processed"], 3)
        self.assertIn("keypoints_extracted", summary)
        self.assertIn("lstm_predictions", summary)


if __name__ == "__main__":
    unittest.main()

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.run_dataset_evaluation import aggregate, process_row, read_dataset_rows
from scripts.run_rtsp_inference import create_detector


class DatasetEvaluationTest(unittest.TestCase):
    def test_reads_dataset_rows_and_resolves_video_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "data" / "clips" / "sample.avi"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"placeholder")
            metadata_dir = root / "data" / "metadata"
            metadata_dir.mkdir(parents=True)
            metadata = metadata_dir / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["clip_path", "label", "split"])
                writer.writeheader()
                writer.writerow({"clip_path": "data/clips/sample.avi", "label": "1", "split": "train"})

            rows = read_dataset_rows(metadata)

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["_resolved_video_path"].endswith("sample.avi"))

    def test_reads_dataset_rows_adds_split_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["clip_id", "clip_path", "label"])
                writer.writeheader()
                for idx in range(6):
                    writer.writerow({"clip_id": f"clip_{idx}", "clip_path": f"missing_{idx}.avi", "label": "1" if idx % 2 else "0"})

            rows = read_dataset_rows(metadata)

        self.assertTrue(all(row.get("split") in {"train", "test", "val"} for row in rows))

    def test_process_row_reports_keypoint_sequence_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "sample.avi"
            import cv2

            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
            for _ in range(4):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            args = Namespace(detector_mode="mock", max_frames=4, sequence_length=2, sequence_stride=1)
            detector = create_detector("mock", "yolov8n-pose.pt", "auto")

            summary = process_row({"_resolved_video_path": str(video), "split": "train", "label": "1", "clip_id": "sample"}, detector, args)

        self.assertEqual(summary["frames_processed"], 4)
        self.assertEqual(summary["bbox_detections"], 4)
        self.assertEqual(summary["keypoints_extracted"], 4)
        self.assertEqual(summary["generated_sequences"], 3)
        self.assertFalse(summary["zero_sequence"])

    def test_aggregate_includes_required_dataset_totals(self):
        args = Namespace(
            metadata_csv="metadata.csv",
            detector_mode="mock",
            yolo_model="yolov8n-pose.pt",
            max_frames=4,
            max_rows_per_split=1,
            sequence_length=2,
            sequence_stride=1,
        )
        rows = [{"split": "train", "label": "1"}]
        clips = [
            {
                "clip_id": "a",
                "split": "train",
                "label": "Faint",
                "video_path": "a.avi",
                "frames_processed": 4,
                "bbox_detections": 4,
                "keypoints_extracted": 4,
                "generated_sequences": 3,
                "fallback_crop_usage_count": 0,
                "zero_sequence": False,
                "error": None,
            }
        ]

        summary = aggregate(rows, clips, args)

        self.assertEqual(summary["totals"]["generated_sequences"], 3)
        self.assertEqual(summary["totals"]["fallback_crop_usage_ratio"], 0.0)
        self.assertIn("zero_sequence_examples", summary)


if __name__ == "__main__":
    unittest.main()

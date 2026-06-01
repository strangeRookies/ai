import unittest

import numpy as np

from ai.action.classifier import crops_to_features
from ai.action.train_lstm import load_training_rows, summarize_metadata, target_ranges_for_row


class LSTMActionClassifierTest(unittest.TestCase):
    def test_crops_to_features_shape(self):
        crops = [
            np.zeros((20, 30, 3), dtype=np.uint8),
            np.full((20, 30, 3), 255, dtype=np.uint8),
        ]

        features = crops_to_features(crops, feature_size=8)

        self.assertEqual(features.shape, (2, 64))
        self.assertEqual(float(features[0].max()), 0.0)
        self.assertEqual(float(features[1].min()), 1.0)

    def test_loads_ai_fall_metadata_csv(self):
        import csv
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "metadata.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["clip_path", "label", "split"])
                writer.writeheader()
                writer.writerow({"clip_path": "clip_a.mp4", "label": "1", "split": "train"})

            rows = load_training_rows(csv_path, split="train")

        self.assertEqual(rows[0]["video_path"], "clip_a.mp4")
        self.assertEqual(rows[0]["label"], 1)

    def test_event_frame_range_is_preferred(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "label.json"
            label_path.write_text(
                json.dumps(
                    {
                        "metadata": {"file_name": "clip.mp4", "frame_count": 100},
                        "annotations": {"event_class": "Faint", "event_frame": [[20, 40]]},
                    }
                ),
                encoding="utf-8",
            )
            row = {"label_path": str(label_path), "label": 1, "start_frame": 0, "end_frame": 99}

            ranges = target_ranges_for_row(row)

        self.assertEqual(ranges, [(20, 40, True)])

    def test_preprocess_summary_reports_fallback_and_zero_sequence(self):
        metadata = [
            {
                "crop_source": "fallback_full_frame",
                "used_event_frame": True,
            }
        ]
        clips = [
            {"sequences_generated": 1, "skipped_frames_no_person": 2},
            {"sequences_generated": 0, "skipped_frames_no_person": 3},
        ]

        summary = summarize_metadata(metadata, clips)

        self.assertEqual(summary["total_sequences_generated"], 1)
        self.assertEqual(summary["sequences_from_event_frame_ranges"], 1)
        self.assertEqual(summary["sequences_using_fallback_full_frame_crops"], 1)
        self.assertEqual(summary["zero_sequence_clips"], 1)
        self.assertEqual(summary["skipped_frames_due_to_no_person"], 5)


if __name__ == "__main__":
    unittest.main()

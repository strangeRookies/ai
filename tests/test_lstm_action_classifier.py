import unittest

import numpy as np

from ai.action.classifier import crops_to_features
from ai.action.train_lstm import load_training_rows


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


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from ai.action.classifier import crops_to_features


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


if __name__ == "__main__":
    unittest.main()

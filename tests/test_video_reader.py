import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ai.streams.video_reader import VideoReader


@unittest.skipIf(not CV2_AVAILABLE, "cv2 is required for video reader tests")
class VideoReaderTest(unittest.TestCase):
    def test_read_increments_frame_idx_by_one_without_sampling(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "sample.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 16))
            for _ in range(3):
                writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
            writer.release()

            with VideoReader(str(video)) as reader:
                packets = [reader.read(), reader.read(), reader.read()]

        self.assertEqual([packet.frame_idx for packet in packets if packet is not None], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

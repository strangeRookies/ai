import unittest

from stream.frame_queue import LatestFrameQueue


class LatestFrameQueueTest(unittest.TestCase):
    def test_put_latest_counts_dropped_frames_when_queue_is_full(self):
        queue = LatestFrameQueue(max_size=1)

        queue.put_latest("first")
        queue.put_latest("second")

        self.assertEqual(queue.drop_count, 1)
        self.assertEqual(queue.get(timeout=0), "second")


if __name__ == "__main__":
    unittest.main()

import unittest

from ai.frame_sync import FrameMetadataBuffer
from ai.streams.video_reader import FramePacket


class FrameSyncTest(unittest.TestCase):
    def test_buffer_assigns_camera_local_frame_ids_and_lookup(self):
        now_values = iter([1000, 1017, 1034])
        buffer = FrameMetadataBuffer(maxlen=2, now_ms=lambda: next(now_values))

        first = buffer.record_capture("cam_04", FramePacket(frame_idx=10, fps=30.0, timestamp=0.3, frame=None), (360, 640, 3))
        second = buffer.record_capture("cam_04", FramePacket(frame_idx=11, fps=30.0, timestamp=0.4, frame=None), (360, 640, 3))
        third = buffer.record_capture("cam_04", FramePacket(frame_idx=12, fps=30.0, timestamp=0.5, frame=None), (720, 1280, 3))

        self.assertEqual(first.frame_id, 1)
        self.assertEqual(second.frame_id, 2)
        self.assertEqual(third.frame_id, 3)
        self.assertIsNone(buffer.get_by_frame_id("cam_04", 1))
        self.assertEqual(buffer.get_latest("cam_04"), third)
        self.assertEqual(buffer.get_nearest_by_timestamp("cam_04", 1018), second)
        self.assertEqual(third.width, 1280)
        self.assertEqual(third.height, 720)

    def test_buffer_tracks_processed_and_published_latency(self):
        now_values = iter([2000, 2055, 2062])
        buffer = FrameMetadataBuffer(maxlen=60, now_ms=lambda: next(now_values))

        captured = buffer.record_capture("cam_04", FramePacket(frame_idx=0, fps=30.0, timestamp=0.0, frame=None), (360, 640, 3))
        processed = buffer.mark_processed("cam_04", captured.frame_id)
        published = buffer.mark_published("cam_04", captured.frame_id)

        self.assertEqual(processed.processed_at_ms, 2055)
        self.assertEqual(published.published_at_ms, 2062)
        self.assertEqual(published.ai_latency_ms, 55)
        self.assertEqual(published.publish_latency_ms, 62)


if __name__ == "__main__":
    unittest.main()

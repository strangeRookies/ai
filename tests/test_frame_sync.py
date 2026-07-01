import unittest

from ai.frame_sync import (
    CameraFrameQueue,
    FrameMetadataBuffer,
    FramePacket as SyncFramePacket,
    evidence_context_from_packet,
)
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

    def test_latest_queue_evidence_context_uses_processed_frame_after_drop(self):
        now_values = iter([1000, 1010, 1020, 1030, 1050, 1060])
        buffer = FrameMetadataBuffer(maxlen=10, now_ms=lambda: next(now_values))
        queue = CameraFrameQueue("cam_04", maxsize=2)

        for index in range(4):
            packet = FramePacket(frame_idx=index, fps=30.0, timestamp=index / 30.0, frame=None)
            metadata = buffer.record_capture("cam_04", packet, (360, 640, 3))
            queue.put_latest(
                SyncFramePacket(
                    camera_login_id="cam_04",
                    frame_id=metadata.frame_id,
                    captured_at_ms=metadata.captured_at_ms,
                    frame=None,
                    width=metadata.width,
                    height=metadata.height,
                    frame_idx=packet.frame_idx,
                    timestamp=packet.timestamp,
                    fps=packet.fps,
                )
            )

        processed_packet = queue.get_latest()
        self.assertIsNotNone(processed_packet)
        self.assertEqual(processed_packet.frame_id, 4)
        processed = buffer.mark_processed("cam_04", processed_packet.frame_id)
        published = buffer.mark_published("cam_04", processed_packet.frame_id)
        evidence = buffer.evidence_context("cam_04", processed_packet.frame_id, queue.dropped_frame_count)

        self.assertEqual(queue.dropped_frame_count, 3)
        self.assertEqual(evidence["cameraLoginId"], "cam_04")
        self.assertEqual(evidence["frameId"], 4)
        self.assertEqual(evidence["timestampMs"], 1030)
        self.assertEqual(evidence["capturedAtMs"], 1030)
        self.assertEqual(evidence["processedAtMs"], 1050)
        self.assertEqual(evidence["publishedAtMs"], 1060)
        self.assertEqual(evidence["aiLatencyMs"], 20)
        self.assertEqual(evidence["publishLatencyMs"], 30)
        self.assertEqual(evidence["droppedFrameCount"], 3)
        self.assertEqual(evidence["evidenceId"], "cam_04-4-1030")
        self.assertTrue(evidence["latencyOrderValid"])
        self.assertEqual(processed.frame_id, published.frame_id)

    def test_evidence_context_from_packet_survives_metadata_eviction(self):
        packet = SyncFramePacket(
            camera_login_id="cam_04",
            frame_id=7,
            captured_at_ms=2000,
            frame=None,
            width=640,
            height=360,
            frame_idx=20,
            timestamp=2.0,
            fps=10.0,
        )

        evidence = evidence_context_from_packet(
            packet,
            processed_at_ms=2030,
            published_at_ms=2045,
            dropped_frame_count=5,
        )

        self.assertEqual(evidence["cameraLoginId"], "cam_04")
        self.assertEqual(evidence["frameId"], 7)
        self.assertEqual(evidence["timestampMs"], 2000)
        self.assertEqual(evidence["evidenceId"], "cam_04-7-2000")
        self.assertEqual(evidence["droppedFrameCount"], 5)
        self.assertTrue(evidence["latencyOrderValid"])

    def test_evidence_context_flags_invalid_latency_order(self):
        packet = SyncFramePacket(
            camera_login_id="cam_04",
            frame_id=8,
            captured_at_ms=3000,
            frame=None,
            width=640,
            height=360,
            frame_idx=21,
            timestamp=2.1,
            fps=10.0,
        )

        evidence = evidence_context_from_packet(packet, processed_at_ms=3100, published_at_ms=3090)

        self.assertFalse(evidence["latencyOrderValid"])


if __name__ == "__main__":
    unittest.main()

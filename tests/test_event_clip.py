import queue
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from ai.events.clip_worker import ClipWriterWorker, enqueue_event_clip
from ai.events.event_clip import CircularFrameBuffer, EventClipBuffer, EventClipTask


def dummy_frame(value=0, width=64, height=48):
    return np.full((height, width, 3), value, dtype=np.uint8)


class EventClipTest(unittest.TestCase):
    def test_circular_buffer_keeps_latest_150_frames(self):
        buffer = CircularFrameBuffer(maxlen=150)
        for index in range(200):
            buffer.append(dummy_frame(index))

        snapshot = buffer.snapshot()
        self.assertEqual(len(buffer), 150)
        self.assertEqual(len(snapshot), 150)
        self.assertEqual(int(snapshot[0][0, 0, 0]), 50)
        self.assertEqual(int(snapshot[-1][0, 0, 0]), 199)

    def test_event_trigger_takes_pre_event_snapshot_copy(self):
        clip_buffer = EventClipBuffer(pre_event_frame_count=3, post_event_frame_count=2, cooldown_seconds=10)
        source_frames = [dummy_frame(index) for index in range(3)]
        for frame in source_frames:
            clip_buffer.add_frame(frame)

        self.assertTrue(clip_buffer.trigger_event("Fall", "cam_01", now=100.0))
        source_frames[-1][:] = 255

        pre_frames = clip_buffer._active_event["pre_frames"]
        self.assertEqual(len(pre_frames), 3)
        self.assertEqual(int(pre_frames[-1][0, 0, 0]), 2)

    def test_post_frames_create_task_and_enqueue(self):
        clip_buffer = EventClipBuffer(pre_event_frame_count=3, post_event_frame_count=2, cooldown_seconds=10)
        for index in range(3):
            clip_buffer.add_frame(dummy_frame(index))
        clip_buffer.trigger_event("Fall", "cam_01", now=100.0)

        self.assertIsNone(clip_buffer.add_frame(dummy_frame(3)))
        task = clip_buffer.add_frame(dummy_frame(4))
        self.assertIsNotNone(task)
        self.assertEqual(len(task.frames), 5)

        task_queue = queue.Queue(maxsize=1)
        self.assertTrue(enqueue_event_clip(task_queue, task))
        self.assertEqual(task_queue.qsize(), 1)

    def test_worker_creates_mp4_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_queue = queue.Queue(maxsize=2)
            worker = ClipWriterWorker(task_queue, uploader=lambda path, metadata: {"uploaded": False})
            worker.start()
            frames = [dummy_frame(index) for index in range(10)]
            task = EventClipTask(
                event_type="Fall",
                camera_id="cam_01",
                frames=frames,
                fps=10.0,
                output_dir=temp_dir,
                metadata={"test": True},
            )

            self.assertTrue(enqueue_event_clip(task_queue, task))
            task_queue.join()
            worker.stop()

            clips = list(Path(temp_dir).glob("Fall_cam_01_*.mp4"))
            self.assertEqual(len(clips), 1)
            self.assertGreater(clips[0].stat().st_size, 0)

    def test_worker_failure_does_not_stop_worker_thread(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_queue = queue.Queue(maxsize=2)
            worker = ClipWriterWorker(task_queue, uploader=lambda path, metadata: None)
            worker.start()
            bad_task = EventClipTask(
                event_type="Fall",
                camera_id="cam_01",
                frames=[{"not": "an image"}],
                fps=10.0,
                output_dir=temp_dir,
            )

            self.assertTrue(enqueue_event_clip(task_queue, bad_task))
            task_queue.join()
            time.sleep(0.1)
            self.assertTrue(worker._thread.is_alive())
            worker.stop()


if __name__ == "__main__":
    unittest.main()

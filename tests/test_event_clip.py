import queue
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from ai.events.clip_worker import ClipWriterWorker, enqueue_event_clip, save_clip_to_mp4
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

        pre_frames = clip_buffer._active_events["cam_01"][0]["pre_frames"]
        self.assertEqual(len(pre_frames), 3)
        self.assertEqual(int(pre_frames[-1][0, 0, 0]), 2)

    def test_post_frames_create_task_and_enqueue(self):
        clip_buffer = EventClipBuffer(pre_event_frame_count=3, post_event_frame_count=2, cooldown_seconds=10)
        for index in range(3):
            clip_buffer.add_frame(dummy_frame(index))
        clip_buffer.trigger_event("Fall", "cam_01", now=100.0)

        self.assertEqual(clip_buffer.add_frame(dummy_frame(3)), [])
        tasks = clip_buffer.add_frame(dummy_frame(4))
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(len(task.frames), 5)

        task_queue = queue.Queue(maxsize=1)
        self.assertTrue(enqueue_event_clip(task_queue, task))
        self.assertEqual(task_queue.qsize(), 1)

    def test_event_clip_task_preserves_evidence_metadata(self):
        clip_buffer = EventClipBuffer(pre_event_frame_count=1, post_event_frame_count=1, cooldown_seconds=10)
        clip_buffer.add_frame(dummy_frame(1))

        self.assertTrue(
            clip_buffer.trigger_event(
                "Fall",
                "cam_01",
                metadata={"evidenceId": "cam_01-7-2000", "traceId": "cam_01-7-2000"},
                now=100.0,
            )
        )
        tasks = clip_buffer.add_frame(dummy_frame(2))
        task = tasks[0]

        self.assertEqual(task.metadata["evidenceId"], "cam_01-7-2000")
        self.assertEqual(task.metadata["traceId"], "cam_01-7-2000")

    def test_clip_worker_prefers_event_id_not_timestamp(self):
        meta = {
            "eventId": "stable-evt-1",
            "evidenceId": "stable-evt-1",
            "event_timestamp": 1_700_000_000.0,
            "track_id": 4,
            "confidence": 0.88,
            "faint_prob": 0.88,
        }
        event_id = meta.get("eventId") or meta.get("event_id") or meta.get("evidenceId")
        self.assertEqual(event_id, "stable-evt-1")
        self.assertNotEqual(event_id, meta["event_timestamp"])

    def test_overlapping_events_on_same_camera_both_tracked_independently(self):
        clip_buffer = EventClipBuffer(pre_event_frame_count=2, post_event_frame_count=2, cooldown_seconds=10, max_concurrent_events=4)
        for index in range(2):
            clip_buffer.add_frame(dummy_frame(index))

        self.assertTrue(clip_buffer.trigger_event("Fall", "cam_01", metadata={"evidenceId": "evt-A"}, now=100.0))
        self.assertTrue(clip_buffer.trigger_event("Faint", "cam_01", metadata={"evidenceId": "evt-B"}, now=100.1))

        tasks = []
        for index in range(2):
            tasks.extend(clip_buffer.add_frame(dummy_frame(10 + index)))

        self.assertEqual(len(tasks), 2)
        ids = sorted(task.metadata["evidenceId"] for task in tasks)
        self.assertEqual(ids, ["evt-A", "evt-B"])

    def test_fifth_trigger_rejected_when_four_already_active(self):
        clip_buffer = EventClipBuffer(pre_event_frame_count=2, post_event_frame_count=2, cooldown_seconds=10, max_concurrent_events=4)
        for index in range(2):
            clip_buffer.add_frame(dummy_frame(index))

        event_types = ["Fall", "Faint", "Collapse", "Hazard"]
        for offset, event_type in enumerate(event_types):
            self.assertTrue(clip_buffer.trigger_event(event_type, "cam_01", metadata={"evidenceId": event_type}, now=100.0 + offset * 0.1))

        self.assertFalse(clip_buffer.trigger_event("Exit", "cam_01", metadata={"evidenceId": "5th"}, now=100.5))
        self.assertEqual(len(clip_buffer._active_events["cam_01"]), 4)

    def test_slot_frees_up_after_completion_for_new_trigger(self):
        clip_buffer = EventClipBuffer(pre_event_frame_count=1, post_event_frame_count=1, cooldown_seconds=10, max_concurrent_events=1)
        clip_buffer.add_frame(dummy_frame(0))
        self.assertTrue(clip_buffer.trigger_event("Fall", "cam_01", metadata={"evidenceId": "first"}, now=100.0))
        self.assertFalse(clip_buffer.trigger_event("Faint", "cam_01", metadata={"evidenceId": "blocked"}, now=100.1))

        tasks = clip_buffer.add_frame(dummy_frame(1))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(clip_buffer._active_events, {})

        self.assertTrue(clip_buffer.trigger_event("Faint", "cam_01", metadata={"evidenceId": "after-free"}, now=110.0))

    def test_different_cameras_are_independent(self):
        clip_buffer = EventClipBuffer(pre_event_frame_count=1, post_event_frame_count=1, cooldown_seconds=10, max_concurrent_events=4)
        clip_buffer.add_frame(dummy_frame(0))

        event_types = ["t1", "t2", "t3", "t4"]
        for offset, event_type in enumerate(event_types):
            self.assertTrue(clip_buffer.trigger_event(event_type, "cam_01", metadata={"evidenceId": event_type}, now=100.0 + offset * 0.1))

        # cam_01 is now full, but a different camera must be unaffected
        self.assertTrue(clip_buffer.trigger_event("Fall", "cam_02", metadata={"evidenceId": "cam2-1"}, now=100.2))
        # and cam_01 itself should still correctly reject its own 5th trigger
        self.assertFalse(clip_buffer.trigger_event("t5", "cam_01", metadata={"evidenceId": "cam1-5th"}, now=100.3))

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

    def test_worker_passes_evidence_metadata_and_safe_clip_name(self):
        uploads = []
        with tempfile.TemporaryDirectory() as temp_dir:
            task_queue = queue.Queue(maxsize=2)
            worker = ClipWriterWorker(task_queue, uploader=lambda path, metadata: uploads.append((path, metadata)))
            worker.start()
            task = EventClipTask(
                event_type="Fall",
                camera_id="cam_01",
                frames=[dummy_frame(index) for index in range(10)],
                fps=10.0,
                output_dir=temp_dir,
                metadata={"evidenceId": "cam_01-7-2000", "traceId": "cam_01-7-2000"},
            )

            self.assertTrue(enqueue_event_clip(task_queue, task))
            task_queue.join()
            worker.stop()

        self.assertEqual(uploads[0][1]["evidenceId"], "cam_01-7-2000")
        self.assertIn("evidence-cam_01-7-2000", uploads[0][0].name)

    def test_save_clip_to_mp4_keeps_legacy_glob_with_evidence_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = EventClipTask(
                event_type="Fall",
                camera_id="cam_01",
                frames=[dummy_frame(index) for index in range(10)],
                fps=10.0,
                output_dir=temp_dir,
                metadata={"evidenceId": "cam_01-7-2000"},
            )

            output_path = save_clip_to_mp4(task)

        self.assertTrue(output_path.name.startswith("Fall_cam_01_"))
        self.assertIn("evidence-cam_01-7-2000", output_path.name)

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

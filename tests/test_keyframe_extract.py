"""Real keyframe extraction tests (fixture MP4, no placeholder bytes)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ai.vlm.keyframe_extract import (
    JPEG_MAGIC,
    MAX_FRAMES,
    extract_keyframes_from_path,
    select_frame_indices,
)


def _make_fixture_mp4(path: Path, *, frames: int = 30, w: int = 64, h: int = 48, fps: float = 10.0) -> Path:
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "VideoWriter failed"
    for i in range(frames):
        # Distinct color per frame for determinism checks
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (i * 7 % 255, 40, 200 - (i * 3 % 100))
        cv2.putText(img, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        writer.write(img)
    writer.release()
    return path


class KeyframeExtractTest(unittest.TestCase):
    def test_uniform_indices_deterministic(self):
        a = select_frame_indices(30, max_frames=6)
        b = select_frame_indices(30, max_frames=6)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 6)
        self.assertEqual(a[0], 0)
        self.assertEqual(a[-1], 29)

    def test_extract_real_jpegs_from_fixture_mp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp4 = Path(tmp) / "sample.mp4"
            _make_fixture_mp4(mp4, frames=24)
            kfs = extract_keyframes_from_path(mp4, max_frames=MAX_FRAMES)
            self.assertGreaterEqual(len(kfs), 1)
            self.assertLessEqual(len(kfs), MAX_FRAMES)
            for kf in kfs:
                self.assertTrue(kf.jpeg_bytes.startswith(JPEG_MAGIC))
                self.assertGreater(len(kf.jpeg_bytes), 100)
                # OpenCV decode
                import cv2
                import numpy as np

                arr = np.frombuffer(kf.jpeg_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                self.assertIsNotNone(img)
                self.assertGreater(img.size, 0)

    def test_temp_cleanup_on_bytes_path(self):
        from ai.vlm.keyframe_extract import extract_keyframes_from_bytes

        with tempfile.TemporaryDirectory() as tmp:
            mp4 = Path(tmp) / "sample.mp4"
            _make_fixture_mp4(mp4, frames=12)
            data = mp4.read_bytes()
            before = set(Path(tempfile.gettempdir()).glob("vlm_clip_*"))
            kfs = extract_keyframes_from_bytes(data, max_frames=3)
            after = set(Path(tempfile.gettempdir()).glob("vlm_clip_*"))
            self.assertEqual(len(kfs), 3)
            # No leftover vlm_clip_ files from this call
            self.assertEqual(before, after)

    def test_no_one_byte_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp4 = Path(tmp) / "sample.mp4"
            _make_fixture_mp4(mp4, frames=10)
            kfs = extract_keyframes_from_path(mp4, max_frames=4)
            for kf in kfs:
                self.assertNotEqual(len(kf.jpeg_bytes), 1)


if __name__ == "__main__":
    unittest.main()

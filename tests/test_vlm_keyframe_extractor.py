from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ai.vlm.contracts import VlmContractError, validate_keyframes, validate_vlm_result
from ai.vlm.keyframe_extractor import KeyframeExtractionError, extract_eight_keyframes, local_video_source


def _write_video(path: Path, frame_count: int, *, fps: float = 8.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (48, 32))
    if not writer.isOpened():
        raise RuntimeError("test video writer unavailable")
    try:
        for index in range(frame_count):
            frame = np.zeros((32, 48, 3), dtype=np.uint8)
            frame[:, :] = ((index * 29) % 256, (index * 53) % 256, (index * 97) % 256)
            cv2.putText(frame, str(index), (3, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            writer.write(frame)
    finally:
        writer.release()


class KeyframeExtractorTest(unittest.TestCase):
    def test_extracts_exactly_eight_ordered_deterministic_unique_jpegs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "sample.avi"
            _write_video(video, 16)

            first = extract_eight_keyframes(video)
            second = extract_eight_keyframes(video)

        self.assertEqual([frame.frame_index for frame in first], [1, 3, 5, 7, 9, 11, 13, 15])
        self.assertEqual([frame.index for frame in first], list(range(8)))
        self.assertEqual([frame.sha256 for frame in first], [frame.sha256 for frame in second])
        self.assertEqual(len({frame.sha256 for frame in first}), 8)
        self.assertTrue(all(frame.jpeg_bytes.startswith(b"\xff\xd8") for frame in first))
        self.assertTrue(all(frame.width == 48 and frame.height == 32 for frame in first))
        self.assertEqual([frame.timestamp_sec for frame in first], sorted(frame.timestamp_sec for frame in first))
        validate_keyframes(first)

    def test_rejects_video_with_fewer_than_eight_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "short.avi"
            _write_video(video, 7)
            with self.assertRaisesRegex(KeyframeExtractionError, "too short"):
                extract_eight_keyframes(video)

    def test_rejects_corrupt_video_and_invalid_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "corrupt.avi"
            corrupt.write_bytes(b"not a video")
            with self.assertRaisesRegex(KeyframeExtractionError, "decode"):
                extract_eight_keyframes(corrupt)

            video = Path(tmp) / "sample.avi"
            _write_video(video, 16)
            with self.assertRaisesRegex(KeyframeExtractionError, "range"):
                extract_eight_keyframes(video, start_sec=2.0, end_sec=1.0)
            with self.assertRaisesRegex(KeyframeExtractionError, "outside"):
                extract_eight_keyframes(video, start_sec=3.0)

    def test_local_and_file_url_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "sample.avi"
            _write_video(video, 8)
            with local_video_source(str(video)) as local:
                self.assertEqual(local, video)
            with local_video_source(video.resolve().as_uri()) as local:
                self.assertEqual(local.resolve(), video.resolve())


class VlmResultContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.valid = {
            "schema_version": "vlm-result-v1",
            "visual_event_type": "person_lying_on_floor",
            "people_count": 1,
            "korean_search_keywords": ["바닥", "쓰러짐"],
            "detailed_description_ko": "한 사람이 바닥 가까이에 있는 장면입니다.",
            "uncertainty_notes": ["영상만으로 원인을 판단하지 않습니다."],
        }

    def test_accepts_exact_schema(self) -> None:
        validate_vlm_result(self.valid)

    def test_rejects_unknown_missing_and_invalid_numeric_fields(self) -> None:
        for mutation in (
            {**self.valid, "unknown": True},
            {key: value for key, value in self.valid.items() if key != "people_count"},
            {**self.valid, "people_count": -1},
            {**self.valid, "people_count": True},
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(VlmContractError):
                    validate_vlm_result(mutation)

    def test_rejects_invalid_keywords_and_forbidden_inference(self) -> None:
        for mutation in (
            {**self.valid, "korean_search_keywords": []},
            {**self.valid, "korean_search_keywords": ["바닥", "바닥"]},
            {**self.valid, "detailed_description_ko": "남성 환자가 뇌졸중으로 쓰러졌습니다."},
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(VlmContractError):
                    validate_vlm_result(mutation)


if __name__ == "__main__":
    unittest.main()

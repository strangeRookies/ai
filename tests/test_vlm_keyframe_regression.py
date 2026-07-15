from __future__ import annotations

import hashlib
import http.server
import os
import subprocess
import sys
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from ai.vlm.contracts import VlmContractError, validate_keyframes, validate_vlm_result
from ai.vlm.keyframe_extractor import (
    KEYFRAME_COUNT,
    KeyframeExtractionError,
    extract_eight_keyframes,
    local_video_source,
)
from ai.vlm_sdk import MockVlmProvider
from scripts.process_vlm import MetadataJson, ProcessVlmArgs, process

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "process_vlm.py"


def _write_video(path: Path, *, frame_count: int, identical: bool = False) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        8.0,
        (48, 32),
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    try:
        for index in range(frame_count):
            value = 64 if identical else index * 11
            frame = np.full((32, 48, 3), value, dtype=np.uint8)
            if not identical:
                cv2.putText(frame, str(index), (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255 - value,) * 3, 1)
            writer.write(frame)
    finally:
        writer.release()


class _QuietFileHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


class VlmKeyframeRegressionTest(unittest.TestCase):
    def test_extracts_exactly_eight_unique_deterministic_ordered_jpegs(self) -> None:
        with TemporaryDirectory() as directory:
            video = Path(directory) / "clip.avi"
            _write_video(video, frame_count=16)

            first = extract_eight_keyframes(video)
            second = extract_eight_keyframes(video)

        self.assertEqual(len(first), KEYFRAME_COUNT)
        self.assertEqual([frame.index for frame in first], list(range(KEYFRAME_COUNT)))
        self.assertEqual([frame.frame_index for frame in first], [1, 3, 5, 7, 9, 11, 13, 15])
        self.assertEqual([frame.sha256 for frame in first], [frame.sha256 for frame in second])
        self.assertEqual([frame.jpeg_bytes for frame in first], [frame.jpeg_bytes for frame in second])
        self.assertEqual(len({frame.sha256 for frame in first}), KEYFRAME_COUNT)
        self.assertTrue(all(left.timestamp_sec < right.timestamp_sec for left, right in zip(first, first[1:])))
        for frame in first:
            self.assertEqual((frame.width, frame.height), (48, 32))
            self.assertTrue(frame.jpeg_bytes.startswith(b"\xff\xd8"))
            self.assertTrue(frame.jpeg_bytes.endswith(b"\xff\xd9"))
            self.assertEqual(frame.sha256, hashlib.sha256(frame.jpeg_bytes).hexdigest())

    def test_rejects_short_corrupt_duplicate_and_invalid_range_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            short = root / "short.avi"
            duplicate = root / "duplicate.avi"
            corrupt = root / "corrupt.avi"
            _write_video(short, frame_count=7)
            _write_video(duplicate, frame_count=8, identical=True)
            corrupt.write_bytes(b"not a video")

            cases = (
                (short, {}, "too short"),
                (duplicate, {}, "duplicate"),
                (corrupt, {}, "decode"),
                (duplicate, {"start_sec": -0.1}, "range"),
                (duplicate, {"start_sec": 1.0, "end_sec": 0.5}, "range"),
                (duplicate, {"end_sec": 2.0}, "outside"),
            )
            for path, kwargs, expected in cases:
                with self.subTest(path=path.name, kwargs=kwargs):
                    with self.assertRaisesRegex(KeyframeExtractionError, expected):
                        extract_eight_keyframes(path, **kwargs)

    def test_file_and_http_sources_resolve_and_download_is_cleaned(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            _write_video(video, frame_count=8)

            with local_video_source(video.as_uri()) as resolved:
                self.assertEqual(resolved.resolve(), video.resolve())

            handler = lambda *args, **kwargs: _QuietFileHandler(*args, directory=directory, **kwargs)
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            downloaded: Path | None = None
            try:
                url = f"http://127.0.0.1:{server.server_port}/{video.name}"
                with local_video_source(url) as resolved:
                    downloaded = resolved
                    self.assertTrue(resolved.is_file())
                    self.assertEqual(len(extract_eight_keyframes(resolved)), KEYFRAME_COUNT)
                self.assertFalse(downloaded.exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_frame_contract_rejects_non_integer_index_and_digest_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            video = Path(directory) / "clip.avi"
            _write_video(video, frame_count=8)
            frames = extract_eight_keyframes(video)

        with self.assertRaisesRegex(VlmContractError, "integer"):
            validate_keyframes((replace(frames[0], frame_index=0.5), *frames[1:]))
        with self.assertRaisesRegex(VlmContractError, "SHA-256"):
            validate_keyframes((replace(frames[0], sha256="0" * 64), *frames[1:]))

    def test_result_contract_rejects_unknown_invalid_and_forbidden_values(self) -> None:
        valid = {
            "schema_version": "vlm-result-v1",
            "visual_event_type": "person_lying_on_floor",
            "people_count": 1,
            "korean_search_keywords": ["바닥", "쓰러짐"],
            "detailed_description_ko": "한 사람이 바닥 가까이에 있습니다.",
            "uncertainty_notes": ["영상에서 직접 관찰되는 내용만 기술했습니다."],
        }
        validate_vlm_result(valid)

        invalid_results = (
            ({**valid, "unknown": True}, "unknown"),
            ({**valid, "people_count": -1}, "nonnegative"),
            ({**valid, "people_count": True}, "integer"),
            ({**valid, "people_count": 1.5}, "integer"),
            ({**valid, "korean_search_keywords": []}, "nonempty"),
            ({**valid, "korean_search_keywords": ["바닥", "바닥"]}, "unique"),
            ({**valid, "detailed_description_ko": "남성으로 보이는 사람이 쓰러졌습니다."}, "forbidden"),
            ({**valid, "visual_event_type": "medical condition"}, "forbidden"),
        )
        for result, expected in invalid_results:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(VlmContractError, expected):
                    validate_vlm_result(result)

    def test_short_video_cli_has_no_success_stdout_and_sanitized_stderr(self) -> None:
        with TemporaryDirectory() as directory:
            video = Path(directory) / "short.avi"
            _write_video(video, frame_count=7)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-url",
                    str(video),
                    "--output-urls",
                    "unused",
                    "--metadata",
                    '{"incident":{"id":"inc-1"}}',
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "VLM_MOCK_MODE": "true"},
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "input video is too short for eight unique keyframes\n")

    def test_process_invokes_provider_exactly_once_with_eight_payloads(self) -> None:
        class CountingProvider:
            def __init__(self) -> None:
                self.calls = 0
                self.frame_count = 0

            def analyze(self, request):
                self.calls += 1
                self.frame_count = len(request.frames)
                return MockVlmProvider().analyze(request)

        with TemporaryDirectory() as directory:
            video = Path(directory) / "clip.avi"
            _write_video(video, frame_count=16)
            provider = CountingProvider()
            with patch("scripts.process_vlm.resolve_vlm_provider", return_value=provider):
                process(
                    ProcessVlmArgs(
                        input_url=str(video),
                        output_urls=("unused",),
                        metadata=MetadataJson('{"incident":{"id":"inc-1"}}'),
                        mock_mode=True,
                    )
                )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.frame_count, KEYFRAME_COUNT)


if __name__ == "__main__":
    unittest.main()

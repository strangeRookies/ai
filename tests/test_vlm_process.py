from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ai.vlm_sdk import MockVlmProvider
from scripts.process_vlm import MetadataJson, ProcessVlmArgs, parse_metadata, process


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "process_vlm.py"


class ProcessVlmCliTest(unittest.TestCase):
    @staticmethod
    def _write_video(path: Path) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            8.0,
            (32, 24),
        )
        if not writer.isOpened():
            raise RuntimeError("test video writer failed to open")
        try:
            for index in range(16):
                frame = np.full((24, 32, 3), index * 12, dtype=np.uint8)
                frame[:, index % 32, :] = 255 - index
                writer.write(frame)
        finally:
            writer.release()

    def test_nested_metadata_is_preserved(self) -> None:
        metadata = parse_metadata(
            MetadataJson(
                '{"scenario_type":"FALL_BED","incident":{"id":"inc-1","signals":[1,true,null]}}'
            )
        )

        self.assertEqual(
            metadata["incident"],
            {"id": "inc-1", "signals": [1, True, None]},
        )

    def test_process_passes_eight_ordered_jpegs_and_nested_metadata(self) -> None:
        class RecordingProvider:
            request = None

            def analyze(self, request):
                self.request = request
                return MockVlmProvider().analyze(request)

        provider = RecordingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "incident.avi"
            self._write_video(video)
            args = ProcessVlmArgs(
                input_url=video.as_uri(),
                output_urls=("http://dummy-put/0",),
                metadata=MetadataJson(
                    '{"scenario_type":"FALL_BED","incident":{"id":"inc-1",'
                    '"flags":[true],"access_token":"do-not-forward"}}'
                ),
                mock_mode=True,
            )
            with patch(
                "scripts.process_vlm.resolve_vlm_provider",
                return_value=provider,
            ):
                process(args)

        self.assertIsNotNone(provider.request)
        frames = provider.request.frames
        self.assertEqual(len(frames), 8)
        self.assertEqual([frame.index for frame in frames], list(range(8)))
        self.assertEqual(
            provider.request.metadata["incident"],
            {"id": "inc-1", "flags": [True], "access_token": "[REDACTED]"},
        )
        self.assertTrue(
            all(frame.jpeg_bytes.startswith(b"\xff\xd8") for frame in frames)
        )
        self.assertEqual(
            [frame.timestamp_sec for frame in frames],
            sorted(frame.timestamp_sec for frame in frames),
        )

    def test_stdout_is_one_json_line_for_local_clip(self) -> None:
        env = {**os.environ, "VLM_MOCK_MODE": "true"}
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "incident.avi"
            self._write_video(video)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-url",
                    str(video),
                    "--output-urls",
                    "http://dummy-put/0,http://dummy-put/1",
                    "--metadata",
                    '{"scenario_type":"FALL_BED","incident":{"id":"inc-1"}}',
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout.count("\n"), 1)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema_version"], "vlm-result-v1")
        self.assertEqual(payload["visual_event_type"], "person_lying_on_floor")
        self.assertIn("detailed_description_ko", payload)
    def test_invalid_metadata_exits_nonzero_with_stderr(self) -> None:
        env = {**os.environ, "VLM_MOCK_MODE": "true"}

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input-url",
                "https://dummy-url/video.mp4",
                "--output-urls",
                "http://dummy-put/0",
                "--metadata",
                "not-json",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("invalid metadata JSON", result.stderr)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
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
                    '{"incident_id":"inc-1","camera_login_id":"cam-01",'
                    '"clip_start_sec":0.5,"clip_end_sec":1.5,'
                    '"scenario_type":"FALL_BED","incident":{"id":"inc-1",'
                    '"flags":[true],"access_token":"do-not-forward",'
                    '"source_url":"https://user:password@example.test/clip"}}'
                ),
                mock_mode=True,
            )
            with patch(
                "scripts.process_vlm.resolve_vlm_provider",
                return_value=provider,
            ):
                analyzed = process(args)

        self.assertIsNotNone(provider.request)
        self.assertEqual(analyzed.incident_id, "inc-1")
        self.assertEqual(analyzed.frame_count, 8)
        self.assertEqual(analyzed.provider, "mock")
        self.assertIs(analyzed.is_mock, True)
        self.assertEqual(
            {field.name for field in fields(type(provider.request))},
            {"frames", "metadata"},
        )
        frames = provider.request.frames
        self.assertEqual(len(frames), 8)
        self.assertEqual([frame.index for frame in frames], list(range(8)))
        self.assertEqual(
            {field.name for field in fields(type(frames[0]))},
            {"index", "timestamp_sec", "jpeg_bytes"},
        )
        for forbidden in (
            "frame_index",
            "width",
            "height",
            "sha256",
            "source_url",
        ):
            self.assertFalse(hasattr(frames[0], forbidden))
        self.assertEqual(
            provider.request.metadata["incident"],
            {
                "id": "inc-1",
                "flags": [True],
                "access_token": "[REDACTED]",
                "source_url": "[REDACTED]",
            },
        )
        self.assertEqual(provider.request.metadata["camera_login_id"], "cam-01")
        self.assertEqual(provider.request.metadata["clip_start_sec"], 0.5)
        self.assertEqual(provider.request.metadata["clip_end_sec"], 1.5)
        serialized_metadata = json.dumps(provider.request.metadata)
        self.assertNotIn("do-not-forward", serialized_metadata)
        self.assertNotIn("user:password", serialized_metadata)
        self.assertTrue(
            all(frame.jpeg_bytes.startswith(b"\xff\xd8") for frame in frames)
        )
        self.assertEqual(
            [frame.timestamp_sec for frame in frames],
            sorted(frame.timestamp_sec for frame in frames),
        )
        self.assertGreaterEqual(frames[0].timestamp_sec, 0.5)
        self.assertLess(frames[-1].timestamp_sec, 1.5)

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
                    '{"scenario_type":"FALL_BED","incident_id":"inc-1",'
                    '"camera_login_id":"cam-01","clip_start_sec":0,'
                    '"clip_end_sec":2}',
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
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "incident_id",
                "visual_event_type",
                "people_count",
                "korean_search_keywords",
                "detailed_description_ko",
                "frame_count",
                "provider",
                "is_mock",
            },
        )
        self.assertEqual(payload["incident_id"], "inc-1")
        self.assertEqual(payload["frame_count"], 8)
        self.assertEqual(payload["provider"], "mock")
        self.assertIs(payload["is_mock"], True)
        self.assertNotIn("uncertainty_notes", payload)
        self.assertEqual(payload["visual_event_type"], "person_lying_on_floor")
        self.assertIn("detailed_description_ko", payload)

    def test_missing_or_invalid_clip_metadata_fails_without_stdout(self) -> None:
        env = {**os.environ, "VLM_MOCK_MODE": "true"}
        valid = {
            "incident_id": "inc-1",
            "camera_login_id": "cam-01",
            "clip_start_sec": 0.5,
            "clip_end_sec": 1.5,
            "nested": {"preserved": True},
        }
        cases = [
            (
                {key: value for key, value in valid.items() if key != "incident_id"},
                "incident_id",
            ),
            ({**valid, "camera_login_id": "  "}, "camera_login_id"),
            ({**valid, "clip_start_sec": True}, "clip_start_sec"),
            ({**valid, "clip_start_sec": -0.1}, "clip_start_sec"),
            ({**valid, "clip_start_sec": float("nan")}, "non-finite"),
            ({**valid, "clip_end_sec": 0.5}, "clip_end_sec"),
            ({**valid, "clip_end_sec": "1.5"}, "clip_end_sec"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "incident.avi"
            self._write_video(video)
            for metadata, expected_error in cases:
                with self.subTest(metadata=metadata):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "--input-url",
                            str(video),
                            "--output-urls",
                            "unused",
                            "--metadata",
                            json.dumps(metadata),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(expected_error, result.stderr)
                    self.assertNotIn(str(video), result.stderr)

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

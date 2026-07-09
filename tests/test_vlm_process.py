from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "process_vlm.py"


class ProcessVlmCliTest(unittest.TestCase):
    def test_stdout_is_json_only_when_mock_mode_enabled(self) -> None:
        env = {**os.environ, "VLM_MOCK_MODE": "true"}

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input-url",
                "https://dummy-url/video.mp4",
                "--output-urls",
                "http://dummy-put/0,http://dummy-put/1",
                "--metadata",
                '{"scenario_type":"FALL_BED"}',
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")
        payload = json.loads(result.stdout)
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

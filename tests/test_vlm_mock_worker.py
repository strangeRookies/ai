import json
import subprocess
import sys
import unittest
from pathlib import Path

from ai.vlm_mock import deterministic_embedding, job_to_json, parse_job


class VlmMockWorkerTest(unittest.TestCase):
    def test_worker_outputs_structured_json_without_gpu_or_api(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "process_vlm_mock.py"),
                "--job",
                str(root / "fixtures" / "vlm" / "demo_job.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(result.stdout)

        self.assertEqual(payload["schemaVersion"], "incident-v1")
        self.assertEqual(payload["incidentId"], "incident-demo-001")
        self.assertTrue(payload["recoveryObserved"])
        self.assertEqual(payload["movementAfterEvent"], "medium")
        self.assertEqual(len(payload["embedding"]), 768)
        self.assertIn("reading mock VLM job", result.stderr)

    def test_same_text_produces_same_768_dimension_embedding(self):
        first = deterministic_embedding("출입문 쓰러짐 recovered")
        second = deterministic_embedding("출입문 쓰러짐 recovered")

        self.assertEqual(len(first), 768)
        self.assertEqual(first, second)

    def test_job_to_json_reflects_faint_suspected_timeline(self):
        job = parse_job(
            {
                "jobId": "job-2",
                "incidentId": "incident-demo-002",
                "originalEventId": "orig-faint-suspected",
                "cameraLoginId": "cam_02",
                "locationName": "복도 B",
                "events": [
                    {
                        "eventId": "event-new-fall",
                        "eventType": "NEW_FALL",
                        "timestamp": "2026-07-10T12:42:00",
                    },
                    {
                        "eventId": "event-faint",
                        "eventType": "FAINT_SUSPECTED",
                        "timestamp": "2026-07-10T12:42:12",
                    },
                ],
            }
        )

        payload = json.loads(job_to_json(job))

        self.assertFalse(payload["recoveryObserved"])
        self.assertEqual(payload["movementAfterEvent"], "low")
        self.assertEqual(payload["estimatedLyingDurationSec"], 18.0)


if __name__ == "__main__":
    unittest.main()

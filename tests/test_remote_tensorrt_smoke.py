import unittest
from pathlib import Path


class RemoteTensorrtSmokeScriptTest(unittest.TestCase):
    def test_smoke_script_exists_and_covers_required_steps(self):
        script = Path("scripts/remote_tensorrt_smoke.sh")
        self.assertTrue(script.is_file(), f"missing {script}")
        text = script.read_text(encoding="utf-8")

        self.assertIn("set -euo pipefail", text)
        self.assertIn("PYTHONPATH=", text)
        self.assertIn("pytest tests/test_tensorrt_runtime.py", text)
        self.assertIn("preflight-only", text)
        self.assertIn("pytorch_fallback", text)
        self.assertIn("engine_validation", text)
        self.assertIn("tensorrt_or_fallback", text)
        self.assertIn("[tensorrt-smoke]", text)
        self.assertIn("SMOKE PASS", text)
        self.assertIn("run_rtsp_inference.py", text)


if __name__ == "__main__":
    unittest.main()

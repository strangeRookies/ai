import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.verify_pose_diagnostics import check_console_log


class VerifyPoseDiagnosticsTest(unittest.TestCase):
    def test_console_check_searches_registered_camera_overlay_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent_log = root / "ai_runner.log"
            overlay_dir = root / "runs" / "registered_cameras"
            overlay_dir.mkdir(parents=True)
            overlay_log = overlay_dir / "cam_04-overlay.log"
            parent_log.write_text("[registered-cameras] overlay started\n", encoding="utf-8")
            overlay_log.write_text(
                "\n".join(
                    [
                        "[pose-tracking-config] "
                        + json.dumps(
                            {
                                "stage": "pose_tracking_config",
                                "bytetrack_constructor_ignored": {},
                            }
                        ),
                        "[pose-diagnostics] "
                        + json.dumps(
                            {
                                "stage": "pose_frame",
                                "cameraLoginId": "cam_04",
                                "raw_detection_count": 1,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            output = StringIO()

            with redirect_stdout(output):
                result = check_console_log(
                    parent_log,
                    extra_log_globs=[str(root / "runs" / "registered_cameras" / "*-overlay.log")],
                )

        self.assertTrue(result)
        text = output.getvalue()
        self.assertIn("SUCCESS: [pose-tracking-config] log found", text)
        self.assertIn("Found 1 '[pose-diagnostics]' log lines", text)
        self.assertIn("cam_04-overlay.log", text)


if __name__ == "__main__":
    unittest.main()

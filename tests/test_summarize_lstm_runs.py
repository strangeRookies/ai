import json
import tempfile
import unittest
from pathlib import Path

from ai.action.summarize_lstm_runs import main


class SummarizeLstmRunsTest(unittest.TestCase):
    def test_writes_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "action_lstm" / "yolov8n"
            run_dir.mkdir(parents=True)
            (run_dir / "history.json").write_text(json.dumps([{"epoch": 1, "val_acc": 0.25}, {"epoch": 2, "val_acc": 0.75}]), encoding="utf-8")
            (run_dir / "best.pt").write_bytes(b"checkpoint")
            output = root / "summary.csv"

            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["summarize_lstm_runs", "--runs-dir", str(root / "runs" / "action_lstm"), "--output", str(output)]
                main()
            finally:
                sys.argv = old_argv

            text = output.read_text(encoding="utf-8")
            self.assertIn("yolov8n", text)
            self.assertIn("0.75", text)
            self.assertIn("True", text)


if __name__ == "__main__":
    unittest.main()

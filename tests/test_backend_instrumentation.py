"""TensorRT/backend instrumentation tests (mock / fake timer only)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai.inference.backend_instrumentation import (
    BackendMetricsCollector,
    run_mock_backend_benchmark,
    select_backend_with_logging,
)


class BackendInstrumentationTest(unittest.TestCase):
    def test_start_log_fields_mock(self):
        log = select_backend_with_logging(
            requested_backend="tensorrt",
            model_path="m.pt",
            engine_path="m.engine",
            device="cpu",
            precision="fp16",
            mock=True,
        )
        d = log.to_dict()
        for key in (
            "requested_backend",
            "actual_backend",
            "fallback_reason",
            "model_path",
            "engine_path",
            "device",
            "precision",
        ):
            self.assertIn(key, d)
        self.assertEqual(d["requested_backend"], "tensorrt")
        self.assertEqual(d["actual_backend"], "mock")
        self.assertIsNotNone(d["fallback_reason"])

    def test_periodic_metrics(self):
        c = BackendMetricsCollector(clock=lambda: 0.0)
        for ms in (10.0, 20.0, 30.0, 40.0):
            c.record_inference_ms(ms)
        c.record_frame_drop(2)
        m = c.periodic_metrics()
        self.assertEqual(m["processed_frame_count"], 4)
        self.assertEqual(m["frame_drop_count"], 2)
        self.assertAlmostEqual(m["avg_inference_ms"], 25.0)
        self.assertIsNotNone(m["p50_inference_ms"])
        self.assertIsNotNone(m["p95_inference_ms"])
        self.assertIsNotNone(m["fps"])

    def test_mock_benchmark_artifacts_no_trt_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_mock_backend_benchmark(
                output_root=tmp,
                iterations=5,
                fake_inference_ms=12.0,
                requested_backend="tensorrt",
            )
            out = Path(result["outputDir"])
            self.assertTrue((out / "manifest.json").is_file())
            self.assertTrue((out / "metrics.json").is_file())
            self.assertTrue((out / "report.md").is_file())
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            report = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("Not a TensorRT performance result", metrics["disclaimer"])
            self.assertIn("Do not interpret", report)
            self.assertNotIn("improved", report.lower())
            self.assertEqual(metrics["start"]["actual_backend"], "mock")
            self.assertEqual(metrics["periodic"]["processed_frame_count"], 5)


if __name__ == "__main__":
    unittest.main()

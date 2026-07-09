from __future__ import annotations

import unittest

from scripts.compare_tensorrt_candidate import BenchmarkArgs, adoption_recommendation, parse_args


class TensorRtDecisionTest(unittest.TestCase):
    def test_engine_export_defaults_to_fp32(self) -> None:
        args = parse_args(["--video", "sample_videos/sample.mp4", "--export-engine"])

        self.assertIsInstance(args, BenchmarkArgs)
        self.assertFalse(args.engine_half)

    def test_recommends_adoption_when_latency_improves_enough(self) -> None:
        result = adoption_recommendation(speedup=1.5, latency_delta_ms=12.0, torch_fps=8.0, tensorrt_fps=13.0)

        self.assertIn("ADOPT_CANDIDATE", result)

    def test_defers_when_speedup_is_too_small(self) -> None:
        result = adoption_recommendation(speedup=1.05, latency_delta_ms=3.0, torch_fps=12.0, tensorrt_fps=13.0)

        self.assertIn("DEFER", result)


if __name__ == "__main__":
    unittest.main()

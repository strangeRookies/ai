from __future__ import annotations

import unittest

from scripts.compare_tensorrt_candidate import (
    BackendResult,
    BenchmarkArgs,
    DetectionEquivalence,
    adoption_recommendation,
    compare_detection_equivalence,
    markdown_equivalence_report,
    non_comparable_recommendation,
    parse_args,
)


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

    def test_non_comparable_recommendation_includes_row_statuses(self) -> None:
        torch = BackendResult("torch", "model.pt", "FAILED: video file not found", 0, 0.0, 0.0, 0.0)
        tensorrt = BackendResult("tensorrt", "model.engine", "OK", 300, 3.0, 4.0, 100.0)

        result = non_comparable_recommendation(torch, tensorrt)

        self.assertIn("torch_status=FAILED: video file not found", result)
        self.assertIn("tensorrt_status=OK", result)

    def test_detection_equivalence_matches_boxes_by_iou_and_confidence(self) -> None:
        torch = [
            {"bbox": [0, 0, 100, 100], "confidence": 0.90, "keypoint_confidence": 0.80},
            {"bbox": [200, 200, 260, 260], "confidence": 0.70, "keypoint_confidence": 0.60},
        ]
        tensorrt = [
            {"bbox": [2, 2, 102, 102], "confidence": 0.88, "keypoint_confidence": 0.75},
            {"bbox": [400, 400, 460, 460], "confidence": 0.50, "keypoint_confidence": 0.40},
        ]

        result = compare_detection_equivalence(frame_index=7, torch_detections=torch, tensorrt_detections=tensorrt)

        self.assertIsInstance(result, DetectionEquivalence)
        self.assertEqual(result.frame_index, 7)
        self.assertEqual(result.torch_count, 2)
        self.assertEqual(result.tensorrt_count, 2)
        self.assertEqual(result.matched_count, 1)
        self.assertEqual(result.detection_count_diff, 0)
        self.assertAlmostEqual(result.avg_keypoint_confidence_diff, -0.05)
        self.assertGreater(result.avg_bbox_iou, 0.90)

    def test_markdown_equivalence_report_includes_event_decision_diff(self) -> None:
        result = DetectionEquivalence(
            frame_index=1,
            torch_count=2,
            tensorrt_count=3,
            matched_count=2,
            detection_count_diff=1,
            avg_bbox_iou=0.91,
            avg_keypoint_confidence_diff=-0.04,
            event_decision_diff=1,
        )

        report = markdown_equivalence_report([result])

        self.assertIn("detection_count_diff", report)
        self.assertIn("keypoint_confidence_diff", report)
        self.assertIn("event_decision_diff", report)


if __name__ == "__main__":
    unittest.main()

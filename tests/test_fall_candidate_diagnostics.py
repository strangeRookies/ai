import unittest

from benchmark.diagnose_fall_candidates import compare_yolo26_against_other_models, summarize_parsing_formats


class FallCandidateDiagnosticsTest(unittest.TestCase):
    def test_compare_yolo26_flags_other_model_candidate_but_yolo26_false(self):
        rows = [
            {
                "frame_idx": 10,
                "model_label": "YOLOv8s-pose",
                "person_idx": 0,
                "candidate": True,
                "reason": "torso_ratio_pass",
                "fall_rule": {"torso_ratio": 1.5},
            },
            {
                "frame_idx": 10,
                "model_label": "YOLO26n-pose",
                "person_idx": 0,
                "candidate": False,
                "reason": "torso_ratio_below_1.3",
                "fall_rule": {"torso_ratio": 0.9},
            },
        ]

        comparison = compare_yolo26_against_other_models(rows)

        self.assertEqual(comparison["other_models_candidate_frames"], [10])
        self.assertEqual(comparison["yolo26_people_frames"], [10])
        self.assertEqual(comparison["yolo26_candidate_frames"], [])
        self.assertEqual(comparison["yolo26_false_while_other_model_true_frames"], [10])
        self.assertEqual(comparison["yolo26_missing_while_other_model_true_frames"], [])

    def test_compare_yolo26_flags_missing_detection_when_others_candidate(self):
        rows = [
            {
                "frame_idx": 20,
                "model_label": "YOLOv11n-pose",
                "person_idx": 0,
                "candidate": True,
                "reason": "torso_ratio_pass",
                "fall_rule": {"torso_ratio": 1.7},
            },
            {
                "frame_idx": 20,
                "model_label": "YOLO26n-pose",
                "person_idx": None,
                "candidate": False,
                "reason": "no_person_result",
                "fall_rule": {"torso_ratio": None},
            },
        ]

        comparison = compare_yolo26_against_other_models(rows)

        self.assertEqual(comparison["yolo26_missing_while_other_model_true_frames"], [20])

    def test_summarize_parsing_formats_flags_normalized_xyxy(self):
        rows = [
            {
                "model_label": "YOLO26n-pose",
                "format_report": {
                    "boxes_xyxy_minmax": [0.1, 0.9],
                    "boxes_xyxyn_minmax": [0.1, 0.9],
                    "keypoints_xy_minmax": [20.0, 300.0],
                    "keypoints_xyn_minmax": [0.1, 0.8],
                    "keypoints_conf_shape": [1, 17],
                },
            }
        ]

        summary = summarize_parsing_formats(rows)

        self.assertEqual(summary["YOLO26n-pose"]["xyxy_normalized_like_rows"], 1)
        self.assertIn("boxes.xyxy looks normalized; benchmark expects pixel xyxy", summary["YOLO26n-pose"]["warnings"])

    def test_summarize_parsing_formats_accepts_pixel_xyxy_and_keypoint_shape(self):
        rows = [
            {
                "model_label": "YOLOv11n-pose",
                "format_report": {
                    "boxes_xyxy_minmax": [10.0, 640.0],
                    "boxes_xyxyn_minmax": [0.01, 0.95],
                    "keypoints_xy_minmax": [20.0, 400.0],
                    "keypoints_xyn_minmax": [0.02, 0.9],
                    "keypoints_conf_shape": [2, 17],
                },
            }
        ]

        summary = summarize_parsing_formats(rows)

        self.assertEqual(summary["YOLOv11n-pose"]["xyxy_pixel_like_rows"], 1)
        self.assertEqual(summary["YOLOv11n-pose"]["warnings"], [])


if __name__ == "__main__":
    unittest.main()

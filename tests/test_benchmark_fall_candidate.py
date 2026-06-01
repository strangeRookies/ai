import unittest

from benchmark.benchmark_models import evaluate_fall_candidate, is_fall_candidate


class BenchmarkFallCandidateTest(unittest.TestCase):
    def test_fall_candidate_reports_torso_ratio_details(self):
        keypoints = [[0.0, 0.0] for _ in range(17)]
        conf = [1.0 for _ in range(17)]
        keypoints[5] = [10.0, 10.0]
        keypoints[6] = [20.0, 10.0]
        keypoints[11] = [80.0, 20.0]
        keypoints[12] = [90.0, 20.0]

        details = evaluate_fall_candidate(keypoints, conf, 0.3)

        self.assertTrue(details["candidate"])
        self.assertEqual(details["reason"], "torso_ratio_pass")
        self.assertGreater(details["torso_ratio"], 1.3)
        self.assertLess(details["torso_angle_degrees"], 45.0)
        self.assertFalse(details["uses_bbox_ratio"])
        self.assertFalse(details["uses_center_height"])
        self.assertFalse(details["uses_torso_angle"])
        self.assertTrue(details["uses_torso_ratio"])
        self.assertTrue(details["uses_confidence_threshold"])
        self.assertTrue(is_fall_candidate(keypoints, conf, 0.3))

    def test_fall_candidate_reports_low_required_confidence(self):
        keypoints = [[0.0, 0.0] for _ in range(17)]
        conf = [1.0 for _ in range(17)]
        conf[5] = 0.1

        details = evaluate_fall_candidate(keypoints, conf, 0.3)

        self.assertFalse(details["candidate"])
        self.assertEqual(details["reason"], "required_keypoint_below_threshold_or_missing")
        self.assertIn(5, details["missing_required"])

    def test_vertical_pose_is_not_candidate(self):
        keypoints = [[0.0, 0.0] for _ in range(17)]
        conf = [1.0 for _ in range(17)]
        keypoints[5] = [10.0, 10.0]
        keypoints[6] = [20.0, 10.0]
        keypoints[11] = [12.0, 90.0]
        keypoints[12] = [22.0, 90.0]

        details = evaluate_fall_candidate(keypoints, conf, 0.3)

        self.assertFalse(details["candidate"])
        self.assertEqual(details["reason"], "torso_ratio_below_1.3")


if __name__ == "__main__":
    unittest.main()

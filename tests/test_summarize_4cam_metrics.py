import unittest

from scripts.summarize_4cam_metrics import decision_hint, format_value, select_latest_valid_per_camera, value_for


class Summarize4CamMetricsTest(unittest.TestCase):
    def test_value_for_uses_per_frame_metric_aliases(self):
        row = {"bbox_per_frame": 1.25, "keypoints_per_frame": 0.75}

        self.assertEqual(value_for(row, "bbox/frame"), 1.25)
        self.assertEqual(value_for(row, "keypoints/frame"), 0.75)

    def test_format_value_reports_null_for_missing_gpu_metric(self):
        self.assertEqual(format_value(None), "null")
        self.assertEqual(format_value(1.23456), "1.235")

    def test_decision_hint_prioritizes_rtsp_read_latency(self):
        rows = [{"avg_frame_read_ms": 120.0, "effective_fps": 20.0, "avg_yolo_inference_ms": 10.0}]

        self.assertIn("GStreamer", decision_hint(rows, target_fps=10.0))

    def test_decision_hint_reports_tensorrt_when_fps_is_low(self):
        rows = [{"avg_frame_read_ms": 10.0, "effective_fps": 4.0, "avg_yolo_inference_ms": 10.0}]

        self.assertIn("TensorRT", decision_hint(rows, target_fps=10.0))

    def test_decision_hint_defers_acceleration_when_healthy(self):
        rows = [{"avg_frame_read_ms": 10.0, "effective_fps": 15.0, "avg_yolo_inference_ms": 20.0}]

        self.assertIn("defer", decision_hint(rows, target_fps=10.0))

    def test_select_latest_valid_per_camera_ignores_zero_frame_rows(self):
        rows = [
            {"camera_id": "camera-1", "frames_processed": 0, "runtime_seconds": 0.006, "_source_mtime": 3},
            {"camera_id": "camera-1", "frames_processed": 300, "runtime_seconds": 10.0, "_source_mtime": 1},
            {"camera_id": "camera-1", "frames_processed": 3000, "runtime_seconds": 100.0, "_source_mtime": 2},
        ]

        selected = select_latest_valid_per_camera(rows)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["frames_processed"], 3000)


if __name__ == "__main__":
    unittest.main()

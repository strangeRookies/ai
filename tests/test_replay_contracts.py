import json
import tempfile
import unittest
from pathlib import Path

from scripts.replay_tracking_from_cache import (
    TrackerConfig,
    classify_iou_failure,
    load_cache,
    run_tracker_config,
    validate_cache,
)


def _detection(bbox, confidence=0.8, keypoints=None):
    return {
        "detection_index": 0,
        "bbox_xyxy": bbox,
        "confidence": confidence,
        "class_id": 0,
        "keypoints": keypoints or [],
        "avg_keypoint_conf": 0.8,
        "valid_keypoints": 0,
    }


class ReplayContractTest(unittest.TestCase):
    def test_cache_schema_and_replay_are_deterministic(self):
        frames = [
            {"frame_id": 0, "timestamp_ms": 0, "detections": [_detection([10, 10, 50, 110])]},
            {"frame_id": 1, "timestamp_ms": 33, "detections": [_detection([12, 10, 52, 110], 0.1)]},
            {"frame_id": 2, "timestamp_ms": 66, "detections": []},
            {"frame_id": 3, "timestamp_ms": 99, "detections": [_detection([0, 10, 80, 70])]},
            {"frame_id": 4, "timestamp_ms": 132, "detections": [_detection([300, 10, 340, 110]), _detection([600, 10, 640, 110])]},
        ]
        meta = {"_type": "meta", "frames_written": len(frames), "source_fps": 30.0}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "synthetic.jsonl"
            cache.write_text("\n".join(json.dumps(row) for row in [meta, *frames]) + "\n", encoding="utf-8")
            loaded_meta, loaded_frames = load_cache(cache)
            self.assertEqual(validate_cache(loaded_meta, loaded_frames)["frames"], 5)
            first = run_tracker_config(loaded_frames, TrackerConfig(name="fixture"), Path(directory) / "one", 30.0)
            second = run_tracker_config(loaded_frames, TrackerConfig(name="fixture"), Path(directory) / "two", 30.0)
        self.assertEqual(first["total_new_tracks"], second["total_new_tracks"])

    def test_iou_failure_reasons_cover_explicit_causes(self):
        self.assertEqual(classify_iou_failure({"confidence": 0.1}, 0.0, [0, 0, 10, 10], [20, 0, 30, 10], None), "LOW_CONFIDENCE_DETECTION")
        self.assertEqual(classify_iou_failure({"confidence": 0.9}, 0.5, [0, 0, 10, 10], [20, 0, 30, 10], None), "FRAME_GAP")
        self.assertEqual(classify_iou_failure({"confidence": 0.9}, 0.0, [0, 0, 10, 100], [0, 0, 100, 10], None), "SCREEN_BOUNDARY")
        self.assertEqual(classify_iou_failure({"confidence": 0.9}, 0.0, [100, 100, 110, 200], [100, 100, 200, 110], None), "BBOX_ASPECT_RATIO_CHANGE")


if __name__ == "__main__":
    unittest.main()

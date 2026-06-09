import unittest

import numpy as np

from ai.postprocess.supervision_postprocessor import (
    SupervisionPostProcessor,
    match_keypoints_by_iou,
)


class SupervisionPostProcessorTest(unittest.TestCase):
    def test_match_keypoints_by_iou_preserves_yolo_keypoints(self):
        source = [
            {
                "bbox": [0, 0, 10, 10],
                "keypoints": [{"x": 1, "y": 2, "confidence": 0.9}],
            }
        ]
        tracked = [{"bbox": [1, 1, 11, 11], "track_id": 3}]

        matched = match_keypoints_by_iou(tracked, source, min_iou=0.5)

        self.assertEqual(matched[0]["track_id"], 3)
        self.assertEqual(matched[0]["keypoints"], source[0]["keypoints"])

    def test_postprocessor_uses_injected_bytetrack_adapter(self):
        adapter = FakeByteTrackAdapter(track_ids=[11])
        processor = SupervisionPostProcessor(byte_tracker=adapter)
        detections = [
            {
                "bbox": [0, 0, 10, 10],
                "confidence": 0.7,
                "keypoints": [{"x": 1, "y": 2, "confidence": 0.9}],
            }
        ]

        output = processor.process(detections, np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual(output[0]["track_id"], 11)
        self.assertEqual(output[0]["keypoints"], detections[0]["keypoints"])
        self.assertEqual(processor.diagnostics()["active_tracks"], 1)


class FakeByteTrackAdapter:
    def __init__(self, track_ids):
        self.track_ids = track_ids

    def update(self, detections):
        output = []
        for index, detection in enumerate(detections):
            item = dict(detection)
            item["track_id"] = self.track_ids[index]
            output.append(item)
        return output

    def diagnostics(self):
        return {"active_tracks": len(self.track_ids), "tracks": {}}


if __name__ == "__main__":
    unittest.main()

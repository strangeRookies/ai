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

    def test_camera_tracker_isolation(self):
        # Create SupervisionByteTrackAdapter and inject FakeRoboflowByteTrack toCam 1 & Cam 2
        from ai.postprocess.supervision_postprocessor import SupervisionByteTrackAdapter
        adapter1 = SupervisionByteTrackAdapter()
        adapter1._tracker = FakeRoboflowByteTrack(track_ids=[1])
        adapter2 = SupervisionByteTrackAdapter()
        adapter2._tracker = FakeRoboflowByteTrack(track_ids=[1])

        proc1 = SupervisionPostProcessor(byte_tracker=adapter1)
        proc2 = SupervisionPostProcessor(byte_tracker=adapter2)

        dets1 = [{"bbox": [10, 10, 50, 50], "confidence": 0.9}]
        dets2 = [{"bbox": [100, 100, 150, 150], "confidence": 0.9}]

        out1 = proc1.process(dets1, np.zeros((200, 200, 3), dtype=np.uint8))
        out2 = proc2.process(dets2, np.zeros((200, 200, 3), dtype=np.uint8))

        # Both should get ID 1 independently
        self.assertEqual(out1[0]["track_id"], 1)
        self.assertEqual(out2[0]["track_id"], 1)

        # Update Cam 1
        dets1_next = [{"bbox": [12, 12, 52, 52], "confidence": 0.9}]
        out1_next = proc1.process(dets1_next, np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(out1_next[0]["track_id"], 1)

        self.assertEqual(proc1.diagnostics()["active_tracks"], 1)
        self.assertEqual(proc2.diagnostics()["active_tracks"], 1)

    def test_consecutive_detections_maintain_track_id(self):
        from ai.postprocess.supervision_postprocessor import SupervisionByteTrackAdapter
        adapter = SupervisionByteTrackAdapter()
        adapter._tracker = FakeRoboflowByteTrack(track_ids=[3])
        proc = SupervisionPostProcessor(byte_tracker=adapter)

        dets = [{"bbox": [20, 20, 60, 60], "confidence": 0.85}]
        out = proc.process(dets, np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(out[0]["track_id"], 3)

    def test_reordered_bytetrack_output_maps_ids_by_bbox_not_index(self):
        from ai.postprocess.supervision_postprocessor import SupervisionByteTrackAdapter
        adapter = SupervisionByteTrackAdapter()
        adapter._tracker = FakeReorderedRoboflowByteTrack()
        proc = SupervisionPostProcessor(byte_tracker=adapter)
        detections = [
            {
                "bbox": [10, 10, 40, 40],
                "confidence": 0.91,
                "class_name": "person",
                "keypoints": [{"x": 12, "y": 13, "confidence": 0.8}],
            },
            {
                "bbox": [100, 100, 150, 160],
                "confidence": 0.82,
                "class_name": "person",
                "keypoints": [{"x": 120, "y": 130, "confidence": 0.7}],
            },
        ]

        output = proc.process(detections, np.zeros((200, 200, 3), dtype=np.uint8))

        self.assertEqual(output[0]["track_id"], 11)
        self.assertEqual(output[0]["bbox"], detections[0]["bbox"])
        self.assertEqual(output[0]["keypoints"], detections[0]["keypoints"])
        self.assertEqual(output[0]["confidence"], 0.91)
        self.assertEqual(output[1]["track_id"], 22)
        self.assertEqual(output[1]["bbox"], detections[1]["bbox"])
        self.assertEqual(output[1]["keypoints"], detections[1]["keypoints"])
        self.assertEqual(output[1]["confidence"], 0.82)

    def test_bbox_smoothing_filters_out_noise(self):
        # Configure postprocessor config with 0.6 smoothing alpha
        from ai.postprocess.supervision_postprocessor import SupervisionPostProcessorConfig, SupervisionByteTrackAdapter
        config = SupervisionPostProcessorConfig(
            track_thresh=0.10,
            track_buffer=30,
            match_thresh=0.20,
            frame_rate=30,
            bbox_smoothing_alpha=0.60
        )
        adapter = SupervisionByteTrackAdapter(
            track_thresh=config.track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh,
            frame_rate=config.frame_rate,
            bbox_smoothing_alpha=config.bbox_smoothing_alpha
        )
        adapter._tracker = FakeRoboflowByteTrack(track_ids=[7])
        proc = SupervisionPostProcessor(config=config, byte_tracker=adapter)

        # Frame 1
        dets = [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.90}]
        out = proc.process(dets, np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(out[0]["bbox"], [10.0, 10.0, 50.0, 50.0])

        # Frame 2 (Noisy jump to [20, 20, 60, 60])
        # Smoothed box = alpha * current + (1 - alpha) * previous
        # x1 = 0.6 * 20.0 + 0.4 * 10.0 = 16.0
        dets_next = [{"bbox": [20.0, 20.0, 60.0, 60.0], "confidence": 0.90}]
        out_next = proc.process(dets_next, np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(out_next[0]["bbox"], [16.0, 16.0, 56.0, 56.0])

    def test_default_supervision_does_not_invent_track_id_when_bytetrack_omits_it(self):
        from ai.postprocess.supervision_postprocessor import SupervisionByteTrackAdapter
        adapter = SupervisionByteTrackAdapter()
        adapter._tracker = FakeNoTrackRoboflowByteTrack()
        proc = SupervisionPostProcessor(byte_tracker=adapter)

        output = proc.process(
            [{"bbox": [20, 20, 60, 60], "confidence": 0.85}],
            np.zeros((200, 200, 3), dtype=np.uint8),
        )

        self.assertIsNone(output[0].get("track_id"))
        self.assertFalse(proc.diagnostics()["stability_fallback"])

    def test_stability_fallback_assigns_stable_track_id_when_bytetrack_omits_it(self):
        from ai.postprocess.supervision_postprocessor import SupervisionByteTrackAdapter
        adapter = SupervisionByteTrackAdapter(
            track_thresh=0.10,
            track_buffer=30,
            match_thresh=0.20,
            bbox_smoothing_alpha=0.60,
            stability_fallback=True,
        )
        adapter._tracker = FakeNoTrackRoboflowByteTrack()
        proc = SupervisionPostProcessor(byte_tracker=adapter)

        first = proc.process(
            [{"bbox": [20, 20, 60, 60], "confidence": 0.85}],
            np.zeros((200, 200, 3), dtype=np.uint8),
        )
        second = proc.process(
            [{"bbox": [22, 22, 62, 62], "confidence": 0.84}],
            np.zeros((200, 200, 3), dtype=np.uint8),
        )

        self.assertIsNotNone(first[0].get("track_id"))
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])
        self.assertEqual(proc.diagnostics()["active_tracks"], 1)
        self.assertTrue(proc.diagnostics()["stability_fallback"])


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


class FakeRoboflowByteTrack:
    def __init__(self, track_ids):
        self.track_ids = track_ids

    def update_with_detections(self, detections):
        import supervision as sv
        tids = []
        for i in range(len(detections.xyxy)):
            tids.append(self.track_ids[i] if i < len(self.track_ids) else 1)
        return sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=np.array(tids, dtype=int)
        )


class FakeReorderedRoboflowByteTrack:
    def update_with_detections(self, detections):
        import supervision as sv
        return sv.Detections(
            xyxy=np.asarray([detections.xyxy[1], detections.xyxy[0]], dtype=np.float32),
            confidence=np.asarray([detections.confidence[1], detections.confidence[0]], dtype=np.float32),
            class_id=np.asarray([detections.class_id[1], detections.class_id[0]], dtype=int),
            tracker_id=np.asarray([22, 11], dtype=int),
        )


class FakeNoTrackRoboflowByteTrack:
    def update_with_detections(self, detections):
        import supervision as sv
        return sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=np.asarray([None for _ in range(len(detections.xyxy))], dtype=object),
        )


if __name__ == "__main__":
    unittest.main()

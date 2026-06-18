class KeypointSequenceBuffer:
    """Build overlapping keypoint sequences without changing input FPS.

    sequence_length is the number of frames in one emitted sequence. stride is
    the next sequence start interval in frames, not FPS sampling. For example,
    8/4 emits 8-frame sequences and permits the next sequence 4 frames later.

    Keypoint detections are preserved as supplied by the detector. Downstream
    LSTM feature conversion represents them as (sequence_length, 51), where
    51 = 17 keypoints x (x, y, confidence).
    TODO: evaluate bbox-relative normalized keypoints + confidence as a future
    model-improvement candidate.
    """

    def __init__(self, sequence_length=16, stride=8):
        self.sequence_length = sequence_length
        self.stride = stride
        self._frames = []
        self._last_emit_frame = -1

    def add(self, frame_idx, detections, frame_shape=None):
        detection = best_detection_with_keypoints(detections)
        if detection is None:
            return None
        self._frames.append({"frame_idx": int(frame_idx), "detection": detection, "frame_shape": frame_shape})
        self._frames = self._frames[-self.sequence_length :]
        if len(self._frames) < self.sequence_length:
            return None
        if self._last_emit_frame >= 0 and frame_idx - self._last_emit_frame < self.stride:
            return None
        self._last_emit_frame = int(frame_idx)
        return {
            "start_frame": self._frames[0]["frame_idx"],
            "end_frame": self._frames[-1]["frame_idx"],
            "detections": [item["detection"] for item in self._frames],
            "frame_shapes": [item["frame_shape"] for item in self._frames],
            "bbox": self._frames[-1]["detection"].get("bbox"),
            "keypoints": self._frames[-1]["detection"].get("keypoints"),
            "track_id": self._frames[-1]["detection"].get("track_id"),
        }


def best_detection_with_keypoints(detections):
    candidates = [item for item in detections if item.get("keypoints")]
    if not candidates:
        return None
    return max(candidates, key=lambda item: _bbox_area(item.get("bbox")))


def _bbox_area(bbox):
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(float(bbox[2]) - float(bbox[0]), 0.0) * max(float(bbox[3]) - float(bbox[1]), 0.0)

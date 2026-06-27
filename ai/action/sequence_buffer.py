class CropSequenceBuffer:
    """Build overlapping crop sequences without changing input FPS.

    sequence_length is the number of frames in one emitted sequence. stride is
    the next sequence start interval in frames, not FPS sampling. For example,
    30/15 emits 30-frame sequences and permits the next sequence 15 frames later.
    """

    def __init__(self, sequence_length=30, stride=15, resize_size=224):
        self.sequence_length = sequence_length
        self.stride = stride
        self.resize_size = resize_size
        self._crops = []
        self._last_emit_frame = -1

    def add(self, frame_idx, frame, boxes, frame_id=None, captured_at_ms=None):
        box = largest_box(boxes)
        if box is None:
            return None
        crop = crop_person(frame, box, self.resize_size)
        self._crops.append(
            {
                "frame_idx": frame_idx,
                "frame_id": int(frame_id) if frame_id is not None else int(frame_idx),
                "captured_at_ms": int(captured_at_ms) if captured_at_ms is not None else None,
                "crop": crop,
                "box": box,
            }
        )
        self._crops = self._crops[-self.sequence_length :]
        if len(self._crops) < self.sequence_length:
            return None
        if self._last_emit_frame >= 0 and frame_idx - self._last_emit_frame < self.stride:
            return None
        self._last_emit_frame = frame_idx
        return {
            "start_frame": self._crops[0]["frame_idx"],
            "end_frame": self._crops[-1]["frame_idx"],
            "sequence_start_frame_id": self._crops[0]["frame_id"],
            "sequence_end_frame_id": self._crops[-1]["frame_id"],
            "sequence_start_at_ms": self._crops[0]["captured_at_ms"],
            "sequence_end_at_ms": self._crops[-1]["captured_at_ms"],
            "crops": [item["crop"] for item in self._crops],
            "box": self._crops[-1]["box"],
        }


def largest_box(boxes):
    if not boxes:
        return None
    return max(boxes, key=lambda b: max(b["x2"] - b["x1"], 0) * max(b["y2"] - b["y1"], 0))


def crop_person(frame, box, resize_size):
    try:
        import cv2
    except ImportError:
        return frame
    h, w = frame.shape[:2]
    x1 = max(0, min(int(box["x1"]), w - 1))
    y1 = max(0, min(int(box["y1"]), h - 1))
    x2 = max(x1 + 1, min(int(box["x2"]), w))
    y2 = max(y1 + 1, min(int(box["y2"]), h))
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (resize_size, resize_size))

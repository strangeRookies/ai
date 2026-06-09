import time
from collections import defaultdict, deque


class PerTrackSequenceBuffer:
    def __init__(self, sequence_length=30, max_track_age_seconds=5.0):
        self.sequence_length = max(1, int(sequence_length))
        self.max_track_age_seconds = float(max_track_age_seconds)
        self._buffers = defaultdict(lambda: deque(maxlen=self.sequence_length))
        self._last_seen_at = {}

    def update(self, detections, now=None):
        now = time.time() if now is None else float(now)
        self._drop_stale_tracks(now)
        output = []
        for detection in detections:
            detection = dict(detection)
            track_id = detection.get("track_id")
            if track_id is None:
                output.append(detection)
                continue
            track_id = int(track_id)
            self._last_seen_at[track_id] = now
            keypoint_confidence = detection.get("keypoint_confidence")
            missing_keypoints = keypoint_confidence is None or float(keypoint_confidence) <= 0
            self._buffers[track_id].append(
                {
                    "bbox": detection.get("bbox"),
                    "keypoints": detection.get("keypoints"),
                    "confidence": detection.get("confidence"),
                    "keypoint_confidence": keypoint_confidence,
                    "missing_keypoints": missing_keypoints,
                    "timestamp": now,
                }
            )
            buffer = self._buffers[track_id]
            detection["sequence_length"] = len(buffer)
            detection["sequence_ready"] = len(buffer) >= self.sequence_length
            detection["keypoint_missing_rate"] = round(
                sum(1 for item in buffer if item["missing_keypoints"]) / len(buffer),
                4,
            )
            output.append(detection)
        return output

    def get_sequence(self, track_id):
        return list(self._buffers.get(int(track_id), []))

    def _drop_stale_tracks(self, now):
        stale_track_ids = [
            track_id
            for track_id, last_seen_at in self._last_seen_at.items()
            if now - last_seen_at > self.max_track_age_seconds
        ]
        for track_id in stale_track_ids:
            self._last_seen_at.pop(track_id, None)
            self._buffers.pop(track_id, None)

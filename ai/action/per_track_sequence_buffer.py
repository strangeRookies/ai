import time

from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.sequence_buffer import CropSequenceBuffer


class PerTrackKeypointSequenceBuffers:
    """Maintain one KeypointSequenceBuffer per track_id.

    sequence_length is frames per sequence. stride is the next sequence start
    interval in frames, not frame sampling. Class defaults are only fallbacks;
    runtime scripts may pass different values, such as 8/4.
    """

    def __init__(self, sequence_length=8, stride=4, max_track_age_seconds=5.0):
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.max_track_age_seconds = float(max_track_age_seconds)
        self._buffers = {}
        self._last_seen_at = {}
        self.sequences_generated_by_track = {}

    def add(self, frame_idx, detections, frame_shape=None, now=None):
        now = time.time() if now is None else float(now)
        self._drop_stale_tracks(now)
        sequences = []
        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None or not detection.get("keypoints"):
                continue
            track_id = int(track_id)
            self._last_seen_at[track_id] = now
            buffer = self._buffers.setdefault(track_id, KeypointSequenceBuffer(self.sequence_length, self.stride))
            sequence = buffer.add(frame_idx, [detection], frame_shape)
            if sequence:
                sequence["track_id"] = track_id
                self.sequences_generated_by_track[track_id] = self.sequences_generated_by_track.get(track_id, 0) + 1
                sequences.append(sequence)
        return sequences

    def active_track_ids(self):
        return sorted(self._buffers.keys())

    def _drop_stale_tracks(self, now):
        stale_track_ids = [
            track_id
            for track_id, last_seen_at in self._last_seen_at.items()
            if now - last_seen_at > self.max_track_age_seconds
        ]
        for track_id in stale_track_ids:
            self._last_seen_at.pop(track_id, None)
            self._buffers.pop(track_id, None)


class PerTrackCropSequenceBuffers:
    """Maintain one CropSequenceBuffer per track_id.

    sequence_length is frames per sequence. stride is the next sequence start
    interval in frames, not frame sampling. Class defaults are only fallbacks;
    runtime scripts may pass different values, such as 8/4.
    """

    def __init__(self, sequence_length=8, stride=4, resize_size=224, max_track_age_seconds=5.0):
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.resize_size = int(resize_size)
        self.max_track_age_seconds = float(max_track_age_seconds)
        self._buffers = {}
        self._last_seen_at = {}
        self.sequences_generated_by_track = {}

    def add(self, frame_idx, frame, boxes, now=None):
        now = time.time() if now is None else float(now)
        self._drop_stale_tracks(now)
        sequences = []
        for box in boxes:
            track_id = box.get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            self._last_seen_at[track_id] = now
            buffer = self._buffers.setdefault(track_id, CropSequenceBuffer(self.sequence_length, self.stride, self.resize_size))
            sequence = buffer.add(frame_idx, frame, [box])
            if sequence:
                sequence["track_id"] = track_id
                sequence["bbox"] = [box["x1"], box["y1"], box["x2"], box["y2"]]
                self.sequences_generated_by_track[track_id] = self.sequences_generated_by_track.get(track_id, 0) + 1
                sequences.append(sequence)
        return sequences

    def active_track_ids(self):
        return sorted(self._buffers.keys())

    def _drop_stale_tracks(self, now):
        stale_track_ids = [
            track_id
            for track_id, last_seen_at in self._last_seen_at.items()
            if now - last_seen_at > self.max_track_age_seconds
        ]
        for track_id in stale_track_ids:
            self._last_seen_at.pop(track_id, None)
            self._buffers.pop(track_id, None)

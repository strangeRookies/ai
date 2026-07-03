import time

from ai.action.cheap_filter import CheapFilterConfig, evaluate_sequence_candidate
from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.sequence_buffer import CropSequenceBuffer


def _avg_keypoint_confidence(detection):
    confidences = [
        float(point.get("confidence", 0.0))
        for point in detection.get("keypoints") or []
        if point.get("confidence") is not None
    ]
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 4)


class PerTrackKeypointSequenceBuffers:
    def __init__(self, sequence_length=30, stride=15, max_track_age_seconds=5.0, cheap_filter_config=None):
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.max_track_age_seconds = float(max_track_age_seconds)
        self.cheap_filter_config = cheap_filter_config or CheapFilterConfig(enabled=False)
        self._buffers = {}
        self._last_seen_at = {}
        self.sequences_generated_by_track = {}
        self.sequences_kept_by_filter = 0
        self.sequences_skipped_by_filter = 0
        self.cheap_filter_reasons = {}
        self.last_sequence_diagnostics = {}

    def add(self, frame_idx, detections, frame_shape=None, now=None, frame_id=None, captured_at_ms=None):
        now = time.time() if now is None else float(now)
        self._drop_stale_tracks(now)
        sequences = []

        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                self.last_sequence_diagnostics[-1] = self._diagnostic(
                    None,
                    "selected_track_missing",
                    0,
                    detection,
                )
                continue

            track_id = int(track_id)
            if not detection.get("keypoints"):
                self.last_sequence_diagnostics[track_id] = self._diagnostic(
                    track_id,
                    "insufficient_keypoints",
                    self.buffer_lengths().get(track_id, 0),
                    detection,
                )
                continue

            self._last_seen_at[track_id] = now
            buffer = self._buffers.setdefault(track_id, KeypointSequenceBuffer(self.sequence_length, self.stride))
            sequence = buffer.add(frame_idx, [detection], frame_shape, frame_id=frame_id, captured_at_ms=captured_at_ms)
            self.last_sequence_diagnostics[track_id] = self._diagnostic(
                track_id,
                "sequence_ready" if sequence else "buffer_not_full",
                len(buffer._frames),
                detection,
            )

            if not sequence:
                continue

            decision = evaluate_sequence_candidate(sequence, self.cheap_filter_config)
            sequence["cheap_filter"] = {
                "keep": decision.keep,
                "risk_score": decision.risk_score,
                "reasons": list(decision.reasons),
            }
            for reason in decision.reasons:
                self.cheap_filter_reasons[reason] = self.cheap_filter_reasons.get(reason, 0) + 1

            if not decision.keep:
                self.sequences_skipped_by_filter += 1
                self.last_sequence_diagnostics[track_id]["reason"] = "cheap_filter_skipped"
                self.last_sequence_diagnostics[track_id]["cheap_filter_reasons"] = list(decision.reasons)
                continue

            self.sequences_kept_by_filter += 1
            sequence["track_id"] = track_id
            self.sequences_generated_by_track[track_id] = self.sequences_generated_by_track.get(track_id, 0) + 1
            sequences.append(sequence)

        return sequences

    def active_track_ids(self):
        return sorted(self._buffers.keys())

    def buffer_lengths(self):
        return {track_id: len(buffer._frames) for track_id, buffer in self._buffers.items()}

    def sequence_diagnostics(self):
        return dict(self.last_sequence_diagnostics)

    def _diagnostic(self, track_id, reason, buffer_length, detection):
        return {
            "track_id": track_id,
            "reason": reason,
            "buffer_length": int(buffer_length),
            "required_sequence_length": self.sequence_length,
            "stride": self.stride,
            "missing_count": max(self.sequence_length - int(buffer_length), 0),
            "keypoint_count": len(detection.get("keypoints") or []),
            "avg_keypoint_confidence": _avg_keypoint_confidence(detection),
        }

    def _drop_stale_tracks(self, now):
        stale_track_ids = [
            track_id
            for track_id, last_seen_at in self._last_seen_at.items()
            if now - last_seen_at > self.max_track_age_seconds
        ]
        for track_id in stale_track_ids:
            self._last_seen_at.pop(track_id, None)
            self._buffers.pop(track_id, None)
            self.last_sequence_diagnostics.pop(track_id, None)


class PerTrackCropSequenceBuffers:
    def __init__(self, sequence_length=30, stride=15, resize_size=224, max_track_age_seconds=5.0):
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.resize_size = int(resize_size)
        self.max_track_age_seconds = float(max_track_age_seconds)
        self._buffers = {}
        self._last_seen_at = {}
        self.sequences_generated_by_track = {}
        self.last_sequence_diagnostics = {}

    def add(self, frame_idx, frame, boxes, now=None, frame_id=None, captured_at_ms=None):
        now = time.time() if now is None else float(now)
        self._drop_stale_tracks(now)
        sequences = []

        for box in boxes:
            track_id = box.get("track_id")
            if track_id is None:
                self.last_sequence_diagnostics[-1] = self._diagnostic(None, "selected_track_missing", 0)
                continue
            track_id = int(track_id)
            self._last_seen_at[track_id] = now

            buffer = self._buffers.setdefault(track_id, CropSequenceBuffer(self.sequence_length, self.stride, self.resize_size))
            sequence = buffer.add(frame_idx, frame, [box], frame_id=frame_id, captured_at_ms=captured_at_ms)
            self.last_sequence_diagnostics[track_id] = self._diagnostic(
                track_id,
                "sequence_ready" if sequence else "buffer_not_full",
                len(buffer._crops),
            )

            if sequence:
                sequence["track_id"] = track_id
                sequence["bbox"] = [box["x1"], box["y1"], box["x2"], box["y2"]]
                self.sequences_generated_by_track[track_id] = self.sequences_generated_by_track.get(track_id, 0) + 1
                sequences.append(sequence)

        return sequences

    def active_track_ids(self):
        return sorted(self._buffers.keys())

    def buffer_lengths(self):
        return {track_id: len(buffer._crops) for track_id, buffer in self._buffers.items()}

    def sequence_diagnostics(self):
        return dict(self.last_sequence_diagnostics)

    def _diagnostic(self, track_id, reason, buffer_length):
        return {
            "track_id": track_id,
            "reason": reason,
            "buffer_length": int(buffer_length),
            "required_sequence_length": self.sequence_length,
            "stride": self.stride,
            "missing_count": max(self.sequence_length - int(buffer_length), 0),
        }

    def _drop_stale_tracks(self, now):
        stale_track_ids = [
            track_id
            for track_id, last_seen_at in self._last_seen_at.items()
            if now - last_seen_at > self.max_track_age_seconds
        ]
        for track_id in stale_track_ids:
            self._last_seen_at.pop(track_id, None)
            self._buffers.pop(track_id, None)
            self.last_sequence_diagnostics.pop(track_id, None)

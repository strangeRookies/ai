import time

from ai.action.cheap_filter import CheapFilterConfig, evaluate_sequence_candidate
from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.sequence_buffer import CropSequenceBuffer
from tracking.simple_tracker import bbox_iou, center_distance_ratio


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
    def __init__(
        self,
        sequence_length=30,
        stride=15,
        max_track_age_seconds=5.0,
        cheap_filter_config=None,
        missing_track_grace_seconds=None,
        relink_iou_threshold=0.30,
        relink_center_distance_ratio=0.70,
        relink_max_time_gap_seconds=2.0,
    ):
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.max_track_age_seconds = float(max_track_age_seconds)
        self.missing_track_grace_seconds = float(
            max_track_age_seconds if missing_track_grace_seconds is None else missing_track_grace_seconds
        )
        self.relink_iou_threshold = float(relink_iou_threshold)
        self.relink_center_distance_ratio = float(relink_center_distance_ratio)
        self.relink_max_time_gap_seconds = float(relink_max_time_gap_seconds)
        self.cheap_filter_config = cheap_filter_config or CheapFilterConfig(enabled=False)
        self._buffers = {}
        self._last_seen_at = {}
        self._last_detection_by_track = {}
        self.sequences_generated_by_track = {}
        self.sequences_kept_by_filter = 0
        self.sequences_skipped_by_filter = 0
        self.cheap_filter_reasons = {}
        self.last_sequence_diagnostics = {}
        self.relink_success_count = 0
        self.relink_failure_count = 0
        self.buffer_retained_count = 0
        self.buffer_deleted_count = 0

    def add(self, frame_idx, detections, frame_shape=None, now=None, frame_id=None, captured_at_ms=None):
        now = time.time() if now is None else float(now)
        self._drop_stale_tracks(now, frame_idx, frame_shape, frame_id, captured_at_ms)
        sequences = []

        for detection in detections:
            detection = dict(detection)
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
            relinked_from = self._relink_track_id(track_id, detection, now)
            if relinked_from is None and track_id not in self._buffers:
                self.relink_failure_count += 1
            elif relinked_from is not None:
                detection["track_id"] = relinked_from
                track_id = relinked_from
                self.relink_success_count += 1
            if not detection.get("keypoints"):
                self.last_sequence_diagnostics[track_id] = self._diagnostic(
                    track_id,
                    "insufficient_keypoints",
                    self.buffer_lengths().get(track_id, 0),
                    detection,
                )
                continue

            self._last_seen_at[track_id] = now
            self._last_detection_by_track[track_id] = dict(detection)
            buffer = self._buffers.setdefault(track_id, KeypointSequenceBuffer(self.sequence_length, self.stride))
            sequence = buffer.add(frame_idx, [detection], frame_shape, frame_id=frame_id, captured_at_ms=captured_at_ms)
            self.last_sequence_diagnostics[track_id] = self._diagnostic(
                track_id,
                "sequence_ready" if sequence else "buffer_not_full",
                len(buffer._frames),
                detection,
            )
            if relinked_from is not None:
                self.last_sequence_diagnostics[track_id]["relink"] = "success"
                self.last_sequence_diagnostics[track_id]["incoming_track_id"] = detection.get("original_track_id", track_id)

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

    def _drop_stale_tracks(self, now, frame_idx, frame_shape, frame_id, captured_at_ms):
        stale_track_ids = [
            track_id
            for track_id, last_seen_at in self._last_seen_at.items()
            if now - last_seen_at > self.missing_track_grace_seconds
        ]
        grace_track_ids = [
            track_id
            for track_id, last_seen_at in self._last_seen_at.items()
            if self.max_track_age_seconds < now - last_seen_at <= self.missing_track_grace_seconds
        ]
        for track_id in grace_track_ids:
            detection = dict(self._last_detection_by_track.get(track_id) or {})
            if not detection:
                continue
            buffer = self._buffers.get(track_id)
            if buffer is None:
                continue
            self.buffer_retained_count += 1
            self.last_sequence_diagnostics[track_id] = self._diagnostic(
                track_id,
                "missing_track_grace",
                len(buffer._frames),
                detection,
            )
        for track_id in stale_track_ids:
            self._last_seen_at.pop(track_id, None)
            self._buffers.pop(track_id, None)
            self._last_detection_by_track.pop(track_id, None)
            self.last_sequence_diagnostics.pop(track_id, None)
            self.buffer_deleted_count += 1

    def _relink_track_id(self, incoming_track_id, detection, now):
        if incoming_track_id in self._buffers:
            return None
        best_track_id = None
        best_score = -1.0
        incoming_bbox = detection.get("bbox")
        for track_id, previous in self._last_detection_by_track.items():
            last_seen_at = float(self._last_seen_at.get(track_id, 0.0))
            if now - last_seen_at > self.relink_max_time_gap_seconds:
                continue
            previous_bbox = previous.get("bbox")
            iou = bbox_iou(incoming_bbox, previous_bbox)
            center_ratio = center_distance_ratio(incoming_bbox, previous_bbox)
            if iou < self.relink_iou_threshold and center_ratio > self.relink_center_distance_ratio:
                continue
            score = iou + max(0.0, self.relink_center_distance_ratio - center_ratio)
            if score > best_score:
                best_score = score
                best_track_id = track_id
        if best_track_id is not None:
            detection["original_track_id"] = incoming_track_id
        return best_track_id


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

import time

from ai.action.cheap_filter import CheapFilterConfig, evaluate_sequence_candidate
from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.sequence_buffer import CropSequenceBuffer
from tracking.simple_tracker import bbox_iou, center_distance_ratio
from ai.action.sequence_buffer_migration import merge_sequence_buffers


def aggregate_sequence_completion(sequence_diagnostics, sequences_generated_by_track):
    """Return serializable completion metrics shared by runtime summaries."""
    diagnostics = dict(sequence_diagnostics or {})
    generated = dict(sequences_generated_by_track or {})
    observed = {int(track_id) for track_id in diagnostics if int(track_id) >= 0}
    observed.update(int(track_id) for track_id in generated)
    eligible = {track_id for track_id in observed if int(diagnostics.get(track_id, {}).get("buffer_length", 0)) > 0}
    completed = {track_id for track_id, count in generated.items() if int(count) > 0}
    reasons = {}
    for track_id in observed - completed:
        reason = str(diagnostics.get(track_id, {}).get("reason") or "unknown")
        reasons[reason] = int(reasons.get(reason, 0)) + 1
    return {"total_observed_tracks": len(observed), "eligible_tracks": len(eligible), "tracks_with_completed_sequence": len(completed), "total_completed_sequences": sum(int(count) for count in generated.values()), "sequence_completion_rate": round(len(completed) / len(eligible), 4) if eligible else None, "average_sequences_per_eligible_track": round(sum(int(count) for count in generated.values()) / len(eligible), 4) if eligible else 0.0, "incomplete_reasons": reasons, "incomplete_buffer_not_full": int(reasons.get("buffer_not_full", 0)), "incomplete_insufficient_keypoints": int(reasons.get("insufficient_keypoints", 0)), "incomplete_missing_track_grace": int(reasons.get("missing_track_grace", 0)), "incomplete_id_switch": int(reasons.get("id_switch", 0)), "incomplete_detector_miss": int(reasons.get("detector_miss", 0)), "incomplete_reset_or_eof": int(reasons.get("reset_or_eof", 0)), "incomplete_cheap_filter_skipped": int(reasons.get("cheap_filter_skipped", 0)), "maximum_frame_gap": 0, "average_frame_gap": 0.0, "identity_consistency_violations": 0}

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
    """track_id별 keypoint frame을 모아 LSTM 입력 sequence를 만든다.

    YOLO Pose/ByteTrack은 매 프레임 결과이고, LSTM은 일정 길이의 시간 창을 필요로 한다.
    이 클래스는 track별 ring buffer를 유지하면서 sequence_length/stride가 찼을 때만
    sequence를 반환한다. track이 잠깐 사라져도 grace period 동안 buffer를 유지하고,
    새 track_id가 이전 bbox와 충분히 겹치면 re-link해서 sequence가 끊기지 않게 한다.
    """

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
        self.migration_conflicts = []
        self.relink_success_count = 0
        self.relink_failure_count = 0
        self.buffer_retained_count = 0
        self.buffer_deleted_count = 0

    def add(self, frame_idx, detections, frame_shape=None, now=None, frame_id=None, captured_at_ms=None):
        """현재 프레임의 tracked detections를 buffer에 넣고 준비된 sequence들을 반환한다.

        detection에 `track_id`가 없으면 LSTM 대상이 될 수 없으므로 diagnostics만 남긴다.
        `keypoints`가 없거나 부족한 경우도 buffer에 넣지 않는다. sequence가 만들어진 뒤에는
        cheap filter가 빠른 1차 위험도 검사를 하고, 통과한 sequence만 LSTM classifier로 간다.
        """

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

    def sequence_completion_summary(self):
        return aggregate_sequence_completion(self.last_sequence_diagnostics, self.sequences_generated_by_track)

    def migrate_track_id(self, old_track_id, new_track_id, *, fresh_start_history: bool = False) -> bool:
        """Move buffer/state from old_track_id to new_track_id.

        Incident Recovery should pass ``fresh_start_history=True`` so pre-gap
        keypoint history is dropped instead of merged — raw motion features
        would otherwise treat the long miss as one huge frame step (velocity spike).
        Ordinary tracker resilience may keep the default merge/move behavior.
        """
        old_id, new_id = int(old_track_id), int(new_track_id)
        if old_id == new_id:
            return False

        if fresh_start_history:
            # Fail-safe for recovery: discard LSTM history on BOTH old and new ids.
            # Fall lifecycle / display-id / overlay are migrated separately.
            side_maps = (
                self._buffers,
                self._last_seen_at,
                self._last_detection_by_track,
                self.last_sequence_diagnostics,
                self.sequences_generated_by_track,
            )
            had = False
            for track_id in (old_id, new_id):
                for mapping in side_maps:
                    if track_id in mapping:
                        had = True
                        mapping.pop(track_id, None)
            self.last_sequence_diagnostics[new_id] = {
                "track_id": new_id,
                "reason": "recovery_sequence_fresh_start",
                "migrated_from": old_id,
                "buffer_length": 0,
                "required_sequence_length": self.sequence_length,
                "stride": self.stride,
            }
            return bool(had)

        if old_id not in self._buffers and old_id not in self._last_seen_at:
            return False

        if new_id in self._buffers and old_id in self._buffers:
            if not merge_sequence_buffers(self._buffers[old_id], self._buffers[new_id], new_id):
                self.migration_conflicts.append({"source_track_id": old_id, "destination_track_id": new_id, "reason": "incompatible_buffer"})
                return False
            self._buffers.pop(old_id, None)
            self._last_seen_at[new_id] = max(self._last_seen_at.get(old_id, 0.0), self._last_seen_at.get(new_id, 0.0))
            self._last_seen_at.pop(old_id, None)
            self._last_detection_by_track.pop(old_id, None)
            self.last_sequence_diagnostics.pop(old_id, None)
            self.sequences_generated_by_track[new_id] = self.sequences_generated_by_track.get(new_id, 0) + self.sequences_generated_by_track.pop(old_id, 0)
            return True
        if old_id in self._buffers:
            self._buffers[new_id] = self._buffers.pop(old_id)
        if old_id in self._last_seen_at:
            self._last_seen_at[new_id] = self._last_seen_at.pop(old_id)
        if old_id in self._last_detection_by_track:
            det = dict(self._last_detection_by_track.pop(old_id))
            det["track_id"] = new_id
            self._last_detection_by_track[new_id] = det
        if old_id in self.last_sequence_diagnostics:
            diag = dict(self.last_sequence_diagnostics.pop(old_id))
            diag["track_id"] = new_id
            diag["migrated_from"] = old_id
            self.last_sequence_diagnostics[new_id] = diag
        if old_id in self.sequences_generated_by_track:
            self.sequences_generated_by_track[new_id] = (
                self.sequences_generated_by_track.get(new_id, 0)
                + self.sequences_generated_by_track.pop(old_id)
            )
        return True

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
        """사라진 track buffer를 grace period 기준으로 유지하거나 삭제한다.

        `max_track_age_seconds`를 넘었지만 `missing_track_grace_seconds` 이내인 track은
        곧 다시 잡힐 수 있으므로 buffer를 남긴다. grace까지 넘은 track은 메모리 누수와
        잘못된 re-link를 막기 위해 삭제한다.
        """

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
        """새 track_id가 기존 buffer의 사람과 같은지 bbox IoU/중심거리로 추정한다.

        ByteTrack이 ID를 바꿨더라도 bbox가 충분히 겹치거나 중심점 이동이 작고, 마지막
        관측 시간 차이가 짧으면 기존 track buffer에 이어 붙인다. 이 로직이 성공하면
        sequence_length를 처음부터 다시 채우지 않아도 되어 LSTM 지연이 줄어든다.
        """

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
        self.migration_conflicts = []

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

    def sequence_completion_summary(self):
        return aggregate_sequence_completion(self.last_sequence_diagnostics, self.sequences_generated_by_track)

    def migrate_track_id(self, old_track_id, new_track_id, *, fresh_start_history: bool = False) -> bool:
        old_id, new_id = int(old_track_id), int(new_track_id)
        if old_id == new_id:
            return False
        if fresh_start_history:
            side_maps = (
                self._buffers,
                self._last_seen_at,
                self.last_sequence_diagnostics,
                self.sequences_generated_by_track,
            )
            had = False
            for track_id in (old_id, new_id):
                for mapping in side_maps:
                    if track_id in mapping:
                        had = True
                        mapping.pop(track_id, None)
            return bool(had)
        if old_id not in self._buffers and old_id not in self._last_seen_at:
            return False
        if new_id in self._buffers and old_id in self._buffers:
            if not merge_sequence_buffers(self._buffers[old_id], self._buffers[new_id], new_id):
                self.migration_conflicts.append({"source_track_id": old_id, "destination_track_id": new_id, "reason": "incompatible_buffer"})
                return False
            self._buffers.pop(old_id, None)
            self._last_seen_at[new_id] = max(self._last_seen_at.get(old_id, 0.0), self._last_seen_at.get(new_id, 0.0))
            self._last_seen_at.pop(old_id, None)
            self.last_sequence_diagnostics.pop(old_id, None)
            self.sequences_generated_by_track[new_id] = self.sequences_generated_by_track.get(new_id, 0) + self.sequences_generated_by_track.pop(old_id, 0)
            return True
        if old_id in self._buffers:
            self._buffers[new_id] = self._buffers.pop(old_id)
        if old_id in self._last_seen_at:
            self._last_seen_at[new_id] = self._last_seen_at.pop(old_id)
        if old_id in self.last_sequence_diagnostics:
            diag = dict(self.last_sequence_diagnostics.pop(old_id))
            diag["track_id"] = new_id
            self.last_sequence_diagnostics[new_id] = diag
        if old_id in self.sequences_generated_by_track:
            self.sequences_generated_by_track[new_id] = (
                self.sequences_generated_by_track.get(new_id, 0)
                + self.sequences_generated_by_track.pop(old_id)
            )
        return True

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

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ByteTrackAdapter(Protocol):
    def update(self, detections: list[dict]) -> list[dict]:
        ...

    def diagnostics(self) -> dict:
        ...


@dataclass(frozen=True, slots=True)
class SupervisionPostProcessorConfig:
    min_iou: float = 0.30


class SupervisionPostProcessor:
    def __init__(
        self,
        config: SupervisionPostProcessorConfig | None = None,
        byte_tracker: ByteTrackAdapter | None = None,
    ) -> None:
        self.config = config or SupervisionPostProcessorConfig()
        self._tracker = byte_tracker or SupervisionByteTrackAdapter()

    def process(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        del frame
        tracked = self._tracker.update(detections)
        return match_keypoints_by_iou(tracked, detections, min_iou=self.config.min_iou)

    def diagnostics(self) -> dict:
        return self._tracker.diagnostics()


class SupervisionByteTrackAdapter:
    def __init__(self) -> None:
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "supervision is required when ENABLE_SUPERVISION_POSTPROCESSING=true",
            ) from exc

        self._sv = sv
        self._tracker = sv.ByteTrack()
        self._active_track_ids: set[int] = set()

    def update(self, detections: list[dict]) -> list[dict]:
        if not detections:
            self._active_track_ids = set()
            return []

        sv_detections = self._to_supervision_detections(detections)
        tracked = self._tracker.update_with_detections(sv_detections)
        track_ids = getattr(tracked, "tracker_id", None)
        output: list[dict] = []
        self._active_track_ids = set()
        for index, detection in enumerate(detections):
            item = dict(detection)
            track_id = _track_id_at(track_ids, index)
            if track_id is not None:
                item["track_id"] = track_id
                self._active_track_ids.add(track_id)
            output.append(item)
        return output

    def diagnostics(self) -> dict:
        return {
            "active_tracks": len(self._active_track_ids),
            "new_tracks": 0,
            "lost_tracks": 0,
            "id_switch_like_events": 0,
            "tracks": {str(track_id): {"track_id": track_id} for track_id in sorted(self._active_track_ids)},
        }

    def _to_supervision_detections(self, detections: list[dict]):
        xyxy = np.asarray([_bbox_xyxy(item) for item in detections], dtype=np.float32)
        confidence = np.asarray([float(item.get("confidence", 0.0)) for item in detections], dtype=np.float32)
        class_id = np.zeros(len(detections), dtype=int)
        return self._sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def match_keypoints_by_iou(
    tracked_detections: list[dict],
    source_detections: list[dict],
    min_iou: float = 0.30,
) -> list[dict]:
    matched: list[dict] = []
    used_source_indexes: set[int] = set()
    for tracked in tracked_detections:
        best_index = _best_iou_source_index(tracked, source_detections, used_source_indexes)
        item = dict(tracked)
        if best_index is not None and _bbox_iou(_bbox_xyxy(tracked), _bbox_xyxy(source_detections[best_index])) >= min_iou:
            source = source_detections[best_index]
            used_source_indexes.add(best_index)
            if source.get("keypoints"):
                item["keypoints"] = source["keypoints"]
            if source.get("keypoint_confidence") is not None:
                item["keypoint_confidence"] = source["keypoint_confidence"]
        matched.append(item)
    return matched


def _best_iou_source_index(
    tracked: dict,
    source_detections: list[dict],
    used_source_indexes: set[int],
) -> int | None:
    best_index: int | None = None
    best_iou = 0.0
    tracked_bbox = _bbox_xyxy(tracked)
    for index, source in enumerate(source_detections):
        if index in used_source_indexes:
            continue
        iou = _bbox_iou(tracked_bbox, _bbox_xyxy(source))
        if iou > best_iou:
            best_iou = iou
            best_index = index
    return best_index


def _track_id_at(track_ids, index: int) -> int | None:
    if track_ids is None or index >= len(track_ids):
        return None
    value = track_ids[index]
    if value is None:
        return None
    return int(value)


def _bbox_xyxy(detection: dict) -> list[float]:
    bbox = detection.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return [float(value) for value in bbox[:4]]


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    second_area = max(second[2] - second[0], 0.0) * max(second[3] - second[1], 0.0)
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union

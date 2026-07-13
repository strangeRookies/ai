from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Final
from typing import Protocol

import numpy as np

from ai.postprocess.person_session import PersonSessionConfig, PersonSessionReconnector
from tracking.simple_tracker import SimpleTrackAssigner


BYTETRACK_MODERN_PARAMETERS: Final = {
    "track_activation_threshold",
    "lost_track_buffer",
    "minimum_matching_threshold",
    "frame_rate",
}


class ByteTrackAdapter(Protocol):
    def update(self, detections: list[dict]) -> list[dict]:
        ...

    def diagnostics(self) -> dict:
        ...


@dataclass(frozen=True, slots=True)
class SupervisionPostProcessorConfig:
    """YOLO Pose raw detections를 ByteTrack에 넣을 때 쓰는 tracking 설정.

    `track_thresh`, `match_thresh`, `track_buffer`, `frame_rate`는 supervision
    버전에 따라 생성자 이름이 달라질 수 있어 adapter에서 지원 여부를 확인한다.
    `stability_fallback`은 ByteTrack 결과가 흔들리는 실험 상황에서만 켜는 보조 경로다.
    """

    min_iou: float = 0.30
    track_thresh: float = 0.10
    track_buffer: int = 90
    match_thresh: float = 0.20
    frame_rate: int = 30
    bbox_smoothing_alpha: float = 1.0  # 1.0 means disabled (raw bbox)
    stability_fallback: bool = False
    fallback_max_missing_seconds: float = 4.0
    fallback_center_match_ratio: float = 0.70
    session_reconnect: bool = False
    session_reconnect_max_missing_seconds: float = 3.0


class SupervisionPostProcessor:
    def __init__(
        self,
        config: SupervisionPostProcessorConfig | None = None,
        byte_tracker: ByteTrackAdapter | None = None,
    ) -> None:
        self.config = config or SupervisionPostProcessorConfig()
        self._tracker = byte_tracker or SupervisionByteTrackAdapter(
            track_thresh=self.config.track_thresh,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_thresh,
            frame_rate=self.config.frame_rate,
            bbox_smoothing_alpha=self.config.bbox_smoothing_alpha,
            stability_fallback=self.config.stability_fallback,
            fallback_max_missing_seconds=self.config.fallback_max_missing_seconds,
            fallback_center_match_ratio=self.config.fallback_center_match_ratio,
            session_reconnect=self.config.session_reconnect,
            session_reconnect_max_missing_seconds=self.config.session_reconnect_max_missing_seconds,
        )

    def process(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """YOLO Pose detection list에 `track_id`를 붙여 downstream으로 넘긴다.

        현재 frame 자체는 ByteTrack 입력에 쓰지 않지만, postprocessor 인터페이스를
        다른 tracker와 맞추기 위해 받는다. 출력 detection은 keypoints/confidence 등
        YOLO raw field를 보존해야 LSTM sequence와 pose diagnostics가 깨지지 않는다.
        """

        del frame
        return self._tracker.update(detections)

    def diagnostics(self) -> dict:
        return self._tracker.diagnostics()


class SupervisionByteTrackAdapter:
    def __init__(
        self,
        track_thresh: float = 0.10,
        track_buffer: int = 90,
        match_thresh: float = 0.20,
        frame_rate: int = 30,
        bbox_smoothing_alpha: float = 1.0,
        stability_fallback: bool = False,
        fallback_max_missing_seconds: float = 4.0,
        fallback_center_match_ratio: float = 0.70,
        session_reconnect: bool = False,
        session_reconnect_max_missing_seconds: float = 3.0,
        sv_module=None,
    ) -> None:
        if sv_module is None:
            try:
                import supervision as sv
            except ImportError as exc:
                raise RuntimeError(
                    "supervision is required when ENABLE_SUPERVISION_POSTPROCESSING=true",
                ) from exc
        else:
            sv = sv_module

        self._sv = sv
        constructor_kwargs, ignored_kwargs = build_bytetrack_constructor_kwargs(
            sv.ByteTrack,
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )
        self._bytetrack_constructor = {
            "used": constructor_kwargs,
            "ignored": ignored_kwargs,
            "supported_parameters": sorted(_constructor_parameters(sv.ByteTrack)),
        }
        self._tracker = sv.ByteTrack(**constructor_kwargs)
        self._active_track_ids: set[int] = set()
        self._previous_active_track_ids: set[int] = set()
        self._lifecycle_events: list[dict] = []
        self._last_new_tracks = 0
        self._last_lost_tracks = 0
        self._bbox_smoothing_alpha = bbox_smoothing_alpha
        self._previous_bboxes: dict[int, list[float]] = {}
        self._stability_fallback = bool(stability_fallback)
        self._session_reconnector = (
            PersonSessionReconnector(
                PersonSessionConfig(
                    track_thresh=track_thresh,
                    match_thresh=match_thresh,
                    track_buffer=track_buffer,
                    max_missing_seconds=session_reconnect_max_missing_seconds,
                    center_match_ratio=fallback_center_match_ratio,
                )
            )
            if session_reconnect
            else None
        )
        # Production defaults after offline A/B + cam_03 live canary (2026-07):
        # NEAR_DUP_SUPPRESS_MODE=hybrid_kp, SIMPLE_TRACK_NEW_TRACK_THRESH=0.30.
        # Rollback: none + 0.25 (see scripts/rollback_tracking_suppression.sh).
        # Known limitations: real two-person GT incomplete; residual new/lost may remain.
        near_dup_mode = (os.getenv("NEAR_DUP_SUPPRESS_MODE") or "hybrid_kp").strip().lower()
        near_dup_sort = (os.getenv("NEAR_DUP_SORT_BY_CONF") or "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        new_track_env = os.getenv("SIMPLE_TRACK_NEW_TRACK_THRESH")
        if new_track_env is not None and str(new_track_env).strip() != "":
            new_track_thresh = float(new_track_env)
        else:
            new_track_thresh = 0.30
        self._fallback_assigner = (
            SimpleTrackAssigner(
                track_thresh=track_thresh,
                match_thresh=match_thresh,
                track_buffer=track_buffer,
                min_box_area=10.0,
                bbox_smoothing_alpha=bbox_smoothing_alpha if bbox_smoothing_alpha < 1.0 else 0.60,
                max_missing_seconds=fallback_max_missing_seconds,
                center_match_ratio=fallback_center_match_ratio,
                # Associate low-conf dets to existing tracks, but mint new IDs only above gate.
                new_track_thresh=new_track_thresh,
                assumed_fps=float(frame_rate),
                near_dup_suppress_mode=near_dup_mode,
                near_dup_iou_thresh=float(os.getenv("NEAR_DUP_IOU_THRESH") or 0.70),
                near_dup_center_ratio=float(os.getenv("NEAR_DUP_CENTER_RATIO") or 0.25),
                near_dup_area_ratio_min=float(os.getenv("NEAR_DUP_AREA_RATIO_MIN") or 0.55),
                near_dup_area_ratio_max=float(os.getenv("NEAR_DUP_AREA_RATIO_MAX") or 1.80),
                near_dup_keypoint_dist=float(os.getenv("NEAR_DUP_KEYPOINT_DIST") or 0.35),
                sort_detections_by_conf=near_dup_sort if near_dup_mode not in {"", "none", "off"} else False,
            )
            if self._stability_fallback
            else None
        )

    def update(self, detections: list[dict]) -> list[dict]:
        """supervision ByteTrack을 호출하고 결과를 원본 YOLO detection에 다시 매핑한다.

        supervision은 low-confidence detection을 필터링하거나 순서를 바꿀 수 있다. 그래서
        tracker 출력 순서를 그대로 믿지 않고 bbox IoU로 원본 detection을 찾아 `track_id`만
        덧붙인다. 이 처리가 없으면 keypoint가 다른 사람 bbox에 붙어 LSTM 입력이 오염된다.
        """

        if not detections:
            self._record_active_set_delta(set())
            self._previous_bboxes.clear()
            if self._fallback_assigner is not None:
                self._fallback_assigner.update([])
            return []

        sv_detections = self._to_supervision_detections(detections)
        tracked = self._tracker.update_with_detections(sv_detections)

        # supervision's update_with_detections may FILTER low-confidence detections
        # and may REORDER them relative to the input.
        # We match each tracked bbox back to the nearest original detection by
        # IoU so the assignment is always correct regardless of supervision's
        # internal filtering or sorting behaviour.
        tracker_bboxes: list[tuple[list[float], int]] = []
        track_ids_arr = getattr(tracked, "tracker_id", None)
        tracked_xyxy = getattr(tracked, "xyxy", None)
        if track_ids_arr is not None and tracked_xyxy is not None:
            for t_idx in range(len(track_ids_arr)):
                tid = track_ids_arr[t_idx]
                if tid is None:
                    continue
                bbox = [float(v) for v in tracked_xyxy[t_idx]]
                tracker_bboxes.append((bbox, int(tid)))

        # Build output preserving ALL original fields (including keypoints)
        output: list[dict] = [dict(d) for d in detections]
        assigned: set[int] = set()
        self._active_track_ids = set()
        current_active_tids = set()

        for tracked_bbox, track_id in tracker_bboxes:
            best_det_idx: int | None = None
            best_iou = 0.0
            for det_idx, detection in enumerate(detections):
                if det_idx in assigned:
                    continue
                iou = _bbox_iou(tracked_bbox, _bbox_xyxy(detection))
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            if best_det_idx is not None and best_iou > 0.0:
                # Optionally apply exponential moving average bounding box smoothing
                raw_box = output[best_det_idx]["bbox"]
                if self._bbox_smoothing_alpha < 1.0 and track_id in self._previous_bboxes:
                    prev_box = self._previous_bboxes[track_id]
                    smoothed_box = [
                        round(self._bbox_smoothing_alpha * raw_box[i] + (1.0 - self._bbox_smoothing_alpha) * prev_box[i], 2)
                        for i in range(4)
                    ]
                    output[best_det_idx]["bbox"] = smoothed_box
                
                self._previous_bboxes[track_id] = output[best_det_idx]["bbox"]
                output[best_det_idx]["track_id"] = track_id
                self._active_track_ids.add(track_id)
                current_active_tids.add(track_id)
                assigned.add(best_det_idx)

        # Clear previous records for lost tracks to avoid memory leaks
        for tid in list(self._previous_bboxes.keys()):
            if tid not in current_active_tids:
                del self._previous_bboxes[tid]

        if self._fallback_assigner is not None:
            fallback_input = []
            for detection in output:
                item = dict(detection)
                item.pop("track_id", None)
                fallback_input.append(item)
            fallback_output = self._fallback_assigner.update(fallback_input)
            self._active_track_ids = {
                int(item["track_id"])
                for item in fallback_output
                if item.get("track_id") is not None
            }
            # Fallback SimpleTrackAssigner already owns detailed lifecycle events.
            self._previous_active_track_ids = set(self._active_track_ids)
            self._lifecycle_events = []
            self._last_new_tracks = 0
            self._last_lost_tracks = 0
            return fallback_output

        if self._session_reconnector is not None:
            output = self._session_reconnector.update(output)
            self._active_track_ids = {
                int(item["track_id"])
                for item in output
                if item.get("track_id") is not None
            }
        self._record_active_set_delta(self._active_track_ids)
        return output

    def _record_active_set_delta(self, active_ids: set[int]) -> None:
        """Synthesize lost/new lifecycle events from ByteTrack active-id set changes."""
        previous = self._previous_active_track_ids
        new_ids = sorted(active_ids - previous)
        lost_ids = sorted(previous - active_ids)
        events: list[dict] = []
        for track_id in lost_ids:
            events.append(
                {
                    "event": "lost",
                    "reason": "bytetrack_inactive",
                    "trackId": int(track_id),
                }
            )
        for track_id in new_ids:
            events.append(
                {
                    "event": "new_track",
                    "reason": "bytetrack_new_or_reactivated",
                    "trackId": int(track_id),
                }
            )
        self._lifecycle_events = events
        self._last_new_tracks = len(new_ids)
        self._last_lost_tracks = len(lost_ids)
        self._previous_active_track_ids = set(active_ids)
        self._active_track_ids = set(active_ids)

    def diagnostics(self) -> dict:
        if self._fallback_assigner is not None:
            diagnostics = self._fallback_assigner.diagnostics()
            diagnostics["stability_fallback"] = True
            return diagnostics
        session_diagnostics = (
            self._session_reconnector.diagnostics()
            if self._session_reconnector is not None
            else {
                "person_session_reconnect_success": 0,
                "person_session_reconnect_failure": 0,
            }
        )
        # Prefer frame-local active-set deltas; fall back to session reconnector counters.
        new_tracks = self._last_new_tracks or int(session_diagnostics.get("new_tracks", 0))
        lost_tracks = self._last_lost_tracks or int(session_diagnostics.get("lost_tracks", 0))
        removed = [
            int(e["trackId"])
            for e in self._lifecycle_events
            if e.get("event") == "lost" and e.get("trackId") is not None
        ]
        return {
            "active_tracks": len(self._active_track_ids),
            "new_tracks": int(new_tracks),
            "lost_tracks": int(lost_tracks),
            "id_switch_like_events": int(session_diagnostics.get("id_switch_like_events", 0)),
            "removed_track_ids": removed,
            "lifecycle_events": list(self._lifecycle_events),
            "tracks": {str(track_id): {"track_id": track_id} for track_id in sorted(self._active_track_ids)},
            "stability_fallback": False,
            "person_session_reconnect_success": int(session_diagnostics.get("person_session_reconnect_success", 0)),
            "person_session_reconnect_failure": int(session_diagnostics.get("person_session_reconnect_failure", 0)),
            "bytetrack_constructor": self._bytetrack_constructor,
        }

    def _to_supervision_detections(self, detections: list[dict]):
        xyxy = np.asarray([_bbox_xyxy(item) for item in detections], dtype=np.float32)
        confidence = np.asarray([float(item.get("confidence", 0.0)) for item in detections], dtype=np.float32)
        class_id = np.zeros(len(detections), dtype=int)
        return self._sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def _bbox_xyxy(detection: dict) -> list[float]:
    bbox = detection.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return [float(value) for value in bbox[:4]]


def build_bytetrack_constructor_kwargs(
    byte_track_cls,
    track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    frame_rate: int,
) -> tuple[dict, dict]:
    """설치된 supervision.ByteTrack 생성자가 실제로 받는 인자만 골라낸다.

    supervision 0.28처럼 wrapper signature가 `*args, **kwargs`로 보일 때는 알려진
    modern parameter name으로 대체한다. 지원하지 않는 값은 `ignored`에 남겨 startup
    config dump에서 확인할 수 있게 한다.
    """

    supported = _constructor_parameters(byte_track_cls)
    candidates = [
        (("track_activation_threshold", "track_thresh"), "track_thresh", float(track_thresh)),
        (("lost_track_buffer", "track_buffer"), "track_buffer", int(track_buffer)),
        (("minimum_matching_threshold", "match_thresh"), "match_thresh", float(match_thresh)),
        (("frame_rate",), "frame_rate", int(frame_rate)),
    ]
    kwargs = {}
    ignored = {}
    for names, public_name, value in candidates:
        matched_name = next((name for name in names if name in supported), None)
        if matched_name is None:
            ignored[public_name] = value
            continue
        kwargs[matched_name] = value
    return kwargs, ignored


def _constructor_parameters(byte_track_cls) -> set[str]:
    inspected_cls = getattr(byte_track_cls, "wrapped", byte_track_cls)
    try:
        signature = inspect.signature(inspected_cls)
    except (TypeError, ValueError):
        try:
            signature = inspect.signature(inspected_cls.__init__)
        except (AttributeError, TypeError, ValueError):
            return set()
    if _signature_is_kwargs_proxy(signature):
        return set(BYTETRACK_MODERN_PARAMETERS)
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
    }


def _signature_is_kwargs_proxy(signature: inspect.Signature) -> bool:
    kinds = {parameter.kind for parameter in signature.parameters.values()}
    return inspect.Parameter.VAR_POSITIONAL in kinds and inspect.Parameter.VAR_KEYWORD in kinds


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


def match_keypoints_by_iou(
    tracked_detections: list[dict],
    source_detections: list[dict],
    min_iou: float = 0.30,
) -> list[dict]:
    """tracker가 bbox만 반환한 경우 원본 YOLO keypoint를 IoU 기준으로 복원한다.

    ByteTrack은 pose keypoint를 알지 못한다. 따라서 bbox association 이후에도
    LSTM이 같은 사람의 keypoint sequence를 받게 하려면, tracker bbox와 가장 잘 겹치는
    raw detection에서 keypoints/keypoint_confidence를 다시 붙여야 한다.
    """

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



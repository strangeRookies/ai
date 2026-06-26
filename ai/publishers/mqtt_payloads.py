from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Final, TypeAlias

from ai.action.faint_post_processing import faint_probability

SCHEMA_VERSION: Final = "1.0"
DEFAULT_EVENT_TYPE: Final = "faint"
DEFAULT_MEMO_TEXT: Final = "쓰러짐 의심!"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap: TypeAlias = Mapping[str, JsonValue]


def current_timestamp_ms() -> int:
    return int(time.time() * 1000)


def build_overlay_payload(
    stream_id: str,
    frame_width: int,
    frame_height: int,
    boxes: Sequence[JsonMap],
    timestamp_ms: int | None = None,
) -> dict[str, JsonValue]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "messageType": "overlay",
        "timestampMs": timestamp_ms if timestamp_ms is not None else current_timestamp_ms(),
        "streamId": stream_id,
        "frameWidth": int(frame_width),
        "frameHeight": int(frame_height),
        "events": [
            _overlay_event(box, frame_width=frame_width, frame_height=frame_height)
            for box in boxes
            if _has_overlay_signal(box)
        ],
    }


def build_confirmed_event_payload(
    stream_id: str,
    frame_width: int,
    frame_height: int,
    prediction: JsonMap,
    sequence: JsonMap | None,
    boxes: Sequence[JsonMap],
    timestamp_ms: int | None = None,
    event_id: str | None = None,
    memo_text: str = DEFAULT_MEMO_TEXT,
) -> dict[str, JsonValue]:
    emitted_at = timestamp_ms if timestamp_ms is not None else current_timestamp_ms()
    event_type = _event_type(prediction)
    confidence = _prediction_confidence(prediction)
    tracking_id = _tracking_id(sequence) if sequence is not None else None
    bbox = _sequence_bbox(sequence) if sequence is not None else None
    if bbox is None:
        bbox = _first_box_bbox(boxes)

    payload: dict[str, JsonValue] = {
        "schemaVersion": SCHEMA_VERSION,
        "messageType": "event",
        "eventId": event_id or _default_event_id(stream_id, emitted_at),
        "timestampMs": emitted_at,
        "streamId": stream_id,
        "type": event_type,
        "memoText": memo_text,
        "confidence": confidence,
        "frameWidth": int(frame_width),
        "frameHeight": int(frame_height),
        "boundingBox": bbox,
    }
    if tracking_id is not None:
        payload["trackingId"] = tracking_id
    return payload


def frame_size_from_shape(frame_shape: Sequence[int]) -> tuple[int, int]:
    if len(frame_shape) < 2:
        return 0, 0
    return int(frame_shape[1]), int(frame_shape[0])


def camera_stream_id(args: JsonMap) -> str:
    camera_login_id = args.get("camera_login_id")
    if camera_login_id:
        return str(camera_login_id)
    return str(args.get("camera_id") or "unknown")


def _has_overlay_signal(box: JsonMap) -> bool:
    return box.get("faint_probability") is not None or bool(box.get("event_triggered"))


def _overlay_event(box: JsonMap, frame_width: int | None = None, frame_height: int | None = None) -> dict[str, JsonValue]:
    confidence = box.get("faint_probability")
    if confidence is None:
        confidence = box.get("score", 0.0)
    event: dict[str, JsonValue] = {
        "type": DEFAULT_EVENT_TYPE,
        "confidence": _clamp_probability(confidence),
        "boundingBox": _box_bbox(box, frame_width=frame_width, frame_height=frame_height),
    }
    tracking_id = box.get("track_id")
    if tracking_id is not None:
        event["trackingId"] = int(float(str(tracking_id)))
    return event


def _box_bbox(box: JsonMap, frame_width: int | None = None, frame_height: int | None = None) -> dict[str, JsonValue]:
    x1 = _float_value(box.get("x1"))
    y1 = _float_value(box.get("y1"))
    x2 = _float_value(box.get("x2"))
    y2 = _float_value(box.get("y2"))
    return _bbox_from_xyxy(x1, y1, x2, y2, frame_width=frame_width, frame_height=frame_height)


def _sequence_bbox(sequence: JsonMap) -> dict[str, JsonValue] | None:
    raw = sequence.get("bbox")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    return _bbox_from_xyxy(
        _float_value(raw[0]),
        _float_value(raw[1]),
        _float_value(raw[2]),
        _float_value(raw[3]),
    )


def _first_box_bbox(boxes: Sequence[JsonMap]) -> dict[str, JsonValue] | None:
    if not boxes:
        return None
    return _box_bbox(boxes[0])


def _bbox_from_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, JsonValue]:
    if frame_width is not None and frame_width > 0:
        x1 = _clamp_value(x1, 0.0, float(frame_width))
        x2 = _clamp_value(x2, 0.0, float(frame_width))
    if frame_height is not None and frame_height > 0:
        y1 = _clamp_value(y1, 0.0, float(frame_height))
        y2 = _clamp_value(y2, 0.0, float(frame_height))
    x = round(min(x1, x2))
    y = round(min(y1, y2))
    right = round(max(x1, x2))
    bottom = round(max(y1, y2))
    return {
        "x": x,
        "y": y,
        "width": max(0, right - x),
        "height": max(0, bottom - y),
    }


def _prediction_confidence(prediction: JsonMap) -> float:
    faint_prob = faint_probability(prediction)
    if faint_prob is not None:
        return _clamp_probability(faint_prob)
    return _clamp_probability(prediction.get("score", 0.0))


def _event_type(prediction: JsonMap) -> str:
    label = str(prediction.get("label") or DEFAULT_EVENT_TYPE).strip().lower()
    match label:
        case "faint":
            return "faint"
        case "fall" | "fall_detected":
            return "fall"
        case "normal":
            return "normal"
        case _:
            return label or DEFAULT_EVENT_TYPE


def _tracking_id(sequence: JsonMap) -> int | None:
    value = sequence.get("track_id")
    if value is None:
        return None
    return int(float(str(value)))


def _float_value(value: JsonValue) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return float(str(value))


def _clamp_probability(value: JsonValue) -> float:
    probability = _float_value(value)
    return max(0.0, min(1.0, probability))


def _clamp_value(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _default_event_id(stream_id: str, timestamp_ms: int) -> str:
    event_date = time.strftime("%Y%m%d", time.localtime(timestamp_ms / 1000.0))
    return f"evt-{event_date}-{stream_id}-{timestamp_ms}"

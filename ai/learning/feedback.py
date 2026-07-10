from __future__ import annotations

import time
from enum import StrEnum
from typing import Final, NotRequired, TypedDict, assert_never


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class FeedbackOutcome(StrEnum):
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    TRUE_POSITIVE = "true_positive"
    TRUE_NEGATIVE = "true_negative"


class EventPayload(TypedDict, total=False):
    eventId: str
    event_id: str
    cameraLoginId: str
    camera_login_id: str
    camera_id: str
    frameId: int
    frame_id: int
    timestampMs: int
    timestamp: float | str
    confidence: float
    bbox: list[JsonValue]
    keypoints: list[JsonValue]
    events: list[dict[str, JsonValue]]
    aiLatencyMs: int
    publishLatencyMs: int
    snapshot_path: str
    snapshotUrl: str
    clip_path: str
    clipUrl: str


class FeedbackRecord(TypedDict):
    event_id: str
    camera_login_id: str
    frame_id: int | None
    timestamp_ms: int | None
    confidence: float | None
    bbox: list[JsonValue]
    keypoints: list[JsonValue]
    ai_latency_ms: int | None
    publish_latency_ms: int | None
    snapshot_path: str | None
    clip_path: str | None
    feedback_outcome: str
    candidate_type: str
    operator_id: str
    feedback_text: str
    feedback_timestamp_ms: int
    source_type: str
    schema_version: str
    privacy_note: str
    vlm_description: NotRequired[str]


SCHEMA_VERSION: Final = "self-improvement-feedback/v1"
PRIVACY_NOTE: Final = "Store snapshots/clips only in the configured evidence directory with retention and de-identification controls."


def candidate_type_for_outcome(outcome: FeedbackOutcome) -> str:
    match outcome:
        case FeedbackOutcome.FALSE_POSITIVE:
            return "hard_negative"
        case FeedbackOutcome.FALSE_NEGATIVE:
            return "faint_fall_reinforcement"
        case FeedbackOutcome.TRUE_POSITIVE:
            return "verified_positive"
        case FeedbackOutcome.TRUE_NEGATIVE:
            return "verified_negative"


def build_feedback_record(
    event_payload: EventPayload,
    outcome: FeedbackOutcome,
    operator_id: str,
    feedback_text: str,
    feedback_timestamp_ms: int | None = None,
) -> FeedbackRecord:
    timestamp_ms = _timestamp_ms(event_payload)
    return FeedbackRecord(
        event_id=_event_id(event_payload, timestamp_ms),
        camera_login_id=_camera_login_id(event_payload),
        frame_id=_optional_int(event_payload.get("frameId", event_payload.get("frame_id"))),
        timestamp_ms=timestamp_ms,
        confidence=_optional_float(event_payload.get("confidence")),
        bbox=_json_list(event_payload.get("bbox")),
        keypoints=_keypoints(event_payload),
        ai_latency_ms=_optional_int(event_payload.get("aiLatencyMs")),
        publish_latency_ms=_optional_int(event_payload.get("publishLatencyMs")),
        snapshot_path=_optional_str(event_payload.get("snapshot_path", event_payload.get("snapshotUrl"))),
        clip_path=_optional_str(event_payload.get("clip_path", event_payload.get("clipUrl"))),
        feedback_outcome=outcome.value,
        candidate_type=candidate_type_for_outcome(outcome),
        operator_id=operator_id,
        feedback_text=feedback_text,
        feedback_timestamp_ms=feedback_timestamp_ms if feedback_timestamp_ms is not None else _current_timestamp_ms(),
        source_type="operator_feedback",
        schema_version=SCHEMA_VERSION,
        privacy_note=PRIVACY_NOTE,
    )


def _current_timestamp_ms() -> int:
    return int(time.time() * 1000)


def _event_id(event_payload: EventPayload, timestamp_ms: int | None) -> str:
    explicit = _optional_str(event_payload.get("eventId", event_payload.get("event_id")))
    if explicit is not None:
        return explicit
    camera_login_id = _camera_login_id(event_payload)
    timestamp_part = "unknown" if timestamp_ms is None else str(timestamp_ms)
    return f"{camera_login_id}:{timestamp_part}"


def _camera_login_id(event_payload: EventPayload) -> str:
    value = event_payload.get("cameraLoginId", event_payload.get("camera_login_id", event_payload.get("camera_id")))
    text = _optional_str(value)
    return text or "unknown-camera"


def _timestamp_ms(event_payload: EventPayload) -> int | None:
    timestamp_ms = _optional_int(event_payload.get("timestampMs"))
    if timestamp_ms is not None:
        return timestamp_ms
    timestamp = event_payload.get("timestamp")
    match timestamp:
        case int() | float():
            return int(timestamp if timestamp > 10_000_000_000 else timestamp * 1000)
        case str():
            return None
        case None:
            return None
        case unreachable:
            assert_never(unreachable)


def _keypoints(event_payload: EventPayload) -> list[JsonValue]:
    direct = _json_list(event_payload.get("keypoints"))
    if direct:
        return direct
    events = event_payload.get("events")
    if not events:
        return []
    first_event = events[0]
    return _json_list(first_event.get("keypoints"))


def _json_list(value: JsonValue | list[JsonValue] | None) -> list[JsonValue]:
    if isinstance(value, list):
        return list(value)
    return []


def _optional_int(value: JsonValue | None) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float | str):
        return int(float(value))
    return None


def _optional_float(value: JsonValue | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float | str):
        return float(value)
    return None


def _optional_str(value: JsonValue | None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None

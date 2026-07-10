from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeAlias, assert_never

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]

EMBEDDING_DIMENSION: Final = 768


class EventType(StrEnum):
    NEW_FALL = "NEW_FALL"
    FAINT_SUSPECTED = "FAINT_SUSPECTED"
    FALL_UNRECOVERED = "FALL_UNRECOVERED"
    RECOVERED = "RECOVERED"


@dataclass(frozen=True, slots=True)
class MockEvent:
    event_id: str
    event_type: EventType
    timestamp: str


@dataclass(frozen=True, slots=True)
class MockVlmJob:
    job_id: str
    incident_id: str
    original_event_id: str
    camera_login_id: str
    location_name: str
    events: tuple[MockEvent, ...]


def parse_job(payload: dict[str, JSONValue]) -> MockVlmJob:
    events_value = payload.get("events")
    if not isinstance(events_value, list):
        raise KeyError("events")
    return MockVlmJob(
        job_id=string_field(payload, "jobId"),
        incident_id=string_field(payload, "incidentId"),
        original_event_id=string_field(payload, "originalEventId"),
        camera_login_id=string_field(payload, "cameraLoginId"),
        location_name=string_field(payload, "locationName"),
        events=tuple(parse_event(event) for event in events_value),
    )


def build_vlm_result(job: MockVlmJob) -> dict[str, JSONValue]:
    recovered = contains(job, EventType.RECOVERED)
    unrecovered = contains(job, EventType.FALL_UNRECOVERED)
    faint_suspected = contains(job, EventType.FAINT_SUSPECTED)
    movement = "low" if faint_suspected or unrecovered else "medium"
    result = {
        "schemaVersion": "incident-v1",
        "incidentId": job.incident_id,
        "summary": summary(recovered, unrecovered),
        "observedAction": "fall_and_remain_lying" if unrecovered else "fall_then_recover",
        "preEventActivity": "walking",
        "postEventPosture": "lying" if unrecovered else "standing_after_recovery",
        "movementAfterEvent": movement,
        "recoveryObserved": recovered,
        "estimatedLyingDurationSec": 18.0 if faint_suspected or unrecovered else 4.0,
        "riskLevel": "CRITICAL" if unrecovered else "HIGH",
        "personDescription": {"topColor": "black", "bottomColor": "dark"},
        "uncertainty": [],
        "evidenceFrameIndexes": [2, 3, 4, 6, 7],
    }
    result["searchDocument"] = build_search_document(job, result)
    result["embedding"] = deterministic_embedding(str(result["searchDocument"]))
    return result


def deterministic_embedding(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    for token in re.findall(r"[\w가-힣]+", text.casefold()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
        vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = sum(value * value for value in vector) ** 0.5
    return vector if norm == 0.0 else [value / norm for value in vector]


def parse_event(value: JSONValue) -> MockEvent:
    if not isinstance(value, dict):
        raise KeyError("event")
    event_type = EventType(string_field(value, "eventType"))
    return MockEvent(
        event_id=string_field(value, "eventId"),
        event_type=event_type,
        timestamp=string_field(value, "timestamp"),
    )


def string_field(payload: dict[str, JSONValue], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise KeyError(key)
    return value


def contains(job: MockVlmJob, event_type: EventType) -> bool:
    return any(event.event_type == event_type for event in job.events)


def summary(recovered: bool, unrecovered: bool) -> str:
    if unrecovered:
        return "보행 중 쓰러진 뒤 일정 시간 움직임이 낮고 회복이 관찰되지 않았습니다."
    if recovered:
        return "보행 중 쓰러진 뒤 일정 시간 움직임이 낮았으나 이후 회복했습니다."
    return "보행 중 쓰러진 뒤 낮은 움직임이 관찰되어 추가 확인이 필요합니다."


def build_search_document(job: MockVlmJob, result: dict[str, JSONValue]) -> str:
    event_types = ", ".join(event.event_type.value for event in job.events)
    recovery_text = "recovered" if result["recoveryObserved"] is True else "not recovered"
    return "\n".join(
        [
            f"장소: {job.location_name}",
            f"카메라: {job.camera_login_id}",
            f"원본 이벤트: {job.original_event_id}",
            f"이벤트 타임라인: {event_types}",
            f"사고 후 자세: {result['postEventPosture']}",
            f"움직임: {result['movementAfterEvent']}",
            f"회복 여부: {recovery_text}",
            f"요약: {result['summary']}",
        ]
    )


def job_to_json(job: MockVlmJob) -> str:
    return json.dumps(build_vlm_result(job), ensure_ascii=False, separators=(",", ":"))


def describe_event_type(event_type: EventType) -> str:
    match event_type:
        case EventType.NEW_FALL:
            return "fall started"
        case EventType.FAINT_SUSPECTED:
            return "low movement after fall"
        case EventType.FALL_UNRECOVERED:
            return "fall remained unrecovered"
        case EventType.RECOVERED:
            return "recovery observed"
        case unreachable:
            assert_never(unreachable)

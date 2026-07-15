"""Strict contracts shared by the incident VLM extraction and provider paths."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ai.vlm.keyframe_extractor import KEYFRAME_COUNT, ExtractedKeyframe


VLM_RESULT_SCHEMA_VERSION = "vlm-result-v1"
VLM_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "incident_id",
        "visual_event_type",
        "people_count",
        "korean_search_keywords",
        "detailed_description_ko",
        "frame_count",
        "provider",
        "is_mock",
    }
)
_FORBIDDEN_INFERENCE = re.compile(
    r"\b(?:identity|identified as|name is|age|aged|gender|male|female|race|ethnicity|"
    r"diagnos(?:is|ed)|medical condition|heart attack|stroke)\b|"
    r"(?:신원|실명|이름은|나이|연령|성별|남성|여성|인종|민족|진단|질환|심장마비|뇌졸중)",
    re.IGNORECASE,
)


class VlmContractError(ValueError):
    """Raised when frames or provider output violate the public VLM contract."""


def validate_keyframes(frames: Sequence[ExtractedKeyframe]) -> None:
    if len(frames) != KEYFRAME_COUNT:
        raise VlmContractError("exactly eight keyframes are required")

    hashes: set[str] = set()
    previous_timestamp = -1.0
    previous_frame_index = -1
    for expected_index, frame in enumerate(frames):
        if isinstance(frame.index, bool) or not isinstance(frame.index, int) or frame.index != expected_index:
            raise VlmContractError("keyframe indices must be contiguous and ordered")
        if (
            isinstance(frame.timestamp_sec, bool)
            or not isinstance(frame.timestamp_sec, (int, float))
            or not math.isfinite(frame.timestamp_sec)
            or frame.timestamp_sec < 0
        ):
            raise VlmContractError("keyframe timestamps must be finite and nonnegative")
        if frame.timestamp_sec <= previous_timestamp:
            raise VlmContractError("keyframe timestamps must be strictly increasing")
        if (
            isinstance(frame.frame_index, bool)
            or not isinstance(frame.frame_index, int)
            or frame.frame_index < 0
        ):
            raise VlmContractError("frame indices must be nonnegative integers")
        if frame.frame_index <= previous_frame_index:
            raise VlmContractError("frame indices must be strictly increasing")
        if (
            isinstance(frame.width, bool)
            or not isinstance(frame.width, int)
            or isinstance(frame.height, bool)
            or not isinstance(frame.height, int)
            or frame.width <= 0
            or frame.height <= 0
        ):
            raise VlmContractError("keyframe dimensions must be positive integers")
        if (
            not isinstance(frame.jpeg_bytes, bytes)
            or not frame.jpeg_bytes.startswith(b"\xff\xd8")
            or not frame.jpeg_bytes.endswith(b"\xff\xd9")
        ):
            raise VlmContractError("keyframe payload must be JPEG")
        if not isinstance(frame.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", frame.sha256):
            raise VlmContractError("keyframe SHA-256 is invalid")
        if frame.sha256 != hashlib.sha256(frame.jpeg_bytes).hexdigest():
            raise VlmContractError("keyframe SHA-256 does not match JPEG payload")
        if frame.sha256 in hashes:
            raise VlmContractError("keyframe payloads must be unique")
        hashes.add(frame.sha256)
        previous_timestamp = frame.timestamp_sec
        previous_frame_index = frame.frame_index


def validate_vlm_result(result: Mapping[str, Any]) -> None:
    """Validate the exact ``vlm-result-v1`` provider response schema."""

    if not isinstance(result, Mapping):
        raise VlmContractError("VLM result must be an object")
    fields = set(result)
    missing = VLM_RESULT_FIELDS - fields
    unknown = fields - VLM_RESULT_FIELDS
    if missing:
        raise VlmContractError(f"VLM result is missing required fields: {sorted(missing)}")
    if unknown:
        raise VlmContractError(
            f"VLM result contains unknown fields: {sorted(unknown, key=str)}"
        )
    if result["schema_version"] != VLM_RESULT_SCHEMA_VERSION:
        raise VlmContractError("schema_version must be vlm-result-v1")

    _required_text(result["incident_id"], "incident_id")
    frame_count = result["frame_count"]
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count != KEYFRAME_COUNT
    ):
        raise VlmContractError("frame_count must equal 8")
    if result["provider"] != "mock":
        raise VlmContractError("provider must be mock")
    if result["is_mock"] is not True:
        raise VlmContractError("is_mock must be true")

    event_type = _required_text(result["visual_event_type"], "visual_event_type")
    description = _required_text(result["detailed_description_ko"], "detailed_description_ko")

    people_count = result["people_count"]
    if isinstance(people_count, bool) or not isinstance(people_count, int) or people_count < 0:
        raise VlmContractError("people_count must be a nonnegative integer")

    keywords = result["korean_search_keywords"]
    if isinstance(keywords, (str, bytes)) or not isinstance(keywords, Sequence) or not keywords:
        raise VlmContractError("korean_search_keywords must be a nonempty array")
    normalized_keywords: list[str] = []
    for keyword in keywords:
        normalized = _required_text(keyword, "korean_search_keywords item")
        if len(normalized) > 100:
            raise VlmContractError("korean_search_keywords items must not exceed 100 characters")
        normalized_keywords.append(normalized)
    if len(set(normalized_keywords)) != len(normalized_keywords):
        raise VlmContractError("korean_search_keywords must be unique")

    inference_text = " ".join([event_type, description, *normalized_keywords])
    if _FORBIDDEN_INFERENCE.search(inference_text):
        raise VlmContractError("VLM result contains forbidden identity, demographic, or medical inference")


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VlmContractError(f"{field_name} must be a nonempty string")
    return value.strip()

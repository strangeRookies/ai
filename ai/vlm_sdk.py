"""VLM provider adapters — direct SDK/HTTP only, no LangChain."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ai.vlm.contracts import VLM_RESULT_SCHEMA_VERSION, validate_vlm_result
from ai.vlm.keyframe_extractor import KEYFRAME_COUNT


@dataclass(frozen=True, slots=True)
class VlmFramePayload:
    index: int
    timestamp_sec: float
    jpeg_bytes: bytes


@dataclass(frozen=True, slots=True)
class VlmAnalyzeRequest:
    frames: tuple[VlmFramePayload, ...]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class VlmAnalyzeResult:
    schema_version: str
    incident_id: str
    visual_event_type: str
    people_count: int
    korean_search_keywords: tuple[str, ...]
    detailed_description_ko: str
    frame_count: int
    provider: str
    is_mock: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "incident_id": self.incident_id,
            "visual_event_type": self.visual_event_type,
            "people_count": self.people_count,
            "korean_search_keywords": list(self.korean_search_keywords),
            "detailed_description_ko": self.detailed_description_ko,
            "frame_count": self.frame_count,
            "provider": self.provider,
            "is_mock": self.is_mock,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> VlmAnalyzeResult:
        validate_vlm_result(value)
        return cls(
            schema_version=str(value["schema_version"]),
            incident_id=str(value["incident_id"]),
            visual_event_type=str(value["visual_event_type"]),
            people_count=int(value["people_count"]),
            korean_search_keywords=tuple(value["korean_search_keywords"]),
            detailed_description_ko=str(value["detailed_description_ko"]),
            frame_count=int(value["frame_count"]),
            provider=str(value["provider"]),
            is_mock=bool(value["is_mock"]),
        )


class VlmProvider(Protocol):
    def analyze(self, request: VlmAnalyzeRequest) -> VlmAnalyzeResult: ...


class MockVlmProvider:
    def analyze(self, request: VlmAnalyzeRequest) -> VlmAnalyzeResult:
        _validate_request(request)
        description = (
            "안전 이벤트 영상에서 작업자 또는 사람이 감시 구역 바닥 근처에 있는 장면입니다. "
            "복도나 출입 구역으로 보이는 환경에서 안전모, 조끼, 바닥, 쓰러짐 여부를 검색할 수 있습니다."
        )
        result = VlmAnalyzeResult(
            schema_version=VLM_RESULT_SCHEMA_VERSION,
            incident_id=_incident_id(request.metadata),
            visual_event_type="person_lying_on_floor",
            people_count=1,
            korean_search_keywords=("바닥", "쓰러짐", "안전모", "조끼", "복도"),
            detailed_description_ko=description,
            frame_count=KEYFRAME_COUNT,
            provider="mock",
            is_mock=True,
        )
        validate_vlm_result(result.to_dict())
        return result


def _validate_request(request: VlmAnalyzeRequest) -> None:
    if len(request.frames) != KEYFRAME_COUNT:
        raise ValueError("exactly eight provider frames are required")
    previous_timestamp = -1.0
    digests: set[str] = set()
    for expected_index, frame in enumerate(request.frames):
        if (
            isinstance(frame.index, bool)
            or not isinstance(frame.index, int)
            or frame.index != expected_index
        ):
            raise ValueError("provider frame indices must be contiguous and ordered")
        if (
            isinstance(frame.timestamp_sec, bool)
            or not isinstance(frame.timestamp_sec, (int, float))
            or not math.isfinite(frame.timestamp_sec)
            or frame.timestamp_sec < 0
            or frame.timestamp_sec <= previous_timestamp
        ):
            raise ValueError("provider frame timestamps must be ordered and nonnegative")
        if (
            not isinstance(frame.jpeg_bytes, bytes)
            or not frame.jpeg_bytes.startswith(b"\xff\xd8")
            or not frame.jpeg_bytes.endswith(b"\xff\xd9")
        ):
            raise ValueError("provider frame payload must be JPEG")
        digest = hashlib.sha256(frame.jpeg_bytes).hexdigest()
        if digest in digests:
            raise ValueError("provider frame payloads must be unique")
        digests.add(digest)
        previous_timestamp = frame.timestamp_sec


def _incident_id(metadata: Mapping[str, object]) -> str:
    candidates = (metadata.get("incident_id"), metadata.get("incidentId"))
    incident = metadata.get("incident")
    if isinstance(incident, Mapping):
        candidates += (
            incident.get("id"),
            incident.get("incident_id"),
            incident.get("incidentId"),
        )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise ValueError("sanitized metadata must include incident_id")


class GeminiVlmProvider:
    """
    Placeholder for multimodal Gemini direct SDK calls.
    Real media download + generateContent wiring is deferred until GPU/API keys are available.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def analyze(self, request: VlmAnalyzeRequest) -> VlmAnalyzeResult:
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is required for VLM_PROVIDER=gemini")
        # Intentionally not calling network in scaffold; keep contract stable for callers.
        raise RuntimeError(
            "Gemini multimodal VLM is scaffolded only. "
            "Wire generateContent with the supplied frame payloads when keys/GPU path are ready."
        )


def resolve_vlm_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
) -> VlmProvider:
    name = (provider_name or os.getenv("VLM_PROVIDER", "mock")).strip().lower()
    key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
    mock_mode = os.getenv("VLM_MOCK_MODE", "true").lower() == "true"
    if mock_mode or name in {"", "mock"}:
        return MockVlmProvider()
    if name == "gemini":
        return GeminiVlmProvider(api_key=key)
    raise ValueError(f"unsupported VLM_PROVIDER: {name}")

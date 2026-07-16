"""VLM provider adapters — direct SDK/HTTP only, no LangChain."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ai.vlm.contracts import VLM_RESULT_SCHEMA_VERSION, validate_vlm_result
from ai.vlm.keyframe_extractor import KEYFRAME_COUNT
from ai.vlm.provider_mode import resolve_vlm_provider_name, vlm_force_mock


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
    requires_deidentified_frames: bool

    def analyze(self, request: VlmAnalyzeRequest) -> VlmAnalyzeResult: ...


class MockVlmProvider:
    requires_deidentified_frames = False
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


class GeminiTransportError(RuntimeError):
    """Sanitized Gemini transport failure safe to surface without request data."""

    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


class GeminiTransport(Protocol):
    def __call__(
        self,
        *,
        api_key: str,
        model: str,
        payload: Mapping[str, object],
        timeout_sec: float,
    ) -> Mapping[str, Any]: ...


_GEMINI_RESULT_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "schema_version": {"type": "STRING"},
        "incident_id": {"type": "STRING"},
        "visual_event_type": {"type": "STRING"},
        "people_count": {"type": "INTEGER"},
        "korean_search_keywords": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "detailed_description_ko": {"type": "STRING"},
        "frame_count": {"type": "INTEGER"},
        "provider": {"type": "STRING"},
        "is_mock": {"type": "BOOLEAN"},
    },
    "required": sorted(
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
    ),
}
_SAFE_METADATA_FIELDS = frozenset(
    {
        "incident_id",
        "camera_login_id",
        "clip_start_sec",
        "clip_end_sec",
        "scenario_type",
        "event_type",
        "severity",
        "timestamp",
    }
)


def _default_gemini_transport(
    *,
    api_key: str,
    model: str,
    payload: Mapping[str, object],
    timeout_sec: float,
) -> Mapping[str, Any]:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model, safe='')}:generateContent"
    )
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read(2_000_001)
    except urllib.error.HTTPError as exc:
        raise GeminiTransportError(
            f"Gemini request failed with HTTP {exc.code}",
            transient=exc.code in {408, 429} or 500 <= exc.code < 600,
        ) from None
    except (TimeoutError, socket.timeout):
        raise GeminiTransportError("Gemini request timed out", transient=True) from None
    except urllib.error.URLError:
        raise GeminiTransportError("Gemini network request failed", transient=True) from None

    if len(raw) > 2_000_000:
        raise GeminiTransportError("Gemini response exceeded size limit", transient=False)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise GeminiTransportError("Gemini returned malformed JSON", transient=False) from None
    if not isinstance(value, Mapping):
        raise GeminiTransportError("Gemini response must be an object", transient=False)
    return value


def _gemini_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Return the minimal non-secret incident context sent to Gemini."""

    safe: dict[str, object] = {}
    for key in _SAFE_METADATA_FIELDS:
        value = metadata.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            safe[key] = value
    safe["incident_id"] = _incident_id(metadata)
    return safe


def _gemini_payload(request: VlmAnalyzeRequest) -> dict[str, object]:
    metadata = _gemini_metadata(request.metadata)
    prompt = (
        "당신은 CCTV 안전사고 분석 보조자입니다. 제공된 8장의 시간순 JPEG만 근거로 "
        "관찰 가능한 사실을 한국어로 분석하세요. 신원, 이름, 나이, 성별, 인종, 질병이나 "
        "의학적 진단을 추론하지 마세요. 보이지 않는 사실은 만들지 마세요. "
        "schema_version은 vlm-result-v1, incident_id는 입력값, frame_count는 8, "
        "provider는 gemini, is_mock은 false로 반환하세요. 검색 키워드는 중복 없는 "
        "한국어 표현으로 작성하세요.\n사고 메타데이터: "
        + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    )
    parts: list[dict[str, object]] = [{"text": prompt}]
    for frame in request.frames:
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": base64.b64encode(frame.jpeg_bytes).decode("ascii"),
                }
            }
        )
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _GEMINI_RESULT_SCHEMA,
            "temperature": 0.1,
        },
    }


def _response_result(response: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        candidates = response["candidates"]
        text = candidates[0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise GeminiTransportError(
            "Gemini response did not contain structured content",
            transient=False,
        ) from None
    if not isinstance(text, str):
        raise GeminiTransportError("Gemini structured content must be text", transient=False)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raise GeminiTransportError("Gemini structured content was malformed", transient=False) from None
    if not isinstance(value, Mapping):
        raise GeminiTransportError("Gemini structured content must be an object", transient=False)
    return value


class GeminiVlmProvider:
    requires_deidentified_frames = True

    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        timeout_sec: float | None = None,
        max_attempts: int = 3,
        transport: GeminiTransport = _default_gemini_transport,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = api_key
        self._model = (model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
        configured_timeout = (
            timeout_sec
            if timeout_sec is not None
            else float(os.getenv("GEMINI_TIMEOUT_SEC", "30"))
        )
        if not math.isfinite(configured_timeout) or configured_timeout <= 0:
            raise ValueError("Gemini timeout must be a positive finite number")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
            raise ValueError("Gemini max_attempts must be between 1 and 5")
        if not self._model:
            raise ValueError("Gemini model must not be empty")
        self._timeout_sec = configured_timeout
        self._max_attempts = max_attempts
        self._transport = transport
        self._sleep = sleep

    def analyze(self, request: VlmAnalyzeRequest) -> VlmAnalyzeResult:
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is required for VLM_PROVIDER=gemini")
        _validate_request(request)
        payload = _gemini_payload(request)
        response: Mapping[str, Any] | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._transport(
                    api_key=self._api_key,
                    model=self._model,
                    payload=payload,
                    timeout_sec=self._timeout_sec,
                )
                break
            except GeminiTransportError as exc:
                if not exc.transient or attempt + 1 >= self._max_attempts:
                    raise
                self._sleep(0.25 * (2**attempt))
        if response is None:  # defensive; loop always returns or raises
            raise GeminiTransportError("Gemini request failed", transient=False)

        result = VlmAnalyzeResult.from_mapping(_response_result(response))
        if result.incident_id != _incident_id(request.metadata):
            raise ValueError("Gemini incident_id does not match request")
        return result


def resolve_vlm_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
) -> VlmProvider:
    try:
        resolved = resolve_vlm_provider_name(provider_name)
    except ValueError:
        if vlm_force_mock():
            return MockVlmProvider()
        raise
    key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
    if resolved == "gemini":
        return GeminiVlmProvider(api_key=key)
    if resolved == "mock":
        return MockVlmProvider()
    raise ValueError(f"unsupported VLM_PROVIDER: {resolved}")

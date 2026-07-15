"""Backend-ready, versioned VLM semantic indexing payload contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai.embedding_sdk import EMBEDDING_DIMENSION
from ai.vlm.contracts import validate_vlm_result
from ai.vlm.search_document import canonical_keywords
from ai.vlm_sdk import VlmAnalyzeResult

INDEX_PAYLOAD_SCHEMA_VERSION = "vlm-index-payload-v1"


class IndexPayloadError(ValueError):
    """Raised when an indexing envelope violates its public contract."""


@dataclass(frozen=True, slots=True)
class SearchPayload:
    document: str
    keywords: tuple[str, ...]
    embedding_model: str
    embedding_dimension: int
    embedding: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "document": self.document,
            "keywords": list(self.keywords),
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "embedding": list(self.embedding),
        }


@dataclass(frozen=True, slots=True)
class VlmIndexPayload:
    schema_version: str
    incident_id: str
    camera_login_id: str
    captured_at: str
    vlm_result: VlmAnalyzeResult
    search: SearchPayload

    def __post_init__(self) -> None:
        validate_index_payload(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "incident_id": self.incident_id,
            "camera_login_id": self.camera_login_id,
            "captured_at": self.captured_at,
            "vlm_result": self.vlm_result.to_dict(),
            "search": self.search.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )


def validate_index_payload(payload: VlmIndexPayload) -> None:
    if payload.schema_version != INDEX_PAYLOAD_SCHEMA_VERSION:
        raise IndexPayloadError("schema_version must be vlm-index-payload-v1")
    incident_id = _required_text(payload.incident_id, "incident_id")
    _required_text(payload.camera_login_id, "camera_login_id")
    _validate_captured_at(payload.captured_at)

    if not isinstance(payload.vlm_result, VlmAnalyzeResult):
        raise IndexPayloadError("vlm_result must be a VlmAnalyzeResult")
    try:
        validate_vlm_result(payload.vlm_result.to_dict())
    except ValueError as exc:
        raise IndexPayloadError("vlm_result is invalid") from exc
    if payload.vlm_result.incident_id != incident_id:
        raise IndexPayloadError("envelope and VLM incident IDs must match")

    search = payload.search
    if not isinstance(search, SearchPayload):
        raise IndexPayloadError("search must be a SearchPayload")
    _required_text(search.document, "search.document")
    model = _required_text(search.embedding_model, "search.embedding_model")
    try:
        keywords = canonical_keywords(search.keywords)
    except ValueError as exc:
        raise IndexPayloadError("search.keywords are invalid") from exc
    if keywords != payload.vlm_result.korean_search_keywords:
        raise IndexPayloadError("search keywords must match the VLM result")

    dimension = search.embedding_dimension
    if dimension != EMBEDDING_DIMENSION:
        raise IndexPayloadError(f"embedding_dimension must equal {EMBEDDING_DIMENSION}")
    if not isinstance(search.embedding, tuple) or len(search.embedding) != dimension:
        raise IndexPayloadError("embedding length must match embedding_dimension")
    for value in search.embedding:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise IndexPayloadError("embedding values must be finite numbers")

    embedding_is_mock = model.startswith("mock-")
    if payload.vlm_result.is_mock != embedding_is_mock:
        raise IndexPayloadError("VLM and embedding provider modes must match")
    if not embedding_is_mock and not model.startswith("gemini-"):
        raise IndexPayloadError("real embedding_model must identify Gemini")


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IndexPayloadError(f"{field_name} must be a nonempty string")
    return value.strip()


def _validate_captured_at(value: object) -> None:
    captured_at = _required_text(value, "captured_at")
    try:
        parsed = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError:
        raise IndexPayloadError("captured_at must be an ISO 8601 timestamp") from None
    if parsed.utcoffset() is None:
        raise IndexPayloadError("captured_at must include a timezone")

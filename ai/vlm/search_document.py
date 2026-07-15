"""Deterministic semantic document construction for validated VLM results."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ai.vlm.contracts import VlmContractError, validate_vlm_result

_MAX_EVENT_TYPE = 200
_MAX_DESCRIPTION = 2_000
_MAX_KEYWORDS = 32
_MAX_KEYWORD = 100
_MAX_METADATA_VALUE = 100
_MAX_DOCUMENT = 4_096
_HANGUL = re.compile(r"[가-힣]")
_SENSITIVE_TEXT = re.compile(
    r"(?:api[_-]?key|authorization|bearer\s+[A-Za-z0-9._~+\-/]+=*|client[_-]?secret|"
    r"private[_-]?key|password|refresh[_-]?token|session[_-]?key|\[REDACTED\])",
    re.IGNORECASE,
)


class SearchDocumentError(VlmContractError):
    """Raised when semantic text is unsafe or cannot be indexed deterministically."""


def build_search_document(result: Any, metadata: Mapping[str, object]) -> str:
    """Build the canonical document without duplicating envelope identifiers.

    Only ``event_type`` and ``severity`` are accepted from metadata. Incident,
    camera, timestamp, URLs, credentials, and arbitrary metadata stay outside
    semantic text.
    """

    values = _result_mapping(result)
    validate_vlm_result(values)
    if not isinstance(metadata, Mapping):
        raise SearchDocumentError("search metadata must be an object")

    visual_event_type = _bounded_text(
        values["visual_event_type"], "visual_event_type", _MAX_EVENT_TYPE
    )
    description = _bounded_text(
        values["detailed_description_ko"], "detailed_description_ko", _MAX_DESCRIPTION
    )
    people_count = values["people_count"]
    if (
        isinstance(people_count, bool)
        or not isinstance(people_count, int)
        or people_count < 0
        or people_count > 1_000_000
    ):
        raise SearchDocumentError("people_count must be a bounded nonnegative integer")

    keywords = canonical_keywords(values["korean_search_keywords"])
    lines = [
        f"visual_event_type: {visual_event_type}",
        f"detailed_description_ko: {description}",
        f"people_count: {people_count}",
        f"korean_search_keywords: {', '.join(keywords)}",
    ]
    for field_name in ("event_type", "severity"):
        value = metadata.get(field_name)
        if value is None:
            continue
        normalized = _bounded_text(value, f"metadata.{field_name}", _MAX_METADATA_VALUE)
        lines.append(f"{field_name}: {normalized}")

    document = "\n".join(lines)
    if len(document) > _MAX_DOCUMENT:
        raise SearchDocumentError("canonical search document exceeds size limit")
    _reject_sensitive(document)
    return document


def canonical_keywords(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise SearchDocumentError("korean_search_keywords must be a nonempty array")
    if len(value) > _MAX_KEYWORDS:
        raise SearchDocumentError("korean_search_keywords exceeds item limit")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        keyword = _bounded_text(item, "korean_search_keywords item", _MAX_KEYWORD)
        if not _HANGUL.search(keyword):
            raise SearchDocumentError("korean_search_keywords items must contain Korean text")
        if keyword in seen:
            raise SearchDocumentError("korean_search_keywords must be unique")
        seen.add(keyword)
        result.append(keyword)
    return tuple(result)


def _result_mapping(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    raise SearchDocumentError("VLM result must be an object")


def _bounded_text(value: object, field_name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SearchDocumentError(f"{field_name} must be a nonempty string")
    normalized = " ".join(value.split())
    if len(normalized) > limit:
        raise SearchDocumentError(f"{field_name} exceeds size limit")
    _reject_sensitive(normalized)
    return normalized


def _reject_sensitive(value: str) -> None:
    if _SENSITIVE_TEXT.search(value):
        raise SearchDocumentError("semantic text contains sensitive content")

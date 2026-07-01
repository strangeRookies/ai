from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from ai.evidence import evidence_id, latency_order_valid

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def add_evidence_fields(
    payload: dict[str, JsonValue],
    stream_id: str,
    frame_id: int | None,
    captured_at_ms: int | None,
    processed_at_ms: int | None,
    published_at_ms: int | None,
    dropped_frame_count: int | None = None,
    snapshot_path: str | None = None,
    clip_path: str | None = None,
) -> None:
    if dropped_frame_count is None and snapshot_path is None and clip_path is None:
        return
    if frame_id is None or captured_at_ms is None:
        return
    evidence_key = evidence_id(stream_id, frame_id, captured_at_ms)
    evidence: dict[str, JsonValue] = {
        "evidenceId": evidence_key,
        "traceId": evidence_key,
        "cameraLoginId": stream_id,
        "frameId": int(frame_id),
        "timestampMs": int(captured_at_ms),
        "capturedAtMs": int(captured_at_ms),
        "processedAtMs": _optional_int(processed_at_ms),
        "publishedAtMs": _optional_int(published_at_ms),
        "latency": _latency(captured_at_ms, processed_at_ms, published_at_ms),
        "droppedFrameCount": int(dropped_frame_count or 0),
        "latencyOrderValid": latency_order_valid(captured_at_ms, processed_at_ms, published_at_ms),
    }
    if snapshot_path is not None:
        evidence["snapshotPath"] = snapshot_path
    if clip_path is not None:
        evidence["clipPath"] = clip_path
    payload["evidenceId"] = evidence_key
    payload["traceId"] = evidence_key
    payload["evidence"] = evidence
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {}
    metadata["evidenceId"] = evidence_key
    if snapshot_path is not None:
        metadata["snapshotPath"] = snapshot_path
    if clip_path is not None:
        metadata["clipPath"] = clip_path
    payload["metadata"] = metadata


def _latency(captured_at_ms: int, processed_at_ms: int | None, published_at_ms: int | None) -> dict[str, JsonValue]:
    return {
        "aiLatencyMs": max(0, int(processed_at_ms) - int(captured_at_ms))
        if processed_at_ms is not None
        else None,
        "publishLatencyMs": max(0, int(published_at_ms) - int(captured_at_ms))
        if published_at_ms is not None
        else None,
    }


def _optional_int(value: int | None) -> int | None:
    return int(value) if value is not None else None

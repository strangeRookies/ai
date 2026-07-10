from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VlmEvidence:
    event_id: str
    camera_login_id: str
    timestamp_ms: int
    snapshot_path: str | None
    clip_path: str | None
    detected_type: str
    confidence: float | None


@dataclass(frozen=True, slots=True)
class VlmDescription:
    event_id: str
    camera_login_id: str
    adapter_name: str
    description: str
    final_decision: bool


class VlmAdapter(Protocol):
    def describe(self, evidence: VlmEvidence) -> VlmDescription: ...


class MockVlmAdapter:
    adapter_name = "mock-vlm"

    def describe(self, evidence: VlmEvidence) -> VlmDescription:
        media = _media_summary(evidence.snapshot_path, evidence.clip_path)
        confidence = "unknown" if evidence.confidence is None else f"{evidence.confidence:.2f}"
        return VlmDescription(
            event_id=evidence.event_id,
            camera_login_id=evidence.camera_login_id,
            adapter_name=self.adapter_name,
            description=(
                "operator-assist mock description: "
                f"{media}; detected_type={evidence.detected_type}; confidence={confidence}; "
                "use only for explanation, labeling support, and FP/FN review."
            ),
            final_decision=False,
        )


def _media_summary(snapshot_path: str | None, clip_path: str | None) -> str:
    parts: list[str] = []
    if snapshot_path:
        parts.append(f"snapshot={snapshot_path}")
    if clip_path:
        parts.append(f"clip={clip_path}")
    if not parts:
        return "no saved media"
    return ", ".join(parts)

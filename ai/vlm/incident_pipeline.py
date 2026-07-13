"""Incident-centric VLM job skeleton without real Vision API or media.

Mock/offline path only — not real VLM understanding performance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class IncidentEventType(StrEnum):
    NEW_FALL = "NEW_FALL"
    FAINT_SUSPECTED = "FAINT_SUSPECTED"
    FALL_UNRECOVERED = "FALL_UNRECOVERED"
    RECOVERED = "RECOVERED"
    NORMAL = "NORMAL"


class VlmJobPolicy(StrEnum):
    FINAL_ONLY = "FINAL_ONLY"
    EVERY_EVENT = "EVERY_EVENT"


class IncidentTerminalStatus(StrEnum):
    """Terminal states when stream ends while incident is still open."""

    INTERRUPTED = "INTERRUPTED"
    STREAM_LOST = "STREAM_LOST"
    CLOSED_UNKNOWN = "CLOSED_UNKNOWN"


DEFAULT_KEYFRAME_OFFSETS_SEC = (-2.0, -1.0, -0.3, 0.0, 0.5, 2.0, 5.0)


@dataclass(frozen=True, slots=True)
class IncidentEvent:
    event_type: IncidentEventType
    timestamp_sec: float
    event_id: str


@dataclass(frozen=True, slots=True)
class Incident:
    incident_id: str
    camera_login_id: str
    original_event_id: str
    events: tuple[IncidentEvent, ...]
    clip_start_sec: float = 0.0
    clip_end_sec: float | None = None
    fps: float | None = None


@dataclass
class VlmJob:
    job_id: str
    incident_id: str
    camera_login_id: str
    status: str = "PENDING"
    keyframe_timestamps_sec: list[float] = field(default_factory=list)
    keyframe_metadata: list[dict[str, Any]] = field(default_factory=list)
    deidentified: bool = False
    structured_result: dict[str, Any] | None = None
    search_document: str | None = None
    error: str | None = None
    vlm_call_count: int = 0


class DeidentificationGate(Protocol):
    def deidentify(self, keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return de-identified keyframe metadata or raise on failure."""


class PassThroughDeid:
    def deidentify(self, keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(item, deidentified=True) for item in keyframes]


class FailingDeid:
    def deidentify(self, keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise RuntimeError("de-identification failed")


class MockStructuredVlm:
    def analyze(self, incident: Incident, keyframes: list[dict[str, Any]]) -> dict[str, Any]:
        types = [event.event_type.value for event in incident.events]
        recovered = IncidentEventType.RECOVERED in {event.event_type for event in incident.events}
        unrecovered = IncidentEventType.FALL_UNRECOVERED in {event.event_type for event in incident.events}
        return {
            "schemaVersion": "incident-v1",
            "incidentId": incident.incident_id,
            "cameraLoginId": incident.camera_login_id,
            "timeline": types,
            "recoveryObserved": recovered,
            "riskLevel": "CRITICAL" if unrecovered else ("HIGH" if recovered or types else "MEDIUM"),
            "keyframeCount": len(keyframes),
            "summary": "mock structured VLM result (not real vision analysis)",
        }


def is_analysis_eligible(incident: Incident, *, policy: VlmJobPolicy = VlmJobPolicy.FINAL_ONLY) -> bool:
    if not incident.events:
        return False
    types = {event.event_type for event in incident.events}
    if policy == VlmJobPolicy.EVERY_EVENT:
        return True
    # FINAL_ONLY: wait until recovered or unrecovered terminal signal, or only NEW_FALL present with clip end
    if IncidentEventType.RECOVERED in types or IncidentEventType.FALL_UNRECOVERED in types:
        return True
    if IncidentEventType.NEW_FALL in types and incident.clip_end_sec is not None:
        return True
    return False


def compute_keyframe_timestamps(
    incident: Incident,
    *,
    offsets_sec: tuple[float, ...] = DEFAULT_KEYFRAME_OFFSETS_SEC,
) -> list[float]:
    """Event-aware keyframe times in clip coordinates; clamps to clip range; de-duplicates."""
    if not incident.events:
        return []
    t0 = float(incident.events[0].timestamp_sec)
    clip_start = float(incident.clip_start_sec)
    clip_end = float(incident.clip_end_sec) if incident.clip_end_sec is not None else t0 + 10.0
    if clip_end < clip_start:
        clip_start, clip_end = clip_end, clip_start

    recovered_ts = next(
        (float(event.timestamp_sec) for event in incident.events if event.event_type == IncidentEventType.RECOVERED),
        None,
    )
    stamps: list[float] = []
    for offset in offsets_sec:
        stamps.append(t0 + float(offset))
    if recovered_ts is not None:
        stamps.append(recovered_ts)
    else:
        stamps.append(clip_end)

    clamped = [min(max(value, clip_start), clip_end) for value in stamps]
    # de-dupe while preserving order (round for float stability)
    unique: list[float] = []
    seen: set[float] = set()
    for value in clamped:
        key = round(value, 4)
        if key in seen:
            continue
        seen.add(key)
        unique.append(float(key))
    return unique


def mock_keyframe_metadata(
    timestamps_sec: list[float],
    *,
    fps: float | None,
    incident_id: str,
) -> list[dict[str, Any]]:
    meta: list[dict[str, Any]] = []
    for index, ts in enumerate(timestamps_sec):
        frame_index = None
        if fps is not None and fps > 0:
            frame_index = int(round(ts * float(fps)))
        meta.append(
            {
                "index": index,
                "timestampSec": ts,
                "frameIndex": frame_index,
                "mockMediaKey": f"mock://{incident_id}/kf_{index}.jpg",
                "fps": fps,
            }
        )
    return meta


def validate_incident_v1(result: dict[str, Any]) -> None:
    required = ["schemaVersion", "incidentId", "summary", "timeline", "recoveryObserved", "riskLevel"]
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"invalid incident-v1 schema, missing: {missing}")
    if result.get("schemaVersion") != "incident-v1":
        raise ValueError("schemaVersion must be incident-v1")


def build_search_document(incident: Incident, result: dict[str, Any]) -> str:
    timeline = ", ".join(result.get("timeline") or [])
    return "\n".join(
        [
            f"incident: {incident.incident_id}",
            f"camera: {incident.camera_login_id}",
            f"timeline: {timeline}",
            f"risk: {result.get('riskLevel')}",
            f"recovery: {result.get('recoveryObserved')}",
            f"summary: {result.get('summary')}",
        ]
    )


class IncidentVlmPipeline:
    def __init__(
        self,
        *,
        policy: VlmJobPolicy = VlmJobPolicy.FINAL_ONLY,
        deid: DeidentificationGate | None = None,
        vlm: MockStructuredVlm | None = None,
    ) -> None:
        self.policy = policy
        self.deid = deid or PassThroughDeid()
        self.vlm = vlm or MockStructuredVlm()
        self._jobs_by_incident: dict[str, VlmJob] = {}
        self._open_incidents: dict[str, Incident] = {}
        self._terminal_status: dict[str, IncidentTerminalStatus] = {}
        self.vlm_calls = 0

    def get_job(self, incident_id: str) -> VlmJob | None:
        return self._jobs_by_incident.get(incident_id)

    def register_open_incident(self, incident: Incident) -> None:
        """Track an in-flight incident until terminal recovery/unrecovered or stream reset."""
        if incident.incident_id in self._terminal_status:
            return
        self._open_incidents[incident.incident_id] = incident

    def open_incident_ids(self) -> list[str]:
        return list(self._open_incidents.keys())

    def terminal_status(self, incident_id: str) -> IncidentTerminalStatus | None:
        return self._terminal_status.get(incident_id)

    def on_stream_reset(
        self,
        *,
        reason: str = "stream_reconnect",
        camera_login_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Close open incidents when analysis stream resets mid-incident.

        Policy:
        - STREAM_RECONNECT / large gap → STREAM_LOST
        - VIDEO_EOF without recovery → INTERRUPTED
        - other / unknown → CLOSED_UNKNOWN
        Does not call real VLM.
        """
        reason_l = str(reason or "").lower()
        if "reconnect" in reason_l or "stream_lost" in reason_l or "large_" in reason_l:
            status = IncidentTerminalStatus.STREAM_LOST
        elif "eof" in reason_l or "video_eof" in reason_l or "interrupted" in reason_l:
            status = IncidentTerminalStatus.INTERRUPTED
        else:
            status = IncidentTerminalStatus.CLOSED_UNKNOWN

        closed: list[dict[str, Any]] = []
        for incident_id, incident in list(self._open_incidents.items()):
            if camera_login_id is not None and incident.camera_login_id != camera_login_id:
                continue
            # Already terminal recovered/unrecovered — leave as-is if a SUCCESS job exists
            existing = self._jobs_by_incident.get(incident_id)
            if existing is not None and existing.status == "SUCCESS":
                self._open_incidents.pop(incident_id, None)
                continue
            self._terminal_status[incident_id] = status
            job = VlmJob(
                job_id=f"job-term-{incident_id[:12]}",
                incident_id=incident_id,
                camera_login_id=incident.camera_login_id,
                status=status.value,
                error=f"stream reset: {reason}",
            )
            self._jobs_by_incident[incident_id] = job
            self._open_incidents.pop(incident_id, None)
            closed.append(
                {
                    "incidentId": incident_id,
                    "cameraLoginId": incident.camera_login_id,
                    "status": status.value,
                    "reason": reason,
                }
            )
        return closed

    def process(self, incident: Incident) -> VlmJob:
        existing = self._jobs_by_incident.get(incident.incident_id)
        if existing is not None:
            # prevent duplicate jobs for same incident
            return existing
        term = self._terminal_status.get(incident.incident_id)
        if term is not None:
            return VlmJob(
                job_id=f"job-term-{incident.incident_id[:12]}",
                incident_id=incident.incident_id,
                camera_login_id=incident.camera_login_id,
                status=term.value,
                error="incident already terminal after stream reset",
            )
        if not is_analysis_eligible(incident, policy=self.policy):
            self.register_open_incident(incident)
            job = VlmJob(
                job_id=f"job-ineligible-{incident.incident_id}",
                incident_id=incident.incident_id,
                camera_login_id=incident.camera_login_id,
                status="INELIGIBLE",
                error="analysis not eligible yet",
            )
            return job

        if self.policy == VlmJobPolicy.FINAL_ONLY and self.vlm_calls >= 1:
            job = VlmJob(
                job_id=f"job-skip-{incident.incident_id}",
                incident_id=incident.incident_id,
                camera_login_id=incident.camera_login_id,
                status="SKIPPED",
                error="FINAL_ONLY already consumed or not eligible",
            )
            return job

        job_id = "job-" + hashlib.sha1(incident.incident_id.encode("utf-8")).hexdigest()[:12]
        job = VlmJob(job_id=job_id, incident_id=incident.incident_id, camera_login_id=incident.camera_login_id)
        self._jobs_by_incident[incident.incident_id] = job
        self.register_open_incident(incident)

        stamps = compute_keyframe_timestamps(incident)
        job.keyframe_timestamps_sec = stamps
        job.keyframe_metadata = mock_keyframe_metadata(stamps, fps=incident.fps, incident_id=incident.incident_id)

        try:
            deid_frames = self.deid.deidentify(job.keyframe_metadata)
            job.deidentified = True
        except Exception as exc:
            job.status = "BLOCKED_DEID"
            job.error = str(exc)
            job.deidentified = False
            return job

        # de-id failure already returned; only call VLM when deidentified
        if self.policy == VlmJobPolicy.FINAL_ONLY and self.vlm_calls >= 1:
            job.status = "SKIPPED"
            job.error = "FINAL_ONLY max one VLM call"
            return job

        result = self.vlm.analyze(incident, deid_frames)
        job.vlm_call_count += 1
        self.vlm_calls += 1
        validate_incident_v1(result)
        job.structured_result = result
        job.search_document = build_search_document(incident, result)
        job.status = "SUCCESS"
        self._open_incidents.pop(incident.incident_id, None)
        return job

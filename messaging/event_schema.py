from datetime import datetime, timezone
from uuid import uuid4


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_safety_event(
    event_type,
    camera_id,
    severity,
    message,
    source="edge-ai",
    track_id=None,
    metadata=None,
    timestamp=None,
    status="confirmed",
    confidence=None,
    bbox=None,
    model=None,
    evidence=None,
    event_id=None,
    schema_version="1.0",
):
    event = {
        "schema_version": schema_version,
        "event_id": event_id or str(uuid4()),
        "type": event_type,
        "event_type": event_type,
        "camera_id": camera_id,
        "timestamp": timestamp or utc_now_iso(),
        "status": status,
        "severity": severity,
        "message": message,
        "source": source,
    }
    if track_id is not None:
        event["track_id"] = int(track_id)
    if confidence is not None:
        event["confidence"] = float(confidence)
    if bbox is not None:
        event["bbox"] = bbox
    if model is not None:
        event["model"] = model
    if evidence is not None:
        event["evidence"] = evidence
    if metadata is not None:
        event["metadata"] = metadata
    return event

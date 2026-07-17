"""Primary snapshot capture + S3 upload (decoupled from VLM).

Contract:
- SNAPSHOT_CAPTURE_ENABLED (default true)
- SNAPSHOT_UPLOAD_ENABLED (default true)
- Keys: snapshots/{eventId}.jpg
- Never emits local paths or clip keys as snapshot_object_key
- Upload failures are non-fatal to callers
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ai.storage.uploader import resolve_s3_bucket_name


def snapshot_capture_enabled() -> bool:
    val = os.getenv("SNAPSHOT_CAPTURE_ENABLED", "true").strip().lower()
    return val in {"1", "true", "yes", "on"}


def snapshot_upload_enabled() -> bool:
    val = os.getenv("SNAPSHOT_UPLOAD_ENABLED", "true").strip().lower()
    return val in {"1", "true", "yes", "on"}


def clip_recording_enabled() -> bool:
    """CLIP_RECORDING_ENABLED aliases EVENT_CLIP_ENABLED for compatibility."""
    explicit = os.getenv("CLIP_RECORDING_ENABLED")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return os.getenv("EVENT_CLIP_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def vlm_enabled() -> bool:
    """VLM analysis/worker gating. Primary snapshot is independent of this."""
    val = os.getenv("VLM_ENABLED")
    if val is not None:
        return val.strip().lower() in {"1", "true", "yes", "on"}
    # Do not change operational defaults: if unset, allow VLM analysis when key present (existing behavior)
    # but primary snapshot must not consult this.
    return True


def encode_frame_jpeg(frame: Any, quality: int = 85) -> bytes | None:
    """Encode BGR frame to JPEG bytes. Pure function; does not depend on VLM."""
    try:
        import cv2  # noqa: PLC0415
    except Exception:
        return None
    if frame is None:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None


def _build_snapshot_key(event_id: str) -> str:
    safe_id = str(event_id or "").strip()
    # Reject dangerous values that would produce bad keys
    if not safe_id or safe_id.lower().startswith(("file:", "http:", "clip", "/")):
        # Use a stable fallback name; caller should still treat as non-uploaded if no real id
        safe_id = "unknown"
    return f"snapshots/{safe_id}.jpg"


def upload_snapshot_jpeg(
    jpeg_bytes: bytes,
    event_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upload JPEG bytes to snapshots/{eventId}.jpg.

    Returns: { "uploaded": bool, "s3_key": str|None, "url": str|None, "metadata": dict }
    Never raises for caller; falls back to local on any error.
    Rejects using clip/local paths as the object key.
    """
    if not jpeg_bytes:
        return {"uploaded": False, "s3_key": None, "url": None, "metadata": metadata or {}}

    # Always compute the canonical key shape for logging/determinism
    s3_key = _build_snapshot_key(event_id)

    # Guard: never allow a clip-style or file path to become the snapshot key
    if s3_key.startswith("clips/") or "://" in s3_key or s3_key.endswith(".mp4"):
        return {"uploaded": False, "s3_key": None, "url": None, "metadata": metadata or {}}

    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket_name = resolve_s3_bucket_name()
    region_name = os.environ.get("AWS_REGION", "ap-northeast-2")

    if not (aws_access_key and aws_secret_key and bucket_name):
        # Local fallback: do not emit object key
        print(
            f"[snapshot-uploader][fallback] AWS S3 credentials missing. Snapshot for eventId={event_id} not uploaded.",
            file=sys.stderr,
        )
        return {"uploaded": False, "s3_key": None, "url": None, "metadata": metadata or {}}

    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name,
        )

        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=jpeg_bytes,
            ContentType="image/jpeg",
        )

        s3_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{s3_key}"
        print(f"[snapshot-uploader] S3 upload successful: key={s3_key}", file=sys.stderr)
        return {
            "uploaded": True,
            "s3_key": s3_key,
            "url": s3_url,
            "metadata": metadata or {},
        }

    except ImportError:
        print(
            "[snapshot-uploader][fallback] boto3 not installed. Snapshot retained locally (no object key).",
            file=sys.stderr,
        )
        return {"uploaded": False, "s3_key": None, "url": None, "metadata": metadata or {}}
    except (NoCredentialsError, ClientError) as exc:
        print(
            f"[snapshot-uploader][fallback] S3 upload failed for eventId={event_id}: {exc}",
            file=sys.stderr,
        )
        return {"uploaded": False, "s3_key": None, "url": None, "metadata": metadata or {}}
    except Exception as exc:
        print(
            f"[snapshot-uploader][fallback] unexpected error eventId={event_id}: {exc}",
            file=sys.stderr,
        )
        return {"uploaded": False, "s3_key": None, "url": None, "metadata": metadata or {}}


def attach_primary_snapshot_if_enabled(payload: dict[str, Any], frame: Any) -> None:
    """Best-effort: if capture+upload enabled, encode frame and upload.

    On success, sets snapshot_object_key + snapshotObjectKey (and snapshot_url/snapshotUrl) on payload.
    Never raises. Uses payload['eventId'] for deterministic key.
    """
    if not snapshot_capture_enabled():
        return
    if not snapshot_upload_enabled():
        return
    if not isinstance(payload, dict):
        return
    event_id = str(payload.get("eventId") or payload.get("event_id") or "").strip()
    if not event_id:
        return
    jpeg = encode_frame_jpeg(frame)
    if not jpeg:
        return
    try:
        result = upload_snapshot_jpeg(jpeg, event_id)
        if result and result.get("uploaded") and result.get("s3_key"):
            key = result["s3_key"]
            # Final guard: only accept canonical snapshots/*.jpg keys
            if isinstance(key, str) and key.startswith("snapshots/") and key.endswith(".jpg"):
                payload["snapshot_object_key"] = key
                payload["snapshotObjectKey"] = key
                if result.get("url"):
                    payload["snapshot_url"] = result["url"]
                    payload["snapshotUrl"] = result["url"]
    except Exception as exc:
        # Absolute last guard; attach must not propagate
        print(f"[snapshot-uploader][warn] attach suppressed error: {exc}", file=sys.stderr)

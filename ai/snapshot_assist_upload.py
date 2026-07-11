"""Async post-event JPEG upload for VLM snapshot assist (side-channel).

Does not create or mutate primary safety alerts. Failures are logged only.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any
from urllib import error, request

logger = logging.getLogger(__name__)


def snapshot_assist_enabled() -> bool:
    return os.getenv("VLM_SNAPSHOT_ASSIST_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def snapshot_assist_url() -> str:
    base = os.getenv("SNAPSHOT_ASSIST_BASE_URL") or os.getenv("BACKEND_BASE_URL") or "http://localhost:18080"
    return base.rstrip("/") + "/api/internal/vlm/snapshot-assist"


def service_token() -> str:
    return (
        os.getenv("VLM_SNAPSHOT_ASSIST_SERVICE_TOKEN")
        or os.getenv("AI_SERVICE_TOKEN")
        or ""
    )


def encode_frame_jpeg(frame: Any, quality: int = 85) -> bytes | None:
    """Encode BGR ndarray to JPEG bytes. Returns None if opencv unavailable or encode fails."""
    try:
        import cv2
    except ImportError:
        logger.warning("snapshot assist: cv2 not available; skip JPEG encode")
        return None
    if frame is None or not hasattr(frame, "shape"):
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return buf.tobytes()


def submit_snapshot_async(
    *,
    event_id: str,
    camera_login_id: str,
    jpeg_bytes: bytes,
    timeout_sec: float = 10.0,
) -> None:
    """Fire-and-forget HTTP upload. Never raises into the inference loop."""
    if not snapshot_assist_enabled():
        return
    token = service_token()
    if not token:
        logger.warning("snapshot assist: service token missing; skip upload eventId=%s", event_id)
        return
    if not jpeg_bytes:
        return
    if not event_id:
        event_id = str(uuid.uuid4())

    def _run() -> None:
        try:
            submit_snapshot(
                event_id=event_id,
                camera_login_id=camera_login_id,
                jpeg_bytes=jpeg_bytes,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            logger.warning("snapshot assist upload failed eventId=%s: %s", event_id, exc)

    threading.Thread(target=_run, name=f"snapshot-assist-{event_id[:8]}", daemon=True).start()


def submit_snapshot(
    *,
    event_id: str,
    camera_login_id: str,
    jpeg_bytes: bytes,
    timeout_sec: float = 10.0,
    base_url: str | None = None,
    token: str | None = None,
) -> int:
    """Synchronous multipart POST. Returns HTTP status code."""
    token = token if token is not None else service_token()
    if not token:
        raise RuntimeError("service token required")
    url = (base_url or snapshot_assist_url()).rstrip("/") + f"/{event_id}"
    boundary = f"----StrangeBoundary{uuid.uuid4().hex}"
    cam = camera_login_id or ""
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="cameraLoginId"\r\n\r\n'
    body += cam.encode("utf-8") + b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="file"; filename="snapshot.jpg"\r\n'
    body += b"Content-Type: image/jpeg\r\n\r\n"
    body += jpeg_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-Service-Token": token,
        },
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            return int(resp.status)
    except error.HTTPError as exc:
        return int(exc.code)

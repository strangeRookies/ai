"""Async post-event JPEG upload for VLM snapshot assist (side-channel).

Does not create or mutate primary safety alerts. Failures are logged only.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from typing import Any
from urllib import error, request

logger = logging.getLogger(__name__)

# Bounded queue (maxsize=50) and lock for daemon uploader thread.
_UPLOAD_QUEUE: queue.Queue[tuple[str, str, bytes, float]] = queue.Queue(maxsize=50)
_WORKER_STARTED = False
_QUEUE_LOCK = threading.Lock()

# Process-local deduplication set to prevent duplicate uploads for the same eventId.
_SENT_EVENTS: set[str] = set()
_SENT_EVENTS_LOCK = threading.Lock()


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


def _upload_worker_loop():
    while True:
        try:
            event_id, camera_login_id, jpeg_bytes, timeout_sec = _UPLOAD_QUEUE.get()
            try:
                submit_snapshot(
                    event_id=event_id,
                    camera_login_id=camera_login_id,
                    jpeg_bytes=jpeg_bytes,
                    timeout_sec=timeout_sec,
                )
            except Exception as exc:
                logger.warning("snapshot assist upload failed eventId=%s: %s", event_id, exc)
            finally:
                _UPLOAD_QUEUE.task_done()
        except Exception as loop_exc:
            logger.error("Snapshot assist upload worker loop error: %s", loop_exc)


def submit_snapshot_async(
    *,
    event_id: str,
    camera_login_id: str,
    jpeg_bytes: bytes,
    timeout_sec: float = 10.0,
) -> None:
    """Fire-and-forget HTTP upload via a bounded background queue. Never raises."""
    if not snapshot_assist_enabled():
        return
    token = service_token()
    if not token:
        logger.warning("snapshot assist: service token missing; skip upload eventId=%s", event_id)
        return
    if not jpeg_bytes:
        return
    if not event_id or not event_id.strip():
        logger.warning("snapshot assist: eventId is empty; reject upload")
        return
    if not camera_login_id or not camera_login_id.strip():
        logger.warning("snapshot assist: cameraLoginId is empty; reject upload eventId=%s", event_id)
        return

    # Enforce process-local deduplication check (never upload same eventId twice)
    with _SENT_EVENTS_LOCK:
        if event_id in _SENT_EVENTS:
            logger.info("snapshot assist: duplicate upload request for eventId=%s dropped", event_id)
            return
        _SENT_EVENTS.add(event_id)
        # Keep deduplication set small (limit to last 1000 events)
        if len(_SENT_EVENTS) > 1000:
            _SENT_EVENTS.clear()
            _SENT_EVENTS.add(event_id)

    global _WORKER_STARTED
    with _QUEUE_LOCK:
        if not _WORKER_STARTED:
            t = threading.Thread(target=_upload_worker_loop, name="snapshot-assist-uploader", daemon=True)
            t.start()
            _WORKER_STARTED = True

    try:
        # non-blocking put to bounded queue; drops if queue is full (protects inference thread)
        _UPLOAD_QUEUE.put_nowait((event_id, camera_login_id, jpeg_bytes, timeout_sec))
    except queue.Full:
        logger.warning("snapshot assist upload queue is full (maxsize=50); dropping request eventId=%s", event_id)


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

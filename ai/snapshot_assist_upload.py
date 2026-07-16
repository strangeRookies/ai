"""Async post-event JPEG upload for VLM snapshot assist (side-channel)."""

from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from typing import Any
from urllib import error, request

logger = logging.getLogger(__name__)

_UPLOAD_QUEUE: queue.Queue[tuple[str, str, bytes, dict[str, Any]]] = queue.Queue(maxsize=50)
_FRAME_QUEUE: queue.Queue[tuple[str, str, Any, int, dict[str, Any]]] = queue.Queue(maxsize=50)
_FRAME_WORKER_STARTED = False
_FRAME_WORKER_THREAD: threading.Thread | None = None
_FRAME_STOP = threading.Event()

_WORKER_STARTED = False
_QUEUE_LOCK = threading.Lock()
_SENT_EVENTS: set[str] = set()
_SENT_EVENTS_LOCK = threading.Lock()


def _frame_worker_loop() -> None:
    while not _FRAME_STOP.is_set():
        try:
            event_id, camera_login_id, frame, quality, meta = _FRAME_QUEUE.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            from ai.snapshot_face_blur import deidentify_event_frame  # noqa: PLC0415

            blurred = deidentify_event_frame(
                frame,
                bbox=meta.get("bbox"),
                keypoints=meta.get("keypoints"),
            )
            jpeg = encode_frame_jpeg(blurred, quality=quality)
            if jpeg:
                upload_fields = {k: v for k, v in meta.items() if k not in {"bbox", "keypoints"}}
                submit_snapshot_async(
                    event_id=event_id,
                    camera_login_id=camera_login_id,
                    jpeg_bytes=jpeg,
                    **upload_fields,
                )
        except Exception as exc:
            logger.warning("snapshot assist frame worker failed eventId=%s: %s", event_id, exc)
        finally:
            _FRAME_QUEUE.task_done()


def submit_frame_snapshot_async(
    event_id: str,
    camera_login_id: str,
    frame: Any,
    quality: int = 85,
    **meta: Any,
) -> bool:
    global _FRAME_WORKER_STARTED, _FRAME_WORKER_THREAD
    if not snapshot_assist_enabled() or frame is None:
        return False
    try:
        frame_copy = frame.copy()
    except Exception:
        return False
    if not _FRAME_WORKER_STARTED:
        _FRAME_STOP.clear()
        _FRAME_WORKER_STARTED = True
        _FRAME_WORKER_THREAD = threading.Thread(
            target=_frame_worker_loop, name="snapshot-assist-encoder", daemon=True
        )
        _FRAME_WORKER_THREAD.start()
    try:
        _FRAME_QUEUE.put_nowait((event_id, camera_login_id, frame_copy, quality, dict(meta)))
    except queue.Full:
        logger.warning("snapshot assist frame queue full eventId=%s", event_id)
        return False
    return True


def stop_snapshot_assist_worker(timeout: float = 2.0) -> None:
    global _FRAME_WORKER_STARTED
    _FRAME_STOP.set()
    if _FRAME_WORKER_THREAD is not None:
        _FRAME_WORKER_THREAD.join(timeout=timeout)
    _FRAME_WORKER_STARTED = False


def snapshot_assist_enabled() -> bool:
    try:
        from ai.vlm.provider_mode import snapshot_assist_should_run  # noqa: PLC0415

        return snapshot_assist_should_run()
    except Exception:
        return os.getenv("VLM_SNAPSHOT_ASSIST_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def snapshot_assist_url() -> str:
    base = os.getenv("SNAPSHOT_ASSIST_BASE_URL") or os.getenv("BACKEND_BASE_URL") or "http://localhost:18080"
    return base.rstrip("/") + "/api/internal/vlm/snapshot-assist"


def service_token() -> str:
    return (
        os.getenv("AI_SERVICE_TOKEN")
        or os.getenv("VLM_SNAPSHOT_ASSIST_SERVICE_TOKEN")
        or ""
    ).strip()


def encode_frame_jpeg(frame: Any, quality: int = 85) -> bytes | None:
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buf.tobytes()


def _upload_worker_loop() -> None:
    while True:
        try:
            event_id, camera_login_id, jpeg_bytes, fields = _UPLOAD_QUEUE.get()
            submit_snapshot(
                event_id=event_id,
                camera_login_id=camera_login_id,
                jpeg_bytes=jpeg_bytes,
                **fields,
            )
        except Exception as loop_exc:
            logger.error("Snapshot assist upload worker loop error: %s", loop_exc)


def submit_snapshot_async(
    *,
    event_id: str,
    camera_login_id: str,
    jpeg_bytes: bytes,
    base_url: str | None = None,
    token: str | None = None,
    **fields: Any,
) -> None:
    if not snapshot_assist_enabled():
        return
    token = token or service_token()
    if not token:
        logger.warning("snapshot assist: missing service token; skip eventId=%s", event_id)
        return
    if not camera_login_id or not camera_login_id.strip():
        logger.warning("snapshot assist: cameraLoginId is empty; reject upload eventId=%s", event_id)
        return
    with _SENT_EVENTS_LOCK:
        if event_id in _SENT_EVENTS:
            return
        _SENT_EVENTS.add(event_id)
    global _WORKER_STARTED
    with _QUEUE_LOCK:
        if not _WORKER_STARTED:
            _WORKER_STARTED = True
            threading.Thread(target=_upload_worker_loop, daemon=True, name="snapshot-assist-upload").start()
    try:
        _UPLOAD_QUEUE.put_nowait((event_id, camera_login_id, jpeg_bytes, dict(fields)))
    except queue.Full:
        logger.warning("snapshot assist upload queue full; drop eventId=%s", event_id)


def submit_snapshot(
    *,
    event_id: str,
    camera_login_id: str,
    jpeg_bytes: bytes,
    base_url: str | None = None,
    token: str | None = None,
    **fields: Any,
) -> int:
    token = token or service_token()
    if not token:
        raise RuntimeError("service token required")
    url = (base_url or snapshot_assist_url()).rstrip("/") + f"/{event_id}"
    boundary = f"----StrangeBoundary{uuid.uuid4().hex}"
    body = b""

    def add_field(name: str, value: Any) -> None:
        nonlocal body
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += text.encode("utf-8") + b"\r\n"

    add_field("cameraLoginId", camera_login_id)
    for key in (
        "eventType",
        "trackId",
        "confidence",
        "faintProbability",
        "lifecycleState",
        "consecutiveCount",
        "detectorReason",
        "capturedAt",
    ):
        if key in fields:
            add_field(key, fields[key])

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
        with request.urlopen(req, timeout=30) as resp:
            return int(resp.status)
    except error.HTTPError as exc:
        return int(exc.code)
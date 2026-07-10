from __future__ import annotations


def evidence_id(
    camera_login_id: str,
    frame_id: int,
    timestamp_ms: int,
    stream_run_id: str | None = None,
) -> str:
    """Stable evidence key.

    Prefer including stream_run_id so the same frameId after a stream reset
    does not collide. When stream_run_id is omitted, legacy
    ``{camera}-{frameId}-{ts}`` form is kept for backward-compatible callers.
    """
    cam = str(camera_login_id)
    if stream_run_id:
        return f"{cam}-{stream_run_id}-{int(frame_id)}-{int(timestamp_ms)}"
    return f"{cam}-{int(frame_id)}-{int(timestamp_ms)}"


def latency_order_valid(
    captured_at_ms: int,
    processed_at_ms: int | None,
    published_at_ms: int | None,
) -> bool:
    if processed_at_ms is None or published_at_ms is None:
        return True
    return int(captured_at_ms) <= int(processed_at_ms) <= int(published_at_ms)

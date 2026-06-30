from __future__ import annotations


def evidence_id(camera_login_id: str, frame_id: int, timestamp_ms: int) -> str:
    return f"{camera_login_id}-{int(frame_id)}-{int(timestamp_ms)}"


def latency_order_valid(
    captured_at_ms: int,
    processed_at_ms: int | None,
    published_at_ms: int | None,
) -> bool:
    if processed_at_ms is None or published_at_ms is None:
        return True
    return int(captured_at_ms) <= int(processed_at_ms) <= int(published_at_ms)

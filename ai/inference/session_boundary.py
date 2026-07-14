"""Pure contract for deciding when a video session must be reset."""
from __future__ import annotations


def session_reset_reason(
    previous_frame_id: int | None,
    current_frame_id: int | None,
    previous_captured_at_ms: int | None,
    current_captured_at_ms: int | None,
    *,
    max_frame_gap: int = 90,
    max_time_gap_ms: int = 3000,
) -> str | None:
    """Return a boundary reason without mutating tracker or lifecycle state."""
    if previous_frame_id is not None and current_frame_id is not None:
        frame_gap = current_frame_id - previous_frame_id
        if frame_gap < 0:
            return "FRAME_ID_RESET"
        if frame_gap > max_frame_gap:
            return "LARGE_FRAME_GAP"
    if previous_captured_at_ms is not None and current_captured_at_ms is not None:
        if current_captured_at_ms - previous_captured_at_ms > max_time_gap_ms:
            return "LARGE_TIME_GAP"
    return None
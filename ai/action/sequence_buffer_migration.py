"""Compatible sequence-buffer merge helpers for recovery migration."""
from __future__ import annotations

def merge_sequence_buffers(source, destination, new_track_id: int) -> bool:
    if type(source) is not type(destination):
        return False
    for attr in ("sequence_length", "stride", "resize_size"):
        if hasattr(source, attr) and getattr(source, attr) != getattr(destination, attr):
            return False
    if hasattr(source, "_frames") and hasattr(destination, "_frames"):
        entries_attr = "_frames"
    elif hasattr(source, "_crops") and hasattr(destination, "_crops"):
        entries_attr = "_crops"
    else:
        return False
    try:
        combined = {}
        for entry in getattr(source, entries_attr):
            combined[(entry["frame_id"], entry["frame_idx"])] = entry
        for entry in getattr(destination, entries_attr):
            combined[(entry["frame_id"], entry["frame_idx"])] = entry
        ordered = sorted(combined.values(), key=lambda item: (item["frame_id"], item["frame_idx"]))
    except (KeyError, TypeError):
        return False
    ordered = ordered[-int(destination.sequence_length):]
    if entries_attr == "_frames":
        for entry in ordered:
            detection = entry.get("detection")
            if isinstance(detection, dict):
                detection["track_id"] = int(new_track_id)
    setattr(destination, entries_attr, ordered)
    destination._last_emit_frame = max(int(getattr(source, "_last_emit_frame", -1)), int(getattr(destination, "_last_emit_frame", -1)))
    return True

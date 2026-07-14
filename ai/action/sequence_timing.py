def sequence_timing(samples, stride):
    """Return capture-time metadata when every sample has a monotonic timestamp."""
    timestamps = [sample.get("captured_at_ms") for sample in samples]
    if not timestamps or any(timestamp is None for timestamp in timestamps):
        return {
            "sequence_timing_mode": "frame_index_fallback",
            "sequence_duration_ms": None,
            "sequence_sample_interval_ms": None,
            "sequence_stride_ms": None,
        }

    normalized = [int(timestamp) for timestamp in timestamps]
    if any(later <= earlier for earlier, later in zip(normalized, normalized[1:])):
        return {
            "sequence_timing_mode": "frame_index_fallback",
            "sequence_duration_ms": None,
            "sequence_sample_interval_ms": None,
            "sequence_stride_ms": None,
        }

    duration_ms = normalized[-1] - normalized[0]
    sample_interval_ms = duration_ms / max(len(normalized) - 1, 1)
    return {
        "sequence_timing_mode": "captured_at_ms",
        "sequence_duration_ms": int(duration_ms),
        "sequence_sample_interval_ms": round(sample_interval_ms, 3),
        "sequence_stride_ms": round(sample_interval_ms * int(stride), 3),
    }


def sequence_is_ready(samples, stride, last_emit_frame, last_emit_at_ms):
    """Use capture time for stride gating and retain frame-index fallback."""
    timing = sequence_timing(samples, stride)
    if timing["sequence_timing_mode"] == "captured_at_ms":
        current_at_ms = int(samples[-1]["captured_at_ms"])
        if last_emit_at_ms is None:
            return True, timing
        required_stride_ms = float(timing["sequence_stride_ms"])
        return current_at_ms - int(last_emit_at_ms) >= required_stride_ms, timing

    current_frame = int(samples[-1]["frame_idx"])
    return last_emit_frame < 0 or current_frame - last_emit_frame >= int(stride), timing

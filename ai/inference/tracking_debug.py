from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence


def tracking_debug_enabled() -> bool:
    return os.getenv("TRACKING_DEBUG", "false").lower() in {"1", "true", "yes", "on"}


def log_sequence_stage(
    camera_login_id: str,
    frame_id: int | None,
    active_track_ids: Sequence[int],
    buffer_lengths: Mapping[int, int],
    sequences_generated: int,
    sequences_generated_by_track: Mapping[int, int],
    latest_faint_prob: float | None,
    sequence_diagnostics: Mapping[int, Mapping[str, object]] | None = None,
    checkpoint_input_size: int | None = None,
    runtime_feature_dim: int | None = None,
    tensor_shape: Sequence[int] | None = None,
    sequence_length: int | None = None,
) -> None:
    if not tracking_debug_enabled():
        return
    diagnostics = {
        str(track_id): dict(value)
        for track_id, value in (sequence_diagnostics or {}).items()
    }
    reasons = {str(value.get("reason", "")) for value in diagnostics.values()}
    if sequences_generated > 0:
        diagnosis = "sequence_ready"
    elif "feature_dim_mismatch" in reasons:
        diagnosis = "feature_dim_mismatch"
    elif "insufficient_keypoints" in reasons:
        diagnosis = "insufficient_keypoints"
    elif "selected_track_missing" in reasons:
        diagnosis = "selected_track_missing"
    elif "buffer_not_full" in reasons:
        diagnosis = "sequence_buffer_not_full"
    elif active_track_ids:
        diagnosis = "sequence_buffer_no_reason"
    else:
        diagnosis = "no_active_tracks"
    record = {
        "stage": "sequence",
        "cameraLoginId": camera_login_id,
        "frameId": frame_id,
        "activeTrackIds": list(active_track_ids),
        "bufferLengths": {str(track_id): int(length) for track_id, length in buffer_lengths.items()},
        "sequencesGenerated": int(sequences_generated),
        "lstmSequenceGenerated": sequences_generated > 0,
        "sequencesGeneratedByTrack": {
            str(track_id): int(count) for track_id, count in sequences_generated_by_track.items()
        },
        "latestFaintProbability": latest_faint_prob,
        "trackDiagnostics": diagnostics,
        "diagnosis": diagnosis,
        "checkpointInputSize": checkpoint_input_size,
        "runtimeFeatureDim": runtime_feature_dim,
        "tensorShape": list(tensor_shape) if tensor_shape is not None else None,
        "sequenceLength": sequence_length,
    }
    print(f"[stage-log] {json.dumps(record, ensure_ascii=False)}", flush=True)

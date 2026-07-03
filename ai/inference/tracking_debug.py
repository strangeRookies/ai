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
) -> None:
    if not tracking_debug_enabled():
        return
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
    }
    print(f"[stage-log] {json.dumps(record, ensure_ascii=False)}", flush=True)

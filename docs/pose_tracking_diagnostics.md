# Pose and Tracking Diagnostics

The default worker path remains YOLO Pose -> Supervision ByteTrack -> LSTM. Diagnostics are opt-in so production logs and diagnostic image writes stay quiet by default.

## Environment

- `TRACK_THRESH`, `TRACK_IOU_THRESHOLD`, `TRACK_BUFFER`, `TRACK_FRAME_RATE`: passed to `supervision.ByteTrack` only when the installed constructor supports the matching parameter. Unsupported values are reported in startup config under `bytetrack_constructor_ignored`.
- `TRACKING_GRACE_PERIOD_SECONDS`: keeps a per-track LSTM sequence buffer after a short `active_tracks=0` gap instead of deleting it immediately.
- `TRACKING_RELINK_IOU_THRESHOLD`, `TRACKING_RELINK_CENTER_RATIO`, `TRACKING_RELINK_MAX_TIME_GAP_SECONDS`: reconnect an incoming changed `track_id` to a recent sequence when bbox IoU or center distance is close enough.
- `TRACKING_DEBUG=true`: enables detailed tracking/sequence stage logs.
- `POSE_DEBUG=true`: enables YOLO Pose raw detection diagnostics.
- `POSE_DEBUG_SUMMARY_EVERY_N`: prints rolling per-camera pose summaries every N observed frames.
- `POSE_MIN_KEYPOINT_CONFIDENCE`: threshold for valid keypoint counts and low-pose-quality classification.
- `POSE_DEBUG_SAVE_IMAGES=false`: default OFF. When true, saves sampled bbox/keypoint diagnostic images under `POSE_DEBUG_IMAGE_DIR`.
- `POSE_DEBUG_IMAGE_EVERY_N`: saves only every Nth frame when image saving is explicitly enabled.
- `POSE_TRACKING_DIAG_JSONL=false`: default OFF. When true, writes compact per-frame pose/tracking diagnostics as JSONL.
- `POSE_TRACKING_DIAG_JSONL_PATH=runs/diagnostics/pose_tracking_diag.jsonl`: output path for JSONL diagnostics.

On worker startup, `[pose-tracking-config]` prints the effective tracking/pose settings once, including all tracking thresholds, grace/re-link settings, pose confidence settings, image-save status, JSONL status, and any `supervision.ByteTrack` constructor parameters ignored by the installed supervision version.

## Reading cam_04/cam_05 Comparisons

Filter console logs by `[pose-diagnostics]`, or enable JSONL for easier offline comparison. Do not assume two camera IDs are comparable just because they look similar:

1. Compare `assignedVideoPath`.
2. If it is empty, compare `sourceUrl`.
3. Prefer frame-by-frame comparison by `frameId` when both streams preserve stable frame IDs.
4. If `frameId` is missing or may reset after reconnects, fall back to `timestampMs`. This is less exact because capture clocks, reconnect gaps, and queue lag can shift timestamps between workers.

Check these fields side by side:

- `raw_detection_count`
- `avg_bbox_confidence`
- `avg_keypoint_confidence`
- `active_tracks`
- `track_ids`
- `sequenceReadyCount`
- `diagnosis`

Rolling summaries under `stage=pose_summary` include `total_frames`, `frames_with_person`, `frames_without_person`, `avg_person_count`, `avg_bbox_conf`, `avg_keypoint_conf`, `avg_valid_keypoints`, `tracker_active_rate`, `sequence_ready_count`, `active_tracks_zero_count`, `relink_success_count`, and `relink_fail_count`.

For a compact camera-level JSON summary:

```powershell
python scripts/summarize_pose_tracking_diag.py runs/diagnostics/pose_tracking_diag.jsonl --pretty
```

The script outputs per-`cameraLoginId` values for `avg_raw_detection_count`, `avg_bbox_confidence`, `avg_keypoint_confidence`, `tracker_active_rate`, and `sequence_ready_count`.

To verify a GPU PC run, use:

```powershell
python scripts/verify_pose_diagnostics.py --log-path ai_runner.log --jsonl-path runs/diagnostics/pose_tracking_diag.jsonl
```

`run_registered_cameras.py` writes parent logs to the console log and child AI worker logs to `runs/registered_cameras/*-overlay.log`; the verifier scans those overlay logs by default because `[pose-tracking-config]` and `[pose-diagnostics]` are emitted by the child worker.

## Diagnosis Rules

- YOLO Pose raw detection missing + `active_tracks=0`: detector is not finding people.
- YOLO Pose raw detection exists + low keypoint confidence: pose quality, input resolution, lighting, or camera quality issue.
- YOLO Pose raw detection exists + acceptable keypoint confidence + `active_tracks=0`: ByteTrack association or threshold issue.
- YOLO Pose raw detection exists + tracker IDs change often: tracking continuity issue; re-link may be needed.
- Tracker is active but `sequenceReadyCount=0`: sequence buffer, frame stride, or keypoint feature conversion issue.

Final worker summaries use these buckets: `detector_missing`, `low_pose_quality`, `tracker_association_issue`, `frequent_track_id_switch`, `sequence_buffer_issue`, and `normal`.

## BoT-SORT and SAM3

BoT-SORT is not the default path. It should be added only as an experiment option after adding a maintained tracker/ReID dependency and wiring it at `ai.inference.rtsp_runtime.create_detection_postprocessor`.

SAM3 is not part of this pipeline change. Treat it as a future experiment candidate for segmentation-assisted person masking before pose/tracking, with ByteTrack fallback kept available if VRAM or dependencies are insufficient.

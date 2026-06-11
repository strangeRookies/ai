# RTSP LSTM Error-Augmented Deployment

This runbook applies the FP/FN error-analysis LSTM checkpoint to the RTSP inference path without changing the YOLO26n-pose detector.

## Applied defaults

```powershell
YOLO_MODEL=yolo26n-pose.pt
ACTION_MODEL=benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/YOLO26n-pose=./yolo26n-pose.pt/best.pt
ACTION_THRESHOLD=0.3
MIN_CONSECUTIVE_FAINT=3
CAMERA_COOLDOWN_SECONDS=10
MQTT_TOPIC=safety/events
```

Camera status events remain on `safety/cameras/status`.

## Preflight

Run this after placing both model files in the workspace:

```powershell
python scripts/run_rtsp_inference.py --preflight-only --detector-mode real --dry-run
```

Expected evidence in the JSON summary:

- `action_model` is the error-augmented `best.pt` path.
- `action_model_exists` is `true`.
- `yolo_model` is `yolo26n-pose.pt`.
- `yolo_model_exists` is `true`.
- `action_threshold` is `0.3`.
- `min_consecutive_faint` is `3`.
- `camera_cooldown_seconds` is `10`.

If the checkpoint is missing, the process exits with `LSTM action checkpoint not found`.

## RTSP smoke test

```powershell
python scripts/run_rtsp_inference.py `
  --detector-mode real `
  --rtsp-url "<RTSP_URL>" `
  --camera-id cam_01 `
  --publisher console `
  --dry-run `
  --debug-every-n 1 `
  --max-frames 120 `
  --output runs/verification/cam_01_error_augmented_summary.json `
  --event-log-dir runs/verification/events
```

Confirm:

- Debug logs include `latest_faint_prob=...`.
- The summary has `bbox_detections`, `keypoints_extracted`, `generated_sequences`, and `per_track_sequences_generated`.
- Events are generated only after the same `camera_id + track_id` reaches 3 consecutive Faint predictions.
- Event JSON includes `event_type`, `camera_id`, `camera_login_id`, `track_id`, `confidence`, `faint_prob`, `bbox`, and `timestamp`.

## MQTT smoke test

```powershell
python scripts/run_rtsp_inference.py `
  --detector-mode real `
  --rtsp-url "<RTSP_URL>" `
  --camera-id cam_01 `
  --publisher mqtt `
  --mqtt-topic safety/events `
  --debug-every-n 1 `
  --max-frames 300
```

Confirm in EMQX/backend logs that duplicate Faint events for the same camera/track are suppressed for `CAMERA_COOLDOWN_SECONDS`.

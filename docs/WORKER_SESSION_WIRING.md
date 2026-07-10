# Worker Session Wiring

Offline-safe documentation for analysis session identity, reset policy, and metrics paths.
Does not claim tracking accuracy, FPS, or TensorRT speedup improvements.

## workerRunId vs streamRunId

| ID | Lifetime | Changes when |
| --- | --- | --- |
| **workerRunId** | Process / OverlayWorker instance | Worker process restart (`WORKER_RESTARTED`) |
| **streamRunId** | Continuous analysis segment | VIDEO_EOF, SOURCE_CHANGED, RTSP reconnect, large frame/time gap, manual reset |
| **frameId** (session) | Per stream segment | Increments each inference frame; restarts at 0 on new stream |
| **evidenceFrameKey** | `streamRunId:frameId` | Unique across restarts even if capture `frameId` wraps |

Capture `frame_id` from `FrameMetadataBuffer` remains the MQTT/frame-sync id. Session fields are **optional** payload enrichments (`workerRunId`, `streamRunId`, `evidenceFrameKey`) for evidence correlation.

## Boundary event policy

| Event | Where it originates | Reset reason | workerRunId | streamRunId |
| --- | --- | --- | --- | --- |
| VIDEO_EOF | Reader `packet is None` / file end | `video_eof` | keep | new |
| SOURCE_CHANGED | Registered-camera source signature → process restart; or explicit in-process change | `source_change` | new if process restarts | new |
| RTSP_RECONNECTED | Reader reconnect loop; also LARGE_*_GAP / FRAME_ID_RESET proxies | `stream_reconnect` | keep | new |
| WORKER_RESTARTED | New process (`run_registered_cameras` / CLI start) | `worker_start` | new | new |
| MANUAL_RESET | Ops / tests | `manual` | keep | new |

Shared path: `AnalysisWorkerRuntime.reset_analysis_session(reason)` → flush metrics → clear components → new `streamRunId`.

## Reset targets

- tracker (`SimpleTrackAssigner.reset` / `SupervisionPostProcessor.reset`)
- keypoint / crop sequence buffers (`clear`)
- Fall/Faint postprocessor + state machine (`reset` / `reset_all`)
- exit / hazard postprocessors (`reset`)
- display ID mapper, overlay publish state, track selector
- session overlay / hazard / displayed IDs / pending events / frame_sync dicts

Optional missing components do not abort reset. Tracker/sequence failures are logged at **error** level.

## Metrics artifacts

On session end or boundary reset:

```text
runs/tracking_metrics/{cameraLoginId}/{streamRunId}/
├─ session-summary.json
├─ track-lifecycle.csv
└─ match-decisions.csv
```

Save failures: warning + `metrics_flush_failed_total` only (inference continues).
Live overlay may flush with `flush_blocking=False` (background thread + snapshot).
Worker **exit** uses `finalize_for_exit()` — flush only, **no** empty next-stream directory.

Tracker events are ingested **once per frame** via `note_tracker_events` (from `last_events`).

## Stale packet / event guards

| Guard | Mechanism |
| --- | --- |
| Queue after reset | `CameraFrameQueue.clear()` + `queue_cleared_frame_count` |
| Packet tag | `FramePacket.session_generation` + `stream_run_id` stamped by reader |
| Pre-inference drop | `accept_packet()` → `stale_packet_dropped_total` |
| Late publish | `enrich_payload(..., capture_generation=)` returns None → `stale_event_dropped_total` |
| Double reconnect | coalesce same reason within 250ms |
| Partial reset | force-clear tracks/buffers + `session_reset_failed_total` |

## Safety counters

```text
stale_packet_dropped_total
stale_event_dropped_total
queue_cleared_frame_count
session_reset_failed_total
metrics_flush_failed_total
metrics_flush_duration_ms
```

## Backend start log

At model init (real detector state only):

- `requested_backend`, `actual_backend`, `fallback_reason`
- `model_path`, `engine_path`, `device`, `precision`
- `workerRunId`, `streamRunId`

If detector is missing: `actual_backend=unavailable`. No fake TensorRT performance numbers in ops logs.

## Live verification checklist (real video / GPU PC)

- [ ] Overlay worker starts; logs show `[worker-backend-session]` and `[analysis-session]`
- [ ] After RTSP reconnect: new `streamRunId`, same `workerRunId`, tracks restart
- [ ] File A then B: no track/sequence/fall leak; metrics dir per stream
- [ ] Overlay / event MQTT payloads include optional session ids without breaking consumers
- [ ] Metrics flush under `runs/tracking_metrics/...` after boundary
- [ ] TensorRT path: `actual_backend` reflects real engine load (not mock timings)

## Entry points

- `scripts/serve_ai_overlay.py` — primary multi-cam overlay worker
- `scripts/run_rtsp_inference.py` — offline/file + RTSP inference CLI
- `scripts/run_registered_cameras.py` / `ai/registered_camera_workers.py` — process supervision (SOURCE_CHANGED → restart)

# Multi-camera isolation & supervisor policy

Offline-safe documentation. Defaults are **not** measured 4-cam GPU optima.

## Process model

| Layer | Isolation |
| --- | --- |
| **Supervisor** (`registered_camera_workers`) | One OS process bundle per `cameraLoginId` (overlay; optional ffmpeg for SIMULATED_RTSP) |
| **Worker process** | Own `AnalysisWorkerRuntime`, tracker, sequence buffers, Fall/Faint state, `CameraFrameQueue`, metrics root |
| **Dynamic cameras** | Active list from backend API — **not** hardcoded `cam_01`–`cam_04` |

## Evidence key

```text
evidenceFrameKey = {cameraLoginId}:{streamRunId}:{frameId}
```

Also: `evidence_id(camera, frameId, capturedAtMs)` for MQTT evidence fields.

## Supervisor restart policy (env)

| Env | Default | Meaning |
| --- | --- | --- |
| `SUPERVISOR_RESTART_INITIAL_DELAY_SEC` | `2.0` | First backoff after exit |
| `SUPERVISOR_RESTART_MAX_DELAY_SEC` | `60.0` | Cap on exponential delay |
| `SUPERVISOR_MAX_CONSECUTIVE_FAILURES` | `10` | Stop restart storm after N consecutive failures |
| `SUPERVISOR_STABLE_RUNTIME_RESET_SEC` | `120.0` | If process lived ≥ this, clear failure streak |
| `SUPERVISOR_STARTUP_STAGGER_SEC` | `0.5` | Delay between successive camera starts at boot |

Formula: `delay = min(max_delay, initial * 2^(consecutive_failures-1))`.

Source change: only the camera whose `source_signature` changed is restarted.

## Queue policy

- Bounded (`maxsize`)
- Overflow: drop oldest (deque maxlen) / `queue_overflow_drop_total`
- Optional `max_packet_age_ms` drops aged packets before `get_latest`
- Session reset: `clear()` + generation/stream stale guards

## Metrics paths

```text
runs/tracking_metrics/{cameraLoginId}/pid-{pid}/{streamRunId}/
  session-summary.json
  track-lifecycle.csv
  match-decisions.csv
```

Writes: temp file + `os.replace` (atomic). Async flush joins on `finalize_for_exit` / `wait_for_flush`.

## Incident stream reset

Open incidents → terminal:

- reconnect / large gap → `STREAM_LOST`
- EOF → `INTERRUPTED`
- other → `CLOSED_UNKNOWN`

## Event de-duplication layers

| Layer | Role |
| --- | --- |
| AI process `EventIdempotencyStore` (via `EventPublisher`) | Best-effort skip of same `eventId` retransmit in one process |
| **Backend DB UNIQUE(eventId)** | **Final durable defense** — implement/confirm on GPU PC with Backend team |

Process-local gate is not a substitute for Backend uniqueness.

## Incident interrupt policy (GPU PC follow-up)

Skeleton: stream reset → `STREAM_LOST` / `INTERRUPTED` / `CLOSED_UNKNOWN` via
`IncidentVlmPipeline.on_stream_reset` called from `AnalysisWorkerRuntime.reset_analysis_session`.

**After GPU PC:** adjust thresholds using real RTSP disconnect duration distribution
(not estimated here). Backend + AI should agree when mid-incident clips are closed.

## evidence_id

When `streamRunId` is known:

```text
{cameraLoginId}-{streamRunId}-{frameId}-{capturedAtMs}
```

Legacy callers without streamRunId keep `{camera}-{frameId}-{ts}`.

## Restart blocked visibility

On max consecutive failures:

```text
[registered-cameras][status] {restart_blocked: true, reason, consecutive_failures, ...}
```

Also: `SupervisorRestartBook.instrumentation()` / `blocked_cameras()`.

## GPU PC checklist (no numbers claimed here)

- [ ] Set env delays for site load
- [ ] Confirm only signature-changed camera restarts
- [ ] Confirm 4 processes, independent metrics dirs
- [ ] Confirm restart storm stops at max consecutive failures + status log
- [ ] Confirm queue overflow/age counters under real RTSP
- [ ] Confirm Backend UNIQUE(eventId)
- [ ] Tune incident interrupt using disconnect time distribution

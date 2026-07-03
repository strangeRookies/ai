# Active Camera Overlay Inference MQTT Plan

## TL;DR
> Summary:      Fix the currently used AI camera path so every active registered camera uses the same YOLO -> tracker -> duplicate removal -> optional selected/preferred track -> per-track buffer -> LSTM -> overlay/frame_sync/event MQTT flow. The plan is TDD-first and keeps default multi-person behavior when no selector is configured.
> Deliverables:
> - Missing track-selection/dedup helper restored and wired into `process_frame`.
> - Optional selected/preferred track selector exposed through env and CLI, then forwarded by the registered-camera runner.
> - Registered-camera dry-run proves arbitrary active camera IDs, not `cam_04`/`cam_05`, are used.
> - Overlay, frame_sync, and confirmed-event payloads preserve cameraLoginId, track_id, faint_probability/action fields, and evidence linkage.
> - Manual QA artifacts captured under `evidence/`.
> Effort:       Medium
> Risk:         Medium - the publish path spans runner, overlay processing, frame sync, and MQTT topics, and the repo already has a missing `ai.inference.track_selection` module referenced by tests.

## Scope
### Must have
- Use only the AI Agent scope: files under `strange_ai/`, including `scripts/`, `ai/`, `tracking/`, and `tests/`.
- Restore or implement `ai/inference/track_selection.py` because `tests/test_track_selection.py:3` imports it but `ai/inference/` currently contains only `__init__.py`, `rtsp_runtime.py`, and `tracking_debug.py`.
- Keep `scripts/run_registered_cameras.py:147` loading backend active cameras and `ai/registered_camera_workers.py:204` syncing workers by `camera_login_id`.
- Preserve registered-camera dynamic identity from `ai/registered_cameras.py:128` and `ai/registered_cameras.py:271`: `cameraLoginId` must be the external key passed to `scripts/serve_ai_overlay.py`.
- Apply duplicate tracked-detection removal after tracker assignment and before `normalize_detections`/sequence buffering in `scripts/serve_ai_overlay.py:140` through `scripts/serve_ai_overlay.py:170`.
- Add optional selection via CLI and env:
  - Canonical CLI: `--selected-track-id <int>`
  - Alias CLI: `--preferred-track-id <int>` using the same destination
  - Canonical env: `AI_SELECTED_TRACK_ID`
  - Alias env: `AI_PREFERRED_TRACK_ID`
  - Default: unset/empty means process all tracks.
- Preserve all existing production tracker/model/cooldown threshold defaults from `scripts/serve_ai_overlay.py:773`, `scripts/serve_ai_overlay.py:794`, `scripts/serve_ai_overlay.py:795`, `scripts/serve_ai_overlay.py:796`, `scripts/serve_ai_overlay.py:797`, `scripts/serve_ai_overlay.py:781`, and `scripts/serve_ai_overlay.py:782`.
- Preserve per-track sequence buffering through `ai/action/per_track_sequence_buffer.py:8` and `ai/action/per_track_sequence_buffer.py:112`.
- Preserve action/faint attachment through `ai/visualization/action_overlay.py:111` before overlay payload creation through `ai/publishers/mqtt_payloads.py:23`.
- Preserve frame_sync payload schema from `ai/publishers/mqtt_payloads.py:68` and confirmed-event schema from `ai/publishers/mqtt_payloads.py:103`.
- Publish overlay and frame_sync payloads to camera topic and confirmed events to event topic using `ai/publishers/event_publisher.py:155`.
- Final review must explicitly assess these risks: accidental single-person default, dropping valid overlapping persons, changing thresholds, breaking cameraLoginId contract, publishing event before frame_sync, and hardcoding test camera IDs into production code.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Do not edit `strange_back/`, `strange_front/`, `strange_infra/`, root `docs/`, or shared contract files.
- Do not modify production tracker thresholds, model defaults, `DEFAULT_MIN_CONSECUTIVE_FAINT`, or cooldown defaults.
- Do not make `cam_04`, `cam_05`, `cam4`, or `cam5` production special cases.
- Do not remove DTO aliases already asserted in `tests/test_mqtt_payloads.py:208`, such as `camera_id`, `cameraLoginId`, `camera_login_id`, `trackingId`, and `track_id`.
- Do not replace the current MQTT publisher abstraction or introduce a new broker/client library.
- Do not skip, delete, xfail, or weaken existing tests.
- Do not make selected/preferred track mandatory. Multi-person all-track behavior must remain default.
- Do not run destructive git operations or merge other branches.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD + Python `unittest`. Each implementation task starts by adding or enabling a focused failing test, captures RED output, then applies the smallest source change and captures GREEN output.
- QA policy: every task has agent-executed scenarios. Manual QA is CLI/data-shaped and uses exact PowerShell/Python invocations against the real modules and scripts.
- Evidence: `evidence/task-<N>-active-camera-overlay-inference-mqtt.<ext>`

## Execution strategy
### Parallel execution waves
> Target 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks to maximize parallelism.

Wave 1 (no dependencies):
- Task 1: Restore tracked-detection selection/dedup helper.
- Task 2: Expose overlay selected/preferred track CLI/env parsing.
- Task 4: Prove registered runner starts arbitrary active cameras.
- Task 5: Lock MQTT overlay/frame_sync/event schema and context.

Wave 2 (after Wave 1):
- Task 3: depends [2] - forward selected/preferred selector from registered runner to overlay command.
- Task 6: depends [1, 2, 5] - wire dedup/selection into `process_frame` before sequence buffering and payload creation.

Wave 3 (after Wave 2):
- Task 7: depends [3, 6] - centralize OverlayWorker publish order as overlay -> frame_sync -> event.

Critical path: Task 1 -> Task 6 -> Task 7

### Dependency matrix
| Task | Depends on | Blocks | Can parallelize with |
|------|------------|--------|----------------------|
| 1    | none       | 6      | 2, 4, 5              |
| 2    | none       | 3, 6   | 1, 4, 5              |
| 3    | 2          | 7      | 6                    |
| 4    | none       | final  | 1, 2, 5              |
| 5    | none       | 6      | 1, 2, 4              |
| 6    | 1, 2, 5    | 7      | 3                    |
| 7    | 3, 6       | final  | none                 |

## Todos
> Implementation + Test = ONE task. Never separate.
> Every task MUST have: References + Acceptance Criteria + QA Scenarios + Commit.

- [ ] 1. Restore tracked-detection selection and dedup helper

  What to do: Add `ai/inference/track_selection.py` with `deduplicate_tracked_detections` and `filter_selected_track`. First capture the existing RED from `tests/test_track_selection.py` importing a missing module. Implement the helper so default `selected_track_id=None` returns all detections unchanged, numeric string selectors match integer track IDs, selected mode returns only the matching track, and duplicate removal removes duplicate entries for the same tracked person while preferring detections with more keypoints, then higher confidence. Keep distinct `track_id` values even if bboxes overlap, because two real people can overlap.
  Must NOT do: Do not change tracker thresholds or `tracking/simple_tracker.py`; do not deduplicate distinct track IDs solely by high IoU; do not drop untracked detections unless selected-track filtering is explicitly configured.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [6] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `tests/test_track_selection.py:3` - imports the missing helper module and is the immediate RED target.
  - Pattern:  `tests/test_track_selection.py:7` - selected-track filtering contract.
  - Pattern:  `tests/test_track_selection.py:19` - default all-track behavior must be preserved.
  - Pattern:  `tests/test_track_selection.py:30` - duplicate removal prefers better keypoints over higher confidence.
  - API/Type: `tracking/simple_tracker.py:30` - tracker assigns `track_id`; helper must run after this without changing tracker behavior.
  - API/Type: `tracking/simple_tracker.py:179` - bbox IoU helper can be imported if needed, but do not use it to merge distinct track IDs.
  - Test:     `tests/test_track_selection.py` - extend with distinct-track overlap and string selector edge cases.

  Acceptance criteria (agent-executable only):
  - [ ] RED captured before implementation: `python -m unittest tests.test_track_selection.TrackSelectionTest` fails with `ModuleNotFoundError: No module named 'ai.inference.track_selection'`.
  - [ ] GREEN captured after implementation: `python -m unittest tests.test_track_selection.TrackSelectionTest` exits 0 and reports all tests OK.
  - [ ] `python -c "from ai.inference.track_selection import deduplicate_tracked_detections, filter_selected_track; print('track-selection-ok')"` prints `track-selection-ok`.

  QA scenarios (MANDATORY - task incomplete without these):
  > Name the exact tool AND its exact invocation - not "verify it works". Browser use: use Chrome to drive the page; if Chrome is not available, download and use agent-browser (https://github.com/vercel-labs/agent-browser). Computer use: OS-level GUI automation for a non-browser desktop app.
  ```
  Scenario: Helper unit contract
    Tool:     powershell
    Steps:    New-Item -ItemType Directory -Force evidence | Out-Null; python -m unittest tests.test_track_selection.TrackSelectionTest *> evidence/task-1-active-camera-overlay-inference-mqtt.txt; exit $LASTEXITCODE
    Expected: Exit code 0 and evidence contains "OK".
    Evidence: evidence/task-1-active-camera-overlay-inference-mqtt.txt

  Scenario: Distinct overlapping track IDs are not deduplicated
    Tool:     powershell
    Steps:    @'
              import json
              from ai.inference.track_selection import deduplicate_tracked_detections
              detections = [
                  {"bbox": [0, 0, 100, 100], "track_id": 1, "confidence": 0.9},
                  {"bbox": [1, 1, 101, 101], "track_id": 2, "confidence": 0.8},
              ]
              unique, removed = deduplicate_tracked_detections(detections)
              assert [item["track_id"] for item in unique] == [1, 2], unique
              assert removed == [], removed
              print(json.dumps({"track_ids": [item["track_id"] for item in unique], "removed": removed}, sort_keys=True))
              '@ | python - > evidence/task-1-active-camera-overlay-inference-mqtt-error.json
    Expected: Exit code 0 and JSON contains `"track_ids": [1, 2]`.
    Evidence: evidence/task-1-active-camera-overlay-inference-mqtt-error.json
  ```

  Commit: YES | Message: `fix(inference): add tracked detection selection helpers` | Files: [`ai/inference/track_selection.py`, `tests/test_track_selection.py`]

- [ ] 2. Expose overlay selected/preferred track CLI and env parsing

  What to do: Refactor `scripts/serve_ai_overlay.py` argument construction into a testable parser helper if needed, then add `--selected-track-id` and alias `--preferred-track-id` with destination `selected_track_id`. Default must read `AI_SELECTED_TRACK_ID`, then `AI_PREFERRED_TRACK_ID`, then unset. Empty strings must behave as unset. Invalid non-integer input must fail parser validation before any worker starts. Keep current default camera, detector, tracker, sequence, cooldown, and MQTT arguments unchanged.
  Must NOT do: Do not start `OverlayWorker` in parser tests; do not change defaults for detector mode, tracker thresholds, sequence length/stride, min consecutive faint, or cooldown.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [3, 6] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `scripts/serve_ai_overlay.py:758` - current parser is inside `main`.
  - Pattern:  `scripts/serve_ai_overlay.py:760` - current RTSP URL default must remain unchanged.
  - Pattern:  `scripts/serve_ai_overlay.py:781` - min consecutive faint default must remain unchanged.
  - Pattern:  `scripts/serve_ai_overlay.py:782` - cooldown default must remain unchanged.
  - Pattern:  `scripts/serve_ai_overlay.py:794` - tracking mode default must remain unchanged.
  - Pattern:  `scripts/serve_ai_overlay.py:795` - track threshold default must remain unchanged.
  - Pattern:  `scripts/serve_ai_overlay.py:796` - match threshold default must remain unchanged.
  - Pattern:  `scripts/serve_ai_overlay.py:837` - `camera_login_id` fallback currently happens after parsing and must remain.
  - Test:     `tests/test_ai_overlay_server.py` - add parser-only tests here or a focused `tests/test_ai_overlay_cli.py`.

  Acceptance criteria (agent-executable only):
  - [ ] RED captured: new parser test `test_overlay_parser_accepts_selected_track_id_without_starting_worker` fails because parser has no `selected_track_id`.
  - [ ] GREEN captured: `python -m unittest tests.test_ai_overlay_server.AiOverlayServerTest.test_overlay_parser_accepts_selected_track_id_without_starting_worker` exits 0.
  - [ ] GREEN captured: `python -m unittest tests.test_ai_overlay_server.AiOverlayServerTest.test_overlay_parser_uses_preferred_track_env_alias` exits 0.
  - [ ] CLI help includes both flags: `python scripts/serve_ai_overlay.py --help` exits 0 and stdout contains `--selected-track-id` and `--preferred-track-id`.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: Overlay CLI advertises selector flags
    Tool:     powershell
    Steps:    New-Item -ItemType Directory -Force evidence | Out-Null; python scripts/serve_ai_overlay.py --help *> evidence/task-2-active-camera-overlay-inference-mqtt.txt; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; Select-String -Path evidence/task-2-active-camera-overlay-inference-mqtt.txt -Pattern "--selected-track-id"; Select-String -Path evidence/task-2-active-camera-overlay-inference-mqtt.txt -Pattern "--preferred-track-id"
    Expected: Exit code 0 and both selectors are found in help output.
    Evidence: evidence/task-2-active-camera-overlay-inference-mqtt.txt

  Scenario: Invalid selector fails before worker start
    Tool:     powershell
    Steps:    $env:AI_SELECTED_TRACK_ID='abc'; python scripts/serve_ai_overlay.py --help *> evidence/task-2-active-camera-overlay-inference-mqtt-error.txt; $env:AI_SELECTED_TRACK_ID=$null; exit 0
    Expected: Help still exits 0 because help must not validate runtime env, and parser unit test separately asserts invalid parse exits with SystemExit before `OverlayWorker.start`.
    Evidence: evidence/task-2-active-camera-overlay-inference-mqtt-error.txt
  ```

  Commit: YES | Message: `feat(overlay): add optional selected track argument` | Files: [`scripts/serve_ai_overlay.py`, `tests/test_ai_overlay_server.py`]

- [ ] 3. Forward selected/preferred selector from registered runner to overlay command

  What to do: Add optional selector fields to `ai/registered_cameras.py:69` `RunnerConfig`, parse the same CLI/env contract in `scripts/run_registered_cameras.py:35`, map it in `config_from_args` at `scripts/run_registered_cameras.py:87`, and append `--selected-track-id <value>` in `ai/registered_cameras.py:271` `build_overlay_command` only when configured. Use canonical outbound flag `--selected-track-id` even if the user supplied `--preferred-track-id` or `AI_PREFERRED_TRACK_ID`.
  Must NOT do: Do not require selector config for runner startup; do not change how `camera.camera_login_id` is passed to `--camera-id` and `--camera-login-id`; do not pass selector when unset.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [7] | Blocked by: [2]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `ai/registered_cameras.py:69` - `RunnerConfig` is the config contract to extend.
  - Pattern:  `ai/registered_cameras.py:271` - `build_overlay_command` owns the child overlay command.
  - Pattern:  `ai/registered_cameras.py:280` - current command passes `--camera-id`.
  - Pattern:  `ai/registered_cameras.py:282` - current command passes `--camera-login-id`.
  - Pattern:  `scripts/run_registered_cameras.py:35` - runner parser location.
  - Pattern:  `scripts/run_registered_cameras.py:87` - runner config mapping location.
  - Test:     `tests/test_registered_camera_runner.py:75` - current overlay command test pattern.
  - Test:     `tests/test_registered_camera_runner.py:281` - `fake_config` must include any new `RunnerConfig` field.

  Acceptance criteria (agent-executable only):
  - [ ] RED captured: new `tests.test_registered_camera_runner.RegisteredCameraRunnerTest.test_overlay_command_forwards_selected_track_id_when_configured` fails before implementation.
  - [ ] GREEN captured: `python -m unittest tests.test_registered_camera_runner.RegisteredCameraRunnerTest.test_overlay_command_forwards_selected_track_id_when_configured` exits 0.
  - [ ] GREEN captured: `python -m unittest tests.test_registered_camera_runner.RegisteredCameraRunnerTest.test_overlay_command_omits_selected_track_id_when_unset` exits 0.
  - [ ] `python scripts/run_registered_cameras.py --help` exits 0 and contains `--selected-track-id` and `--preferred-track-id`.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: Runner dry-run forwards selected track for arbitrary active cameras
    Tool:     powershell
    Steps:    New-Item -ItemType Directory -Force evidence | Out-Null; $server = Start-Job -ScriptBlock { @'
              import json
              from http.server import BaseHTTPRequestHandler, HTTPServer
              class Handler(BaseHTTPRequestHandler):
                  def do_GET(self):
                      body = json.dumps({"success": True, "data": [
                          {"cameraId": 101, "cameraLoginId": "ward_a", "rtspUrl": "rtsp://cctv/ward_a", "sourceType": "REAL_RTSP", "aiEnabled": True, "status": "ACTIVE"},
                          {"cameraId": 102, "cameraLoginId": "lobby_02", "rtspUrl": "rtsp://cctv/lobby_02", "sourceType": "REAL_RTSP", "aiEnabled": True, "status": "ACTIVE"}
                      ]}).encode("utf-8")
                      self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                  def log_message(self, *_): pass
              HTTPServer(("127.0.0.1", 18080), Handler).serve_forever()
              '@ | python - }; Start-Sleep -Seconds 1; try { python scripts/run_registered_cameras.py --backend-base-url http://127.0.0.1:18080 --dry-run --publisher console --selected-track-id 2 *> evidence/task-3-active-camera-overlay-inference-mqtt.txt; $code=$LASTEXITCODE } finally { Stop-Job $server; Remove-Job $server }; if ($code -ne 0) { exit $code }; Select-String -Path evidence/task-3-active-camera-overlay-inference-mqtt.txt -Pattern "ward_a"; Select-String -Path evidence/task-3-active-camera-overlay-inference-mqtt.txt -Pattern "lobby_02"; Select-String -Path evidence/task-3-active-camera-overlay-inference-mqtt.txt -Pattern "--selected-track-id 2"
    Expected: Exit code 0; output contains both arbitrary camera IDs and `--selected-track-id 2`.
    Evidence: evidence/task-3-active-camera-overlay-inference-mqtt.txt

  Scenario: Unset selector does not add single-track behavior
    Tool:     powershell
    Steps:    python -m unittest tests.test_registered_camera_runner.RegisteredCameraRunnerTest.test_overlay_command_omits_selected_track_id_when_unset *> evidence/task-3-active-camera-overlay-inference-mqtt-error.txt; exit $LASTEXITCODE
    Expected: Exit code 0 and evidence contains "OK".
    Evidence: evidence/task-3-active-camera-overlay-inference-mqtt-error.txt
  ```

  Commit: YES | Message: `feat(registered-cameras): forward optional selected track` | Files: [`ai/registered_cameras.py`, `scripts/run_registered_cameras.py`, `tests/test_registered_camera_runner.py`]

- [ ] 4. Prove registered runner starts every active camera without hardcoded IDs

  What to do: Add a regression test that feeds backend active-camera data containing at least two arbitrary non-`cam_04`/`cam_05` IDs, such as `ward_a` and `lobby_02`, and proves `sync_camera_workers` starts both. Also add a negative source scan assertion or test helper proving `scripts/run_registered_cameras.py`, `ai/registered_cameras.py`, and `ai/registered_camera_workers.py` contain no production `cam_04`/`cam_05` literals. Existing test fixtures may mention camera-looking strings, but production runner files must not.
  Must NOT do: Do not delete existing tests that use sample IDs; do not change backend DTO parsing unless the new test exposes a defect; do not hardcode the arbitrary QA camera IDs into production code.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [final] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `scripts/run_registered_cameras.py:147` - active cameras are loaded from backend.
  - Pattern:  `scripts/run_registered_cameras.py:170` - empty active-camera warning.
  - Pattern:  `ai/registered_camera_workers.py:204` - sync uses a camera list.
  - Pattern:  `ai/registered_camera_workers.py:209` - active map key is `camera.camera_login_id`.
  - Pattern:  `ai/registered_camera_workers.py:229` - starts missing or changed workers.
  - Pattern:  `ai/registered_camera_workers.py:241` - dry-run returns after one sync, useful for QA.
  - Test:     `tests/test_registered_camera_runner.py:174` - existing single-camera start pattern.
  - Test:     `tests/test_registered_camera_runner.py:116` - backend envelope parsing pattern.

  Acceptance criteria (agent-executable only):
  - [ ] RED captured: new `tests.test_registered_camera_runner.RegisteredCameraRunnerTest.test_sync_camera_workers_starts_all_backend_active_camera_login_ids` fails before any needed fix.
  - [ ] GREEN captured: `python -m unittest tests.test_registered_camera_runner.RegisteredCameraRunnerTest.test_sync_camera_workers_starts_all_backend_active_camera_login_ids` exits 0.
  - [ ] `rg -n "cam_04|cam_05" scripts/run_registered_cameras.py ai/registered_cameras.py ai/registered_camera_workers.py` exits with no matches.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: Production runner files contain no cam_04/cam_05 special cases
    Tool:     powershell
    Steps:    New-Item -ItemType Directory -Force evidence | Out-Null; rg -n "cam_04|cam_05" scripts/run_registered_cameras.py ai/registered_cameras.py ai/registered_camera_workers.py *> evidence/task-4-active-camera-overlay-inference-mqtt.txt; if ($LASTEXITCODE -eq 1) { exit 0 } else { exit 1 }
    Expected: Exit code 0 because ripgrep returns 1 for no matches and the wrapper converts no matches to pass.
    Evidence: evidence/task-4-active-camera-overlay-inference-mqtt.txt

  Scenario: Two arbitrary active cameras both produce overlay commands in dry-run
    Tool:     powershell
    Steps:    New-Item -ItemType Directory -Force evidence | Out-Null; $server = Start-Job -ScriptBlock { @'
              import json
              from http.server import BaseHTTPRequestHandler, HTTPServer
              class Handler(BaseHTTPRequestHandler):
                  def do_GET(self):
                      body = json.dumps({"success": True, "data": [
                          {"cameraId": 201, "cameraLoginId": "entrance_west", "rtspUrl": "rtsp://cctv/entrance_west", "sourceType": "REAL_RTSP", "aiEnabled": True, "status": "ACTIVE"},
                          {"cameraId": 202, "cameraLoginId": "icu_bed_7", "rtspUrl": "rtsp://cctv/icu_bed_7", "sourceType": "REAL_RTSP", "aiEnabled": True, "status": "ACTIVE"}
                      ]}).encode("utf-8")
                      self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                  def log_message(self, *_): pass
              HTTPServer(("127.0.0.1", 18081), Handler).serve_forever()
              '@ | python - }; Start-Sleep -Seconds 1; try { python scripts/run_registered_cameras.py --backend-base-url http://127.0.0.1:18081 --dry-run --publisher console *> evidence/task-4-active-camera-overlay-inference-mqtt-error.txt; $code=$LASTEXITCODE } finally { Stop-Job $server; Remove-Job $server }; if ($code -ne 0) { exit $code }; Select-String -Path evidence/task-4-active-camera-overlay-inference-mqtt-error.txt -Pattern "entrance_west"; Select-String -Path evidence/task-4-active-camera-overlay-inference-mqtt-error.txt -Pattern "icu_bed_7"
    Expected: Exit code 0 and output contains both arbitrary camera IDs.
    Evidence: evidence/task-4-active-camera-overlay-inference-mqtt-error.txt
  ```

  Commit: YES | Message: `test(registered-cameras): prove dynamic active camera workers` | Files: [`tests/test_registered_camera_runner.py`]

- [ ] 5. Lock MQTT payload schema, evidence linkage, and publish diagnostics

  What to do: Add focused tests around payload builders and publisher context before touching publish ownership. Assert overlay events include every tracked bbox, `faint_probability`, `action_overlay` or action-equivalent field already attached by overlay processing, raw `track_id`, display aliases when present, and evidence fields. Assert frame_sync payload uses the same cameraLoginId/frame/capture evidence key. Assert confirmed events preserve backend DTO aliases and `track_id`. Assert `_payload_context` logs `messageType`, `streamId`, `cameraLoginId`, `frameId`, `eventId` when present, and target topic.
  Must NOT do: Do not rename existing payload fields; do not remove legacy aliases; do not change MQTT default topics unless a test proves an existing bug directly related to this request.

  Parallelization: Can parallel: YES | Wave 1 | Blocks: [6] | Blocked by: []

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `ai/publishers/mqtt_payloads.py:23` - overlay payload builder includes all boxes.
  - Pattern:  `ai/publishers/mqtt_payloads.py:68` - frame_sync payload builder.
  - Pattern:  `ai/publishers/mqtt_payloads.py:103` - confirmed event payload builder.
  - Pattern:  `ai/publishers/mqtt_payloads.py:207` - `_overlay_event` converts box metadata to frontend/MQTT event fields.
  - Pattern:  `ai/publishers/event_publisher.py:65` - MQTT publish path.
  - Pattern:  `ai/publishers/event_publisher.py:165` - payload context diagnostic builder.
  - Test:     `tests/test_mqtt_payloads.py:7` - overlay schema pattern.
  - Test:     `tests/test_mqtt_payloads.py:285` - frame_sync evidence pattern.
  - Test:     `tests/test_mqtt_event_publisher.py:66` - context diagnostic pattern.

  Acceptance criteria (agent-executable only):
  - [ ] RED captured: new payload test `test_overlay_payload_preserves_action_and_faint_signal_per_track` fails before any necessary payload fix.
  - [ ] GREEN captured: `python -m unittest tests.test_mqtt_payloads.MqttPayloadsTest.test_overlay_payload_preserves_action_and_faint_signal_per_track` exits 0.
  - [ ] GREEN captured: `python -m unittest tests.test_mqtt_payloads.MqttPayloadsTest.test_frame_sync_payload_uses_same_evidence_key` exits 0.
  - [ ] GREEN captured: `python -m unittest tests.test_mqtt_event_publisher.MqttEventPublisherTest.test_payload_context_includes_publish_diagnostics` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: Payload builders emit linked overlay frame_sync event records
    Tool:     powershell
    Steps:    New-Item -ItemType Directory -Force evidence | Out-Null; @'
              import json
              from ai.publishers.mqtt_payloads import build_overlay_payload, build_frame_sync_payload, build_confirmed_event_payload
              overlay = build_overlay_payload("ward_a", 640, 360, 1782180000123, frame_id=9, captured_at_ms=1782180000100, processed_at_ms=1782180000110, published_at_ms=1782180000123, boxes=[{"x1":1,"y1":2,"x2":101,"y2":202,"score":0.9,"track_id":7,"faint_probability":0.82,"action_overlay":"ALERT Faint: 0.82","event_triggered":True}])
              sync = build_frame_sync_payload("ward_a", 9, 1782180000100, 1782180000123, 23, processed_at_ms=1782180000110, dropped_frame_count=0)
              event = build_confirmed_event_payload("ward_a", 640, 360, {"label":"Faint","score":0.82,"probabilities":{"Faint":0.82}}, {"bbox":[1,2,101,202],"track_id":7}, [], frame_id=9, captured_at_ms=1782180000100, processed_at_ms=1782180000110, published_at_ms=1782180000123)
              assert overlay["evidenceId"] == sync["evidenceId"] == event["evidenceId"]
              assert overlay["events"][0]["track_id"] == 7
              assert event["track_id"] == 7
              print(json.dumps({"overlay": overlay["messageType"], "sync": sync["messageType"], "event": event["messageType"], "evidenceId": overlay["evidenceId"]}, sort_keys=True))
              '@ | python - > evidence/task-5-active-camera-overlay-inference-mqtt.json
    Expected: Exit code 0 and JSON contains `"evidenceId": "ward_a-9-1782180000100"`.
    Evidence: evidence/task-5-active-camera-overlay-inference-mqtt.json

  Scenario: MQTT context includes event id for confirmed events
    Tool:     powershell
    Steps:    @'
              from ai.publishers.event_publisher import _payload_context
              text = _payload_context({"messageType":"event","streamId":"ward_a","cameraLoginId":"ward_a","frameId":9,"eventId":"evt-ward_a-9"}, "event", connected=True, rc=0)
              assert "topic=event" in text and "eventId=evt-ward_a-9" in text and "cameraLoginId=ward_a" in text, text
              print(text)
              '@ | python - > evidence/task-5-active-camera-overlay-inference-mqtt-error.txt
    Expected: Exit code 0 and output contains `eventId=evt-ward_a-9`.
    Evidence: evidence/task-5-active-camera-overlay-inference-mqtt-error.txt
  ```

  Commit: YES | Message: `test(mqtt): lock overlay frame sync event payload contracts` | Files: [`tests/test_mqtt_payloads.py`, `tests/test_mqtt_event_publisher.py`, `ai/publishers/mqtt_payloads.py`, `ai/publishers/event_publisher.py`]

- [ ] 6. Wire dedup and optional selection into `process_frame` before sequence buffering

  What to do: Import the Task 1 helper into `scripts/serve_ai_overlay.py`. After `update_detections_with_postprocessor` at `scripts/serve_ai_overlay.py:142` and before `normalize_detections` at `scripts/serve_ai_overlay.py:170`, run duplicate removal and then selected/preferred filtering using `args.selected_track_id`. Use the filtered detections for normalized boxes, per-track sequence buffers, LSTM predictions, action/faint attachment, overlay payloads, and confirmed events. Record skipped/dedup counts in `summary` using additive keys such as `deduplicated_detection_count`, `selected_track_skipped_count`, and `track_selection_skipped`, but do not change existing summary keys. Ensure `annotate_boxes_with_track_actions` at `scripts/serve_ai_overlay.py:296` still attaches per-track `faint_probability`, `action_overlay`, `event_triggered`, and threshold metadata.
  Must NOT do: Do not change detector output, tracker internals, model thresholds, or cooldown decisions. Do not filter all tracks when selector is unset. Do not let selected-track filtering affect exit ROI event state unless the selector is explicitly configured and documented in tests.

  Parallelization: Can parallel: YES | Wave 2 | Blocks: [7] | Blocked by: [1, 2, 5]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `scripts/serve_ai_overlay.py:99` - `process_frame` is the current processing unit.
  - Pattern:  `scripts/serve_ai_overlay.py:118` - stream ID uses `camera_login_id` fallback.
  - Pattern:  `scripts/serve_ai_overlay.py:140` - detections before tracker.
  - Pattern:  `scripts/serve_ai_overlay.py:142` - tracker postprocessor call.
  - Pattern:  `scripts/serve_ai_overlay.py:170` - normalized boxes currently happen immediately after tracking.
  - Pattern:  `scripts/serve_ai_overlay.py:224` - crop sequence buffer consumes `boxes`.
  - Pattern:  `scripts/serve_ai_overlay.py:233` - keypoint sequence buffer consumes detections.
  - Pattern:  `scripts/serve_ai_overlay.py:272` - per-track postprocessor/cooldown state is keyed by track.
  - Pattern:  `scripts/serve_ai_overlay.py:296` - action/faint is attached per track.
  - Pattern:  `scripts/serve_ai_overlay.py:320` - overlay payload builder consumes boxes.
  - API/Type: `ai/action/per_track_sequence_buffer.py:8` - keypoint buffer is per `track_id`.
  - API/Type: `ai/action/per_track_sequence_buffer.py:112` - crop buffer is per `track_id`.
  - Test:     `tests/test_ai_overlay_server.py:37` - process_frame count pattern.
  - Test:     `tests/test_ai_overlay_server.py:77` - fake publisher pattern.
  - Test:     `tests/test_ai_overlay_server.py:304` - `FakePublisher`.
  - Test:     `tests/test_ai_overlay_server.py:313` - fixed tracker pattern to extend for multi-track fixtures.

  Acceptance criteria (agent-executable only):
  - [ ] RED captured: new `tests.test_ai_overlay_server.AiOverlayServerTest.test_process_frame_default_keeps_multiple_tracks_after_dedup` fails before implementation.
  - [ ] RED captured: new `tests.test_ai_overlay_server.AiOverlayServerTest.test_process_frame_selected_track_filters_overlay_sequence_and_event` fails before implementation.
  - [ ] GREEN captured: `python -m unittest tests.test_ai_overlay_server.AiOverlayServerTest.test_process_frame_default_keeps_multiple_tracks_after_dedup` exits 0.
  - [ ] GREEN captured: `python -m unittest tests.test_ai_overlay_server.AiOverlayServerTest.test_process_frame_selected_track_filters_overlay_sequence_and_event` exits 0.
  - [ ] Regression command exits 0: `python -m unittest tests.test_ai_overlay_server tests.test_track_selection tests.test_mqtt_payloads tests.test_rtsp_inference`.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: Default multi-track flow keeps all unique tracks
    Tool:     powershell
    Steps:    New-Item -ItemType Directory -Force evidence | Out-Null; @'
              import json
              from argparse import Namespace
              import numpy as np
              from ai.action.per_track_sequence_buffer import PerTrackCropSequenceBuffers
              from ai.streams.video_reader import FramePacket
              from scripts.serve_ai_overlay import initial_summary, process_frame
              class Detector:
                  def detect(self, _frame):
                      return [
                          {"bbox":[0,0,40,80],"track_id":7,"confidence":0.91,"keypoints":[{"x":1,"y":1,"confidence":0.9}]},
                          {"bbox":[1,1,41,81],"track_id":7,"confidence":0.80,"keypoints":[{"x":1,"y":1,"confidence":0.9},{"x":2,"y":2,"confidence":0.8}]},
                          {"bbox":[100,0,160,90],"track_id":8,"confidence":0.88,"keypoints":[{"x":3,"y":3,"confidence":0.9}]},
                      ]
              class Classifier:
                  def predict(self, sequence):
                      return {"label":"Faint","score":0.8,"probabilities":{"Faint":0.8}}
              class Publisher:
                  def __init__(self): self.published = []
                  def publish(self, payload, topic=None): self.published.append((topic, payload)); return True
              args = Namespace(camera_id="ward_a", camera_login_id="ward_a", detector_mode="real", classifier_input="crops", selected_track_id=None, mqtt_camera_topic="camera", mqtt_event_topic="event", mqtt_topic=None, action_threshold=0.3, min_consecutive_faint=1, camera_cooldown_seconds=0, print_events=False, overlay_debug_tracks=False, sequence_length=2, sequence_stride=1)
              frame = np.zeros((96, 192, 3), dtype=np.uint8)
              summary = initial_summary(); buffer = PerTrackCropSequenceBuffers(sequence_length=2, stride=1, resize_size=16); publisher = Publisher()
              process_frame(FramePacket(0, 10.0, 0.0, frame), Detector(), Classifier(), buffer, summary, args, publisher=publisher)
              process_frame(FramePacket(1, 10.0, 0.1, frame), Detector(), Classifier(), buffer, summary, args, publisher=publisher)
              overlay = [payload for topic, payload in publisher.published if payload.get("messageType") == "overlay"][-1]
              track_ids = [event.get("track_id") for event in overlay["events"]]
              assert track_ids == [7, 8], track_ids
              assert summary["per_track_sequences_generated"] == {"7": 1, "8": 1}, summary["per_track_sequences_generated"]
              print(json.dumps({"track_ids": track_ids, "per_track": summary["per_track_sequences_generated"]}, sort_keys=True))
              '@ | python - > evidence/task-6-active-camera-overlay-inference-mqtt.json
    Expected: Exit code 0 and JSON contains `"track_ids": [7, 8]`.
    Evidence: evidence/task-6-active-camera-overlay-inference-mqtt.json

  Scenario: Selected track filters overlay, sequence, and event
    Tool:     powershell
    Steps:    Same inline Python as the happy path, except set `selected_track_id=8` and assert overlay track IDs are `[8]`, `summary["per_track_sequences_generated"] == {"8": 1}`, and no event payload has `track_id` 7; write stdout to `evidence/task-6-active-camera-overlay-inference-mqtt-error.json`.
    Expected: Exit code 0 and JSON contains `"track_ids": [8]`.
    Evidence: evidence/task-6-active-camera-overlay-inference-mqtt-error.json
  ```

  Commit: YES | Message: `fix(overlay): apply tracked detection selection before inference` | Files: [`scripts/serve_ai_overlay.py`, `tests/test_ai_overlay_server.py`]

- [ ] 7. Centralize OverlayWorker publish order as overlay -> frame_sync -> event

  What to do: Make the currently used `OverlayWorker._run` path own MQTT publish order. Add a small internal result/bundle abstraction in `scripts/serve_ai_overlay.py` so `process_frame` can return overlay image plus overlay payload plus confirmed-event payloads without immediately publishing when called by `OverlayWorker`. Preserve the default `process_frame` return shape and immediate publishing behavior for existing unit callers unless tests are intentionally updated. In `_run`, call `process_frame` in deferred mode, build `build_frame_sync_payload` at `scripts/serve_ai_overlay.py:695`, then publish in exact order: overlay payload to camera topic, frame_sync payload to camera topic, each confirmed event payload to event topic. Count publish attempts consistently in `mqtt_publish_count`.
  Must NOT do: Do not publish confirmed events before frame_sync in the OverlayWorker path. Do not break `process_frame` tests that call it directly. Do not change event cooldown/min-consecutive semantics; event payloads must still be generated only after `FaintEventPostProcessor.should_trigger`.

  Parallelization: Can parallel: NO | Wave 3 | Blocks: [final] | Blocked by: [3, 6]

  References (executor has NO interview context - be exhaustive):
  - Pattern:  `scripts/serve_ai_overlay.py:337` - current immediate overlay publish inside `process_frame`.
  - Pattern:  `scripts/serve_ai_overlay.py:342` - current confirmed-event loop inside `process_frame`.
  - Pattern:  `scripts/serve_ai_overlay.py:361` - current immediate event publish inside `process_frame`.
  - Pattern:  `scripts/serve_ai_overlay.py:678` - `OverlayWorker._run` calls `process_frame`.
  - Pattern:  `scripts/serve_ai_overlay.py:695` - frame_sync payload is currently built after `process_frame`.
  - Pattern:  `scripts/serve_ai_overlay.py:705` - frame_sync currently publishes after `process_frame`.
  - Pattern:  `ai/publishers/event_publisher.py:155` - topic settings helper.
  - Test:     `tests/test_ai_overlay_server.py:77` - publish metadata test to update or preserve.
  - Test:     `tests/test_ai_overlay_server.py:142` - evidence ID alignment test.
  - Test:     `tests/test_ai_overlay_server.py:304` - fake publisher pattern.

  Acceptance criteria (agent-executable only):
  - [ ] RED captured: new `tests.test_ai_overlay_server.AiOverlayServerTest.test_overlay_worker_publish_helper_orders_overlay_frame_sync_event` fails because there is no centralized helper/order yet.
  - [ ] GREEN captured: `python -m unittest tests.test_ai_overlay_server.AiOverlayServerTest.test_overlay_worker_publish_helper_orders_overlay_frame_sync_event` exits 0.
  - [ ] GREEN captured: `python -m unittest tests.test_ai_overlay_server.AiOverlayServerTest.test_process_frame_keeps_overlay_and_event_on_same_evidence_id` exits 0.
  - [ ] GREEN captured: `python -m unittest tests.test_ai_overlay_server tests.test_mqtt_payloads tests.test_mqtt_event_publisher tests.test_registered_camera_runner tests.test_track_selection` exits 0.

  QA scenarios (MANDATORY - task incomplete without these):
  ```
  Scenario: Publish helper emits overlay then frame_sync then event
    Tool:     powershell
    Steps:    New-Item -ItemType Directory -Force evidence | Out-Null; @'
              import json
              from scripts.serve_ai_overlay import publish_frame_bundle
              class Publisher:
                  def __init__(self): self.published = []
                  def publish(self, payload, topic=None): self.published.append((topic, payload)); return True
              publisher = Publisher()
              overlay = {"messageType": "overlay", "streamId": "ward_a", "cameraLoginId": "ward_a", "frameId": 9}
              sync = {"messageType": "frame_sync", "streamId": "ward_a", "cameraLoginId": "ward_a", "frameId": 9}
              event = {"messageType": "event", "streamId": "ward_a", "cameraLoginId": "ward_a", "frameId": 9, "eventId": "evt-ward_a-9", "track_id": 8}
              count = publish_frame_bundle(publisher, {"camera_topic": "camera", "event_topic": "event"}, overlay, sync, [event], "ward_a")
              observed = [(topic, payload["messageType"]) for topic, payload in publisher.published]
              assert observed == [("camera", "overlay"), ("camera", "frame_sync"), ("event", "event")], observed
              assert count == 3, count
              print(json.dumps({"observed": observed, "count": count}, sort_keys=True))
              '@ | python - > evidence/task-7-active-camera-overlay-inference-mqtt.json
    Expected: Exit code 0 and JSON shows camera overlay, camera frame_sync, event event in order.
    Evidence: evidence/task-7-active-camera-overlay-inference-mqtt.json

  Scenario: Deferred process_frame still generates event after cooldown gate
    Tool:     powershell
    Steps:    python -m unittest tests.test_ai_overlay_server.AiOverlayServerTest.test_process_frame_keeps_overlay_and_event_on_same_evidence_id *> evidence/task-7-active-camera-overlay-inference-mqtt-error.txt; exit $LASTEXITCODE
    Expected: Exit code 0 and evidence contains "OK".
    Evidence: evidence/task-7-active-camera-overlay-inference-mqtt-error.txt
  ```

  Commit: YES | Message: `fix(overlay): publish frame sync before confirmed events` | Files: [`scripts/serve_ai_overlay.py`, `tests/test_ai_overlay_server.py`]

## Final verification wave (MANDATORY - after all implementation tasks)
> Runs in PARALLEL. ALL must APPROVE. Surface results to the caller and wait for an explicit "okay" before declaring complete.
- [ ] F1. Plan compliance audit - every task done, every acceptance criterion met
- [ ] F2. Code quality review - diagnostics clean, idioms match, no dead code
- [ ] F3. Real manual QA - every QA scenario executed with evidence captured
- [ ] F4. Scope fidelity - nothing extra shipped beyond Must-Have, nothing Must-NOT-Have introduced

Required final commands:
- `python -m unittest tests.test_track_selection tests.test_ai_overlay_server tests.test_mqtt_payloads tests.test_mqtt_event_publisher tests.test_registered_camera_runner tests.test_rtsp_inference`
- `rg -n "cam_04|cam_05" scripts/run_registered_cameras.py ai/registered_cameras.py ai/registered_camera_workers.py` and treat no matches as pass.
- `python scripts/run_registered_cameras.py --help`
- `python scripts/serve_ai_overlay.py --help`

Required review risk checklist:
- Default multi-person behavior remains enabled when selector is unset.
- Selected/preferred track mode is opt-in through env/CLI and is forwarded only when configured.
- Duplicate removal cannot collapse distinct valid track IDs.
- Production tracker/model/cooldown thresholds are unchanged.
- Overlay/frame_sync/event payloads preserve evidence linkage and cameraLoginId.
- OverlayWorker publish order is overlay -> frame_sync -> event.
- No production runner code hardcodes `cam_04` or `cam_05`.

## Commit strategy
- One logical change per commit. Conventional Commits (`<type>(<scope>): <subject>` body + footer).
- Atomic: every commit builds and passes tests on its own.
- No "WIP" / "fix typo squash later" commits on the final branch - clean up before merge.
- Reference the plan file path in the final commit footer: `Plan: plans/active-camera-overlay-inference-mqtt.md`.

Suggested final commit order:
- `fix(inference): add tracked detection selection helpers`
- `feat(overlay): add optional selected track argument`
- `feat(registered-cameras): forward optional selected track`
- `test(registered-cameras): prove dynamic active camera workers`
- `test(mqtt): lock overlay frame sync event payload contracts`
- `fix(overlay): apply tracked detection selection before inference`
- `fix(overlay): publish frame sync before confirmed events`

## Success criteria
- All Must-Have shipped; all QA scenarios pass with captured evidence; F1-F4 approved; commit history clean.
- `tests/test_track_selection.py` passes and `ai/inference/track_selection.py` exists.
- Active-camera runner dry-run uses arbitrary backend cameraLoginIds and contains no production `cam_04`/`cam_05` special cases.
- `process_frame` default mode processes all unique tracks, and selected mode filters only when explicitly configured.
- Overlay payloads contain all active boxes after dedup/selection with per-track faint/action fields where LSTM has produced a signal.
- Frame_sync payloads and confirmed events carry matching evidence IDs for the same frame.
- MQTT publish order in the actual OverlayWorker path is overlay -> frame_sync -> event.

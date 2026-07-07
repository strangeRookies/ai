# MJPEG display normalization notepad

## Goal

Alerts arrive, but the browser-facing MJPEG video and AI overlay path must be normalized after the WebRTC/HLS experiment. The default display path is MJPEG while keeping dynamic `cameraLoginId` registration and preserving HLS/WebRTC as selectable fallback modes.

## Success criteria

- Frontend generates MJPEG URLs using `cameraLoginId` suffix ports: `cam_01 -> 8010`, `cam_02 -> 8011`, `cam_05 -> 8014`, without limiting the camera count to four.
- AI registered runner launches MJPEG workers with overlay baking enabled by default; explicit `--no-mjpeg-enable-overlay` remains available for rollback.
- Root AI launcher forwards the configured MJPEG port range for the GPU PC SSH tunnel.
- Runtime diagnostics can distinguish URL/port mismatch, worker not ready, RTSP unavailable, and overlay disabled.
- Evidence includes automated checks plus at least one real HTTP smoke check against live MJPEG/health endpoints.

## Current diagnosis

- Local tunneled worker health for ports `8011..8014` previously showed frame processing, bbox detections, keypoint extraction, LSTM predictions, and nonzero MJPEG frame counts.
- The active GPU worker commands previously included `--no-mjpeg-enable-overlay`, so a stream could arrive while visual AI overlay/tracking bake was absent. Code now defaults MJPEG overlay bake to enabled.
- The live GPU processes may still need restart or redeploy before the new default takes effect.

## Evidence artifacts

- `.omo/ulw-loop/evidence/mjpeg-display-http-smoke.txt`
- `docs/wiki/content/MJPEG-Display-Port-Normalization.md`

## Cleanup

No long-running local server is intentionally spawned by this notepad. HTTP smoke checks must remove temporary stream files after capture.

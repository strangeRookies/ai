# keypoint_motion54_v2 follow-ups (deferred)

This note records work **not** done in the recovery velocity-spike fix. Production
`keypoint_motion54` remains **raw per-frame hip displacement** as trained.

## Done in current release (v1 contract preserved)

- Incident Recovery relink: keypoint LSTM buffer **fresh-start** (no merge of pre-miss history; old and new track sequence state cleared).
- Motion append: discontinuity mask zeros `center_drop`/`velocity` only at recovery markers or large frame gaps (timestamp is fallback when frame metadata is missing); continuous samples unchanged.
- Feature order still `51=center_drop`, `52=velocity`, `53=torso_angle_norm`.

## Deferred to keypoint_motion54_v2 (requires retrain)

1. **Timestamp-normalized velocity**
   `velocity = displacement / max(Δt, ε)` so variable FPS / queue drops share one scale.
2. **Frame-gap normalized displacement**
   Divide by frame index step when `frame_id` skips.
3. **Confidence-aware hip midpoint**
   Weight shoulders/hips by keypoint confidence; fallback when hips missing.
4. **Acceleration / jerk channels**
   Extra dims beyond 54 — new schema id required.
5. **Learned gap token**
   Explicit "discontinuity" embedding instead of hard zero.

Any of the above changes the train/serve distribution relative to existing
`keypoint_motion54` checkpoints and must ship as a **new** `feature_schema_version`
with packaging metadata and a new training run — never as a silent rewrite of v1.

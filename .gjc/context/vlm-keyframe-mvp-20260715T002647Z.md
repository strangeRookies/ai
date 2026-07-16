# VLM keyframe MVP context

## Task statement
Convert the current VLM scaffold into an MVP E2E path that extracts exactly eight deterministic JPEG keyframes from a finalized incident clip and sends them in timestamp order to MockVlmProvider.

## Desired outcome
Support local paths, file URLs, and HTTP(S) inputs; preserve nested metadata; validate strict VLM JSON; fail safely; add focused synthetic-video tests; leave realtime detection and MQTT paths unchanged.

## Known facts and evidence
- Required repository scope is strange_ai only.
- Required checks include tests/test_vlm_process.py, tests/test_incident_vlm_pipeline.py, compileall, and git diff --check.
- Current strange_ai worktree HEAD is detached at 631895f.
- origin/develop is a167744 and is not an ancestor of detached HEAD.
- Assigned branch codex/ai-worker-flow-improvements is 3b3a6bb and is not based on current origin/develop.
- The only current untracked content before intake was .gjc session state.
- tmux is active, tmux 3.3.6 resolves, and gjc resolves to C:/Users/user/.bun/bin/gjc.exe.

## Constraints
- Do not commit directly to develop.
- Follow AGENTS.md: AI changes stay inside strange_ai and use the assigned feature branch; do not merge peer feature branches.
- Do not implement Gemini, backend/DB/pgvector, MQTT/realtime changes, S3 PUT, or advanced anonymization.
- Keep original frames in memory and clean temporary downloads/files in finally blocks.

## Unknowns / open questions
- No current feature branch is checked out. The mandated assigned branch is not based on current origin/develop, so the safe implementation branch/baseline must be selected explicitly before mutation.

## Likely codebase touchpoints
- scripts/process_vlm.py
- ai/vlm_sdk.py
- ai/vlm/incident_pipeline.py
- ai/vlm/keyframe_extractor.py (optional)
- ai/vlm/contracts.py (optional)
- tests/test_vlm_process.py
- tests/test_incident_vlm_pipeline.py

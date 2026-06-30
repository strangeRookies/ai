# Self-Improving AI Frame Evidence Prerequisite

Self-Improving AI should start after frame evidence consistency is stable.

## Reason

The RTSP reader can receive frames faster than YOLO Pose plus tracking plus LSTM can process them. The AI runtime uses a latest-frame queue, so old frames can be dropped or overwritten. In that situation, a reader-side frame counter or camera-latest metadata can point to a different frame than the one that actually entered inference.

## Stable Feedback Key

Operator feedback for FP, FN, TP, and TN should attach to:

- `evidenceId`
- `cameraLoginId`
- `frameId`
- `capturedAtMs`
- `processedAtMs`
- `publishedAtMs`
- `bbox`
- `keypoints`
- `confidence`
- `latency`
- optional `snapshotPath`
- optional `clipPath`

Current evidence identity:

```text
evidenceId = {cameraLoginId}-{frameId}-{capturedAtMs}
traceId    = evidenceId
```

## Candidate Routing

- False positive feedback becomes a hard-negative candidate.
- False negative feedback becomes a faint/fall reinforcement candidate.
- True positive feedback becomes a verified-positive candidate.
- True negative feedback becomes a verified-negative candidate.

This routing is only reliable when the feedback points to the same processed-frame evidence unit used by the AI payload, overlay, event, and stored clip metadata.

## Current Implementation Boundary

This change stabilizes evidence fields in the AI runtime and payload builders. It does not implement a new backend feedback API, external VLM API, generated-video synthetic data, or frontend parser changes.

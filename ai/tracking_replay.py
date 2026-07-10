"""Tracking replay over frame detection JSONL (no real video).

Results are synthetic-fixture logic validation only — not real video performance.
"""

from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ai.tracking_runtime_metrics import TrackingRuntimeMetrics
from ai.worker_session import SessionResetReason, WorkerSession, begin_session
from tracking.simple_tracker import SimpleTrackAssigner

DISCLAIMER = "Synthetic fixture 기반 로직 검증이며 실제 영상 성능이 아님"


@dataclass
class ReplayFrame:
    frame_index: int
    detections: list[dict]
    now: float | None = None
    source_id: str | None = None
    video_boundary: bool = False


def load_detections_jsonl(path: str | Path) -> list[ReplayFrame]:
    frames: list[ReplayFrame] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            frames.append(
                ReplayFrame(
                    frame_index=int(payload.get("frame_index", line_no - 1)),
                    detections=list(payload.get("detections") or []),
                    now=payload.get("now"),
                    source_id=payload.get("source_id"),
                    video_boundary=bool(payload.get("video_boundary", False)),
                )
            )
    return frames


def run_tracking_replay(
    frames: Iterable[ReplayFrame],
    *,
    camera_login_id: str = "cam_synthetic",
    tracker: SimpleTrackAssigner | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    tracker = tracker or SimpleTrackAssigner(match_thresh=0.2, track_buffer=10, min_box_area=1)
    session = begin_session(camera_login_id, tracker=tracker)
    metrics = TrackingRuntimeMetrics()

    frame_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    previous_source: str | None = None
    id_switch_count = 0
    fragmentation_count = 0
    reacquisition_count = 0
    hard = soft = sole = 0
    unmatched = 0
    seen_track_ids: set[int] = set()
    lost_then_seen: set[int] = set()

    for frame in frames:
        if frame.video_boundary or (previous_source is not None and frame.source_id and frame.source_id != previous_source):
            session.reset(SessionResetReason.SOURCE_CHANGE if frame.source_id else SessionResetReason.VIDEO_EOF)
            # Track IDs restart after reset; do not treat reused IDs as reacquisition.
            lost_then_seen.clear()
            seen_track_ids.clear()
            # Keep cumulative metrics across sources; tracker/session state is reset above.
        previous_source = frame.source_id or previous_source

        frame_id = session.next_frame_id()
        identity = session.identity()
        now = float(frame.now if frame.now is not None else frame.frame_index)
        detections_in = [dict(det) for det in frame.detections]
        tracked = tracker.update(detections_in, now=now)
        events = list(getattr(tracker, "last_events", []) or [])
        metrics.ingest_tracker_events(events, now=now)
        metrics.observe_active(len(getattr(tracker, "_tracks", {}) or {}))

        for event in events:
            kind = event.get("event")
            if kind == "id_switch_like":
                id_switch_count += 1
                fragmentation_count += 1
            if kind == "match":
                reason = str(event.get("reason") or "")
                if "soft" in reason:
                    soft += 1
                elif "sole" in reason:
                    sole += 1
                else:
                    hard += 1
            if kind == "new_track" and event.get("reason") == "no_match":
                unmatched += 1
            decision_rows.append(
                {
                    "frameIndex": frame.frame_index,
                    "frameId": frame_id,
                    "streamRunId": identity.stream_run_id,
                    "event": kind,
                    "reason": event.get("reason"),
                    "trackId": event.get("trackId"),
                    "iou": event.get("iou"),
                    "centerRatio": event.get("centerRatio"),
                }
            )
            if kind in {"new_track", "lost", "match", "id_switch_like"}:
                lifecycle_rows.append(
                    {
                        "frameIndex": frame.frame_index,
                        "streamRunId": identity.stream_run_id,
                        "event": kind,
                        "trackId": event.get("trackId"),
                        "reason": event.get("reason"),
                    }
                )

        for det in tracked:
            tid = det.get("track_id")
            if tid is None:
                continue
            tid = int(tid)
            if tid in lost_then_seen:
                reacquisition_count += 1
                lost_then_seen.discard(tid)
            seen_track_ids.add(tid)

        for event in events:
            if event.get("event") == "lost" and event.get("trackId") is not None:
                lost_then_seen.add(int(event["trackId"]))

        frame_rows.append(
            {
                "frameIndex": frame.frame_index,
                "frameId": frame_id,
                "streamRunId": identity.stream_run_id,
                "workerRunId": identity.worker_run_id,
                "cameraLoginId": identity.camera_login_id,
                "evidenceFrameKey": identity.evidence_frame_key(),
                "sourceId": frame.source_id,
                "inputDetections": len(frame.detections),
                "tracked": tracked,
                "diagnostics": tracker.diagnostics(),
            }
        )

    rid = run_id or f"replay-{uuid.uuid4().hex[:10]}"
    out = Path(output_dir or Path("runs") / "tracking_replay" / rid)
    out.mkdir(parents=True, exist_ok=True)

    total_matches = hard + soft + sole
    summary_metrics = {
        "disclaimer": DISCLAIMER,
        "synthetic_id_switch_count": id_switch_count,
        "fragmentation_count": fragmentation_count,
        "track_created_count": metrics.track_created_total,
        "track_removed_count": metrics.track_removed_total,
        "reacquisition_count": reacquisition_count,
        "average_track_duration": metrics.average_track_duration,
        "match_hard_count": hard,
        "match_soft_count": soft,
        "match_sole_count": sole,
        "match_hard_ratio": (hard / total_matches) if total_matches else None,
        "match_soft_ratio": (soft / total_matches) if total_matches else None,
        "match_sole_ratio": (sole / total_matches) if total_matches else None,
        "rejection_reason_distribution": dict(metrics.rejection_reasons),
        "unmatched_detection_count": unmatched,
        "frames": len(frame_rows),
        "session_reset_count": session.reset_count,
        "last_reset_reason": session.last_reset_reason,
        "runtime_metrics": metrics.summary(),
    }

    manifest = {
        "runId": rid,
        "disclaimer": DISCLAIMER,
        "cameraLoginId": camera_login_id,
        "frameCount": len(frame_rows),
        "outputDir": str(out).replace("\\", "/"),
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(summary_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "frame-results.jsonl").open("w", encoding="utf-8") as handle:
        for row in frame_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_csv(out / "track-lifecycle.csv", lifecycle_rows)
    _write_csv(out / "match-decisions.csv", decision_rows)
    report = _render_report(manifest, summary_metrics)
    (out / "report.md").write_text(report, encoding="utf-8")
    metrics.save_session_summary(out, run_id=rid)

    return {
        "runId": rid,
        "outputDir": str(out),
        "metrics": summary_metrics,
        "manifest": manifest,
        "session": session.snapshot(),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        "# Tracking Replay Report",
        "",
        f"> **{DISCLAIMER}**",
        "",
        f"- runId: `{manifest['runId']}`",
        f"- frames: {manifest['frameCount']}",
        f"- cameraLoginId: `{manifest['cameraLoginId']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "synthetic_id_switch_count",
        "fragmentation_count",
        "track_created_count",
        "track_removed_count",
        "reacquisition_count",
        "average_track_duration",
        "match_hard_count",
        "match_soft_count",
        "match_sole_count",
        "unmatched_detection_count",
        "session_reset_count",
    ]:
        lines.append(f"| {key} | {metrics.get(key)} |")
    lines.extend(["", f"Last reset reason: `{metrics.get('last_reset_reason')}`", ""])
    return "\n".join(lines) + "\n"

#!/usr/bin/env python3
"""Synthetic offline measurement for incident ROI recovery (no global conf change).

Runs a fixed scripted sequence:
  frames 0-4: track present + fall suspected register
  frames 5-20: track missing → recovery ROI attempts
  inject single recovery candidate via detect_fn on frames 8-10 only

Reports recovery attempts/successes/latency and continuity of incident_id.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.postprocess.incident_recovery import IncidentRecoveryManager, RecoveryConfig  # noqa: E402


class FakeFrame:
    def __init__(self, h=720, w=1280):
        self.shape = (h, w, 3)

    def __getitem__(self, item):
        class C:
            size = 100

        return C()


def main() -> int:
    cfg = RecoveryConfig(miss_frames_to_start=2, recovery_interval_frames=2, max_recovery_seconds=3.0)
    mgr = IncidentRecoveryManager(cfg)
    frame = FakeFrame()
    t0 = time.perf_counter()
    recovery_frames = {8, 9, 10}

    def detect(crop, conf, imgsz):
        # Only succeed once on first recovery attempt after miss>=2
        return [{"bbox": [30, 30, 90, 150], "confidence": 0.35, "keypoints": []}]

    continuity_ok = False
    incident_id = None
    for fid in range(0, 25):
        ts = fid / 30.0
        if fid < 5:
            tracked = [{"track_id": 3, "bbox": [200, 300, 280, 480], "confidence": 0.8}]
            if fid == 2:
                rec = mgr.note_fall_faint_suspected(
                    camera_login_id="cam_meas",
                    track_id=3,
                    bbox=[200, 300, 280, 480],
                    timestamp=ts,
                    frame_id=fid,
                    incident_id="inc-meas-1",
                )
                incident_id = rec.incident_id
            out = mgr.on_tracked_frame(
                camera_login_id="cam_meas",
                tracked=tracked,
                timestamp=ts,
                frame_id=fid,
                frame_shape=(720, 1280),
                frame_bgr=frame,
                detect_roi_fn=detect if fid in recovery_frames else None,
            )
        else:
            # missing track
            out = mgr.on_tracked_frame(
                camera_login_id="cam_meas",
                tracked=[],
                timestamp=ts,
                frame_id=fid,
                frame_shape=(720, 1280),
                frame_bgr=frame,
                detect_roi_fn=detect,
            )
            if any(d.get("incident_id") == incident_id and d.get("recovery_relink") for d in out):
                continuity_ok = True
                # Finalize assigns a NEW track id (never force source track_id=3).
                from ai.postprocess.track_state_migration import finalize_recovery_detections
                from tracking.simple_tracker import SimpleTrackAssigner

                tracker = SimpleTrackAssigner()
                finalized, migrations = finalize_recovery_detections(
                    [d for d in out if d.get("recovery_relink")],
                    camera_login_id="cam_meas",
                    incident_recovery=mgr,
                    tracker=tracker,
                    now=ts,
                )
                if finalized and int(finalized[0]["track_id"]) == 3:
                    continuity_ok = False  # forced source id is a fail
                if migrations and migrations[0].get("from_track_id") != 3:
                    continuity_ok = False
                # after success stop detecting further to avoid spam
                def detect(_c, conf, imgsz):  # noqa: F811
                    return []

    elapsed = time.perf_counter() - t0
    diag = mgr.diagnostics("cam_meas")
    report = {
        "elapsed_sec": round(elapsed, 4),
        "analysis_fps_proxy": round(25 / max(elapsed, 1e-6), 2),
        "incident_continuity": continuity_ok,
        "incident_id": incident_id,
        "diagnostics": diag,
        "wrong_relink": diag.get("wrong_relink"),
        "wrong_relink_evaluation_status": diag.get("wrong_relink_evaluation_status", "not_evaluated"),
        "note": "synthetic offline measure; not a real CCTV clip",
    }
    print(json.dumps(report, indent=2))
    out_path = ROOT / "runs" / "incident_recovery" / "synthetic_measure.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # soft success criterion for synthetic
    ok = continuity_ok and diag.get("recovery_successes", 0) >= 1 and diag.get("wrong_relink_evaluation_status") == "not_evaluated"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

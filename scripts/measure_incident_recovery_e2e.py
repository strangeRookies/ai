#!/usr/bin/env python3
"""End-to-end incident ROI recovery measure on a fixed MP4 frame range.

Runs global detector + simple tracker + IncidentRecoveryManager over
[--start-frame, --end-frame). Optionally injects a global-detector miss window
to exercise ROI recovery deterministically.

Metrics:
  - end-to-end latency samples (recovery attempt wall time)
  - recovery success / reject / timeout
  - wrong relink (manager counter)
  - incident continuity (same incident_id after relink)
  - analysis FPS over the measured range

Does not change global detector conf/imgsz. Recovery YOLO uses conf=0.05 imgsz=640
on crop only.

Example:
  python scripts/measure_incident_recovery_e2e.py \\
    --video path/to/clip.mp4 --start-frame 0 --end-frame 150 \\
    --inject-miss-start 40 --inject-miss-end 70 --register-track-frame 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.postprocess.incident_recovery import (  # noqa: E402
    IncidentRecoveryManager,
    RecoveryConfig,
    make_detect_roi_fn_from_yolo_pose,
)
from ai.postprocess.track_state_migration import finalize_recovery_detections  # noqa: E402
from tracking.simple_tracker import SimpleTrackAssigner  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Path to MP4 (or any OpenCV-readable video)")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=0, help="Exclusive end; 0 = full video")
    p.add_argument("--yolo-model", default="yolo26n-pose.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--detector-conf", type=float, default=0.25)
    p.add_argument(
        "--scripted-person",
        action="store_true",
        help="Use scripted person bbox on real MP4 decode (no YOLO weights required)",
    )
    p.add_argument("--camera-login-id", default="cam_e2e")
    p.add_argument("--inject-miss-start", type=int, default=-1, help="Inclusive frame to blank global dets")
    p.add_argument("--inject-miss-end", type=int, default=-1, help="Exclusive end of blank window")
    p.add_argument(
        "--register-track-frame",
        type=int,
        default=-1,
        help="Frame index to register fall-suspected on the largest track (default: mid of pre-miss)",
    )
    p.add_argument("--miss-frames-to-start", type=int, default=2)
    p.add_argument("--output", default=str(ROOT / "runs" / "incident_recovery" / "e2e_measure.json"))
    p.add_argument("--max-frames", type=int, default=0, help="Safety cap after start")
    return p.parse_args()


def _largest_track(tracked):
    best = None
    best_area = -1.0
    for d in tracked:
        if d.get("track_id") is None:
            continue
        bb = d.get("bbox") or d.get("smoothed_bbox")
        if not bb or len(bb) < 4:
            continue
        area = max(0.0, float(bb[2]) - float(bb[0])) * max(0.0, float(bb[3]) - float(bb[1]))
        if area > best_area:
            best_area = area
            best = d
    return best


def _scripted_person_bbox(fid: int) -> list[float]:
    # Slowly drifting lying-like box matching synth_person.mp4 authoring.
    x1 = 200.0 + fid / 10.0
    y1 = 280.0
    x2 = 320.0 + fid / 10.0
    y2 = 360.0
    return [x1, y1, x2, y2]


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(json.dumps({"error": "video_not_found", "path": str(video_path)}), flush=True)
        return 2

    import cv2

    detector = None
    detect_roi = None
    scripted = bool(args.scripted_person)
    if not scripted:
        try:
            from detector.yolo_pose_detector import YoloPoseDetector

            detector = YoloPoseDetector(
                model_path=args.yolo_model,
                device=args.device,
                imgsz=int(args.imgsz),
                conf=float(args.detector_conf),
            )
            detect_roi = make_detect_roi_fn_from_yolo_pose(detector)
        except Exception as exc:
            print(json.dumps({"warn": "yolo_unavailable_fallback_scripted", "error": str(exc)}), flush=True)
            scripted = True

    if scripted:
        # ROI detect returns crop-local person box relative to expanded ROI origin.
        def detect_roi(crop, conf, imgsz):  # noqa: ARG001
            # Person is roughly centered in recovery ROI for the scripted path.
            ch = int(getattr(crop, "shape", [80, 80])[0]) if hasattr(crop, "shape") else 80
            cw = int(getattr(crop, "shape", [80, 80])[1]) if hasattr(crop, "shape") else 80
            # Prefer a stable mid-size box inside crop so geometry gates pass after offset map.
            bw, bh = max(40, cw // 3), max(50, ch // 2)
            x1 = max(0, (cw - bw) // 2)
            y1 = max(0, (ch - bh) // 2)
            return [{"bbox": [float(x1), float(y1), float(x1 + bw), float(y1 + bh)], "confidence": 0.4, "keypoints": []}]

    tracker = SimpleTrackAssigner()
    recovery = IncidentRecoveryManager(
        RecoveryConfig(miss_frames_to_start=int(args.miss_frames_to_start), recovery_interval_frames=1)
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(json.dumps({"error": "video_open_failed", "path": str(video_path)}), flush=True)
        return 2

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start = max(0, int(args.start_frame))
    end = int(args.end_frame) if int(args.end_frame) > 0 else (total if total > 0 else start + 10_000)
    if args.max_frames > 0:
        end = min(end, start + int(args.max_frames))

    miss_s = int(args.inject_miss_start)
    miss_e = int(args.inject_miss_end)
    reg_frame = int(args.register_track_frame)
    if reg_frame < 0 and miss_s > start:
        reg_frame = max(start, miss_s - 5)

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = 0
    continuity_ok = False
    incident_id = None
    source_track_id = None
    linked_track_ids: list[int] = []
    migrations: list[dict] = []
    first_recovery_success_frame = None
    forced_source = False
    t0 = time.perf_counter()
    fid = start

    while fid < end:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ts = fid / 30.0
        if scripted:
            raw = [
                {
                    "bbox": _scripted_person_bbox(fid),
                    "confidence": 0.85,
                    "keypoints": [{"x": 0, "y": 0, "confidence": 0.5}] * 17,
                }
            ]
        else:
            raw = detector.detect(frame)
        if miss_s >= 0 and miss_e > miss_s and miss_s <= fid < miss_e:
            # Force global detector miss window (ROI recovery only path).
            raw = []
        tracked = tracker.update(raw, now=ts)

        if reg_frame >= 0 and fid == reg_frame:
            target = _largest_track(tracked)
            if target is not None:
                bb = target.get("bbox") or target.get("smoothed_bbox")
                source_track_id = int(target["track_id"])
                rec = recovery.note_fall_faint_suspected(
                    camera_login_id=args.camera_login_id,
                    track_id=source_track_id,
                    bbox=bb,
                    timestamp=ts,
                    frame_id=fid,
                    incident_id="inc-e2e-1",
                )
                incident_id = rec.incident_id

        h, w = frame.shape[:2]
        tracked = recovery.on_tracked_frame(
            camera_login_id=args.camera_login_id,
            tracked=tracked,
            timestamp=ts,
            frame_id=fid,
            frame_shape=(h, w),
            frame_bgr=frame,
            detect_roi_fn=detect_roi,
        )
        tracked, migs = finalize_recovery_detections(
            tracked,
            camera_login_id=args.camera_login_id,
            incident_recovery=recovery,
            tracker=tracker,
            now=ts,
        )
        if migs:
            migrations.extend(migs)
            for m in migs:
                linked_track_ids.append(int(m["to_track_id"]))
                if first_recovery_success_frame is None:
                    first_recovery_success_frame = fid
                if incident_id and m.get("incident_id") == incident_id:
                    continuity_ok = True
                if source_track_id is not None and int(m["to_track_id"]) == int(source_track_id):
                    forced_source = True

        for d in tracked:
            if d.get("incident_id") == incident_id and d.get("recovery_relink"):
                continuity_ok = True
                if d.get("track_id") is not None:
                    linked_track_ids.append(int(d["track_id"]))
                    if source_track_id is not None and int(d["track_id"]) == int(source_track_id):
                        forced_source = True

        frames += 1
        fid += 1

    cap.release()
    elapsed = time.perf_counter() - t0
    diag = recovery.diagnostics(args.camera_login_id)
    # Boundary reset validation (EOF / stream end)
    recovery.reset_camera(args.camera_login_id)
    open_after_reset = len(recovery.open_incidents(args.camera_login_id))

    wrong_relink = diag.get("wrong_relink")
    wrong_relink_evaluation_status = diag.get("wrong_relink_evaluation_status", "not_evaluated")
    recovery_successes = int(diag.get("recovery_successes") or 0)

    report = {
        "video": str(video_path),
        "mode": "scripted_person" if scripted else "yolo_pose",
        "start_frame": start,
        "end_frame_exclusive": end,
        "frames_processed": frames,
        "elapsed_sec": round(elapsed, 4),
        "analysis_fps": round(frames / max(elapsed, 1e-6), 3),
        "inject_miss": {"start": miss_s, "end": miss_e},
        "register_track_frame": reg_frame,
        "source_track_id": source_track_id,
        "incident_id": incident_id,
        "incident_continuity": continuity_ok,
        "first_recovery_success_frame": first_recovery_success_frame,
        "recovery_latency_to_success_frames": (
            None
            if first_recovery_success_frame is None or miss_s < 0
            else int(first_recovery_success_frame) - int(miss_s)
        ),
        "migrations": migrations,
        "linked_track_ids": sorted(set(linked_track_ids)),
        "forced_source_id": forced_source,
        "wrong_relink": wrong_relink,
        "wrong_relink_evaluation_status": wrong_relink_evaluation_status,
        "recovery_successes": recovery_successes,
        "mean_recovery_latency_ms": diag.get("mean_recovery_latency_ms"),
        "diagnostics": diag,
        "open_incidents_after_eof_reset": open_after_reset,
        "pass_criteria": {
            "recovery_success_ge_1": recovery_successes >= 1,
            "wrong_relink_not_evaluated": wrong_relink_evaluation_status == "not_evaluated",
            "incident_continuity": continuity_ok,
            "no_forced_source_id": not forced_source,
            "eof_reset_clears_context": open_after_reset == 0,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    ok = all(report["pass_criteria"].values())
    # If no miss window / no person, still exit 0 with report (measure completed).
    if miss_s < 0 or frames == 0:
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

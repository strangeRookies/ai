#!/usr/bin/env python3
"""Synthetic P1/P2 GT validation for near-dup suppression safety.

Creates 200 frames with two separated walkers + occasional near-dup on P1.
Measures hijacking, wrong_suppression (true P2 filtered), purity/coverage.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_tracking_from_cache import CONFIGS, run_tracker_config  # noqa: E402


def make_frames(n=200, fps=30.0):
    frames = []
    for i in range(n):
        # P1 walks left-to-right on left half
        p1 = [40 + i * 0.8, 200, 100 + i * 0.8, 420]
        # P2 walks right-to-left on right half (never IoU>0.1 with P1)
        p2 = [900 - i * 0.6, 180, 960 - i * 0.6, 400]
        dets = [
            {
                "detection_index": 0,
                "bbox_xyxy": p1,
                "confidence": 0.85,
                "class_id": 0,
                "keypoints": [],
                "avg_keypoint_conf": 0.5,
                "valid_keypoints": 0,
                "gt_label": "P1",
            },
            {
                "detection_index": 1,
                "bbox_xyxy": p2,
                "confidence": 0.82,
                "class_id": 0,
                "keypoints": [],
                "avg_keypoint_conf": 0.5,
                "valid_keypoints": 0,
                "gt_label": "P2",
            },
        ]
        # Inject near-dup of P1 every 10 frames (should be suppressed, not mint / not steal P2)
        if i % 10 == 5:
            nd = [p1[0] + 2, p1[1] + 2, p1[2] + 2, p1[3] + 2]
            dets.append(
                {
                    "detection_index": 2,
                    "bbox_xyxy": nd,
                    "confidence": 0.40,
                    "class_id": 0,
                    "keypoints": [],
                    "avg_keypoint_conf": 0.4,
                    "valid_keypoints": 0,
                    "gt_label": "P1_DUP",
                }
            )
        frames.append(
            {
                "frame_id": i,
                "timestamp_ms": int(i * 1000 / fps),
                "source_width": 1280,
                "source_height": 720,
                "detections": dets,
            }
        )
    return frames


def evaluate(frame_results_path: Path, frames: list[dict]) -> dict:
    cache = {int(f["frame_id"]): f for f in frames}
    # track_id -> hist of gt labels (excluding DUP)
    track_hist = defaultdict(lambda: defaultdict(int))
    person_track = defaultdict(lambda: defaultdict(int))
    person_vis = defaultdict(int)
    wrong_suppress = 0
    missed_person = 0
    merged_frames = 0
    hijack = 0
    last_owner_center = {}
    last_track_by_person = {}
    person_identity_switches = defaultdict(int)

    with frame_results_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fid = int(row["frame_id"])
            crec = cache[fid]
            dets = crec["detections"]
            tracked = row.get("tracked") or []
            # person presence by GT labels in raw dets
            labels = {d.get("gt_label") for d in dets if d.get("gt_label") in {"P1", "P2"}}
            for lab in labels:
                person_vis[lab] += 1
            if len(tracked) < len(labels):
                missed_person += len(labels) - len(tracked)
            # greedy match track to non-dup dets by IoU
            from tracking.simple_tracker import bbox_iou

            used = set()
            for t in tracked:
                tbb = t.get("bbox")
                tid = t.get("track_id")
                if tid is None or not tbb:
                    continue
                best, best_iou, best_lab = None, 0.0, None
                for d in dets:
                    lab = d.get("gt_label")
                    if lab not in {"P1", "P2"} or lab in used:
                        continue
                    iou = bbox_iou(tbb, d["bbox_xyxy"])
                    if iou > best_iou:
                        best_iou, best, best_lab = iou, d, lab
                if best is not None and best_iou >= 0.1:
                    used.add(best_lab)
                    track_hist[int(tid)][best_lab] += 1
                    person_track[best_lab][int(tid)] += 1
                    previous_track_id = last_track_by_person.get(best_lab)
                    if previous_track_id is not None and previous_track_id != int(tid):
                        person_identity_switches[best_lab] += 1
                    last_track_by_person[best_lab] = int(tid)
                    c = ((float(tbb[0]) + float(tbb[2])) / 2, (float(tbb[1]) + float(tbb[3])) / 2)
                    if int(tid) in last_owner_center:
                        prev_lab, pc = last_owner_center[int(tid)]
                        if prev_lab != best_lab and prev_lab in {"P1", "P2"} and best_lab in {"P1", "P2"}:
                            # same track id jumped to other person label
                            hijack += 1
                    last_owner_center[int(tid)] = (best_lab, c)
            # merged: one track box covers both P1 and P2 with high IoU
            for t in tracked:
                tbb = t.get("bbox")
                if not tbb:
                    continue
                hits = 0
                for d in dets:
                    if d.get("gt_label") in {"P1", "P2"} and bbox_iou(tbb, d["bbox_xyxy"]) >= 0.5:
                        hits += 1
                if hits >= 2:
                    merged_frames += 1

    # wrong suppress approx: frames where P2 present but no track matched P2
    wrong_suppress = 0
    with frame_results_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fid = int(row["frame_id"])
            crec = cache[fid]
            labels = {d.get("gt_label") for d in crec["detections"] if d.get("gt_label") in {"P1", "P2"}}
            tracked = row.get("tracked") or []
            from tracking.simple_tracker import bbox_iou

            covered = set()
            for t in tracked:
                tbb = t.get("bbox")
                if not tbb:
                    continue
                for d in crec["detections"]:
                    lab = d.get("gt_label")
                    if lab in {"P1", "P2"} and bbox_iou(tbb, d["bbox_xyxy"]) >= 0.2:
                        covered.add(lab)
            for lab in labels - covered:
                wrong_suppress += 1  # missed coverage of a real person

    purities = []
    for tid, hist in track_hist.items():
        total = sum(hist.values()) or 1
        purities.append(max(hist.values()) / total)
    coverages = []
    for lab, hist in person_track.items():
        vis = person_vis.get(lab) or 1
        coverages.append((max(hist.values()) if hist else 0) / vis)

    return {
        "frames": len(frames),
        "hijack_count": hijack,
        "track_owner_switch_count": hijack,
        "person_identity_switches": {label: int(person_identity_switches.get(label, 0)) for label in ("P1", "P2")},
        "person_identity_switch_count": sum(person_identity_switches.values()),
        "wrong_suppression_or_miss": wrong_suppress,
        "missed_person_count": missed_person,
        "two_person_merged_frames": merged_frames,
        "mean_track_purity": round(sum(purities) / len(purities), 4) if purities else None,
        "mean_track_coverage": round(sum(coverages) / len(coverages), 4) if coverages else None,
        "person_visible": dict(person_vis),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--configs",
        default="I_new030_prev_bbox,K_I_claimed_iou_suppress,L_I_hybrid_suppress,M_I_hybrid_kp_safe",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = make_frames(200)
    results = {}
    for name in [n.strip() for n in args.configs.split(",") if n.strip()]:
        cfg = CONFIGS[name]
        sub = out_dir / name
        summary = run_tracker_config(frames, cfg, sub, source_fps=30.0)
        metrics = evaluate(sub / "frame_results.jsonl", frames)
        metrics["tracker_summary"] = {
            k: summary.get(k)
            for k in (
                "unexpected_new_tracks",
                "multi_det_extra",
                "near_dup_suppress_count",
                "duplicate_frames",
                "id_retention_rate",
            )
        }
        results[name] = metrics
        print(name, json.dumps(metrics, ensure_ascii=False), flush=True)
    (out_dir / "two_person_synthetic_gt.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Best-available multi-person metrics without full GT.

Pseudo-identity labeling on multi-det frames:
- Sort detections left-to-right by bbox center x → P1, P2, ...
- Each frame's tracked boxes are matched to detection pseudo-ids by max IoU
- Build track_id → pseudo-person histogram
- track_purity: for each track_id, max_person_count / track_frames
- track_coverage: for each person, dominant_track_frames / person_visible_frames
- hijack_proxy: same track_id jumps far between consecutive multi frames

Not a substitute for full P1/P2 GT annotation, but enough to compare configs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def center(b):
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def bbox_iou(left, right):
    if not left or not right or len(left) < 4 or len(right) < 4:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(v) for v in left[:4]]
    rx1, ry1, rx2, ry2 = [float(v) for v in right[:4]]
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1) + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-results", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = {}
    multi_frames = 0
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("_type") == "meta":
                continue
            cache[int(o["frame_id"])] = o
            if len(o.get("detections") or []) >= 2:
                multi_frames += 1

    # track_id -> Counter(person_label)
    track_person_hist: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # person_label -> Counter(track_id)
    person_track_hist: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    person_visible: dict[str, int] = defaultdict(int)
    track_frames: dict[int, int] = defaultdict(int)

    last_owner = {}
    hijack = 0
    multi_tracked = 0
    id_switches_on_multi = 0
    prev_ids = set()

    with open(args.frame_results, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fid = int(row["frame_id"])
            tracked = row.get("tracked") or []
            ids = {int(t["track_id"]) for t in tracked if t.get("track_id") is not None}
            crec = cache.get(fid)
            if not crec:
                prev_ids = ids
                continue
            dets = list(crec.get("detections") or [])
            if len(dets) < 2:
                prev_ids = ids
                continue

            multi_tracked += 1
            if prev_ids and ids and ids != prev_ids:
                id_switches_on_multi += 1

            # Pseudo persons left-to-right
            labeled = []
            for d in dets:
                bb = d.get("bbox_xyxy") or []
                if len(bb) < 4:
                    continue
                labeled.append((center(bb)[0], bb, d))
            labeled.sort(key=lambda x: x[0])
            persons = []
            for i, (_, bb, d) in enumerate(labeled):
                plabel = f"P{i + 1}"
                persons.append((plabel, bb))
                person_visible[plabel] += 1

            # Greedy match track -> person by IoU
            used_persons = set()
            for t in tracked:
                tid = t.get("track_id")
                tbb = t.get("bbox")
                if tid is None or not tbb or len(tbb) < 4:
                    continue
                tid = int(tid)
                track_frames[tid] += 1
                best_p, best_iou = None, 0.0
                for plabel, pbb in persons:
                    if plabel in used_persons:
                        continue
                    iou = bbox_iou(tbb, pbb)
                    if iou > best_iou:
                        best_iou = iou
                        best_p = plabel
                if best_p is not None and best_iou >= 0.1:
                    used_persons.add(best_p)
                    track_person_hist[tid][best_p] += 1
                    person_track_hist[best_p][tid] += 1

                c = center(tbb)
                if tid in last_owner:
                    pf, pc = last_owner[tid]
                    if fid - pf <= 15:
                        diag = max(1.0, ((tbb[2] - tbb[0]) ** 2 + (tbb[3] - tbb[1]) ** 2) ** 0.5)
                        if dist(c, pc) / diag > 1.5:
                            hijack += 1
                last_owner[tid] = (fid, c)

            prev_ids = ids

    purities = []
    for tid, hist in track_person_hist.items():
        total = sum(hist.values()) or 1
        purities.append(max(hist.values()) / total)
    mean_purity = sum(purities) / len(purities) if purities else None

    coverages = []
    for plabel, hist in person_track_hist.items():
        vis = person_visible.get(plabel) or 1
        if not hist:
            coverages.append(0.0)
            continue
        dominant = max(hist.values())
        coverages.append(dominant / vis)
    mean_coverage = sum(coverages) / len(coverages) if coverages else None

    out = {
        "multi_det_frames_in_cache": multi_frames,
        "multi_det_frames_seen_in_results": multi_tracked,
        "id_set_changes_on_multi_frames": id_switches_on_multi,
        "hijack_proxy_count": hijack,
        "mean_track_purity": None if mean_purity is None else round(mean_purity, 4),
        "mean_track_coverage": None if mean_coverage is None else round(mean_coverage, 4),
        "person_labels": sorted(person_visible.keys()),
        "person_visible_frames": dict(person_visible),
        "note": (
            "Pseudo-identity: left-to-right P1..Pn on multi-det frames; "
            "purity/coverage are best-available without manual GT."
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

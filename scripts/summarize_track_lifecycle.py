#!/usr/bin/env python3
"""Summarize [track-lifecycle] JSON lines from overlay logs (stdin or files).

Usage on GPU host:
  python scripts/summarize_track_lifecycle.py runs/registered_cameras/*-overlay.log
  tail -n 5000 runs/registered_cameras/*-overlay.log | python scripts/summarize_track_lifecycle.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PREFIX = "[track-lifecycle] "
LINE_RE = re.compile(r"\[track-lifecycle\]\s*(\{.*\})\s*$")


def iter_records(paths: list[str]):
    if not paths:
        for line in sys.stdin:
            rec = parse_line(line)
            if rec is not None:
                yield rec
        return
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"[warn] skip {path}: {exc}", file=sys.stderr)
            continue
        for line in text.splitlines():
            rec = parse_line(line)
            if rec is not None:
                yield rec


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if PREFIX not in line:
        return None
    m = LINE_RE.search(line)
    raw = m.group(1) if m else line.split(PREFIX, 1)[-1].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def main(argv: list[str]) -> int:
    records = list(iter_records(argv[1:]))
    if not records:
        print("no [track-lifecycle] records found")
        return 1

    by_camera: dict[str, Counter] = defaultdict(Counter)
    reason_counter: Counter = Counter()
    event_counter: Counter = Counter()
    confs_new: list[float] = []
    lost_missing: list[float] = []
    samples: list[str] = []

    for rec in records:
        cam = str(rec.get("cameraLoginId") or "?")
        for ev in rec.get("events") or []:
            et = str(ev.get("event") or "?")
            reason = str(ev.get("reason") or "-")
            key = f"{et}:{reason}"
            by_camera[cam][key] += 1
            event_counter[et] += 1
            reason_counter[key] += 1
            if et == "new_track" and ev.get("confidence") is not None:
                confs_new.append(float(ev["confidence"]))
            if et == "lost" and ev.get("missingSeconds") is not None:
                lost_missing.append(float(ev["missingSeconds"]))
            if len(samples) < 8 and et in {"lost", "new_track", "filter"}:
                samples.append(
                    f"{cam} f={rec.get('frameId')} {et}/{reason} "
                    f"tid={ev.get('trackId')} conf={ev.get('confidence')} "
                    f"iou={(ev.get('bestRejected') or {}).get('rejectedCandidates', [{}])[0].get('iou') if et == 'new_track' else None} "
                    f"missSec={ev.get('missingSeconds')}"
                )

    print("=== track-lifecycle summary ===")
    print(f"records: {len(records)}")
    print(f"events:  {sum(event_counter.values())}")
    print()
    print("-- by event --")
    for k, v in event_counter.most_common():
        print(f"  {k:16s} {v}")
    print()
    print("-- by event:reason --")
    for k, v in reason_counter.most_common(20):
        print(f"  {k:40s} {v}")
    print()
    print("-- by camera (top reasons) --")
    for cam in sorted(by_camera):
        top = ", ".join(f"{k}={v}" for k, v in by_camera[cam].most_common(6))
        print(f"  {cam}: {top}")
    if confs_new:
        confs_new.sort()
        print()
        print("-- new_track confidence --")
        print(
            f"  n={len(confs_new)} min={confs_new[0]:.3f} "
            f"p50={confs_new[len(confs_new)//2]:.3f} max={confs_new[-1]:.3f}"
        )
        under = sum(1 for c in confs_new if c < 0.25)
        print(f"  conf<0.25: {under}/{len(confs_new)} ({100*under/len(confs_new):.0f}%)")
    if lost_missing:
        lost_missing.sort()
        print()
        print("-- lost missingSeconds --")
        print(
            f"  n={len(lost_missing)} min={lost_missing[0]:.2f} "
            f"p50={lost_missing[len(lost_missing)//2]:.2f} max={lost_missing[-1]:.2f}"
        )
    if samples:
        print()
        print("-- sample events (up to 8) --")
        for s in samples:
            print(f"  {s}")
    print()
    print("Paste this whole summary (not raw logs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

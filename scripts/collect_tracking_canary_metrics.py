#!/usr/bin/env python3
"""Collect tracking-canary / fps-audit windows from a camera overlay log for N seconds."""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path


BLOCK_HEADERS = (
    "[tracking-canary]",
    "[fps-audit]",
    "[tracking-config]",
    "[near-dup-suppress]",
    "[track-id-switch]",
)


def parse_blocks(text: str) -> list[dict]:
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        header = None
        for h in BLOCK_HEADERS:
            if line.startswith(h):
                header = h.strip("[]")
                break
        if header is None:
            i += 1
            continue
        rec = {"_type": header}
        i += 1
        # same-line key=value after header?
        rest = line[line.find("]") + 1 :].strip()
        if rest:
            for part in rest.split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    rec[k] = v
        while i < len(lines):
            l2 = lines[i]
            if l2.startswith("[") and any(l2.startswith(h) for h in BLOCK_HEADERS):
                break
            if l2.startswith("[") and not l2.startswith("[tracking") and not l2.startswith("[fps") and not l2.startswith("[near") and not l2.startswith("[track-id"):
                # other log section
                if "=" not in l2:
                    break
            if "=" in l2 and not l2.strip().startswith("{"):
                k, v = l2.split("=", 1)
                rec[k.strip()] = v.strip()
                i += 1
                continue
            break
        blocks.append(rec)
    return blocks


def to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def summarize(blocks: list[dict], camera: str) -> dict:
    canary_windows = [b for b in blocks if b.get("_type") == "tracking-canary" and b.get("cameraLoginId") == camera]
    fps_windows = [b for b in blocks if b.get("_type") == "fps-audit" and b.get("cameraLoginId") == camera]
    configs = [b for b in blocks if b.get("_type") == "tracking-config" and b.get("cameraLoginId") == camera]
    suppress = [b for b in blocks if b.get("_type") == "near-dup-suppress" and b.get("cameraLoginId") == camera]
    switches = [b for b in blocks if b.get("_type") == "track-id-switch" and b.get("cameraLoginId") == camera]

    def avg(key, rows):
        vals = [to_float(r.get(key)) for r in rows if to_float(r.get(key)) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    multi = sum(int(to_float(r.get("multiDetExtra"), 0) or 0) for r in canary_windows)
    new_sum = sum(int(to_float(r.get("newTracks"), 0) or 0) for r in canary_windows)
    lost_sum = sum(int(to_float(r.get("lostTracks"), 0) or 0) for r in canary_windows)
    dup_sum = sum(int(to_float(r.get("duplicateFrames"), 0) or 0) for r in canary_windows)
    window_sec = sum(to_float(r.get("windowSec"), 0) or 0 for r in canary_windows)
    minutes = max(window_sec / 60.0, 1e-9)

    return {
        "cameraLoginId": camera,
        "tracking_config_samples": configs[-3:],
        "canary_windows": len(canary_windows),
        "fps_windows": len(fps_windows),
        "window_sec_total": round(window_sec, 1),
        "analysis_fps_avg": avg("analysisFps", canary_windows) or avg("analysisFps", fps_windows),
        "captured_fps_avg": avg("capturedFps", fps_windows),
        "tracker_update_fps_avg": avg("trackerUpdateFps", fps_windows),
        "mjpeg_emit_fps_avg": avg("mjpegEmitFps", fps_windows),
        "new_tracks": new_sum,
        "new_tracks_per_min": round(new_sum / minutes, 4),
        "lost_tracks": lost_sum,
        "lost_tracks_per_min": round(lost_sum / minutes, 4),
        "multi_det_extra": multi,
        "duplicate_frames": dup_sum,
        "suppressed_detections": sum(int(to_float(r.get("suppressedDetections"), 0) or 0) for r in canary_windows),
        "near_dup_suppress_events": len(suppress),
        "track_id_switch_events": len(switches),
        "id_retention_proxy_avg": avg("idRetentionProxy", canary_windows),
        "max_active_tracks_max": max(
            [int(to_float(r.get("maxActiveTracks"), 0) or 0) for r in canary_windows] or [0]
        ),
        "max_ghost_duration_sec_max": max(
            [to_float(r.get("maxGhostDurationSec"), 0) or 0 for r in canary_windows] or [0]
        ),
        "last_canary_window": canary_windows[-1] if canary_windows else None,
        "last_fps_window": fps_windows[-1] if fps_windows else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="overlay log path")
    ap.add_argument("--camera", default="cam_03")
    ap.add_argument("--duration-sec", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poll-sec", type=float, default=5.0)
    args = ap.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    # follow log from current end (only new data during window)
    offset = log_path.stat().st_size if log_path.is_file() else 0
    chunks: list[str] = []
    print(f"[collect] start camera={args.camera} duration={args.duration_sec}s log={log_path} offset={offset}", flush=True)
    while time.time() - start < args.duration_sec:
        if log_path.is_file():
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                data = f.read()
                offset = f.tell()
            if data:
                chunks.append(data)
                # live progress: count new canary windows in this chunk
                n = data.count("[tracking-canary]")
                if n:
                    print(f"[collect] +{n} tracking-canary blocks elapsed={int(time.time()-start)}s", flush=True)
        time.sleep(args.poll_sec)
    text = "".join(chunks)
    blocks = parse_blocks(text)
    summary = summarize(blocks, args.camera)
    summary["duration_sec_requested"] = args.duration_sec
    summary["duration_sec_elapsed"] = round(time.time() - start, 1)
    summary["raw_block_count"] = len(blocks)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize_jsonl_path(path: Path) -> dict[str, dict]:
    stats: dict[str, dict] = defaultdict(_empty_stats)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("stage") != "pose_frame":
                continue
            camera_id = str(record.get("cameraLoginId") or "unknown")
            _observe(stats[camera_id], record)
    return {camera_id: _finalize(values) for camera_id, values in sorted(stats.items())}


def _empty_stats() -> dict:
    return {
        "frames": 0,
        "raw_detection_total": 0,
        "bbox_conf_total": 0.0,
        "bbox_conf_count": 0,
        "keypoint_conf_total": 0.0,
        "keypoint_conf_count": 0,
        "tracker_active_frames": 0,
        "sequence_ready_count": 0,
    }


def _observe(stats: dict, record: dict) -> None:
    stats["frames"] += 1
    stats["raw_detection_total"] += int(record.get("raw_detection_count") or 0)
    _add_optional_average(stats, "bbox", record.get("avg_bbox_confidence"))
    _add_optional_average(stats, "keypoint", record.get("avg_keypoint_confidence"))
    if int(record.get("active_tracks") or 0) > 0:
        stats["tracker_active_frames"] += 1
    stats["sequence_ready_count"] += int(record.get("sequenceReadyCount") or 0)


def _add_optional_average(stats: dict, prefix: str, value) -> None:
    if value is None:
        return
    stats[f"{prefix}_conf_total"] += float(value)
    stats[f"{prefix}_conf_count"] += 1


def _finalize(stats: dict) -> dict:
    frames = int(stats["frames"])
    return {
        "frames": frames,
        "avg_raw_detection_count": _rounded_div(stats["raw_detection_total"], frames),
        "avg_bbox_confidence": _rounded_div(stats["bbox_conf_total"], stats["bbox_conf_count"]),
        "avg_keypoint_confidence": _rounded_div(stats["keypoint_conf_total"], stats["keypoint_conf_count"]),
        "tracker_active_rate": _rounded_div(stats["tracker_active_frames"], frames),
        "sequence_ready_count": int(stats["sequence_ready_count"]),
    }


def _rounded_div(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize YOLO Pose/Tracking JSONL diagnostics by cameraLoginId.")
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument("--pretty", action="store_true", help="Print indented JSON instead of compact JSON.")
    args = parser.parse_args(argv)

    summary = summarize_jsonl_path(args.jsonl_path)
    indent = 2 if args.pretty else None
    print(json.dumps(summary, ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

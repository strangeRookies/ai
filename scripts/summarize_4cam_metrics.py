import argparse
import json
from pathlib import Path


COLUMNS = [
    ("camera_id", 10),
    ("rtsp_url", 28),
    ("frames_processed", 16),
    ("bbox_detections", 16),
    ("bbox/frame", 10),
    ("keypoints/frame", 15),
    ("generated_sequences", 19),
    ("lstm_predictions", 16),
    ("events_generated", 16),
    ("active_tracks", 13),
    ("max_active_tracks", 17),
    ("new_tracks", 11),
    ("lost_tracks", 12),
    ("id_switch_like_events", 22),
    ("faint_predictions", 17),
    ("normal_predictions", 18),
    ("effective_fps", 14),
    ("avg_yolo_inference_ms", 22),
    ("avg_lstm_inference_ms", 22),
    ("runtime_seconds", 15),
]


def load_metrics(path):
    row = json.loads(path.read_text(encoding="utf-8"))
    row["_source_path"] = str(path)
    row["_source_mtime"] = path.stat().st_mtime
    return row


def value_for(row, column):
    if column == "bbox/frame":
        return row.get("bbox_per_frame", 0.0)
    if column == "keypoints/frame":
        return row.get("keypoints_per_frame", 0.0)
    return row.get(column, "")


def format_value(value):
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(rows):
    header = " | ".join(name.ljust(width) for name, width in COLUMNS)
    rule = "-+-".join("-" * width for _name, width in COLUMNS)
    print(header)
    print(rule)
    for row in rows:
        cells = []
        for name, width in COLUMNS:
            cells.append(format_value(value_for(row, name)).ljust(width))
        print(" | ".join(cells))


def is_valid_measurement(row):
    return int(row.get("frames_processed") or 0) > 0 and float(row.get("runtime_seconds") or 0.0) > 1.0


def select_latest_valid_per_camera(rows):
    latest = {}
    for row in rows:
        if not is_valid_measurement(row):
            continue
        camera_id = row.get("camera_id") or row.get("rtsp_url") or row.get("_source_path")
        previous = latest.get(camera_id)
        if previous is None or float(row.get("_source_mtime") or 0.0) > float(previous.get("_source_mtime") or 0.0):
            latest[camera_id] = row
    return [latest[key] for key in sorted(latest)]


def decision_hint(rows, target_fps):
    if any(float(row.get("avg_frame_read_ms") or 0.0) > 80.0 for row in rows):
        return "RTSP/read slow or unstable -> investigate GStreamer."
    if any(float(row.get("effective_fps") or 0.0) < target_fps for row in rows):
        return "YOLO latency high or FPS below target -> investigate TensorRT."
    if any(float(row.get("avg_yolo_inference_ms") or 0.0) > 100.0 for row in rows):
        return "YOLO latency high or FPS below target -> investigate TensorRT."
    return "Metrics healthy -> defer GStreamer/TensorRT."


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize 4-camera RTSP AI metrics JSON files.")
    parser.add_argument("paths", nargs="*", default=None)
    parser.add_argument("--dir", default="runs/verification")
    parser.add_argument("--pattern", default="cam*_rtsp_metrics*.json")
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--all", action="store_true", help="Show every matching file, including stale and zero-frame rows.")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = [Path(path) for path in args.paths] if args.paths else sorted(Path(args.dir).glob(args.pattern))
    if not paths:
        raise SystemExit(f"No metrics JSON files found in {args.dir} matching {args.pattern}")
    rows = [load_metrics(path) for path in paths]
    if not args.all:
        rows = select_latest_valid_per_camera(rows)
        if not rows:
            raise SystemExit("No valid metrics rows found. Use --all to inspect stale or failed rows.")
    print_table(rows)
    print()
    print(f"Decision hint: {decision_hint(rows, args.target_fps)}")
    warnings = [row.get("gpu_memory_warning") for row in rows if row.get("gpu_memory_warning")]
    for warning in sorted(set(warnings)):
        print(f"GPU memory warning: {warning}")


if __name__ == "__main__":
    main()

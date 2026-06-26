import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.streams.video_reader import VideoReader
from scripts.check_dataset_split import stratified_group_split
from scripts.run_rtsp_demo import resolve_existing_video
from scripts.run_rtsp_inference import create_detector, ensure_mock_keypoints, normalize_detections


SPLITS = ("train", "test", "val")


def label_name(row):
    if row.get("label_name"):
        return row["label_name"]
    if row.get("event_class"):
        return row["event_class"]
    return "Faint" if str(row.get("label", "0")) == "1" else "Normal"


def read_dataset_rows(path):
    metadata_path = Path(path).resolve()
    base_dirs = [Path.cwd(), metadata_path.parent, metadata_path.parent.parent, metadata_path.parent.parent.parent]
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            video = row.get("video_path") or row.get("clip_path")
            resolved = resolve_existing_video(video, base_dirs)
            if not resolved:
                row = dict(row)
                row["_resolved_video_path"] = ""
                rows.append(row)
                continue
            row = dict(row)
            row["_resolved_video_path"] = str(resolved)
            rows.append(row)
    if rows and ("split" not in rows[0] or not all(row.get("split") for row in rows)):
        rows = stratified_group_split(rows)
    return rows


def limit_rows_by_split(rows, max_rows_per_split):
    if max_rows_per_split <= 0:
        return rows
    counts = Counter()
    limited = []
    for row in rows:
        split = row.get("split") or "unspecified"
        if counts[split] >= max_rows_per_split:
            continue
        limited.append(row)
        counts[split] += 1
    return limited


def class_counts(rows):
    result = {}
    by_split = defaultdict(list)
    for row in rows:
        by_split[row.get("split") or "unspecified"].append(row)
    for split in sorted(by_split):
        counts = Counter(label_name(row) for row in by_split[split])
        result[split] = {"total": len(by_split[split]), "Faint": counts.get("Faint", 0), "Normal": counts.get("Normal", 0)}
    return result


def process_row(row, detector, args):
    split = row.get("split") or "unspecified"
    label = label_name(row)
    video_path = row.get("_resolved_video_path") or ""
    clip_id = row.get("clip_id") or Path(row.get("video_path") or row.get("clip_path") or video_path).stem
    summary = {
        "clip_id": clip_id,
        "split": split,
        "label": label,
        "video_path": video_path,
        "frames_processed": 0,
        "bbox_detections": 0,
        "keypoints_extracted": 0,
        "generated_sequences": 0,
        "fallback_crop_usage_count": 0,
        "zero_sequence": True,
        "error": None,
    }
    if not video_path:
        summary["error"] = "video_not_found"
        return summary

    buffer = KeypointSequenceBuffer(args.sequence_length, args.sequence_stride)
    try:
        with VideoReader(video_path) as reader:
            while True:
                packet = reader.read()
                if packet is None:
                    break
                if args.max_frames > 0 and summary["frames_processed"] >= args.max_frames:
                    break
                detections = detector.detect(packet.frame)
                if args.detector_mode == "mock":
                    detections = ensure_mock_keypoints(detections)
                boxes = normalize_detections(detections)
                summary["frames_processed"] += 1
                summary["bbox_detections"] += len(boxes)
                summary["keypoints_extracted"] += sum(1 for item in detections if item.get("keypoints"))
                if buffer.add(packet.frame_idx, detections):
                    summary["generated_sequences"] += 1
    except Exception as exc:
        summary["error"] = str(exc)

    summary["zero_sequence"] = summary["generated_sequences"] == 0
    return summary


def aggregate(rows, clip_summaries, args):
    totals = {
        "clips_requested": len(rows),
        "clips_processed": sum(1 for item in clip_summaries if item["frames_processed"] > 0),
        "clips_missing_video": sum(1 for item in clip_summaries if item["error"] == "video_not_found"),
        "decoding_errors": sum(1 for item in clip_summaries if item["error"] and item["error"] != "video_not_found"),
        "frames_processed": sum(item["frames_processed"] for item in clip_summaries),
        "bbox_detections": sum(item["bbox_detections"] for item in clip_summaries),
        "keypoints_extracted": sum(item["keypoints_extracted"] for item in clip_summaries),
        "generated_sequences": sum(item["generated_sequences"] for item in clip_summaries),
        "zero_sequence_clips": sum(1 for item in clip_summaries if item["zero_sequence"]),
        "fallback_crop_usage_count": sum(item["fallback_crop_usage_count"] for item in clip_summaries),
    }
    totals["fallback_crop_usage_ratio"] = 0.0
    if totals["generated_sequences"]:
        totals["fallback_crop_usage_ratio"] = round(totals["fallback_crop_usage_count"] / totals["generated_sequences"], 4)

    by_split = {}
    for split in sorted({item["split"] for item in clip_summaries}):
        items = [item for item in clip_summaries if item["split"] == split]
        by_split[split] = {
            "clips": len(items),
            "Faint": sum(1 for item in items if item["label"] == "Faint"),
            "Normal": sum(1 for item in items if item["label"] == "Normal"),
            "frames_processed": sum(item["frames_processed"] for item in items),
            "bbox_detections": sum(item["bbox_detections"] for item in items),
            "keypoints_extracted": sum(item["keypoints_extracted"] for item in items),
            "generated_sequences": sum(item["generated_sequences"] for item in items),
            "zero_sequence_clips": sum(1 for item in items if item["zero_sequence"]),
        }

    return {
        "metadata_csv": args.metadata_csv,
        "detector_mode": args.detector_mode,
        "yolo_model": args.yolo_model if args.detector_mode == "real" else None,
        "max_frames_per_clip": args.max_frames,
        "max_rows_per_split": args.max_rows_per_split,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "class_counts_selected_rows": class_counts(rows),
        "fallback_crop_enabled": False,
        "totals": totals,
        "by_split": by_split,
        "zero_sequence_examples": [item for item in clip_summaries if item["zero_sequence"]][:5],
        "sample_clip": next((item for item in clip_summaries if item["generated_sequences"] > 0), None),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a safe dataset pose/keypoint sequence evaluation dry-run.")
    parser.add_argument("--metadata-csv", default="../ai_fall_experiments/data/metadata/metadata.csv")
    parser.add_argument("--detector-mode", choices=["real", "mock"], default="mock")
    parser.add_argument("--yolo-model", default="yolov8n-pose.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--max-rows-per-split", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    print(
        "[dataset-evaluation] sequence config: "
        f"sequence_length={args.sequence_length} "
        f"sequence_stride={args.sequence_stride} "
        "defaults=8/4 frame_sampling=disabled",
        flush=True,
    )

    rows = limit_rows_by_split(read_dataset_rows(args.metadata_csv), args.max_rows_per_split)
    detector = create_detector(args.detector_mode, args.yolo_model, args.device)
    clip_summaries = [process_row(row, detector, args) for row in rows]
    summary = aggregate(rows, clip_summaries, args)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

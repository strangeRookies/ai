import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.classifier import LSTMActionModel
from ai.streams.video_reader import VideoReader
from detector.mock_detector import MockDetector
from detector.yolo_pose_detector import YoloPoseDetector
try:
    from benchmark.keypoint_cache_loader import keypoint_array_to_frames, load_keypoint_cache, resolve_keypoint_cache_path
except ModuleNotFoundError:
    from keypoint_cache_loader import keypoint_array_to_frames, load_keypoint_cache, resolve_keypoint_cache_path
from scripts.run_dataset_evaluation import label_name, read_dataset_rows
from scripts.run_rtsp_inference import ensure_mock_keypoints


MODEL_SPECS = [
    {"label": "YOLOv11n-pose", "model": "yolo11n-pose.pt"},
    {"label": "YOLO26n-pose", "model": "yolo26n-pose.pt"},
    {"label": "YOLOv8s-pose", "model": "yolov8s-pose.pt"},
]
CLASS_TO_ID = {"Normal": 0, "Faint": 1}
ID_TO_CLASS = {0: "Normal", 1: "Faint"}
CUDA_CPU_FALLBACK_WARNING = "CUDA requested but unavailable; LSTM ran on CPU fallback."


class CpuFallbackDisabledError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Compare pose extractors as keypoint sequence sources for the LSTM pipeline.")
    parser.add_argument("--metadata-csv", default="../ai_fall_experiments/data/metadata/metadata.csv")
    parser.add_argument("--output-dir", default="benchmark/results/lstm_extractor_comparison")
    parser.add_argument("--models", default="YOLOv11n-pose:yolo11n-pose.pt,YOLO26n-pose:yolo26n-pose.pt,YOLOv8s-pose:yolov8s-pose.pt")
    parser.add_argument("--detector-mode", choices=["real", "mock", "cache"], default="real")
    parser.add_argument("--keypoint-cache-dir", default="../ai_fall_experiments/data/keypoints/yolo26n-pose")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--keypoint-conf-threshold", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detector-conf", type=float, default=0.15, help="YOLO Pose detector confidence threshold.")
    parser.add_argument("--dry-run", action="store_true", help="Generate extractor sequence stats only; skip LSTM training.")
    parser.add_argument("--no-cpu-fallback", action="store_true", help="Fail LSTM training when CUDA is requested but unavailable.")
    parser.add_argument("--prefilter-normal-clips", action="store_true", help="Use the first configured detector to keep Normal clip candidates only when person/keypoint signal is present.")
    parser.add_argument("--prefilter-max-frames", type=int, default=120, help="Maximum frames per Normal candidate during prefiltering.")
    parser.add_argument("--audit-thresholds", default="0.3,0.4,0.5,0.6,0.7", help="Comma-separated Faint probability thresholds for prediction audit.")
    parser.add_argument("--repeat-seeds", type=int, default=1, help="Train/evaluate LSTM repeatedly for N deterministic seeds and report recall/F1 mean/std.")
    parser.add_argument("--loss", choices=["ce", "weighted-ce", "focal", "oversample"], default="ce", help="Loss function to use (default: ce). weighted-ce or focal handles class imbalance.")
    return parser.parse_args()


def parse_model_specs(raw):
    specs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            label, model = item.split(":", 1)
        else:
            label, model = item, item
        specs.append({"label": label.strip(), "model": model.strip()})
    return specs or MODEL_SPECS


def create_pose_detector(mode, model_name, device, imgsz, conf=0.25):
    if mode == "mock":
        return MockDetector(model_name="mock-pose-detector")
    if mode == "cache":
        return None
    return YoloPoseDetector(model_name, device=device, imgsz=imgsz, conf=conf)


def row_label_id(row):
    return CLASS_TO_ID.get(label_name(row), 0)


def limit_rows_by_split_and_class(rows, max_rows_per_split, seed=42):
    if max_rows_per_split <= 0:
        return rows
    limited = []
    by_split = {}
    for row in rows:
        split = row.get("split") or "unspecified"
        by_split.setdefault(split, []).append(row)
    for split in sorted(by_split):
        split_rows = by_split[split]
        faint_rows = [row for row in split_rows if label_name(row) == "Faint"]
        normal_rows = [row for row in split_rows if label_name(row) == "Normal"]
        selected_faint = deterministic_order(faint_rows, seed, split, "Faint")[:max_rows_per_split]
        limited.extend(selected_faint)
        limited.extend(order_normal_candidates(normal_rows, selected_faint, seed, split)[:max_rows_per_split])
    return limited


def deterministic_order(rows, seed, split, label):
    rng = random.Random(f"{seed}:{split}:{label}")
    decorated = [(rng.random(), index, row) for index, row in enumerate(rows)]
    return [row for _, _, row in sorted(decorated)]


def order_normal_candidates(rows, selected_faint_rows, seed, split):
    rng = random.Random(f"{seed}:{split}:Normal")
    faint_contexts = [row_context(row) for row in selected_faint_rows]
    faint_sources = {item["source_key"] for item in faint_contexts if item["source_key"]}
    decorated = []
    for index, row in enumerate(rows):
        context = row_context(row)
        source_rank = 0 if context["source_key"] and context["source_key"] in faint_sources else 1
        distance = nearest_non_overlapping_distance(context["frame_start"], context["frame_end"], faint_contexts)
        decorated.append((source_rank, distance, rng.random(), index, row))
    return [row for _, _, _, _, row in sorted(decorated)]


def row_context(row):
    video_path = row.get("_resolved_video_path") or row.get("video_path") or row.get("clip_path") or ""
    start, end = parse_frame_range(video_path)
    return {"source_key": source_key(video_path), "frame_start": start, "frame_end": end}


def source_key(video_path):
    stem = Path(str(video_path)).stem
    return re.sub(r"__\d{6,}_\d{6,}$", "", stem)


def parse_frame_range(video_path):
    match = re.search(r"__(\d{6,})_(\d{6,})(?:$|\D)", Path(str(video_path)).stem)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def nearest_non_overlapping_distance(start, end, faint_contexts):
    if start is None or end is None:
        return 10**12
    best = 10**12
    for context in faint_contexts:
        faint_start = context["frame_start"]
        faint_end = context["frame_end"]
        if faint_start is None or faint_end is None:
            continue
        if ranges_overlap(start, end, faint_start, faint_end):
            continue
        center = (start + end) / 2.0
        faint_center = (faint_start + faint_end) / 2.0
        best = min(best, abs(center - faint_center))
    return best


def ranges_overlap(first_start, first_end, second_start, second_end):
    return max(first_start, second_start) <= min(first_end, second_end)


def dataset_class_counts(rows):
    counts = {}
    for row in rows:
        split = row.get("split") or "unspecified"
        label = label_name(row)
        split_counts = counts.setdefault(split, {"Normal": 0, "Faint": 0, "total": 0})
        if label not in split_counts:
            split_counts[label] = 0
        split_counts[label] += 1
        split_counts["total"] += 1
    return {split: counts[split] for split in sorted(counts)}


def keypoints_to_feature(detection, frame_shape, keypoint_conf_threshold):
    """Return one image-normalized 51-dim keypoint feature vector.

    LSTM batches use (Batch, sequence_length, feature_dim), where feature_dim
    is 51 = 17 keypoints x (x, y, confidence). x/y are normalized by image
    width/height here, not pixel or bbox-relative coordinates.
    TODO: evaluate bbox-relative normalized keypoints + confidence.
    """

    height, width = frame_shape[:2]
    keypoints = detection.get("keypoints") or []
    features = []
    missing = 0
    total = 17
    for idx in range(total):
        if idx >= len(keypoints) or keypoints[idx] is None:
            features.extend([0.0, 0.0, 0.0])
            missing += 1
            continue
        point = keypoints[idx]
        conf = float(point.get("confidence", 0.0))
        if conf < keypoint_conf_threshold:
            missing += 1
        features.extend(
            [
                float(point.get("x", 0.0)) / max(float(width), 1.0),
                float(point.get("y", 0.0)) / max(float(height), 1.0),
                conf,
            ]
        )
    return np.asarray(features, dtype=np.float32), missing, total


def sequence_to_features(sequence, frame_shapes, keypoint_conf_threshold):
    rows = []
    missing = 0
    total = 0
    for detection, shape in zip(sequence["detections"], frame_shapes):
        features, current_missing, current_total = keypoints_to_feature(detection, shape, keypoint_conf_threshold)
        rows.append(features)
        missing += current_missing
        total += current_total
    
    base_features = np.stack(rows, axis=0)
    try:
        from ai.action.motion_features import append_motion_features
        final_features = append_motion_features(base_features)
    except ImportError:
        final_features = base_features
        
    return final_features, missing, total


def collect_cached_split_sequences(rows, split_name, args):
    x_rows = []
    y_rows = []
    sequence_rows = []
    clip_summaries = []
    totals = Counter()
    selected = [row for row in rows if (row.get("split") or "") == split_name]
    for row in selected:
        video_path = row.get("_resolved_video_path") or ""
        clip_id = row.get("clip_id") or Path(row.get("video_path") or row.get("clip_path") or video_path).stem
        frame_start, frame_end = parse_frame_range(video_path or row.get("video_path") or row.get("clip_path") or "")
        cache_path = resolve_keypoint_cache_path(row, args.keypoint_cache_dir)
        clip = {
            "clip_id": clip_id,
            "split": split_name,
            "label": label_name(row),
            "video_path": video_path,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frames_processed": 0,
            "person_detections": 0,
            "keypoints_extracted": 0,
            "generated_sequences": 0,
            "zero_sequence": True,
            "reason_if_zero_sequence": "",
            "fallback_usage": 0,
            "missing_keypoints": 0,
            "total_keypoints": 0,
            "error": None,
        }
        if cache_path is None:
            clip["error"] = "keypoint_cache_not_found"
            clip["reason_if_zero_sequence"] = "keypoint_cache_not_found"
            clip_summaries.append(clip)
            continue
        try:
            frames = keypoint_array_to_frames(load_keypoint_cache(cache_path))
            buffer = KeypointSequenceBuffer(args.sequence_length, args.sequence_stride)
            shape_buffer = []
            for frame in frames:
                if args.max_frames > 0 and clip["frames_processed"] >= args.max_frames:
                    break
                detections = frame["detections"]
                clip["frames_processed"] += 1
                clip["person_detections"] += len(detections)
                clip["keypoints_extracted"] += sum(1 for item in detections if item.get("keypoints"))
                sequence = buffer.add(frame["frame_idx"], detections)
                shape_buffer.append(frame["frame_shape"])
                shape_buffer = shape_buffer[-args.sequence_length :]
                if sequence is None:
                    continue
                features, missing, total = sequence_to_features(sequence, shape_buffer, args.keypoint_conf_threshold)
                x_rows.append(features)
                y_rows.append(row_label_id(row))
                clip["generated_sequences"] += 1
                clip["missing_keypoints"] += missing
                clip["total_keypoints"] += total
                sequence_rows.append(
                    {
                        "clip_id": clip_id,
                        "split": split_name,
                        "label": label_name(row),
                        "frame_start": sequence["start_frame"],
                        "frame_end": sequence["end_frame"],
                        "missing_keypoints": missing,
                        "total_keypoints": total,
                    }
                )
        except (OSError, KeyError, ValueError) as exc:
            clip["error"] = str(exc)
        clip["zero_sequence"] = clip["generated_sequences"] == 0
        if clip["zero_sequence"] and not clip["reason_if_zero_sequence"]:
            clip["reason_if_zero_sequence"] = zero_sequence_reason(clip)
        clip_summaries.append(clip)
    for clip in clip_summaries:
        totals["clips_processed"] += 1 if clip["frames_processed"] > 0 else 0
        totals["clips_requested"] += 1
        totals["person_detections"] += clip["person_detections"]
        totals["keypoints_extracted"] += clip["keypoints_extracted"]
        totals["generated_sequences"] += clip["generated_sequences"]
        totals["zero_sequence_clips"] += 1 if clip["zero_sequence"] else 0
        totals["fallback_usage"] += clip["fallback_usage"]
        totals["missing_keypoints"] += clip["missing_keypoints"]
        totals["total_keypoints"] += clip["total_keypoints"]
    return x_rows, y_rows, clip_summaries, sequence_rows, dict(totals)


def collect_split_sequences(rows, split_name, detector, args):
    if args.detector_mode == "cache":
        return collect_cached_split_sequences(rows, split_name, args)
    x_rows = []
    y_rows = []
    sequence_rows = []
    clip_summaries = []
    totals = Counter()
    selected = [row for row in rows if (row.get("split") or "") == split_name]
    for row in selected:
        video_path = row.get("_resolved_video_path") or ""
        clip_id = row.get("clip_id") or Path(row.get("video_path") or row.get("clip_path") or video_path).stem
        frame_start, frame_end = parse_frame_range(video_path or row.get("video_path") or row.get("clip_path") or "")
        clip = {
            "clip_id": clip_id,
            "split": split_name,
            "label": label_name(row),
            "video_path": video_path,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frames_processed": 0,
            "person_detections": 0,
            "keypoints_extracted": 0,
            "generated_sequences": 0,
            "zero_sequence": True,
            "reason_if_zero_sequence": "",
            "fallback_usage": 0,
            "missing_keypoints": 0,
            "total_keypoints": 0,
            "error": None,
        }
        if not video_path:
            clip["error"] = "video_not_found"
            clip["reason_if_zero_sequence"] = "video_not_found"
            clip_summaries.append(clip)
            continue
        buffer = KeypointSequenceBuffer(args.sequence_length, args.sequence_stride)
        shape_buffer = []
        try:
            with VideoReader(video_path) as reader:
                start_frame = int(row.get("start_frame") or 0)
                # 경로에 processed 또는 clips가 있으면 잘려진 32프레임짜리 클립이므로 점프하지 않습니다.
                is_processed_clip = "processed" in str(video_path).lower() or "clips" in str(video_path).lower()
                if start_frame > 0 and not is_processed_clip and getattr(reader, "cap", None) is not None:
                    reader.cap.set(reader.cv2.CAP_PROP_POS_FRAMES, start_frame)
                    reader.frame_idx = start_frame
                while True:
                    packet = reader.read()
                    if packet is None:
                        break
                    if args.max_frames > 0 and clip["frames_processed"] >= args.max_frames:
                        break
                    detections = detector.detect(packet.frame)
                    if args.detector_mode == "mock":
                        detections = ensure_mock_keypoints(detections)
                    clip["frames_processed"] += 1
                    clip["person_detections"] += len(detections)
                    clip["keypoints_extracted"] += sum(1 for item in detections if item.get("keypoints"))
                    sequence = buffer.add(packet.frame_idx, detections)
                    shape_buffer.append(packet.frame.shape)
                    shape_buffer = shape_buffer[-args.sequence_length :]
                    if sequence is None:
                        continue
                    features, missing, total = sequence_to_features(sequence, shape_buffer, args.keypoint_conf_threshold)
                    x_rows.append(features)
                    y_rows.append(row_label_id(row))
                    clip["generated_sequences"] += 1
                    clip["missing_keypoints"] += missing
                    clip["total_keypoints"] += total
                    sequence_rows.append(
                        {
                            "clip_id": clip_id,
                            "split": split_name,
                            "label": label_name(row),
                            "frame_start": sequence["start_frame"],
                            "frame_end": sequence["end_frame"],
                            "missing_keypoints": missing,
                            "total_keypoints": total,
                        }
                    )
        except Exception as exc:
            clip["error"] = str(exc)
        if clip["frames_processed"] == 0 and not clip["error"]:
            clip["error"] = "zero_frames_decoded (가능성: 코덱 불일치 또는 비디오 인코딩 오류)"
        clip["zero_sequence"] = clip["generated_sequences"] == 0
        if clip["zero_sequence"] and not clip["reason_if_zero_sequence"]:
            clip["reason_if_zero_sequence"] = zero_sequence_reason(clip)
        clip_summaries.append(clip)

    for clip in clip_summaries:
        totals["clips_processed"] += 1 if clip["frames_processed"] > 0 else 0
        totals["clips_requested"] += 1
        totals["person_detections"] += clip["person_detections"]
        totals["keypoints_extracted"] += clip["keypoints_extracted"]
        totals["generated_sequences"] += clip["generated_sequences"]
        totals["zero_sequence_clips"] += 1 if clip["zero_sequence"] else 0
        totals["fallback_usage"] += clip["fallback_usage"]
        totals["missing_keypoints"] += clip["missing_keypoints"]
        totals["total_keypoints"] += clip["total_keypoints"]
    return x_rows, y_rows, clip_summaries, sequence_rows, dict(totals)


def zero_sequence_reason(clip):
    if clip.get("error"):
        return str(clip["error"])
    if int(clip.get("person_detections", 0)) <= 0:
        return "no_person_detections"
    if int(clip.get("keypoints_extracted", 0)) <= 0:
        return "no_keypoints_extracted"
    return "no_complete_sequence_window"


def write_clip_diagnostics_csv(path, clip_summaries):
    rows = []
    for clip in clip_summaries:
        rows.append(
            {
                "label": clip.get("label"),
                "video_path": clip.get("video_path"),
                "frame_start": clip.get("frame_start"),
                "frame_end": clip.get("frame_end"),
                "person_detections": clip.get("person_detections"),
                "keypoints_extracted": clip.get("keypoints_extracted"),
                "generated_sequences": clip.get("generated_sequences"),
                "reason_if_zero_sequence": clip.get("reason_if_zero_sequence", ""),
            }
        )
    write_csv(path, rows)


def prefilter_normal_rows(rows, args, specs, output_dir):
    if not args.prefilter_normal_clips or args.max_rows_per_split <= 0:
        return rows
    detector = create_pose_detector(args.detector_mode, specs[0]["model"], args.device, args.imgsz, conf=args.detector_conf)
    selected = []
    diagnostics = []
    by_split = {}
    for row in rows:
        split = row.get("split") or "unspecified"
        by_split.setdefault(split, []).append(row)
    for split in sorted(by_split):
        split_rows = by_split[split]
        faint_rows = [row for row in split_rows if label_name(row) == "Faint"]
        selected_faint = deterministic_order(faint_rows, args.seed, split, "Faint")[: args.max_rows_per_split]
        selected.extend(selected_faint)
        normal_rows = [row for row in split_rows if label_name(row) == "Normal"]
        accepted_normal = []
        for row in order_normal_candidates(normal_rows, selected_faint, args.seed, split):
            if len(accepted_normal) >= args.max_rows_per_split:
                break
            diagnostic = inspect_clip_signal(row, detector, args)
            diagnostics.append(diagnostic)
            if diagnostic["person_detections"] > 0 and diagnostic["keypoints_extracted"] > 0:
                accepted_normal.append(row)
        selected.extend(accepted_normal)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "normal_prefilter_diagnostics.csv", diagnostics)
    return selected


def inspect_clip_signal(row, detector, args):
    video_path = row.get("_resolved_video_path") or ""
    frame_start, frame_end = parse_frame_range(video_path or row.get("video_path") or row.get("clip_path") or "")
    diagnostic = {
        "label": label_name(row),
        "video_path": video_path,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "person_detections": 0,
        "keypoints_extracted": 0,
        "generated_sequences": 0,
        "reason_if_zero_sequence": "",
    }
    if not video_path:
        diagnostic["reason_if_zero_sequence"] = "video_not_found"
        return diagnostic
    buffer = KeypointSequenceBuffer(args.sequence_length, args.sequence_stride)
    frames_processed = 0
    try:
        with VideoReader(video_path) as reader:
            while True:
                packet = reader.read()
                if packet is None:
                    break
                if args.prefilter_max_frames > 0 and frames_processed >= args.prefilter_max_frames:
                    break
                detections = detector.detect(packet.frame)
                if args.detector_mode == "mock":
                    detections = ensure_mock_keypoints(detections)
                frames_processed += 1
                diagnostic["person_detections"] += len(detections)
                diagnostic["keypoints_extracted"] += sum(1 for item in detections if item.get("keypoints"))
                if buffer.add(packet.frame_idx, detections):
                    diagnostic["generated_sequences"] += 1
    except Exception as exc:
        diagnostic["reason_if_zero_sequence"] = str(exc)
        return diagnostic
    if diagnostic["generated_sequences"] <= 0:
        diagnostic["reason_if_zero_sequence"] = zero_sequence_reason(diagnostic)
    return diagnostic


def summarize_split(rows, y_rows, totals, requested_class_counts=None):
    counts = Counter(ID_TO_CLASS[int(label)] for label in y_rows)
    total_keypoints = totals.get("total_keypoints", 0)
    return {
        "clips_requested": int(totals.get("clips_requested", 0)),
        "clips_processed": int(totals.get("clips_processed", 0)),
        "person_detections": int(totals.get("person_detections", 0)),
        "keypoints_extracted": int(totals.get("keypoints_extracted", 0)),
        "generated_sequences": int(totals.get("generated_sequences", 0)),
        "zero_sequence_clips": int(totals.get("zero_sequence_clips", 0)),
        "keypoint_missing_rate": round(float(totals.get("missing_keypoints", 0)) / max(float(total_keypoints), 1.0), 6),
        "fallback_usage": int(totals.get("fallback_usage", 0)),
        "fallback_usage_ratio": 0.0,
        "sequence_class_counts": {"Normal": counts.get("Normal", 0), "Faint": counts.get("Faint", 0)},
        "requested_class_counts": requested_class_counts or {"Normal": 0, "Faint": 0, "total": 0},
    }


def parse_thresholds(raw):
    thresholds = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(round(float(item), 6))
    return thresholds or [0.5]


def train_and_evaluate(train_x, train_y, eval_x, eval_y, eval_sequences, args, output_dir):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if args.loss == "oversample":
        train_y_arr = np.asarray(train_y, dtype=np.int64)
        faint_indices = np.where(train_y_arr == 1)[0]
        normal_indices = np.where(train_y_arr == 0)[0]
        if len(faint_indices) > 0 and len(normal_indices) > len(faint_indices):
            repeats = len(normal_indices) // len(faint_indices)
            remainder = len(normal_indices) % len(faint_indices)
            oversampled_faint = np.concatenate([np.repeat(faint_indices, repeats), faint_indices[:remainder]])
            new_indices = np.concatenate([normal_indices, oversampled_faint])
            np.random.shuffle(new_indices)
            train_x = [train_x[i] for i in new_indices]
            train_y = [train_y[i] for i in new_indices]

    resolved_device = normalize_torch_device(args.device, torch, no_cpu_fallback=args.no_cpu_fallback)
    device = torch.device(resolved_device)
    train_tensor = torch.from_numpy(np.stack(train_x).astype(np.float32))
    train_labels = torch.from_numpy(np.asarray(train_y, dtype=np.int64))
    eval_tensor = torch.from_numpy(np.stack(eval_x).astype(np.float32))
    eval_labels = torch.from_numpy(np.asarray(eval_y, dtype=np.int64))
    train_dataset = TensorDataset(train_tensor, train_labels)
    eval_loader = DataLoader(TensorDataset(eval_tensor, eval_labels), batch_size=args.batch_size)
    model_config = {
        "input_size": int(train_tensor.shape[-1]),
        "hidden_size": args.hidden_size,
        "num_layers": 1,
        "num_classes": 2,
        "dropout": 0.0,
    }
    thresholds = parse_thresholds(args.audit_thresholds)
    seed_metrics = []
    base_metrics = None
    repeat_count = max(1, int(args.repeat_seeds))
    for offset in range(repeat_count):
        seed = int(args.seed) + offset
        set_seed(seed, torch)
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
        current_metrics = train_single_lstm(
            train_loader=train_loader,
            eval_loader=eval_loader,
            model_config=model_config,
            device=device,
            args=args,
            output_dir=output_dir if offset == 0 else None,
            thresholds=thresholds,
            eval_sequences=eval_sequences,
            train_y=train_y,
        )
        seed_metrics.append({"seed": seed, "faint_recall": current_metrics.get("recall"), "f1_score": current_metrics.get("f1_score")})
        if offset == 0:
            base_metrics = current_metrics
    metrics = base_metrics or empty_metrics()
    metrics["torch_device"] = str(device)
    metrics["warnings"] = torch_device_warnings(args.device, torch, resolved_device)
    metrics["repeated_seed_audit"] = repeated_seed_audit(seed_metrics)
    return metrics


def train_single_lstm(train_loader, eval_loader, model_config, device, args, output_dir, thresholds, eval_sequences, train_y=None):
    import torch
    from torch import nn
    from collections import Counter

    model = LSTMActionModel(**model_config).model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    alpha = None
    if getattr(args, "loss", "ce") in ["weighted-ce", "focal"] and train_y is not None:
        counts = Counter(train_y)
        total = len(train_y)
        weights = [total / max(counts.get(i, 1), 1) for i in range(model_config["num_classes"])]
        sum_w = sum(weights)
        weights = [w * model_config["num_classes"] / max(sum_w, 1e-6) for w in weights]
        alpha = torch.FloatTensor(weights).to(device)

    if args.loss == "weighted-ce":
        # Calculate class counts from train_y
        labels, counts = np.unique(train_y, return_counts=True)
        # Handle missing classes if any
        class_counts = [0, 0]
        for l, c in zip(labels, counts):
            class_counts[l] = c
        # Basic inverse frequency
        counts_arr = np.array(class_counts)
        weights = torch.tensor([1.0 - (count / counts_arr.sum()) for count in counts_arr], dtype=torch.float32)
        if args.device != "cpu":
            weights = weights.to(args.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif args.loss == "focal":
        class FocalLoss(nn.Module):
            def __init__(self, weight=None, gamma=2.0):
                super().__init__()
                self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')
                self.gamma = gamma
            def forward(self, inputs, targets):
                ce_loss = self.ce(inputs, targets)
                pt = torch.exp(-ce_loss)
                return ((1 - pt) ** self.gamma * ce_loss).mean()
        # Do not use extreme alpha weights with focal loss, let gamma handle the imbalance
        criterion = FocalLoss(weight=None, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(y_batch.numel())
            total += int(y_batch.numel())
        metrics = evaluate_lstm(model, eval_loader, device)
        record = {"epoch": epoch, "train_loss": total_loss / max(1, total), **metrics}
        history.append(record)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "model_config": model_config, "classes": ["Normal", "Faint"]}, output_dir / "best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metrics = evaluate_lstm(model, eval_loader, device, include_probabilities=True)
    predictions = metrics.pop("predictions", [])
    metrics["threshold_audit"] = threshold_audit_metrics(
        [row["true_id"] for row in predictions],
        [row["faint_prob"] for row in predictions],
        thresholds,
    )
    metrics["prediction_counts"] = prediction_counts([row["pred_id"] for row in predictions])
    metrics["prediction_rows"] = prediction_audit_rows(
        [row["true_id"] for row in predictions],
        [[row["normal_prob"], row["faint_prob"]] for row in predictions],
        eval_sequences,
    )
    return metrics


def is_cuda_requested(device_arg):
    raw = str(device_arg).strip().lower()
    return raw.startswith("cuda") or raw.isdigit()


def normalize_torch_device(device_arg, torch_module, no_cpu_fallback=False):
    raw = str(device_arg).strip().lower()
    cuda_available = bool(torch_module.cuda.is_available())
    if raw in {"auto", ""}:
        return "cuda:0" if cuda_available else "cpu"
    if raw == "cpu":
        return "cpu"
    if raw.startswith("cuda"):
        if not cuda_available and no_cpu_fallback:
            raise CpuFallbackDisabledError(f"{CUDA_CPU_FALLBACK_WARNING} Use --device cpu or enable CUDA.")
        return raw if cuda_available else "cpu"
    if raw.isdigit():
        if not cuda_available and no_cpu_fallback:
            raise CpuFallbackDisabledError(f"{CUDA_CPU_FALLBACK_WARNING} Use --device cpu or enable CUDA.")
        return f"cuda:{raw}" if cuda_available else "cpu"
    return "cpu"


def torch_device_warnings(device_arg, torch_module, resolved_device):
    if is_cuda_requested(device_arg) and str(resolved_device).lower() == "cpu" and not torch_module.cuda.is_available():
        return [CUDA_CPU_FALLBACK_WARNING]
    return []


def evaluate_lstm(model, loader, device, include_probabilities=False):
    import torch

    model.eval()
    y_true = []
    y_pred = []
    predictions = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            pred = probs.argmax(dim=1).tolist()
            y_pred.extend(pred)
            true_batch = y_batch.cpu().tolist()
            y_true.extend(true_batch)
            if include_probabilities:
                for truth, predicted, prob_row in zip(true_batch, pred, probs.tolist()):
                    predictions.append(
                        {
                            "true_id": int(truth),
                            "pred_id": int(predicted),
                            "normal_prob": round(float(prob_row[0]), 6),
                            "faint_prob": round(float(prob_row[1]), 6),
                        }
                    )
    metrics = classification_metrics(y_true, y_pred)
    if include_probabilities:
        metrics["predictions"] = predictions
    return metrics


def prediction_audit_rows(y_true, probabilities, sequence_rows=None):
    rows = []
    sequence_rows = sequence_rows or []
    for index, (truth, probs) in enumerate(zip(y_true, probabilities)):
        normal_prob = round(float(probs[0]), 6)
        faint_prob = round(float(probs[1]), 6)
        pred_id = 1 if faint_prob > normal_prob else 0
        sequence = sequence_rows[index] if index < len(sequence_rows) else {}
        rows.append(
            {
                "sequence_index": index,
                "clip_id": sequence.get("clip_id", ""),
                "frame_start": sequence.get("frame_start", ""),
                "frame_end": sequence.get("frame_end", ""),
                "true_label": ID_TO_CLASS[int(truth)],
                "pred_label": ID_TO_CLASS[pred_id],
                "normal_prob": normal_prob,
                "faint_prob": faint_prob,
            }
        )
    return rows


def prediction_counts(y_pred):
    counts = Counter(ID_TO_CLASS[int(label)] for label in y_pred)
    return {"Normal": counts.get("Normal", 0), "Faint": counts.get("Faint", 0)}


def threshold_audit_metrics(y_true, faint_probs, thresholds):
    rows = []
    for threshold in thresholds:
        y_pred = [1 if float(prob) >= float(threshold) else 0 for prob in faint_probs]
        metrics = classification_metrics(y_true, y_pred)
        rows.append(
            {
                "threshold": round(float(threshold), 6),
                "faint_recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "precision": metrics["precision"],
                "predicted_normal": prediction_counts(y_pred)["Normal"],
                "predicted_faint": prediction_counts(y_pred)["Faint"],
            }
        )
    return rows


def repeated_seed_audit(seed_metrics):
    valid = [item for item in seed_metrics if item.get("faint_recall") is not None and item.get("f1_score") is not None]
    enabled = len(seed_metrics) > 1
    if not valid:
        return {"enabled": enabled, "seeds": [item["seed"] for item in seed_metrics]}
    recalls = np.asarray([float(item["faint_recall"]) for item in valid], dtype=np.float64)
    f1_scores = np.asarray([float(item["f1_score"]) for item in valid], dtype=np.float64)
    return {
        "enabled": enabled,
        "seeds": [item["seed"] for item in seed_metrics],
        "faint_recall_mean": round(float(recalls.mean()), 6),
        "faint_recall_std": round(float(recalls.std(ddof=0)), 6),
        "f1_score_mean": round(float(f1_scores.mean()), 6),
        "f1_score_std": round(float(f1_scores.std(ddof=0)), 6),
        "runs": seed_metrics,
    }


def classification_metrics(y_true, y_pred):
    matrix = [[0, 0], [0, 0]]
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth)][int(pred)] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "confusion_matrix": {"labels": ["Normal", "Faint"], "matrix": matrix},
        "per_class_metrics": per_class_metrics(matrix),
    }


def per_class_metrics(matrix):
    labels = ["Normal", "Faint"]
    total = sum(sum(row) for row in matrix)
    metrics = {}
    for class_id, label in enumerate(labels):
        true_positive = matrix[class_id][class_id]
        false_positive = sum(matrix[row][class_id] for row in range(len(labels)) if row != class_id)
        false_negative = sum(matrix[class_id][column] for column in range(len(labels)) if column != class_id)
        true_negative = total - true_positive - false_positive - false_negative
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
        metrics[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
            "support": int(sum(matrix[class_id])),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_negative": int(true_negative),
        }
    return metrics


def empty_metrics(reason="not_run"):
    return {
        "status": reason,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "confusion_matrix": {"labels": ["Normal", "Faint"], "matrix": [[0, 0], [0, 0]]},
        "per_class_metrics": {},
        "train_sequence_class_counts": {"Normal": 0, "Faint": 0},
        "eval_sequence_class_counts": {"Normal": 0, "Faint": 0},
        "prediction_counts": {"Normal": 0, "Faint": 0},
        "threshold_audit": [],
        "repeated_seed_audit": {"enabled": False, "seeds": []},
        "torch_device": None,
        "warnings": [],
    }


def set_seed(seed, torch_module=None):
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sequence_class_counts(y_rows):
    counts = Counter(ID_TO_CLASS[int(label)] for label in y_rows)
    return {"Normal": counts.get("Normal", 0), "Faint": counts.get("Faint", 0)}


def lstm_readiness_status(train_y, eval_y):
    if not train_y:
        return "no_train_sequences"
    if not eval_y:
        return "no_eval_sequences"
    if len(train_y) < 2 or len(eval_y) < 2:
        return "insufficient_sequences"
    train_counts = sequence_class_counts(train_y)
    eval_counts = sequence_class_counts(eval_y)
    if train_counts["Normal"] == 0 or train_counts["Faint"] == 0:
        return "missing_class_in_train"
    if eval_counts["Normal"] == 0 or eval_counts["Faint"] == 0:
        return "missing_class_in_eval"
    return "OK"


def compare_model(spec, rows, args, output_dir, selected_class_counts):
    started = time.perf_counter()
    detector = create_pose_detector(args.detector_mode, spec["model"], args.device, args.imgsz, conf=args.detector_conf)
    model_dir = output_dir / spec["label"]
    model_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, train_clips, train_sequences, train_totals = collect_split_sequences(rows, args.train_split, detector, args)
    eval_x, eval_y, eval_clips, eval_sequences, eval_totals = collect_split_sequences(rows, args.eval_split, detector, args)
    write_csv(model_dir / "train_sequences.csv", train_sequences)
    write_csv(model_dir / "eval_sequences.csv", eval_sequences)
    write_clip_diagnostics_csv(model_dir / "train_clip_diagnostics.csv", train_clips)
    write_clip_diagnostics_csv(model_dir / "eval_clip_diagnostics.csv", eval_clips)
    (model_dir / "train_clips.json").write_text(json.dumps(train_clips, indent=2, ensure_ascii=False), encoding="utf-8")
    (model_dir / "eval_clips.json").write_text(json.dumps(eval_clips, indent=2, ensure_ascii=False), encoding="utf-8")
    train_summary = summarize_split(rows, train_y, train_totals, selected_class_counts.get(args.train_split))
    eval_summary = summarize_split(rows, eval_y, eval_totals, selected_class_counts.get(args.eval_split))
    train_sequence_counts = sequence_class_counts(train_y)
    eval_sequence_counts = sequence_class_counts(eval_y)
    if args.dry_run:
        metrics = empty_metrics("dry_run")
    else:
        readiness_status = lstm_readiness_status(train_y, eval_y)
        if readiness_status != "OK":
            metrics = empty_metrics(readiness_status)
        else:
            try:
                metrics = train_and_evaluate(train_x, train_y, eval_x, eval_y, eval_sequences, args, model_dir)
                metrics["status"] = "OK"
            except CpuFallbackDisabledError:
                raise
            except Exception as exc:
                metrics = empty_metrics(f"failed: {exc}")
    metrics["train_sequence_class_counts"] = train_sequence_counts
    metrics["eval_sequence_class_counts"] = eval_sequence_counts
    runtime_seconds = round(time.perf_counter() - started, 4)
    write_confusion_matrix_csv(model_dir / "confusion_matrix.csv", metrics.get("confusion_matrix", {}))
    prediction_rows = metrics.pop("prediction_rows", [])
    write_csv(model_dir / "eval_predictions.csv", prediction_rows)
    write_csv(model_dir / "threshold_audit.csv", metrics.get("threshold_audit", []))
    (model_dir / "repeated_seed_audit.json").write_text(json.dumps(metrics.get("repeated_seed_audit", {}), indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "model_label": spec["label"],
        "pose_model": spec["model"] if args.detector_mode in {"real", "cache"} else "mock",
        "fall_candidate_count_policy": "reference_only_not_model_selection",
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_sequence_summary": train_summary,
        "eval_sequence_summary": eval_summary,
        "selected_dataset_class_counts": selected_class_counts,
        "lstm_metrics": metrics,
        "runtime_seconds": runtime_seconds,
    }
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def choose_best_model(summaries):
    candidates = [item for item in summaries if item["lstm_metrics"].get("recall") is not None]
    if candidates:
        return max(
            candidates,
            key=lambda item: (
                item["lstm_metrics"]["recall"],
                item["lstm_metrics"]["f1_score"],
                -false_alarm_count(item["lstm_metrics"].get("confusion_matrix", {})),
                item["eval_sequence_summary"]["generated_sequences"],
                -item["eval_sequence_summary"]["zero_sequence_clips"],
                -item["eval_sequence_summary"]["keypoint_missing_rate"],
            ),
        )["model_label"]
    return max(
        summaries,
        key=lambda item: (
            item["eval_sequence_summary"]["generated_sequences"],
            -item["eval_sequence_summary"]["zero_sequence_clips"],
            -item["eval_sequence_summary"]["keypoint_missing_rate"],
        ),
    )["model_label"] if summaries else None


def false_alarm_count(confusion_matrix):
    matrix = confusion_matrix.get("matrix") or [[0, 0], [0, 0]]
    if not matrix or not matrix[0] or len(matrix[0]) < 2:
        return 0
    return int(matrix[0][1])


def write_final_summary(output_dir, summaries, selected_class_counts=None):
    rows = []
    for item in summaries:
        eval_summary = item["eval_sequence_summary"]
        metrics = item["lstm_metrics"]
        rows.append(
            {
                "model_label": item["model_label"],
                "pose_model": item["pose_model"],
                "clips_requested": eval_summary["clips_requested"],
                "clips_processed": eval_summary["clips_processed"],
                "person_detections": eval_summary["person_detections"],
                "keypoints_extracted": eval_summary["keypoints_extracted"],
                "generated_sequences": eval_summary["generated_sequences"],
                "zero_sequence_clips": eval_summary["zero_sequence_clips"],
                "keypoint_missing_rate": eval_summary["keypoint_missing_rate"],
                "fallback_usage": eval_summary["fallback_usage"],
                "train_sequence_normal": metrics.get("train_sequence_class_counts", {}).get("Normal", 0),
                "train_sequence_faint": metrics.get("train_sequence_class_counts", {}).get("Faint", 0),
                "eval_sequence_normal": metrics.get("eval_sequence_class_counts", {}).get("Normal", 0),
                "eval_sequence_faint": metrics.get("eval_sequence_class_counts", {}).get("Faint", 0),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "predicted_normal": metrics.get("prediction_counts", {}).get("Normal", 0),
                "predicted_faint": metrics.get("prediction_counts", {}).get("Faint", 0),
                "metrics_status": metrics.get("status"),
                "torch_device": metrics.get("torch_device"),
                "warnings": "; ".join(metrics.get("warnings", [])),
                "runtime_seconds": item.get("runtime_seconds"),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", rows)
    final = {
        "selection_policy": "Prioritize Faint recall and stable sequence generation; fall_candidate_count is reference-only.",
        "best_model_for_downstream_lstm": choose_best_model(summaries),
        "selected_dataset_class_counts": selected_class_counts or {},
        "warnings": sorted({warning for item in summaries for warning in item["lstm_metrics"].get("warnings", [])}),
        "models": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(output_dir / "report.md", final, rows)
    return final


def write_confusion_matrix_csv(path, confusion_matrix):
    labels = confusion_matrix.get("labels") or ["Normal", "Faint"]
    matrix = confusion_matrix.get("matrix") or [[0, 0], [0, 0]]
    rows = []
    for label, values in zip(labels, matrix):
        row = {"actual": label}
        for predicted_label, value in zip(labels, values):
            row[f"predicted_{predicted_label}"] = int(value)
        rows.append(row)
    write_csv(path, rows)


def write_markdown_report(path, final, rows):
    lines = [
        "# Final LSTM Pose Extractor Benchmark",
        "",
        "## Scope",
        "",
        "This report focuses on the LSTM classification benchmark. Pose-only model speed/quality is a separate benchmark, and sequence generation is reported here only as the input stability layer for LSTM training/evaluation.",
        "",
        "## Pose-Only Benchmark",
        "",
        "Not rerun by this script. Previous pose-only results should not be used alone to select the final fall/Faint classifier.",
        "",
    ]
    if final.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in final["warnings"])
        lines.append("")
    lines.extend(
        [
            "## Selected Dataset Class Counts",
            "",
            "| split | Normal | Faint | total |",
            "| --- | --- | --- | --- |",
        ]
    )
    for split, counts in final.get("selected_dataset_class_counts", {}).items():
        lines.append(f"| {split} | {counts.get('Normal', 0)} | {counts.get('Faint', 0)} | {counts.get('total', 0)} |")
    lines.extend(
        [
        "",
        "## Sequence Generation Benchmark",
        "",
        "| model_label | clips_requested | clips_processed | person_detections | keypoints_extracted | generated_sequences | zero_sequence_clips | keypoint_missing_rate | runtime_seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {model_label} | {clips_requested} | {clips_processed} | {person_detections} | {keypoints_extracted} | {generated_sequences} | {zero_sequence_clips} | {keypoint_missing_rate} | {runtime_seconds} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## LSTM Classification Benchmark",
            "",
            "| model_label | train Normal | train Faint | eval Normal | eval Faint | accuracy | precision | Faint recall | F1-score | metrics_status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {model_label} | {train_sequence_normal} | {train_sequence_faint} | {eval_sequence_normal} | {eval_sequence_faint} | {accuracy} | {precision} | {recall} | {f1_score} | {metrics_status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Prediction Distribution Audit",
            "",
            "| model_label | predicted Normal | predicted Faint |",
            "| --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append("| {model_label} | {predicted_normal} | {predicted_faint} |".format(**row))
    lines.extend(
        [
            "",
            "## Threshold Audit",
            "",
            "| model_label | threshold | Faint recall | F1-score | precision | predicted Normal | predicted Faint |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in final.get("models", []):
        for audit_row in item.get("lstm_metrics", {}).get("threshold_audit", []):
            lines.append(
                f"| {item.get('model_label')} | {audit_row.get('threshold')} | {audit_row.get('faint_recall')} | {audit_row.get('f1_score')} | {audit_row.get('precision')} | {audit_row.get('predicted_normal')} | {audit_row.get('predicted_faint')} |"
            )
    lines.extend(
        [
            "",
            "## Repeated Seed Audit",
            "",
            "| model_label | seeds | Faint recall mean | Faint recall std | F1-score mean | F1-score std |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in final.get("models", []):
        seed_audit = item.get("lstm_metrics", {}).get("repeated_seed_audit", {})
        if not seed_audit.get("enabled"):
            continue
        seeds = ",".join(str(seed) for seed in seed_audit.get("seeds", []))
        lines.append(
            "| {model_label} | {seeds} | {recall_mean} | {recall_std} | {f1_mean} | {f1_std} |".format(
                model_label=item.get("model_label"),
                seeds=seeds,
                recall_mean=seed_audit.get("faint_recall_mean"),
                recall_std=seed_audit.get("faint_recall_std"),
                f1_mean=seed_audit.get("f1_score_mean"),
                f1_std=seed_audit.get("f1_score_std"),
            )
        )
    lines.extend(
        [
            "",
            "## Selection Policy",
            "",
            final["selection_policy"],
            "",
            f"Current best model for downstream LSTM: `{final['best_model_for_downstream_lstm']}`",
            "",
            "Evaluation priority: Faint recall, F1-score, false alarm tendency from confusion matrix, sequence stability, then runtime feasibility.",
            "",
            "Per-model details are saved under each model directory as `summary.json`, `train_clips.json`, `eval_clips.json`, `train_clip_diagnostics.csv`, `eval_clip_diagnostics.csv`, sequence CSV files, `eval_predictions.csv`, `threshold_audit.csv`, `repeated_seed_audit.json`, `history.json` when training runs, and `confusion_matrix.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    print(
        "[lstm-extractor-compare] sequence config: "
        f"sequence_length={args.sequence_length} "
        f"sequence_stride={args.sequence_stride} "
        "defaults=16/8 frame_sampling=disabled keypoint_feature=(sequence_length,51)",
        flush=True,
    )
    specs = parse_model_specs(args.models)
    rows = read_dataset_rows(args.metadata_csv)
    rows = prefilter_normal_rows(rows, args, specs, output_dir)
    rows = limit_rows_by_split_and_class(rows, args.max_rows_per_split, seed=args.seed)
    selected_class_counts = dataset_class_counts(rows)
    summaries = [compare_model(spec, rows, args, output_dir, selected_class_counts) for spec in specs]
    final = write_final_summary(output_dir, summaries, selected_class_counts)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

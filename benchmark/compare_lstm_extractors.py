import argparse
import csv
import json
import random
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
from scripts.run_dataset_evaluation import label_name, limit_rows_by_split, read_dataset_rows
from scripts.run_rtsp_inference import ensure_mock_keypoints


MODEL_SPECS = [
    {"label": "YOLOv11n-pose", "model": "yolo11n-pose.pt"},
    {"label": "YOLO26n-pose", "model": "yolo26n-pose.pt"},
    {"label": "YOLOv8s-pose", "model": "yolov8s-pose.pt"},
]
CLASS_TO_ID = {"Normal": 0, "Faint": 1}
ID_TO_CLASS = {0: "Normal", 1: "Faint"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare pose extractors as keypoint sequence sources for the LSTM pipeline.")
    parser.add_argument("--metadata-csv", default="../ai_fall_experiments/data/metadata/metadata.csv")
    parser.add_argument("--output-dir", default="benchmark/results/lstm_extractor_comparison")
    parser.add_argument("--models", default="YOLOv11n-pose:yolo11n-pose.pt,YOLO26n-pose:yolo26n-pose.pt,YOLOv8s-pose:yolov8s-pose.pt")
    parser.add_argument("--detector-mode", choices=["real", "mock"], default="real")
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
    return YoloPoseDetector(model_name, device=device, imgsz=imgsz, conf=conf)


def row_label_id(row):
    return CLASS_TO_ID.get(label_name(row), 0)


def keypoints_to_feature(detection, frame_shape, keypoint_conf_threshold):
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
    return np.stack(rows, axis=0), missing, total


def collect_split_sequences(rows, split_name, detector, args):
    x_rows = []
    y_rows = []
    sequence_rows = []
    clip_summaries = []
    totals = Counter()
    selected = [row for row in rows if (row.get("split") or "") == split_name]
    for row in selected:
        video_path = row.get("_resolved_video_path") or ""
        clip_id = row.get("clip_id") or Path(row.get("video_path") or row.get("clip_path") or video_path).stem
        clip = {
            "clip_id": clip_id,
            "split": split_name,
            "label": label_name(row),
            "video_path": video_path,
            "frames_processed": 0,
            "person_detections": 0,
            "keypoints_extracted": 0,
            "generated_sequences": 0,
            "zero_sequence": True,
            "fallback_usage": 0,
            "missing_keypoints": 0,
            "total_keypoints": 0,
            "error": None,
        }
        if not video_path:
            clip["error"] = "video_not_found"
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


def summarize_split(rows, y_rows, totals):
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
    }


def train_and_evaluate(train_x, train_y, eval_x, eval_y, args, output_dir):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(args.seed, torch)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    train_tensor = torch.from_numpy(np.stack(train_x).astype(np.float32))
    train_labels = torch.from_numpy(np.asarray(train_y, dtype=np.int64))
    eval_tensor = torch.from_numpy(np.stack(eval_x).astype(np.float32))
    eval_labels = torch.from_numpy(np.asarray(eval_y, dtype=np.int64))
    train_loader = DataLoader(TensorDataset(train_tensor, train_labels), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(eval_tensor, eval_labels), batch_size=args.batch_size)
    model_config = {
        "input_size": int(train_tensor.shape[-1]),
        "hidden_size": args.hidden_size,
        "num_layers": 1,
        "num_classes": 2,
        "dropout": 0.0,
    }
    model = LSTMActionModel(**model_config).model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": model_config, "classes": ["Normal", "Faint"]}, output_dir / "best.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history[-1] if history else empty_metrics()


def evaluate_lstm(model, loader, device):
    import torch

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            pred = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y_batch.cpu().tolist())
    return classification_metrics(y_true, y_pred)


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


def compare_model(spec, rows, args, output_dir):
    started = time.perf_counter()
    detector = create_pose_detector(args.detector_mode, spec["model"], args.device, args.imgsz, conf=args.detector_conf)
    model_dir = output_dir / spec["label"]
    model_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, train_clips, train_sequences, train_totals = collect_split_sequences(rows, args.train_split, detector, args)
    eval_x, eval_y, eval_clips, eval_sequences, eval_totals = collect_split_sequences(rows, args.eval_split, detector, args)
    write_csv(model_dir / "train_sequences.csv", train_sequences)
    write_csv(model_dir / "eval_sequences.csv", eval_sequences)
    (model_dir / "train_clips.json").write_text(json.dumps(train_clips, indent=2, ensure_ascii=False), encoding="utf-8")
    (model_dir / "eval_clips.json").write_text(json.dumps(eval_clips, indent=2, ensure_ascii=False), encoding="utf-8")
    train_summary = summarize_split(rows, train_y, train_totals)
    eval_summary = summarize_split(rows, eval_y, eval_totals)
    if args.dry_run:
        metrics = empty_metrics("dry_run")
    elif not train_x or not eval_x:
        metrics = empty_metrics("no_sequences")
    else:
        try:
            metrics = train_and_evaluate(train_x, train_y, eval_x, eval_y, args, model_dir)
            metrics["status"] = "OK"
        except Exception as exc:
            metrics = empty_metrics(f"failed: {exc}")
    runtime_seconds = round(time.perf_counter() - started, 4)
    write_confusion_matrix_csv(model_dir / "confusion_matrix.csv", metrics.get("confusion_matrix", {}))
    summary = {
        "model_label": spec["label"],
        "pose_model": spec["model"] if args.detector_mode == "real" else "mock",
        "fall_candidate_count_policy": "reference_only_not_model_selection",
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_sequence_summary": train_summary,
        "eval_sequence_summary": eval_summary,
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


def write_final_summary(output_dir, summaries):
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
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "metrics_status": metrics.get("status"),
                "runtime_seconds": item.get("runtime_seconds"),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", rows)
    final = {
        "selection_policy": "Prioritize Faint recall and stable sequence generation; fall_candidate_count is reference-only.",
        "best_model_for_downstream_lstm": choose_best_model(summaries),
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
        "## Sequence Generation Benchmark",
        "",
        "| model_label | clips_requested | clips_processed | person_detections | keypoints_extracted | generated_sequences | zero_sequence_clips | keypoint_missing_rate | runtime_seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
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
            "| model_label | accuracy | precision | Faint recall | F1-score | metrics_status |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {model_label} | {accuracy} | {precision} | {recall} | {f1_score} | {metrics_status} |".format(
                **row
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
            "Per-model details are saved under each model directory as `summary.json`, `train_clips.json`, `eval_clips.json`, sequence CSV files, `history.json` when training runs, and `confusion_matrix.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = limit_rows_by_split(read_dataset_rows(args.metadata_csv), args.max_rows_per_split)
    specs = parse_model_specs(args.models)
    summaries = [compare_model(spec, rows, args, output_dir) for spec in specs]
    final = write_final_summary(output_dir, summaries)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from ai.action.classifier import DEFAULT_CLASSES, LSTMActionModel, crops_to_features
from ai.action.lstm_contract import DEFAULT_LSTM_SEQUENCE_LENGTH, DEFAULT_LSTM_SEQUENCE_STRIDE
from ai.action.sequence_buffer import CropSequenceBuffer
from ai.detection.yolo_person_detector import MockPersonDetector, YoloPersonDetector
from ai.labels.event_label_loader import load_event_label
from ai.streams.video_reader import VideoReader


torch = None
nn = None
DataLoader = None
TensorDataset = None
LABEL_MAPPING = {"Normal": 0, "Faint": 1}


def print_cuda_diagnostics():
    try:
        import torch as torch_module

        print("\n========== CUDA / cuDNN Diagnostics ==========")
        print(f"torch version        : {torch_module.__version__}")
        print(f"cuda available       : {torch_module.cuda.is_available()}")
        print(f"torch cuda version   : {torch_module.version.cuda}")
        print(f"cuDNN enabled        : {torch_module.backends.cudnn.enabled}")
        print(f"cuDNN version        : {torch_module.backends.cudnn.version()}")

        if torch_module.cuda.is_available():
            device_count = torch_module.cuda.device_count()
            print(f"cuda device count    : {device_count}")

            for idx in range(device_count):
                props = torch_module.cuda.get_device_properties(idx)
                print(f"device {idx} name     : {props.name}")
                print(f"device {idx} memory   : {props.total_memory / 1024**3:.2f} GB")
                print(f"device {idx} capability: {props.major}.{props.minor}")

            current = torch_module.cuda.current_device()
            print(f"current cuda device  : cuda:{current}")
            print(f"current device name  : {torch_module.cuda.get_device_name(current)}")

            test_tensor = torch_module.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"test tensor device   : {test_tensor.device}")
        else:
            print("[train-lstm][warning] CUDA is not available. Training will run on CPU.")

        print("==============================================\n")
    except Exception as exc:  # noqa: BROAD_EXCEPT_OK - diagnostics must not abort training.
        print("\n========== CUDA / cuDNN Diagnostics ==========")
        print(f"[train-lstm][warning] Failed to check CUDA diagnostics: {exc}")
        print("==============================================\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_detector(mode, yolo_model, conf, iou, imgsz):
    if mode == "mock":
        return MockPersonDetector(conf=conf)
    if mode == "none":
        return NoPersonDetector()
    return YoloPersonDetector(yolo_model, conf=conf, iou=iou, imgsz=imgsz)


class NoPersonDetector:
    def detect(self, frame, frame_idx, conf=None):
        return {"frame_idx": int(frame_idx), "boxes": []}


def full_frame_box(frame):
    height, width = frame.shape[:2]
    return {"x1": 0.0, "y1": 0.0, "x2": float(width), "y2": float(height), "score": 0.0, "class_name": "fallback_frame"}


def load_training_rows(csv_path, split=None):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if split and row.get("split") and row.get("split") != split:
                continue
            video_path = row.get("video_path") or row.get("clip_path")
            if not video_path:
                continue
            label_value = row.get("label")
            annotation_path = row.get("label_path") or row.get("annotation_path", "")
            rows.append(
                {
                    "clip_id": row.get("clip_id") or Path(video_path).stem,
                    "video_path": video_path,
                    "label_path": annotation_path,
                    "label": int(label_value) if label_value not in (None, "") else None,
                    "label_name": row.get("label_name", ""),
                    "event_class": row.get("event_class", ""),
                    "start_frame": int(row.get("start_frame", 0) or 0),
                    "end_frame": int(row.get("end_frame", 0) or 0),
                    "split": row.get("split", split or ""),
                }
            )
    return rows


def load_row_label(row):
    label_path = row.get("label_path")
    if label_path and Path(label_path).exists():
        try:
            return load_event_label(label_path)
        except Exception:
            return None
    return None


def row_is_active(row, frame_idx):
    if row.get("label") is not None:
        return int(row["label"]) == 1
    label = load_row_label(row)
    return bool(label and label.is_active(frame_idx))


def row_label_name(row):
    if row.get("label_name"):
        return row["label_name"]
    if row.get("event_class"):
        return row["event_class"]
    return "Faint" if int(row.get("label") or 0) == 1 else "Normal"


def target_ranges_for_row(row):
    label = load_row_label(row)
    if label and label.event_frames:
        return [(max(0, int(start)), max(0, int(end)), True) for start, end in label.event_frames]
    if int(row.get("label") or 0) == 1 and row.get("start_frame") is not None and row.get("end_frame"):
        return [(max(0, int(row["start_frame"])), max(0, int(row["end_frame"])), False)]
    return [(0, None, False)]


def box_confidence(box):
    return float(box.get("score", 0.0)) if box else None


def detect_person(detector, frame, frame_idx, args):
    detection = detector.detect(frame, frame_idx)
    if detection["boxes"]:
        return detection, False
    if args.detector_mode == "yolo" and args.yolo_retry_conf < args.yolo_conf:
        retry = detector.detect(frame, frame_idx, conf=args.yolo_retry_conf)
        if retry["boxes"]:
            return retry, True
    return detection, False


def summarize_metadata(sequence_metadata, clip_summaries):
    total_sequences = len(sequence_metadata)
    fallback_sequences = sum(1 for item in sequence_metadata if item["crop_source"] == "fallback_full_frame")
    yolo_sequences = sum(1 for item in sequence_metadata if item["crop_source"] == "yolo_person_box")
    event_sequences = sum(1 for item in sequence_metadata if item["used_event_frame"])
    skipped_frames = sum(item.get("skipped_frames_no_person", 0) for item in clip_summaries)
    zero_sequence_clips = sum(1 for item in clip_summaries if item["sequences_generated"] == 0)
    return {
        "total_clips_processed": len(clip_summaries),
        "total_sequences_generated": total_sequences,
        "sequences_from_event_frame_ranges": event_sequences,
        "sequences_using_yolo_person_boxes": yolo_sequences,
        "sequences_using_fallback_full_frame_crops": fallback_sequences,
        "skipped_frames_due_to_no_person": skipped_frames,
        "skipped_clips_due_to_no_person": zero_sequence_clips,
        "zero_sequence_clips": zero_sequence_clips,
        "fallback_ratio": round(fallback_sequences / total_sequences, 4) if total_sequences else 0.0,
    }


def write_preprocess_outputs(output_dir, split_name, sequence_metadata, clip_summaries, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"preprocess_sequences_{split_name}.csv"
    if sequence_metadata:
        with metadata_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(sequence_metadata[0].keys()))
            writer.writeheader()
            writer.writerows(sequence_metadata)
    else:
        metadata_path.write_text("", encoding="utf-8")
    (output_dir / f"preprocess_clips_{split_name}.json").write_text(json.dumps(clip_summaries, indent=2), encoding="utf-8")
    (output_dir / f"preprocess_summary_{split_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_checkpoint_payload(model_state, model_config, args, best_acc, train_summary, val_summary):
    input_size = int(model_config["input_size"])
    crop_feature_size = int(args.feature_size)
    schema = getattr(args, "feature_schema", "keypoint51" if input_size == 51 else "keypoint_motion54")
    
    names = [f"kp{i}_{coord}" for i in range(17) for coord in ("x", "y", "conf")]
    if input_size == 54:
        if schema == "keypoint_bbox54":
            names += ["bbox_width_norm", "bbox_height_norm", "bbox_area_norm"]
        else:
            names += ["center_drop", "velocity", "torso_angle_norm"]
            
    return {
        "model_state": model_state,
        "model_config": model_config,
        "classes": list(DEFAULT_CLASSES),
        "feature_size": crop_feature_size,
        "sequence_length": int(args.sequence_length),
        "sequence_stride": int(args.sequence_stride),
        "feature_type": "crop" if input_size != 51 and input_size != 54 else "keypoint",
        "crop_feature_size": crop_feature_size,
        "input_size": input_size,
        "feature_schema_version": schema,
        "feature_names": names,
        "label_mapping": dict(LABEL_MAPPING),
        "best_val_acc": best_acc,
        "preprocess_summary": {"train": train_summary, "val": val_summary},
    }


def collect_sequences(rows, args, split_name, output_dir):
    x_rows = []
    y_rows = []
    sequence_metadata = []
    clip_summaries = []
    detector = create_detector(args.detector_mode, args.yolo_model, args.yolo_conf, args.yolo_iou, args.imgsz)
    for row in rows:
        buffer = CropSequenceBuffer(args.sequence_length, args.sequence_stride, args.resize_size)
        ranges = target_ranges_for_row(row)
        clip_generated = 0
        clip_skipped = 0
        current_source_counts = {"yolo_person_box": 0, "fallback_full_frame": 0, "skipped_no_person": 0}
        with VideoReader(row["video_path"]) as reader:
            while True:
                packet = reader.read()
                if packet is None:
                    break
                if args.max_frames > 0 and packet.frame_idx >= args.max_frames:
                    break
                active_ranges = [(start, end, used_event) for start, end, used_event in ranges if packet.frame_idx >= start and (end is None or packet.frame_idx <= end)]
                if not active_ranges:
                    continue
                range_start, range_end, used_event_frame = active_ranges[0]
                detection, retried = detect_person(detector, packet.frame, packet.frame_idx, args)
                detected_person = bool(detection["boxes"])
                crop_source = "yolo_person_box" if detected_person else "skipped_no_person"
                if args.fallback_full_frame and not detected_person:
                    detection["boxes"] = [full_frame_box(packet.frame)]
                    crop_source = "fallback_full_frame"
                elif not detected_person:
                    clip_skipped += 1
                    current_source_counts["skipped_no_person"] += 1
                    continue
                sequence = buffer.add(packet.frame_idx, packet.frame, detection["boxes"])
                if sequence is None:
                    continue
                x_rows.append(crops_to_features(sequence["crops"], args.feature_size))
                y_rows.append(1 if row_is_active(row, sequence["end_frame"]) else 0)
                clip_generated += 1
                current_source_counts[crop_source] += 1
                sequence_metadata.append(
                    {
                        "clip_id": row.get("clip_id") or Path(row["video_path"]).stem,
                        "video_filename": Path(row["video_path"]).name,
                        "label": row.get("label"),
                        "event_class": row_label_name(row),
                        "frame_start": sequence["start_frame"],
                        "frame_end": sequence["end_frame"],
                        "target_range_start": range_start,
                        "target_range_end": "" if range_end is None else range_end,
                        "detected_person": detected_person,
                        "crop_source": crop_source,
                        "detector_confidence": "" if not detection["boxes"] else box_confidence(detection["boxes"][0]),
                        "used_event_frame": used_event_frame,
                        "retried_lower_conf": retried,
                    }
                )
        clip_summaries.append(
            {
                "clip_id": row.get("clip_id") or Path(row["video_path"]).stem,
                "video_filename": Path(row["video_path"]).name,
                "label": row.get("label"),
                "event_class": row_label_name(row),
                "target_ranges": ranges,
                "sequences_generated": clip_generated,
                "skipped_frames_no_person": clip_skipped,
                "source_counts": current_source_counts,
                "zero_sequence": clip_generated == 0,
            }
        )
    summary = summarize_metadata(sequence_metadata, clip_summaries)
    write_preprocess_outputs(output_dir, split_name, sequence_metadata, clip_summaries, summary)
    print(f"[preprocess:{split_name}] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    if summary["fallback_ratio"] >= args.high_fallback_ratio:
        print("High fallback ratio detected. Detector recall must be improved before trusting model accuracy.", flush=True)
    if not x_rows:
        raise RuntimeError("No training sequences were generated. Check videos, labels, and detector settings.")
    return np.stack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.int64), summary


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred = logits.argmax(dim=1).cpu()
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)


def main():
    global torch, nn, DataLoader, TensorDataset
    parser = argparse.ArgumentParser(description="Train an LSTM action classifier from event-frame videos.")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output-dir", default="runs/action_lstm")
    parser.add_argument("--detector-mode", choices=["mock", "yolo", "none"], default="mock")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-retry-conf", type=float, default=0.15)
    parser.add_argument("--yolo-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fallback-full-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--high-fallback-ratio", type=float, default=0.5)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_LSTM_SEQUENCE_LENGTH)
    parser.add_argument("--sequence-stride", type=int, default=DEFAULT_LSTM_SEQUENCE_STRIDE)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--feature-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run-preprocess", action="store_true")
    args = parser.parse_args()

    print_cuda_diagnostics()
    print(f"[train-lstm] requested device: {args.device}")

    if not args.dry_run_preprocess:
        import torch as torch_module
        from torch import nn as nn_module
        from torch.utils.data import DataLoader as data_loader_cls
        from torch.utils.data import TensorDataset as tensor_dataset_cls

        torch = torch_module
        nn = nn_module
        DataLoader = data_loader_cls
        TensorDataset = tensor_dataset_cls

        set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_training_rows(args.dataset_csv, split=args.train_split)
    val_rows = load_training_rows(args.dataset_csv, split=args.val_split)
    if not val_rows and args.val_split != args.train_split:
        all_rows = load_training_rows(args.dataset_csv, split=None)
        split_at = max(1, int(len(all_rows) * 0.8))
        train_rows = all_rows[:split_at]
        val_rows = all_rows[split_at:] or all_rows[-1:]
    if args.max_rows_per_split > 0:
        train_rows = train_rows[: args.max_rows_per_split]
        val_rows = val_rows[: args.max_rows_per_split]
    if not train_rows:
        raise RuntimeError(f"No rows found for train split={args.train_split}")
    if not val_rows:
        raise RuntimeError(f"No rows found for val split={args.val_split}")

    train_x, train_y, train_summary = collect_sequences(train_rows, args, "train", output_dir)
    val_x, val_y, val_summary = collect_sequences(val_rows, args, "val", output_dir)
    if args.dry_run_preprocess:
        print("[train-lstm] dry-run preprocess complete")
        return
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=args.batch_size)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model_config = {
        "input_size": int(train_x.shape[-1]),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_classes": 2,
        "dropout": args.dropout,
    }
    model = LSTMActionModel(**model_config).model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
        val_acc = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": total_loss / max(1, total), "val_acc": val_acc}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                build_checkpoint_payload(model.state_dict(), model_config, args, best_acc, train_summary, val_summary),
                output_dir / "best.pt",
            )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[train-lstm] saved {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

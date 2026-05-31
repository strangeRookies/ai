import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai.action.classifier import LSTMActionModel, crops_to_features
from ai.action.sequence_buffer import CropSequenceBuffer
from ai.detection.yolo_person_detector import MockPersonDetector, YoloPersonDetector
from ai.labels.event_label_loader import load_dataset_rows, load_event_label
from ai.streams.video_reader import VideoReader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_detector(mode, yolo_model, conf, iou):
    if mode == "mock":
        return MockPersonDetector(conf=conf)
    return YoloPersonDetector(yolo_model, conf=conf, iou=iou)


def collect_sequences(rows, args):
    x_rows = []
    y_rows = []
    detector = create_detector(args.detector_mode, args.yolo_model, args.yolo_conf, args.yolo_iou)
    for row in rows:
        label = load_event_label(row["label_path"])
        buffer = CropSequenceBuffer(args.sequence_length, args.sequence_stride, args.resize_size)
        with VideoReader(row["video_path"]) as reader:
            while True:
                packet = reader.read()
                if packet is None:
                    break
                if args.max_frames > 0 and packet.frame_idx >= args.max_frames:
                    break
                detection = detector.detect(packet.frame, packet.frame_idx)
                sequence = buffer.add(packet.frame_idx, packet.frame, detection["boxes"])
                if sequence is None:
                    continue
                x_rows.append(crops_to_features(sequence["crops"], args.feature_size))
                y_rows.append(1 if label.is_active(sequence["end_frame"]) else 0)
    if not x_rows:
        raise RuntimeError("No training sequences were generated. Check videos, labels, and detector settings.")
    return np.stack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.int64)


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
    parser = argparse.ArgumentParser(description="Train an LSTM action classifier from event-frame videos.")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output-dir", default="runs/action_lstm")
    parser.add_argument("--detector-mode", choices=["mock", "yolo"], default="mock")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--yolo-iou", type=float, default=0.5)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--feature-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_dataset_rows(args.dataset_csv, split=args.train_split)
    val_rows = load_dataset_rows(args.dataset_csv, split=args.val_split)
    if not train_rows:
        raise RuntimeError(f"No rows found for train split={args.train_split}")
    if not val_rows:
        raise RuntimeError(f"No rows found for val split={args.val_split}")

    train_x, train_y = collect_sequences(train_rows, args)
    val_x, val_y = collect_sequences(val_rows, args)
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
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "classes": ["Normal", "Fall"],
                    "feature_size": args.feature_size,
                    "sequence_length": args.sequence_length,
                    "best_val_acc": best_acc,
                },
                output_dir / "best.pt",
            )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[train-lstm] saved {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

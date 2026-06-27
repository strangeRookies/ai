from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.classifier import LSTMActionModel, normalize_torch_device
from ai.action.lstm_contract import DEFAULT_LSTM_SEQUENCE_LENGTH, DEFAULT_LSTM_SEQUENCE_STRIDE
from ai.action.fight_vs_normal_dataset import (
    LABEL_TO_ID,
    SequenceBatch,
    collect_sequences,
    inspect_npz,
    load_fight_rows,
    stratified_split,
    validate_rows,
)
from ai.action.fight_vs_normal_metrics import CLASS_NAMES, classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Normal/Fight LSTM classifier from keypoint NPZ CSV.")
    parser.add_argument("--csv", default="data/fight_vs_normal_npz.csv")
    parser.add_argument("--output-dir", default="runs/fight_vs_normal_lstm")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_LSTM_SEQUENCE_LENGTH)
    parser.add_argument("--sequence-stride", type=int, default=DEFAULT_LSTM_SEQUENCE_STRIDE)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def set_seed(seed: int, torch_module: object) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if hasattr(torch_module, "cuda"):
        torch_module.cuda.manual_seed_all(seed)


def make_loader(batch: SequenceBatch, batch_size: int, shuffle: bool, torch_module: object) -> object:
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch_module.from_numpy(batch.x), torch_module.from_numpy(batch.y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: object, loader: object, device: object, torch_module: object) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    with torch_module.no_grad():
        for x_batch, y_batch in loader:
            probs = torch_module.softmax(model(x_batch.to(device)), dim=1).cpu()
            y_pred.extend(int(item) for item in probs.argmax(dim=1).tolist())
            y_true.extend(int(item) for item in y_batch.cpu().tolist())
            probabilities.extend([[round(float(value), 6) for value in row] for row in probs.tolist()])
    metrics = classification_metrics(y_true, y_pred)
    metrics["predictions"] = prediction_rows(y_true, y_pred, probabilities)
    return metrics


def prediction_rows(y_true: list[int], y_pred: list[int], probabilities: list[list[float]]) -> list[dict[str, object]]:
    return [
        {
            "true_label": CLASS_NAMES[truth],
            "pred_label": CLASS_NAMES[prediction],
            "normal_prob": probabilities[index][0],
            "fight_prob": probabilities[index][1],
        }
        for index, (truth, prediction) in enumerate(zip(y_true, y_pred))
    ]


def train(args: argparse.Namespace, train_batch: SequenceBatch, val_batch: SequenceBatch, test_batch: SequenceBatch) -> tuple[dict[str, object], list[dict[str, object]], object]:
    import torch
    from torch import nn

    set_seed(args.seed, torch)
    device = torch.device(normalize_torch_device(args.device, torch))
    train_loader = make_loader(train_batch, args.batch_size, True, torch)
    val_loader = make_loader(val_batch, args.batch_size, False, torch)
    test_loader = make_loader(test_batch, args.batch_size, False, torch)
    model_config = {
        "input_size": int(train_batch.x.shape[-1]),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_classes": len(CLASS_NAMES),
        "dropout": args.dropout,
    }
    model = LSTMActionModel(**model_config).model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    best_state = None
    best_f1 = -1.0
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
        val_metrics = evaluate(model, val_loader, device, torch)
        record = {"epoch": epoch, "train_loss": round(total_loss / max(total, 1), 6), "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        current_f1 = float(val_metrics["f1_score"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, torch)
    checkpoint = {"model_state": model.state_dict(), "model_config": model_config, "classes": list(CLASS_NAMES)}
    return test_metrics, history, checkpoint


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path: Path, metrics: dict[str, object]) -> None:
    matrix = metrics["confusion_matrix"]["matrix"]
    rows = [{"label": label, "Normal": matrix[index][0], "Fight": matrix[index][1]} for index, label in enumerate(CLASS_NAMES)]
    write_csv(path, rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_fight_rows(Path(args.csv))
    validation = validate_rows(rows)
    if validation["missing_files"]:
        raise RuntimeError(f"Missing NPZ files: {validation}")
    splits = stratified_split(rows, seed=args.seed)
    split_counts = {
        "train": dict(Counter(row.label_name for row in splits.train)),
        "val": dict(Counter(row.label_name for row in splits.val)),
        "test": dict(Counter(row.label_name for row in splits.test)),
    }
    train_batch = collect_sequences(splits.train, args.sequence_length, args.sequence_stride)
    val_batch = collect_sequences(splits.val, args.sequence_length, args.sequence_stride)
    test_batch = collect_sequences(splits.test, args.sequence_length, args.sequence_stride)
    test_metrics, history, checkpoint = train(args, train_batch, val_batch, test_batch)
    import torch

    torch.save({**checkpoint, "label_mapping": dict(LABEL_TO_ID), "feature_type": "keypoints", "sequence_length": args.sequence_length, "sequence_stride": args.sequence_stride}, output_dir / "best.pt")
    summary = {
        "csv": args.csv,
        "validation": validation,
        "npz_inspection": inspect_npz(rows[0].npz_path),
        "split_counts": split_counts,
        "sequence_counts": {"train": int(train_batch.y.size), "val": int(val_batch.y.size), "test": int(test_batch.y.size)},
        "test_metrics": test_metrics,
        "classes": list(CLASS_NAMES),
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "npz_inspection.json").write_text(json.dumps(summary["npz_inspection"], indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "test_predictions.csv", test_metrics.get("predictions", []))
    write_confusion_matrix(output_dir / "confusion_matrix.csv", test_metrics)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

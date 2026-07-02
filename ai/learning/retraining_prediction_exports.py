from __future__ import annotations

import csv
from pathlib import Path
from typing import Final, Mapping, Sequence


PREDICTION_EXPORT_FIELDS: Final = [
    "error_type",
    "clip_id",
    "clip_path",
    "label",
    "label_name",
    "ground_truth_label",
    "prediction",
    "prediction_label",
    "faint_prob",
    "prediction_score",
    "threshold",
    "evidence_id",
    "source_video",
    "frame_id",
    "frameId",
    "start_frame",
    "end_frame",
    "sequence_index",
    "scenario_tag",
    "augmentation_type",
    "split",
    "split_group_id",
    "parent_clip_id",
    "feature_schema",
    "feature_dim",
]

CLASS_NAMES: Final = {0: "Normal", 1: "Faint"}


def write_prediction_exports(
    output_dir: Path,
    model_prefix: str,
    test_y: Sequence[int],
    preds: Sequence[int],
    faint_probs: Sequence[float],
    seq_metadata: Sequence[Mapping[str, str | int]],
    threshold: float = 0.5,
    write_unprefixed_aliases: bool = False,
) -> dict[str, int | str]:
    rows = [
        _prediction_row(index, truth, pred, prob, seq_metadata[index], threshold)
        for index, (truth, pred, prob) in enumerate(zip(test_y, preds, faint_probs, strict=True))
    ]
    fp_rows = [row for row in rows if row["error_type"] == "FP"]
    fn_rows = [row for row in rows if row["error_type"] == "FN"]

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{model_prefix}_predictions.csv"
    false_positives_path = output_dir / f"{model_prefix}_false_positives.csv"
    false_negatives_path = output_dir / f"{model_prefix}_false_negatives.csv"

    _write_rows(predictions_path, rows)
    _write_rows(false_positives_path, fp_rows)
    _write_rows(false_negatives_path, fn_rows)

    if write_unprefixed_aliases:
        _write_rows(output_dir / "predictions.csv", rows)
        _write_rows(output_dir / "false_positives.csv", fp_rows)
        _write_rows(output_dir / "false_negatives.csv", fn_rows)

    return {
        "model_prefix": model_prefix,
        "predictions": len(rows),
        "false_positives": len(fp_rows),
        "false_negatives": len(fn_rows),
        "predictions_csv": str(predictions_path),
        "false_positives_csv": str(false_positives_path),
        "false_negatives_csv": str(false_negatives_path),
    }


def _prediction_row(
    sequence_index: int,
    truth: int,
    pred: int,
    faint_prob: float,
    metadata: Mapping[str, str | int],
    threshold: float,
) -> dict[str, str]:
    truth_label = CLASS_NAMES[int(truth)]
    pred_label = CLASS_NAMES[int(pred)]
    error_type = _error_type(int(truth), int(pred))
    clip_id = _text(metadata, "clip_id")
    frame_id = _text(metadata, "frame_id") or _text(metadata, "frameId")
    return {
        "error_type": error_type,
        "clip_id": clip_id,
        "clip_path": _text(metadata, "clip_path"),
        "label": str(int(truth)),
        "label_name": truth_label,
        "ground_truth_label": truth_label,
        "prediction": pred_label,
        "prediction_label": pred_label,
        "faint_prob": f"{float(faint_prob):.6f}",
        "prediction_score": f"{float(faint_prob):.6f}",
        "threshold": f"{float(threshold):.6f}",
        "evidence_id": _evidence_id(clip_id, frame_id, sequence_index),
        "source_video": _text(metadata, "source_video"),
        "frame_id": frame_id,
        "frameId": _text(metadata, "frameId") or frame_id,
        "start_frame": _text(metadata, "start_frame"),
        "end_frame": _text(metadata, "end_frame"),
        "sequence_index": str(sequence_index),
        "scenario_tag": _text(metadata, "scenario_tag"),
        "augmentation_type": _text(metadata, "augmentation_type"),
        "split": _text(metadata, "split"),
        "split_group_id": _text(metadata, "split_group_id"),
        "parent_clip_id": _text(metadata, "parent_clip_id"),
        "feature_schema": _text(metadata, "feature_schema"),
        "feature_dim": _text(metadata, "feature_dim"),
    }


def _error_type(truth: int, pred: int) -> str:
    if truth == 0 and pred == 1:
        return "FP"
    if truth == 1 and pred == 0:
        return "FN"
    return ""


def _evidence_id(clip_id: str, frame_id: str, sequence_index: int) -> str:
    if frame_id:
        return f"{clip_id}:{frame_id}"
    return f"{clip_id}:seq-{sequence_index}"


def _text(metadata: Mapping[str, str | int], key: str) -> str:
    value = metadata.get(key, "")
    return str(value).strip()


def _write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=PREDICTION_EXPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

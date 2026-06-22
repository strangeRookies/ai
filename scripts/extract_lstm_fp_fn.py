import argparse
import csv
import json
from pathlib import Path
from typing import Final, TypedDict


FIELDNAMES: Final = [
    "error_type",
    "clip_id",
    "label",
    "prediction",
    "confidence",
    "normal_prob",
    "faint_prob",
    "sequence_start",
    "sequence_end",
    "sequence_length",
    "sequence_index",
    "source_video",
    "clip_path",
]


class CsvRow(TypedDict, total=False):
    sequence_index: str
    clip_id: str
    frame_start: str
    frame_end: str
    true_label: str
    pred_label: str
    normal_prob: str
    faint_prob: str
    source_video: str
    clip_path: str


class ErrorRow(TypedDict):
    error_type: str
    clip_id: str
    label: str
    prediction: str
    confidence: str
    normal_prob: str
    faint_prob: str
    sequence_start: str
    sequence_end: str
    sequence_length: str
    sequence_index: str
    source_video: str
    clip_path: str


def read_csv(path: Path) -> list[CsvRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return [CsvRow(row) for row in csv.DictReader(fp)]


def metadata_by_clip(path: Path | None) -> dict[str, CsvRow]:
    if path is None:
        return {}
    return {row.get("clip_id", ""): row for row in read_csv(path) if row.get("clip_id")}


def predicted_label(row: CsvRow, threshold: float | None) -> str:
    if threshold is None:
        return row.get("pred_label", "")
    return "Faint" if float(row.get("faint_prob", "0") or "0") >= threshold else "Normal"


def prediction_confidence(row: CsvRow, prediction: str) -> str:
    prob = row.get("faint_prob", "") if prediction == "Faint" else row.get("normal_prob", "")
    return str(round(float(prob or "0"), 6))


def classify_error(label: str, prediction: str) -> str:
    if label == "Faint" and prediction == "Normal":
        return "FN"
    if label == "Normal" and prediction == "Faint":
        return "FP"
    return ""


def build_error_rows(predictions: list[CsvRow], metadata: dict[str, CsvRow], threshold: float | None, sequence_length: int) -> list[ErrorRow]:
    rows = []
    for row in predictions:
        label = row.get("true_label", "")
        prediction = predicted_label(row, threshold)
        error_type = classify_error(label, prediction)
        if not error_type:
            continue
        clip_id = row.get("clip_id", "")
        meta = metadata.get(clip_id, {})
        rows.append(
            ErrorRow(
                error_type=error_type,
                clip_id=clip_id,
                label=label,
                prediction=prediction,
                confidence=prediction_confidence(row, prediction),
                normal_prob=row.get("normal_prob", ""),
                faint_prob=row.get("faint_prob", ""),
                sequence_start=row.get("frame_start", ""),
                sequence_end=row.get("frame_end", ""),
                sequence_length=str(sequence_length),
                sequence_index=row.get("sequence_index", ""),
                source_video=meta.get("source_video", row.get("source_video", "")),
                clip_path=meta.get("clip_path", row.get("clip_path", "")),
            )
        )
    return rows


def write_rows(path: Path, rows: list[ErrorRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def resolve_predictions_path(predictions_path: Path | None, run_dir: Path | None) -> Path:
    if predictions_path is not None:
        return predictions_path
    if run_dir is None:
        raise RuntimeError("Provide --predictions or --run-dir.")
    candidate = run_dir / "sequence_length_30" / "YOLO26n-pose" / "eval_predictions.csv"
    if candidate.exists():
        return candidate
    fallback = run_dir / "YOLO26n-pose" / "eval_predictions.csv"
    if fallback.exists():
        return fallback
    raise RuntimeError(f"No sequence_length=30 eval_predictions.csv found under: {run_dir}")


def extract_errors(predictions_path: Path, output_dir: Path, metadata_path: Path | None, threshold: float | None, sequence_length: int) -> dict[str, int | str | None]:
    prediction_rows = read_csv(predictions_path)
    metadata_rows = metadata_by_clip(metadata_path)
    error_rows = build_error_rows(prediction_rows, metadata_rows, threshold, sequence_length)
    false_negatives = [row for row in error_rows if row["error_type"] == "FN"]
    false_positives = [row for row in error_rows if row["error_type"] == "FP"]
    write_rows(output_dir / "false_negatives.csv", false_negatives)
    write_rows(output_dir / "false_positives.csv", false_positives)
    summary = {
        "predictions": str(predictions_path),
        "metadata": str(metadata_path) if metadata_path is not None else None,
        "sequence_length": sequence_length,
        "threshold": threshold,
        "false_negatives": len(false_negatives),
        "false_positives": len(false_positives),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fp_fn_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FP/FN rows from LSTM eval_predictions.csv.")
    parser.add_argument("--predictions")
    parser.add_argument("--run-dir", help="Sequence comparison output dir containing sequence_length_30/YOLO26n-pose/eval_predictions.csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-csv")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--sequence-length", type=int, default=30)
    args = parser.parse_args()
    metadata = Path(args.metadata_csv) if args.metadata_csv else None
    predictions = resolve_predictions_path(Path(args.predictions) if args.predictions else None, Path(args.run_dir) if args.run_dir else None)
    payload = extract_errors(predictions, Path(args.output_dir), metadata, args.threshold, args.sequence_length)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

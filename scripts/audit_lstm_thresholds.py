import argparse
import csv
import json
from pathlib import Path
from typing import Final


THRESHOLDS: Final = (0.3, 0.4, 0.5, 0.6, 0.7)


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def label_id(label: str) -> int:
    return 1 if label.strip() == "Faint" else 0


def metrics_at_threshold(rows: list[dict[str, str]], threshold: float) -> dict[str, float | int]:
    tn = fp = fn = tp = 0
    for row in rows:
        truth = label_id(row.get("true_label", ""))
        pred = 1 if float(row.get("faint_prob", 0.0)) >= threshold else 0
        if truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1
        else:
            tp += 1
    total = max(tn + fp + fn + tp, 1)
    precision = tp / max(tp + fp, 1)
    faint_recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * faint_recall) / max(precision + faint_recall, 1e-12)
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round((tp + tn) / total, 6),
        "precision": round(precision, 6),
        "faint_recall": round(faint_recall, 6),
        "f1_score": round(f1, 6),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
    }


def recommend_threshold(rows: list[dict[str, float | int]]) -> float:
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["faint_recall"]),
            float(row["f1_score"]),
            -int(row["false_positives"]),
        ),
        reverse=True,
    )
    return float(ranked[0]["threshold"]) if ranked else 0.5


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, float | int]], recommended: float) -> None:
    lines = [
        "# Test Threshold Audit",
        "",
        "| threshold | accuracy | precision | Faint recall | F1 | false positives | false negatives |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']} | {row['accuracy']} | {row['precision']} | {row['faint_recall']} | "
            f"{row['f1_score']} | {row['false_positives']} | {row['false_negatives']} |"
        )
    lines.extend(
        [
            "",
            f"Recommended threshold: `{recommended}`",
            "",
            "Selection priority: Faint recall first, then F1, then false alarm count.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(predictions_path: Path, output_dir: Path) -> dict[str, object]:
    prediction_rows = read_prediction_rows(predictions_path)
    if not prediction_rows:
        raise RuntimeError(f"No prediction rows found: {predictions_path}")
    rows = [metrics_at_threshold(prediction_rows, threshold) for threshold in THRESHOLDS]
    recommended = recommend_threshold(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "threshold_audit.csv", rows)
    write_markdown(output_dir / "threshold_audit.md", rows, recommended)
    payload = {
        "predictions": str(predictions_path),
        "thresholds": rows,
        "recommended_threshold": recommended,
        "selection_rule": "Faint recall first, then F1, then false alarm count.",
    }
    (output_dir / "threshold_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final LSTM test thresholds from eval_predictions.csv.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.predictions), Path(args.output_dir)), indent=2))


if __name__ == "__main__":
    main()

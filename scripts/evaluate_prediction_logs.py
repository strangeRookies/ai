import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.evaluation.prediction_metrics import DEFAULT_EVAL_FOLDERS, metrics, read_prediction_logs, threshold_sweep


DEFAULT_THRESHOLDS = {
    "faint_confidence_threshold": [0.3],
    "consecutive_faint_count": [2],
    "event_cooldown_seconds": [10.0],
    "max_keypoint_missing_rate": [1.0],
    "min_avg_keypoint_conf": [0.0],
}


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_threshold_grid(args):
    return {
        "faint_confidence_threshold": parse_float_list(args.faint_confidence_thresholds),
        "consecutive_faint_count": parse_int_list(args.consecutive_faint_counts),
        "event_cooldown_seconds": parse_float_list(args.event_cooldown_seconds),
        "max_keypoint_missing_rate": parse_float_list(args.max_keypoint_missing_rates),
        "min_avg_keypoint_conf": parse_float_list(args.min_avg_keypoint_confs),
    }


def evaluate(args):
    rows = read_prediction_logs(args.sample_root)
    base_metrics = metrics(rows)
    grid = build_threshold_grid(args)
    sweep_rows = threshold_sweep(rows, grid)
    payload = {
        "sample_root": str(Path(args.sample_root)),
        "expected_folders": list(DEFAULT_EVAL_FOLDERS),
        "row_count": len(rows),
        "metrics": base_metrics,
        "threshold_sweep": sweep_rows,
        "best_thresholds": sweep_rows[0]["thresholds"] if sweep_rows else None,
    }
    if args.output:
        write_json(Path(args.output), payload)
    if args.sweep_csv:
        write_sweep_csv(Path(args.sweep_csv), sweep_rows)
    return payload


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_sweep_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "faint_confidence_threshold",
        "consecutive_faint_count",
        "event_cooldown_seconds",
        "max_keypoint_missing_rate",
        "min_avg_keypoint_conf",
        "precision",
        "recall",
        "f1_score",
        "false_positive_count",
        "false_negative_count",
        "TP",
        "FP",
        "FN",
        "TN",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            thresholds = row["thresholds"]
            matrix = row["confusion_matrix"]
            writer.writerow(
                {
                    **thresholds,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1_score"],
                    "false_positive_count": row["false_positive_count"],
                    "false_negative_count": row["false_negative_count"],
                    "TP": matrix["TP"],
                    "FP": matrix["FP"],
                    "FN": matrix["FN"],
                    "TN": matrix["TN"],
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate RTSP/LSTM prediction JSONL logs from labeled sample folders.")
    parser.add_argument("--sample-root", default="samples/evaluation")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sweep-csv", default=None)
    parser.add_argument("--faint-confidence-thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS["faint_confidence_threshold"]))
    parser.add_argument("--consecutive-faint-counts", default=",".join(str(v) for v in DEFAULT_THRESHOLDS["consecutive_faint_count"]))
    parser.add_argument("--event-cooldown-seconds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS["event_cooldown_seconds"]))
    parser.add_argument("--max-keypoint-missing-rates", default=",".join(str(v) for v in DEFAULT_THRESHOLDS["max_keypoint_missing_rate"]))
    parser.add_argument("--min-avg-keypoint-confs", default=",".join(str(v) for v in DEFAULT_THRESHOLDS["min_avg_keypoint_conf"]))
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

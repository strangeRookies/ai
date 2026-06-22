from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


SEQUENCE_LENGTHS = (8, 16, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO26n-pose LSTM sequence-length comparison.")
    parser.add_argument("--metadata-csv", default="../ai_fall_experiments/data/metadata/metadata.csv")
    parser.add_argument("--output-dir", default="benchmark/results/lstm_sequence_length_8_16_30")
    parser.add_argument("--detector-mode", choices=["real", "mock", "cache"], default="real")
    parser.add_argument("--keypoint-cache-dir", default="../ai_fall_experiments/data/keypoints/yolo26n-pose")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sequence-stride", type=int, default=4)
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
    parser.add_argument("--detector-conf", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repeat-seeds", type=int, default=1)
    parser.add_argument("--audit-thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--loss", choices=["ce", "weighted-ce", "focal", "oversample"], default="ce")
    parser.add_argument("--prefilter-normal-clips", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, sequence_length: int, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "benchmark/compare_lstm_extractors.py",
        "--metadata-csv",
        args.metadata_csv,
        "--output-dir",
        str(run_dir),
        "--models",
        "YOLO26n-pose:yolo26n-pose.pt",
        "--detector-mode",
        args.detector_mode,
        "--keypoint-cache-dir",
        args.keypoint_cache_dir,
        "--device",
        args.device,
        "--imgsz",
        str(args.imgsz),
        "--sequence-length",
        str(sequence_length),
        "--sequence-stride",
        str(args.sequence_stride),
        "--keypoint-conf-threshold",
        str(args.keypoint_conf_threshold),
        "--max-frames",
        str(args.max_frames),
        "--max-rows-per-split",
        str(args.max_rows_per_split),
        "--train-split",
        args.train_split,
        "--eval-split",
        args.eval_split,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-size",
        str(args.hidden_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--detector-conf",
        str(args.detector_conf),
        "--repeat-seeds",
        str(args.repeat_seeds),
        "--audit-thresholds",
        args.audit_thresholds,
        "--loss",
        args.loss,
    ] + (["--dry-run"] if args.dry_run else []) + (["--prefilter-normal-clips"] if args.prefilter_normal_clips else [])


def run_one(args: argparse.Namespace, sequence_length: int, output_dir: Path) -> dict[str, str | int | float | bool | None]:
    run_dir = output_dir / f"sequence_length_{sequence_length}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, sequence_length, run_dir)
    started = time.perf_counter()
    if not Path(args.metadata_csv).exists():
        record = {
            "sequence_length": sequence_length,
            "status": "missing_metadata",
            "metadata_csv": args.metadata_csv,
            "command": " ".join(command),
            "runtime_seconds": 0.0,
        }
        (run_dir / "raw_result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary_row(record, run_dir)
    completed = subprocess.run(command, cwd=Path.cwd(), capture_output=True, text=True, check=False)
    runtime_seconds = round(time.perf_counter() - started, 4)
    record = {
        "sequence_length": sequence_length,
        "status": "OK" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": " ".join(command),
        "runtime_seconds": runtime_seconds,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    (run_dir / "raw_result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_row(record, run_dir)


def summary_row(record: dict[str, str | int | float | bool | None], run_dir: Path) -> dict[str, str | int | float | bool | None]:
    model_summary_path = run_dir / "YOLO26n-pose" / "summary.json"
    metrics: dict[str, str | int | float | bool | None] = {}
    eval_summary: dict[str, str | int | float | bool | None] = {}
    if model_summary_path.exists():
        summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("lstm_metrics", {})
        eval_summary = summary.get("eval_sequence_summary", {})
    matrix = metrics.get("confusion_matrix", {}).get("matrix", [[0, 0], [0, 0]]) if metrics else [[0, 0], [0, 0]]
    fp = matrix[0][1] if len(matrix) >= 1 and len(matrix[0]) >= 2 else 0
    fn = matrix[1][0] if len(matrix) >= 2 and len(matrix[1]) >= 1 else 0
    return {
        "sequence_length": record["sequence_length"],
        "status": record["status"],
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "faint_recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "false_positive": fp,
        "false_negative": fn,
        "generated_sequences": eval_summary.get("generated_sequences"),
        "zero_sequence_clips": eval_summary.get("zero_sequence_clips"),
        "keypoint_missing_rate": eval_summary.get("keypoint_missing_rate"),
        "fallback_usage_ratio": eval_summary.get("fallback_usage_ratio"),
        "estimated_alert_delay_frames": record["sequence_length"],
        "runtime_seconds": record["runtime_seconds"],
        "best_checkpoint": str(run_dir / "YOLO26n-pose" / "best.pt"),
        "best_checkpoint_exists": (run_dir / "YOLO26n-pose" / "best.pt").exists(),
    }


def write_summary(output_dir: Path, rows: list[dict[str, str | int | float | bool | None]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# YOLO26n-pose LSTM Sequence Length 8/16/30 Comparison",
        "",
        "## Existing Structure",
        "",
        "The active keypoint benchmark path is `benchmark/compare_lstm_extractors.py`; it keeps 17 keypoints x,y,confidence as 51 features and trains the same LSTM model while sequence_length changes.",
        "",
        "## Result Table",
        "",
        "| sequence_length | status | Faint recall | F1 | FP | FN | generated sequences | zero sequence clips | estimated alert delay frames | checkpoint |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sequence_length']} | {row['status']} | {row['faint_recall']} | {row['f1_score']} | {row['false_positive']} | {row['false_negative']} | {row['generated_sequences']} | {row['zero_sequence_clips']} | {row['estimated_alert_delay_frames']} | {row['best_checkpoint_exists']} |"
        )
    lines.extend(
        [
            "",
            "## Cheap Filter Structure",
            "",
            "RTSP keypoint sequences now pass through a conservative score-based cheap filter before LSTM inference. The existing torso slope ratio 1.3 signal is retained and combined with keypoint confidence, bbox size, persistence, center drop, and bbox ratio change.",
            "",
            "## Recommendation Rule",
            "",
            "Prefer 16 frames when Faint recall/F1 are close because it balances alert delay and context. Use 8 frames for faster candidate compression and 30 frames only when it materially reduces false positives without hurting Faint recall.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_one(args, sequence_length, output_dir) for sequence_length in SEQUENCE_LENGTHS]
    write_summary(output_dir, rows)
    print(json.dumps({"output_dir": str(output_dir), "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import csv
import json
from pathlib import Path


def read_best_val_acc(run_dir):
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        return None
    return max(float(row.get("val_acc", 0.0)) for row in history)


def main():
    parser = argparse.ArgumentParser(description="Summarize LSTM action classifier runs.")
    parser.add_argument("--runs-dir", default="runs/action_lstm")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_path = Path(args.output) if args.output else runs_dir / "summary.csv"
    rows = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        checkpoint_path = run_dir / "best.pt"
        best_val_acc = read_best_val_acc(run_dir)
        rows.append(
            {
                "run": run_dir.name,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint_path),
                "checkpoint_exists": checkpoint_path.exists(),
                "best_val_acc": "" if best_val_acc is None else best_val_acc,
            }
        )
    if not rows:
        raise RuntimeError(f"No run directories found under {runs_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[lstm-summary] saved {output_path}")


if __name__ == "__main__":
    main()

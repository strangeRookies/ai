import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LSTM experiment result CSV files.")
    parser.add_argument("--runs", required=True)
    args = parser.parse_args()
    runs_dir = Path(args.runs)
    rows = []
    for result_path in sorted(runs_dir.rglob("results.csv")):
        experiment = str(result_path.parent.relative_to(runs_dir)).replace("\\", "/")
        df = pd.read_csv(result_path)
        df.insert(0, "experiment", experiment)
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No results.csv found under {runs_dir}")
    summary = pd.concat(rows, ignore_index=True)
    output = runs_dir / "summary.csv"
    summary.to_csv(output, index=False)
    print(f"[summary] saved {output}")


if __name__ == "__main__":
    main()

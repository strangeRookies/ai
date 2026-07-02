#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///

# --- How to run ---
# uv run scripts/run_self_improving_error_mining.py --sample --output-dir runs/self_improving_error_mining
# ------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.learning.self_improving_error_mining import append_jsonl, mine_prediction_rows
from ai.learning.self_improving_samples import sample_prediction_rows
from ai.learning.self_improving_synthetic import build_synthetic_preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine FP/FN candidates and create bbox54 synthetic feature previews.")
    parser.add_argument("--output-dir", default="runs/self_improving_error_mining")
    parser.add_argument("--sample", action="store_true", help="Use mock/sample rows when no real dataset is available.")
    parser.add_argument("--include-invalid", action="store_true", default=True)
    parser.add_argument("--synthetic-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_path = output_dir / "pipeline.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_lines = ["[self-improving] start"]
    if not args.sample:
        raise RuntimeError("Only --sample mode is implemented in local preflight; real dataset mining is a follow-up.")

    rows = sample_prediction_rows(include_invalid=args.include_invalid)
    log_lines.append(f"[self-improving] current_file=<sample> samples={len(rows)}")
    summary = mine_prediction_rows(rows, output_dir)
    log_lines.append(
        "[self-improving] "
        f"processed={summary['processed']} "
        f"accepted={summary['accepted_counts']} "
        f"quarantine={summary['quarantine_count']} "
        f"quarantine_reasons={summary['quarantine_reasons']}"
    )

    synthetic_count = 0
    if args.synthetic_preview:
        candidates = read_jsonl(output_dir / "hard_negative_candidates.jsonl")
        candidates.extend(read_jsonl(output_dir / "faint_fall_reinforcement_candidates.jsonl"))
        previews = build_synthetic_preview(candidates, seed=args.seed)
        for preview in previews:
            append_jsonl(output_dir / "synthetic_preview.jsonl", preview)
        synthetic_count = len(previews)
        log_lines.append(f"[self-improving] synthetic_preview_count={synthetic_count}")

    payload = dict(summary)
    payload["synthetic_preview_count"] = synthetic_count
    payload["synthetic_preview_path"] = str(output_dir / "synthetic_preview.jsonl")
    payload["real_dataset_verification"] = "not_run_local_data_missing"
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log_lines.append("[self-improving] done")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()

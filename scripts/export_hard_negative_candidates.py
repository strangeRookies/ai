#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.learning.candidate_manifests import build_error_candidates
from ai.learning.manifest_samples import sample_false_negative_rows, sample_false_positive_rows, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FP/FN error CSV rows into retraining candidate manifests.")
    parser.add_argument("--false-positives-csv")
    parser.add_argument("--false-negatives-csv")
    parser.add_argument("--metadata-csv")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--sample", nargs="?", const=100, type=int, help="Use built-in FP/FN sample rows, optionally capped to N rows.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(args.output_dir)
        metadata = Path(args.metadata_csv) if args.metadata_csv else None
        false_positives = Path(args.false_positives_csv) if args.false_positives_csv else None
        false_negatives = Path(args.false_negatives_csv) if args.false_negatives_csv else None
        if args.sample is not None:
            false_positives = Path(tmp) / "false_positives.csv"
            false_negatives = Path(tmp) / "false_negatives.csv"
            write_rows(false_positives, sample_false_positive_rows()[: args.sample])
            write_rows(false_negatives, sample_false_negative_rows()[: args.sample])
        if false_positives is None:
            raise RuntimeError("--false-positives-csv is required unless --sample is used")
        summaries = {
            "hard_negative": build_error_candidates(
                false_positives,
                output_dir / "hard_negative_candidates.csv",
                "hard_negative",
                metadata_csv=metadata,
                dry_run=args.dry_run,
            )
        }
        if false_negatives is not None:
            summaries["faint_reinforcement"] = build_error_candidates(
                false_negatives,
                output_dir / "faint_reinforcement_candidates.csv",
                "faint_reinforcement",
                metadata_csv=metadata,
                dry_run=args.dry_run,
            )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

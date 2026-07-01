#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.learning.candidate_manifests import build_training_manifest
from ai.learning.manifest_samples import sample_metadata_rows, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge base metadata with reviewed candidate manifests into training_manifest_v2.csv.")
    parser.add_argument("--base-metadata-csv", default="data/splits/final_source_video_split/all.csv")
    parser.add_argument("--hard-negative-csv", default="data/manifests/hard_negative_candidates.csv")
    parser.add_argument("--faint-reinforcement-csv", default="data/manifests/faint_reinforcement_candidates.csv")
    parser.add_argument("--synthetic-csv", default="data/manifests/synthetic_candidates.csv")
    parser.add_argument("--output-csv", default="data/manifests/training_manifest_v2.csv")
    parser.add_argument("--max-synthetic-ratio", type=float, default=0.3)
    parser.add_argument("--sample", nargs="?", const=100, type=int, help="Use built-in sample rows, optionally capped to N rows.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        base_metadata = Path(args.base_metadata_csv)
        if args.sample is not None:
            base_metadata = Path(tmp) / "sample_metadata.csv"
            write_rows(base_metadata, sample_metadata_rows()[: args.sample])
        candidate_csvs = [
            Path(args.hard_negative_csv),
            Path(args.faint_reinforcement_csv),
            Path(args.synthetic_csv),
        ]
        summary = build_training_manifest(
            base_metadata,
            Path(args.output_csv),
            candidate_csvs,
            max_synthetic_ratio=args.max_synthetic_ratio,
            sample=args.sample,
            dry_run=args.dry_run,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

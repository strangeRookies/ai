#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.learning.candidate_manifests import check_manifest_leakage


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate training_manifest_v2.csv review, synthetic, and split leakage rules.")
    parser.add_argument("--manifest", default="data/manifests/training_manifest_v2.csv")
    parser.add_argument("--max-synthetic-ratio", type=float, default=0.3)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    try:
        summary = check_manifest_leakage(Path(args.manifest), max_synthetic_ratio=args.max_synthetic_ratio)
    except RuntimeError as exc:
        if not args.report_only:
            raise
        summary = {"manifest": args.manifest, "status": "failed", "error": str(exc)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

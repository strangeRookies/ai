#!/usr/bin/env python
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Inspect candidate manifest files (CSV or JSONL).")
    parser.add_argument("--input", required=True, help="Path to input CSV or JSONL file.")
    parser.add_argument("--group-by", help="Group by a specific key or column name.")
    parser.add_argument("--require-feature-schema", help="Enforce specific feature schema version.")
    parser.add_argument("--require-feature-dim", type=int, help="Enforce specific dimension size.")
    parser.add_argument("--exclude-source-type", help="Filter out rows matching specific source type.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    rows = []
    # Load file based on extension
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    print(f"[WARNING] Failed to parse line {i} as JSON: {e}")
    else:  # Assume CSV
        with input_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))

    total_count = len(rows)
    print(f"Total candidates loaded: {total_count}")

    # Exclude source type
    if args.exclude_source_type:
        exclude_val = args.exclude_source_type.strip()
        original_count = len(rows)
        rows = [r for r in rows if r.get("source_type") != exclude_val]
        excluded = original_count - len(rows)
        print(f"Excluded {excluded} rows with source_type='{exclude_val}'. Remaining: {len(rows)}")

    # Enforce quality validation checks
    violations = 0
    if args.require_feature_schema or args.require_feature_dim:
        print("\n--- Validating Feature Quality ---")
        for i, row in enumerate(rows):
            if args.require_feature_schema:
                schema = row.get("feature_schema")
                if schema != args.require_feature_schema:
                    print(f"[VIOLATION] Row {i}: schema mismatch. Expected '{args.require_feature_schema}', got '{schema}'")
                    violations += 1
                    continue
            if args.require_feature_dim is not None:
                dim_str = row.get("feature_dim")
                try:
                    dim = int(dim_str) if dim_str is not None else None
                except ValueError:
                    dim = None
                if dim != args.require_feature_dim:
                    print(f"[VIOLATION] Row {i}: dimension mismatch. Expected {args.require_feature_dim}, got '{dim_str}'")
                    violations += 1
                    continue

        if violations == 0:
            print("[OK] All rows satisfy feature requirements.")
        else:
            print(f"[WARNING] Found {violations} validation violations.")

    # Group by logic
    if args.group_by:
        group_key = args.group_by.strip()
        print(f"\n--- Group counts by '{group_key}' ---")
        counter = Counter()
        for r in rows:
            val = r.get(group_key, "N/A")
            counter[val] += 1

        for key, count in counter.most_common():
            pct = (count / len(rows)) * 100 if len(rows) > 0 else 0
            print(f"  {key}: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import csv
import json
import random
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export ratio-specific retraining manifests with hard negatives.")
    parser.add_argument("--baseline-manifest", required=True, help="Path to baseline manifest CSV (e.g. data/splits/final_source_video_split/all.csv)")
    parser.add_argument("--hard-negative-candidates", required=True, help="Path to candidates CSV or JSONL file.")
    parser.add_argument("--ratios", default="0.05,0.10,0.20", help="Comma-separated float ratios (e.g., 0.05,0.10,0.20).")
    parser.add_argument("--feature-schema", default="keypoint_bbox54", help="Enforce feature schema.")
    parser.add_argument("--feature-dim", type=int, default=54, help="Enforce feature dimension.")
    parser.add_argument("--output-dir", default="runs/hard_negative_retraining_comparison/train_exports", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")

    args = parser.parse_args()

    baseline_path = Path(args.baseline_manifest)
    candidates_path = Path(args.hard_negative_candidates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        print(f"[ERROR] Baseline manifest not found: {baseline_path}")
        sys.exit(1)
    if not candidates_path.exists():
        print(f"[ERROR] Candidates manifest not found: {candidates_path}")
        sys.exit(1)

    # 1. Read baseline manifest and split
    baseline_rows = []
    with baseline_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            baseline_rows.append(dict(row))

    train_baseline = [r for r in baseline_rows if r.get("split") == "train"]
    val_baseline = [r for r in baseline_rows if r.get("split") == "val"]
    test_baseline = [r for r in baseline_rows if r.get("split") == "test"]

    n_baseline_train = len(train_baseline)
    print(f"Baseline statistics:")
    print(f"  Total rows: {len(baseline_rows)}")
    print(f"  Train count (N): {n_baseline_train}")
    print(f"  Val count: {len(val_baseline)}")
    print(f"  Test count: {len(test_baseline)}")

    # 2. Read and filter candidates
    candidates = []
    if candidates_path.suffix.lower() == ".jsonl":
        with candidates_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    candidates.append(json.loads(line))
                except Exception as e:
                    print(f"[WARNING] Failed to parse line {i} as JSON: {e}")
    else:
        with candidates_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidates.append(dict(row))

    approved_candidates = []
    excluded_counts = {"review_status": 0, "candidate_type": 0, "feature_schema": 0, "feature_dim": 0, "zero_bbox": 0}

    for cand in candidates:
        # Check review status
        if cand.get("review_status", "").strip().lower() != "approved":
            excluded_counts["review_status"] += 1
            continue
        # Check candidate type
        if cand.get("candidate_type", "").strip().lower() != "hard_negative":
            excluded_counts["candidate_type"] += 1
            continue
        # Check feature schema
        if cand.get("feature_schema") != args.feature_schema:
            excluded_counts["feature_schema"] += 1
            continue
        # Check feature dim
        try:
            cand_dim = int(cand.get("feature_dim", 0))
        except ValueError:
            cand_dim = 0
        if cand_dim != args.feature_dim:
            excluded_counts["feature_dim"] += 1
            continue

        # Check bbox validity (must not be zero)
        bbox = cand.get("bbox_features")
        if isinstance(bbox, str):
            try:
                bbox = json.loads(bbox)
            except Exception:
                bbox = None

        if bbox is not None:
            try:
                if all(float(x) == 0.0 for x in bbox):
                    excluded_counts["zero_bbox"] += 1
                    continue
            except Exception:
                pass

        approved_candidates.append(cand)

    print(f"Candidates statistics:")
    print(f"  Total loaded candidates: {len(candidates)}")
    print(f"  Approved strict HN candidates: {len(approved_candidates)}")
    print(f"  Excluded: {excluded_counts}")

    # 3. Export ratios
    ratios = [float(r.strip()) for r in args.ratios.split(",") if r.strip()]
    if 0.0 not in ratios:
        ratios.insert(0, 0.0)

    random.seed(args.seed)
    actual_exports = {}

    for ratio in ratios:
        target_count = int(round(n_baseline_train * ratio))
        if ratio == 0.0:
            sampled_hn = []
        else:
            if target_count > len(approved_candidates):
                print(f"[WARNING] Target HN count ({target_count}) for ratio {ratio} exceeds available approved candidates ({len(approved_candidates)}). Sampling all available.")
                sampled_hn = approved_candidates
            else:
                sampled_hn = random.sample(approved_candidates, target_count)

        # Make copy of baseline train rows and append mapped hard negatives
        new_train = list(train_baseline)
        for hn in sampled_hn:
            mapped_row = {}
            for col in fieldnames:
                val = hn.get(col)
                if val is None:
                    if col == "split":
                        val = "train"
                    elif col == "label":
                        val = "0"
                    elif col == "label_name":
                        val = "Normal"
                    elif col == "source_type":
                        val = "hard_negative"
                    else:
                        val = ""
                mapped_row[col] = val
            new_train.append(mapped_row)

        combined_rows = new_train + val_baseline + test_baseline

        # Save to csv
        filename = "baseline.csv" if ratio == 0.0 else f"hn_{ratio}.csv"
        output_path = output_dir / filename
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in combined_rows:
                writer.writerow(r)

        actual_exports[f"hn_{ratio}"] = {
            "ratio": ratio,
            "target_count": target_count,
            "exported_hn_count": len(sampled_hn),
            "total_train_count": len(new_train),
            "output_file": str(output_path)
        }
        print(f"Exported variant to {output_path} (HN added: {len(sampled_hn)}, Total Train: {len(new_train)})")

    # Save summary
    summary = {
        "baseline_train_count": n_baseline_train,
        "baseline_val_count": len(val_baseline),
        "baseline_test_count": len(test_baseline),
        "approved_hard_negative_available": len(approved_candidates),
        "excluded_candidates_by_reason": excluded_counts,
        "variants": actual_exports,
        "seed": args.seed
    }

    summary_path = output_dir / "export_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved export summary to {summary_path}")


if __name__ == "__main__":
    main()

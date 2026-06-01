import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


SPLITS = ("train", "test", "val")


def label_name(row):
    if row.get("label_name"):
        return row["label_name"]
    return "Faint" if str(row.get("label", "0")) == "1" else "Normal"


def group_key(row):
    return row.get("source_video") or row.get("video_path") or row.get("clip_path") or row.get("clip_id")


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def summarize(rows):
    summary = []
    total = max(1, len(rows))
    by_split = defaultdict(list)
    for row in rows:
        by_split[row.get("split", "")].append(row)
    for split in SPLITS:
        split_rows = by_split.get(split, [])
        counts = Counter(label_name(row) for row in split_rows)
        summary.append(
            {
                "split": split,
                "total": len(split_rows),
                "Faint": counts.get("Faint", 0),
                "Normal": counts.get("Normal", 0),
                "ratio": round(len(split_rows) / total, 4),
            }
        )
    return summary


def leakage_report(rows):
    seen = defaultdict(set)
    for row in rows:
        key = group_key(row)
        if key:
            seen[key].add(row.get("split", ""))
    leaked = {key: sorted(splits) for key, splits in seen.items() if len(splits - {""}) > 1}
    return leaked


def approximately_ok(summary, tolerance=0.05):
    expected = {"train": 0.70, "test": 0.15, "val": 0.15}
    return all(abs(item["ratio"] - expected[item["split"]]) <= tolerance for item in summary)


def stratified_group_split(rows, seed=42, train_ratio=0.70, test_ratio=0.15):
    groups = {}
    for row in rows:
        key = group_key(row)
        if not key:
            key = row.get("clip_id") or row.get("clip_path") or str(len(groups))
        group = groups.setdefault(key, {"rows": [], "positive": 0})
        group["rows"].append(row)
        group["positive"] = max(group["positive"], 1 if label_name(row) == "Faint" else 0)

    rng = random.Random(seed)
    positives = [item for item in groups.items() if item[1]["positive"]]
    negatives = [item for item in groups.items() if not item[1]["positive"]]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    split_map = {}
    for bucket in (positives, negatives):
        n = len(bucket)
        train_end = int(round(n * train_ratio))
        test_end = train_end + int(round(n * test_ratio))
        for key, _ in bucket[:train_end]:
            split_map[key] = "train"
        for key, _ in bucket[train_end:test_end]:
            split_map[key] = "test"
        for key, _ in bucket[test_end:]:
            split_map[key] = "val"

    fixed = []
    for row in rows:
        item = dict(row)
        item["split"] = split_map[group_key(row)]
        fixed.append(item)
    return fixed


def write_rows(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Check dataset split ratio and source-video leakage.")
    parser.add_argument("--metadata-csv", default="../ai_fall_experiments/data/metadata/metadata.csv")
    parser.add_argument("--write-fixed", default=None, help="Optional output CSV for stratified source-video split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_rows(args.metadata_csv)
    if not rows:
        raise RuntimeError(f"No rows found: {args.metadata_csv}")
    if "split" not in rows[0] or not all(row.get("split") for row in rows):
        rows = stratified_group_split(rows, seed=args.seed)

    summary = summarize(rows)
    leaked = leakage_report(rows)
    result = {
        "metadata_csv": args.metadata_csv,
        "total": len(rows),
        "summary": summary,
        "target_ratio": {"train": 0.70, "test": 0.15, "val": 0.15},
        "ratio_ok_approx": approximately_ok(summary),
        "leakage_count": len(leaked),
        "leakage_examples": dict(list(leaked.items())[:5]),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if (not result["ratio_ok_approx"] or leaked) and args.write_fixed:
        fixed = stratified_group_split(read_rows(args.metadata_csv), seed=args.seed)
        write_rows(args.write_fixed, fixed)
        fixed_summary = summarize(fixed)
        print(json.dumps({"fixed_output": args.write_fixed, "summary": fixed_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final


SPLITS: Final = ("train", "val", "test")
RATIOS: Final = {"train": 0.70, "val": 0.15, "test": 0.15}
LABELS: Final = ("Normal", "Faint")


class SplitValidationError(RuntimeError):
    pass


def label_name(row: dict[str, str]) -> str:
    value = row.get("label_name", "").strip()
    if value:
        return value
    return "Faint" if row.get("label", "").strip() == "1" else "Normal"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def source_video(row: dict[str, str]) -> str:
    value = row.get("source_video", "").strip()
    if not value:
        raise SplitValidationError("metadata row is missing source_video")
    return value


def primary_domain(rows: list[dict[str, str]]) -> str:
    counts = Counter(row.get("domain", "unknown") or "unknown" for row in rows)
    return counts.most_common(1)[0][0]


def split_counts(total: int) -> dict[str, int]:
    floors = {split: int(total * ratio) for split, ratio in RATIOS.items()}
    remaining = total - sum(floors.values())
    order = sorted(SPLITS, key=lambda split: (total * RATIOS[split]) - floors[split], reverse=True)
    for split in order[:remaining]:
        floors[split] += 1
    return floors


def build_source_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[source_video(row)].append(row)
    return dict(groups)


def assign_source_splits(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    groups = build_source_groups(rows)
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for key, group_rows in groups.items():
        has_faint = any(label_name(row) == "Faint" for row in group_rows)
        strata[(primary_domain(group_rows), has_faint)].append(key)

    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for stratum_key in sorted(strata):
        keys = strata[stratum_key]
        rng.shuffle(keys)
        counts = split_counts(len(keys))
        cursor = 0
        for split in SPLITS:
            for key in keys[cursor : cursor + counts[split]]:
                assignments[key] = split
            cursor += counts[split]
    return assignments


def attach_split(rows: list[dict[str, str]], assignments: dict[str, str]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item["split"] = assignments[source_video(row)]
        output.append(item)
    return output


def deterministic_sample(rows: list[dict[str, str]], seed: int, split: str, label: str, limit: int) -> list[dict[str, str]]:
    rng = random.Random(f"{seed}:{split}:{label}")
    decorated = [(rng.random(), index, row) for index, row in enumerate(rows)]
    return [row for _, _, row in sorted(decorated)[:limit]]


def class_balance_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for split in SPLITS:
        split_rows = [row for row in rows if row.get("split") == split]
        faint_rows = [row for row in split_rows if label_name(row) == "Faint"]
        normal_rows = [row for row in split_rows if label_name(row) == "Normal"]
        normal_limit = min(len(normal_rows), len(faint_rows))
        selected.extend(deterministic_sample(faint_rows, seed, split, "Faint", len(faint_rows)))
        selected.extend(deterministic_sample(normal_rows, seed, split, "Normal", normal_limit))
    return selected


def count_by_split_label(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts = Counter(label_name(row) for row in rows if row.get("split") == split)
        result[split] = {"Normal": counts.get("Normal", 0), "Faint": counts.get("Faint", 0), "total": sum(counts.values())}
    return result


def source_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {split: len({source_video(row) for row in rows if row.get("split") == split}) for split in SPLITS}


def domain_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts = Counter((row.get("domain", "unknown") or "unknown") for row in rows if row.get("split") == split)
        result[split] = dict(sorted(counts.items()))
    return result


def leakage(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        seen[source_video(row)].add(row.get("split", ""))
    return {key: sorted(values) for key, values in seen.items() if len(values) > 1}


def validate_outputs(rows: list[dict[str, str]], output_dir: Path) -> None:
    leaked = leakage(rows)
    if leaked:
        raise SplitValidationError(f"source_video leakage detected: {dict(list(leaked.items())[:5])}")
    counts = count_by_split_label(rows)
    for split in SPLITS:
        if counts[split]["Normal"] <= 0 or counts[split]["Faint"] <= 0:
            raise SplitValidationError(f"{split} split is missing Normal or Faint rows: {counts[split]}")
    for name in ("train.csv", "val.csv", "test.csv"):
        if not (output_dir / name).exists():
            raise SplitValidationError(f"missing output CSV: {output_dir / name}")


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = summary["class_counts"]
    sources = summary["source_video_counts"]
    domains = summary["domain_distribution"]
    lines = [
        "# Final Source-Video Split Report",
        "",
        "YOLO26n-pose + LSTM Normal/Faint final training split.",
        "",
        "| split | rows | Normal | Faint | source_videos |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        split_counts = counts[split]
        lines.append(f"| {split} | {split_counts['total']} | {split_counts['Normal']} | {split_counts['Faint']} | {sources[split]} |")
    lines.extend(["", "## Domain Distribution", ""])
    for split in SPLITS:
        lines.append(f"- {split}: {domains[split]}")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Leakage check: {summary['leakage_check']}",
            f"- Class balance check: {summary['class_balance_check']}",
            f"- Seed: {summary['seed']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(rows: list[dict[str, str]], seed: int, output_dir: Path) -> dict[str, object]:
    counts = count_by_split_label(rows)
    return {
        "seed": seed,
        "ratio": RATIOS,
        "output_dir": str(output_dir),
        "class_counts": counts,
        "source_video_counts": source_counts(rows),
        "domain_distribution": domain_counts(rows),
        "leakage_check": "PASS" if not leakage(rows) else "FAIL",
        "class_balance_check": "PASS" if all(counts[split]["Normal"] == counts[split]["Faint"] for split in SPLITS) else "FAIL",
        "outputs": {
            "train": str(output_dir / "train.csv"),
            "val": str(output_dir / "val.csv"),
            "test": str(output_dir / "test.csv"),
            "all": str(output_dir / "all.csv"),
        },
    }


def create_split(metadata_csv: Path, output_dir: Path, seed: int) -> dict[str, object]:
    rows = read_rows(metadata_csv)
    if not rows:
        raise SplitValidationError(f"No rows found: {metadata_csv}")
    selected = class_balance_rows(attach_split(rows, assign_source_splits(rows, seed)), seed)
    fieldnames = list(dict.fromkeys([*rows[0].keys(), "split"]))
    for split in SPLITS:
        write_rows(output_dir / f"{split}.csv", [row for row in selected if row["split"] == split], fieldnames)
    write_rows(output_dir / "all.csv", selected, fieldnames)
    validate_outputs(selected, output_dir)
    summary = build_summary(selected, seed, output_dir)
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir / "split_report.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final source_video-based YOLO26n/LSTM train/val/test split.")
    parser.add_argument("--metadata-csv", default="../ai_fall_experiments/data/metadata/metadata.csv")
    parser.add_argument("--output-dir", default="data/splits/final_source_video_split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(create_split(Path(args.metadata_csv), Path(args.output_dir), args.seed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

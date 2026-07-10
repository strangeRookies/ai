from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from ai.learning.manifest_schema import ManifestSummary


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"CSV file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV file has no header: {path}")
        return [dict(row) for row in reader]


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str], dry_run: bool = False) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_csv: Path, summary: ManifestSummary, dry_run: bool) -> None:
    if dry_run:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def summarize(rows: list[dict[str, str]], output_csv: Path, written: bool) -> ManifestSummary:
    source_type_counts = Counter(row.get("source_type", "unknown") or "unknown" for row in rows)
    class_counts = Counter(row.get("label_name", "unknown") or "unknown" for row in rows)
    counts = Counter(source_type_counts)
    counts.update(class_counts)
    return {
        "rows": len(rows),
        "output_csv": str(output_csv),
        "written": written,
        "counts": dict(sorted(counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
    }

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ai.learning.manifest_csv import read_csv_rows
from ai.learning.manifest_schema import LeakageSummary


def check_manifest_leakage(manifest_csv: Path, max_synthetic_ratio: float = 0.3) -> LeakageSummary:
    rows = read_csv_rows(manifest_csv)
    return check_manifest_rows(rows, max_synthetic_ratio=max_synthetic_ratio)


def check_manifest_rows(rows: list[dict[str, str]], max_synthetic_ratio: float = 0.3) -> LeakageSummary:
    split_group_leaks = _count_split_leaks(rows, "split_group_id")
    parent_clip_leaks = _count_split_leaks(rows, "parent_clip_id")
    synthetic_rows = [row for row in rows if row.get("source_type", "").strip() == "synthetic"]
    pending_or_rejected = [row for row in rows if row.get("review_status", "").strip() in {"pending", "rejected", "needs_review"}]
    synthetic_missing_parent = [row for row in synthetic_rows if not row.get("parent_clip_id", "").strip()]
    synthetic_ratio = len(synthetic_rows) / len(rows) if rows else 0.0
    summary: LeakageSummary = {
        "rows": len(rows),
        "synthetic_rows": len(synthetic_rows),
        "synthetic_ratio": round(synthetic_ratio, 6),
        "split_group_leaks": split_group_leaks,
        "parent_clip_leaks": parent_clip_leaks,
        "pending_or_rejected_rows": len(pending_or_rejected),
        "synthetic_missing_parent_rows": len(synthetic_missing_parent),
    }
    problems = _problems(summary, max_synthetic_ratio)
    if problems:
        raise RuntimeError("; ".join(problems))
    return summary


def _count_split_leaks(rows: list[dict[str, str]], group_field: str) -> int:
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_id = _group_id(row, group_field)
        split = row.get("split", "").strip()
        if group_id and split:
            splits_by_group[group_id].add(split)
    return sum(1 for splits in splits_by_group.values() if len(splits) > 1)


def _group_id(row: dict[str, str], group_field: str) -> str:
    value = row.get(group_field, "").strip()
    if value:
        return value
    if group_field == "parent_clip_id":
        return row.get("clip_id", "").strip()
    return ""


def _problems(summary: LeakageSummary, max_synthetic_ratio: float) -> list[str]:
    problems: list[str] = []
    if summary["split_group_leaks"] > 0:
        problems.append(f"split_group_id leakage={summary['split_group_leaks']}")
    if summary["parent_clip_leaks"] > 0:
        problems.append(f"parent_clip_id leakage={summary['parent_clip_leaks']}")
    if summary["pending_or_rejected_rows"] > 0:
        problems.append(f"pending/rejected rows={summary['pending_or_rejected_rows']}")
    if summary["synthetic_missing_parent_rows"] > 0:
        problems.append(f"synthetic rows missing parent_clip_id={summary['synthetic_missing_parent_rows']}")
    if summary["synthetic_ratio"] > max_synthetic_ratio:
        problems.append(f"synthetic ratio {summary['synthetic_ratio']} > {max_synthetic_ratio}")
    return problems

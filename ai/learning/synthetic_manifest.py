from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final, TypedDict


SYNTHETIC_TYPES: Final = ("brightness", "noise", "blur", "crop", "occlusion", "distance")


class SyntheticSummary(TypedDict):
    input_rows: int
    candidate_rows: int
    output_csv: str


def build_synthetic_candidate_manifest(metadata_csv: Path, output_csv: Path) -> SyntheticSummary:
    rows = _read_rows(metadata_csv)
    output_rows: list[dict[str, str]] = []
    for row in rows:
        for synthetic_type in SYNTHETIC_TYPES:
            candidate = dict(row)
            candidate["source_type"] = "synthetic_candidate"
            candidate["synthetic_type"] = synthetic_type
            candidate["reason"] = _candidate_reason(row, synthetic_type)
            output_rows.append(candidate)
    _write_rows(output_csv, output_rows)
    summary = SyntheticSummary(input_rows=len(rows), candidate_rows=len(output_rows), output_csv=str(output_csv))
    (output_csv.parent / "synthetic_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: list[dict[str, str]]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    for key in ("source_type", "synthetic_type", "reason"):
        if key not in ordered:
            ordered.append(key)
    return ordered


def _candidate_reason(row: dict[str, str], synthetic_type: str) -> str:
    clip_id = row.get("clip_id") or Path(row.get("clip_path", "unknown")).stem
    domain = row.get("domain") or row.get("environment") or "unknown-domain"
    return f"weak-condition:{synthetic_type};clip:{clip_id};domain:{domain}"

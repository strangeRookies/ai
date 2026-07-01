from __future__ import annotations

from datetime import UTC, datetime
from math import floor
from pathlib import Path

from ai.learning.manifest_csv import read_csv_rows, summarize, write_csv_rows, write_summary
from ai.learning.manifest_leakage import check_manifest_rows
from ai.learning.manifest_helpers import (
    augmentation_config,
    augmentation_type,
    clip_id,
    dedupe,
    enum_reason,
    estimated_visibility,
    fieldnames,
    first,
    label,
    label_name,
    predicted_label,
    sample_rows,
    scenario_tag,
    split_group_id,
    synthetic_review_status,
)
from ai.learning.manifest_schema import (
    CANDIDATE_FIELDS,
    REQUIRED_TRAINING_MANIFEST_V2_FIELDS,
    SYNTHETIC_FIELDS,
    SYNTHETIC_TYPES,
    CandidateKind,
    ManifestSummary,
)


def build_error_candidates(
    error_csv: Path,
    output_csv: Path,
    kind: CandidateKind,
    metadata_csv: Path | None = None,
    dry_run: bool = False,
) -> ManifestSummary:
    rows = read_csv_rows(error_csv)
    metadata = _metadata_by_clip(metadata_csv)
    candidates = [_candidate_row(row, metadata, kind) for row in rows if _matches_kind(row, kind)]
    deduped = dedupe(candidates, ("evidence_id", "clip_path"))
    write_csv_rows(output_csv, deduped, CANDIDATE_FIELDS, dry_run=dry_run)
    summary = summarize(deduped, output_csv, not dry_run)
    write_summary(output_csv, summary, dry_run)
    return summary


def build_synthetic_candidates(
    input_csv: Path,
    output_csv: Path,
    synthetic_types: tuple[str, ...] = SYNTHETIC_TYPES,
    seed: int = 42,
    min_visibility_for_approved: float = 0.6,
    dry_run: bool = False,
) -> ManifestSummary:
    rows = read_csv_rows(input_csv)
    candidates: list[dict[str, str]] = []
    for row in rows:
        parent_clip_id = clip_id(row)
        parent_clip_path = row.get("clip_path", "").strip()
        parent_split = row.get("split", row.get("parent_split", "train")).strip() or "train"
        if not parent_clip_path:
            raise RuntimeError(f"Missing clip_path for synthetic parent clip_id={parent_clip_id}")
        for raw_type in synthetic_types:
            normalized_type = augmentation_type(raw_type)
            synthetic_clip_id = f"{parent_clip_id}__{normalized_type}"
            visibility = estimated_visibility(normalized_type)
            candidates.append(
                _synthetic_row(
                    row,
                    synthetic_clip_id,
                    parent_clip_id,
                    parent_clip_path,
                    parent_split,
                    normalized_type,
                    seed,
                    visibility,
                    min_visibility_for_approved,
                )
            )
    deduped = dedupe(candidates, ("clip_id", "clip_path"))
    write_csv_rows(output_csv, deduped, SYNTHETIC_FIELDS, dry_run=dry_run)
    summary = summarize(deduped, output_csv, not dry_run)
    write_summary(output_csv, summary, dry_run)
    return summary


def build_training_manifest(
    base_metadata_csv: Path,
    output_csv: Path,
    candidate_csvs: list[Path],
    max_synthetic_ratio: float = 0.3,
    sample: int | None = None,
    dry_run: bool = False,
) -> ManifestSummary:
    if not 0 <= max_synthetic_ratio < 1:
        raise RuntimeError("--max-synthetic-ratio must be >= 0 and < 1")
    if base_metadata_csv.resolve() == output_csv.resolve():
        raise RuntimeError("Refusing to overwrite base metadata.csv; choose a separate training_manifest_v2.csv output path")
    base_rows = read_csv_rows(base_metadata_csv)
    rows = [_normalize_manifest_row(row, "real") for row in sample_rows(base_rows, sample)]
    for path in candidate_csvs:
        if path.exists():
            source_rows = [
                _normalize_manifest_row(row, row.get("source_type", "candidate"))
                for row in read_csv_rows(path)
                if _candidate_is_training_ready(row)
            ]
            rows.extend(sample_rows(source_rows, sample))
    deduped = _cap_synthetic_train_rows(dedupe(rows, ("clip_id", "clip_path")), max_synthetic_ratio)
    check_manifest_rows(deduped, max_synthetic_ratio=max_synthetic_ratio)
    write_csv_rows(output_csv, deduped, fieldnames(deduped), dry_run=dry_run)
    summary = summarize(deduped, output_csv, not dry_run)
    write_summary(output_csv, summary, dry_run)
    return summary


def _candidate_row(row: dict[str, str], metadata: dict[str, dict[str, str]], kind: CandidateKind) -> dict[str, str]:
    meta = metadata.get(row.get("clip_id", "").strip(), {})
    label, label_name = ("0", "Normal") if kind == "hard_negative" else ("1", "Faint")
    failure_type = "false_positive" if kind == "hard_negative" else "false_negative"
    scenario = scenario_tag(row, kind)
    return _with_required_fields(
        {
            "clip_id": first(row, meta, "clip_id"),
            "clip_path": first(row, meta, "clip_path", "video_path"),
            "label": label,
            "label_name": label_name,
            "source_type": kind,
            "review_status": row.get("review_status", "").strip() or "pending",
            "failure_type": failure_type,
            "scenario_tag": scenario,
            "reason": scenario,
            "evidence_id": row.get("evidence_id", "").strip(),
            "camera_login_id": row.get("camera_login_id", row.get("cameraLoginId", "")).strip(),
            "frame_id": row.get("frame_id", row.get("frameId", "")).strip(),
            "captured_at_ms": row.get("captured_at_ms", row.get("capturedAtMs", "")).strip(),
            "predicted_label": predicted_label(row),
            "faint_prob": row.get("faint_prob", row.get("faint_probability", "")).strip(),
            "source_video": first(row, meta, "source_video"),
            "split": first(row, meta, "split"),
            "split_group_id": split_group_id(meta | row, first(row, meta, "clip_id")),
        }
    )


def _synthetic_row(
    row: dict[str, str],
    synthetic_clip_id: str,
    parent_clip_id: str,
    parent_clip_path: str,
    parent_split: str,
    augmentation_type: str,
    seed: int,
    visibility: float,
    min_visibility_for_approved: float,
) -> dict[str, str]:
    scenario = enum_reason(row.get("scenario_tag", row.get("reason", "")).strip() or "weak_condition_augmentation_candidate")
    return _with_required_fields(
        {
            "clip_id": synthetic_clip_id,
            "clip_path": f"data/synthetic/{augmentation_type}/{synthetic_clip_id}.mp4",
            "label": label(row),
            "label_name": label_name(row),
            "source_type": "synthetic",
            "parent_clip_id": parent_clip_id,
            "review_status": synthetic_review_status(visibility, min_visibility_for_approved),
            "failure_type": row.get("failure_type", "").strip(),
            "scenario_tag": scenario,
            "augmentation_type": augmentation_type,
            "augmentation_config": augmentation_config(augmentation_type, seed),
            "random_seed": str(seed),
            "split_group_id": split_group_id(row, parent_clip_id),
            "synthetic_type": augmentation_type,
            "parent_clip_path": parent_clip_path,
            "parent_split": parent_split,
            "split": parent_split,
            "reason": f"{scenario}; augmentation_type={augmentation_type}",
            "evidence_id": row.get("evidence_id", "").strip(),
            "augmentation_seed": str(seed),
            "estimated_visibility": str(visibility),
        }
    )


def _metadata_by_clip(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    return {row.get("clip_id", "").strip(): row for row in read_csv_rows(path) if row.get("clip_id", "").strip()}


def _matches_kind(row: dict[str, str], kind: CandidateKind) -> bool:
    error_type = row.get("error_type", "").strip().lower()
    if kind == "hard_negative":
        return error_type in {"fp", "false_positive", "hard_negative"} or (label_name(row) == "Normal" and predicted_label(row) == "Faint")
    return error_type in {"fn", "false_negative", "faint_reinforcement"} or (label_name(row) == "Faint" and predicted_label(row) == "Normal")


def _normalize_manifest_row(row: dict[str, str], fallback_source_type: str) -> dict[str, str]:
    item = _with_required_fields(dict(row))
    item["clip_id"] = clip_id(row)
    item["clip_path"] = row.get("clip_path", row.get("video_path", "")).strip()
    item["label"] = label(row)
    item["label_name"] = label_name(row)
    item["source_type"] = row.get("source_type", "").strip() or fallback_source_type
    item["review_status"] = row.get("review_status", "").strip() or "approved"
    item["scenario_tag"] = row.get("scenario_tag", row.get("reason", "")).strip()
    item["augmentation_type"] = augmentation_type(row.get("augmentation_type", row.get("synthetic_type", "")).strip())
    item["random_seed"] = row.get("random_seed", row.get("augmentation_seed", "")).strip()
    item["split_group_id"] = split_group_id(row, item["clip_id"])
    if item["source_type"] == "synthetic":
        item["parent_clip_id"] = row.get("parent_clip_id", "").strip()
        item["split"] = row.get("parent_split", row.get("split", "train")).strip() or "train"
        item["parent_split"] = item["split"]
    return item


def _with_required_fields(row: dict[str, str]) -> dict[str, str]:
    output = {field: row.get(field, "").strip() for field in REQUIRED_TRAINING_MANIFEST_V2_FIELDS}
    output.update(row)
    output["created_at"] = output.get("created_at", "").strip() or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return output


def _candidate_is_training_ready(row: dict[str, str]) -> bool:
    source_type = row.get("source_type", "").strip()
    if source_type in {"hard_negative", "faint_reinforcement", "synthetic"} and row.get("review_status", "").strip() != "approved":
        return False
    if source_type == "synthetic" and not row.get("parent_clip_id", "").strip():
        return False
    if source_type == "synthetic" and row.get("parent_split", row.get("split", "")).strip() == "test":
        return False
    return True


def _cap_synthetic_train_rows(rows: list[dict[str, str]], max_synthetic_ratio: float) -> list[dict[str, str]]:
    if max_synthetic_ratio <= 0:
        return [row for row in rows if not _is_train_synthetic(row)]
    train_non_synthetic = [row for row in rows if row.get("split", "train") == "train" and row.get("source_type") != "synthetic"]
    train_synthetic = [row for row in rows if _is_train_synthetic(row)]
    max_synthetic = floor((len(train_non_synthetic) * max_synthetic_ratio) / (1 - max_synthetic_ratio))
    allowed = {id(row) for row in train_synthetic[:max_synthetic]}
    return [row for row in rows if not _is_train_synthetic(row) or id(row) in allowed]


def _is_train_synthetic(row: dict[str, str]) -> bool:
    return row.get("source_type") == "synthetic" and row.get("split", "train") == "train"

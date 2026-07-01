#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.learning.candidate_manifests import SYNTHETIC_TYPES, build_synthetic_candidates
from ai.learning.manifest_csv import read_csv_rows, write_csv_rows
from ai.learning.manifest_samples import sample_faint_candidate_rows, write_rows

SUPPORTED_AUGMENTATIONS = set(SYNTHETIC_TYPES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create lightweight synthetic augmentation candidate metadata and optional OpenCV media.")
    parser.add_argument("--input-csv")
    parser.add_argument("--output-csv", default="data/manifests/synthetic_candidates.csv")
    parser.add_argument("--synthetic-types", default=",".join(SYNTHETIC_TYPES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visibility-for-approved", type=float, default=0.6)
    parser.add_argument("--sample", nargs="?", const=100, type=int, help="Use built-in approved sample candidate rows, optionally capped to N rows.")
    parser.add_argument("--generate-media", action="store_true")
    parser.add_argument("--media-output-dir", default="data/synthetic")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    synthetic_types = tuple(item.strip() for item in args.synthetic_types.split(",") if item.strip())
    invalid_types = sorted(set(synthetic_types) - SUPPORTED_AUGMENTATIONS)
    if invalid_types:
        raise RuntimeError(f"Unsupported synthetic types: {', '.join(invalid_types)}")
    with tempfile.TemporaryDirectory() as tmp:
        input_csv = Path(args.input_csv) if args.input_csv else None
        if args.sample is not None:
            input_csv = Path(tmp) / "sample_faint_candidates.csv"
            write_rows(input_csv, sample_faint_candidate_rows()[: args.sample])
        if input_csv is None:
            raise RuntimeError("--input-csv is required unless --sample is used")
        summary = build_synthetic_candidates(
            input_csv,
            Path(args.output_csv),
            synthetic_types=synthetic_types,
            seed=args.seed,
            min_visibility_for_approved=args.min_visibility_for_approved,
            dry_run=args.dry_run,
        )
        if args.generate_media and not args.dry_run:
            rows = _generate_media(Path(args.output_csv), Path(args.media_output_dir), args.seed)
            write_csv_rows(Path(args.output_csv), rows, list(rows[0].keys()) if rows else [], dry_run=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _generate_media(manifest_csv: Path, media_output_dir: Path, seed: int) -> list[dict[str, str]]:
    from ai.learning.video_augmentation import generate_augmented_video

    rows = read_csv_rows(manifest_csv)
    media_root = media_output_dir.resolve()
    for row in rows:
        parent_path = Path(row.get("parent_clip_path", ""))
        augmentation_type = row.get("augmentation_type", "")
        output_path = (media_root / augmentation_type / f"{_safe_filename(row['clip_id'])}.mp4").resolve()
        if not output_path.is_relative_to(media_root):
            raise RuntimeError(f"Refusing to write synthetic media outside {media_root}: {output_path}")
        result = generate_augmented_video(parent_path, output_path, augmentation_type, random_seed=seed)
        row["clip_path"] = str(result.output_path)
        row["augmentation_config"] = result.augmentation_config
    return rows


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "synthetic_clip"


if __name__ == "__main__":
    main()

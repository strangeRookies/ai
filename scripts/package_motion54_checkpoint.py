#!/usr/bin/env python3
"""Package a keypoint_motion54 LSTM checkpoint with schema metadata (weights unchanged).

Does not overwrite the input file. Does not train or mutate tensor values.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.action.feature_schema import (  # noqa: E402
    KEYPOINT_BBOX54_SCHEMA_VERSION,
    KEYPOINT_MOTION54_INPUT_SIZE,
    KEYPOINT_MOTION54_SCHEMA_VERSION,
    keypoint_motion54_feature_names,
)
from ai.action.lstm_contract import DEFAULT_LSTM_SEQUENCE_LENGTH, DEFAULT_LSTM_SEQUENCE_STRIDE  # noqa: E402


REQUIRED_CLASSES = ("Normal", "Faint")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model_state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(model_state.keys()):
        digest.update(key.encode("utf-8"))
        tensor = model_state[key]
        if hasattr(tensor, "detach"):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(array.tobytes())
            digest.update(str(tuple(array.shape)).encode("utf-8"))
        else:
            digest.update(repr(tensor).encode("utf-8"))
    return digest.hexdigest()


def model_states_equal(before: dict[str, Any], after: dict[str, Any]) -> bool:
    import torch

    if set(before.keys()) != set(after.keys()):
        return False
    for key in before:
        left = before[key]
        right = after[key]
        if hasattr(left, "detach") and hasattr(right, "detach"):
            if not torch.equal(left.detach().cpu(), right.detach().cpu()):
                return False
        elif left != right:
            return False
    return True


def resolve_sequence_length(checkpoint: dict[str, Any], explicit: int | None, input_path: Path) -> tuple[int, str]:
    if explicit is not None:
        return int(explicit), "cli"
    value = checkpoint.get("sequence_length")
    if value not in (None, ""):
        return int(value), "checkpoint"
    path_text = str(input_path).replace("\\", "/").lower()
    if "sequence30" in path_text or "lstm_sequence30" in path_text:
        return int(DEFAULT_LSTM_SEQUENCE_LENGTH), "path_name_sequence30"
    return int(DEFAULT_LSTM_SEQUENCE_LENGTH), "experiment_default_30"


def resolve_sequence_stride(checkpoint: dict[str, Any], explicit: int | None) -> tuple[int | None, str]:
    if explicit is not None:
        return int(explicit), "cli"
    value = checkpoint.get("sequence_stride")
    if value not in (None, ""):
        return int(value), "checkpoint"
    # compare_lstm_extractors / runtime defaults historically use 15 for sequence30 motion runs.
    return int(DEFAULT_LSTM_SEQUENCE_STRIDE), "INFERRED_FROM_EXPERIMENT_DEFAULT=15"


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a dict, got {type(payload).__name__}")
    return payload


def validate_motion54_source(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint missing model_config dict")
    input_size = int(model_config.get("input_size", checkpoint.get("input_size") or 0))
    if input_size != KEYPOINT_MOTION54_INPUT_SIZE:
        raise ValueError(f"model_config.input_size must be 54, got {input_size}")
    classes = list(checkpoint.get("classes") or [])
    if classes != list(REQUIRED_CLASSES):
        raise ValueError(f"classes must be {list(REQUIRED_CLASSES)}, got {classes}")
    schema = checkpoint.get("feature_schema_version")
    if schema == KEYPOINT_BBOX54_SCHEMA_VERSION:
        raise ValueError("refusing to package keypoint_bbox54 checkpoint as motion54")
    if schema not in (None, "", KEYPOINT_MOTION54_SCHEMA_VERSION):
        raise ValueError(f"unsupported existing feature_schema_version={schema!r}")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("checkpoint missing non-empty model_state")
    return {
        "input_size": input_size,
        "classes": classes,
        "feature_schema_version": schema,
        "model_state_keys": len(model_state),
    }


def package_checkpoint(
    input_path: Path,
    output_path: Path,
    *,
    sequence_length: int | None,
    sequence_stride: int | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    import torch

    original_sha = sha256_file(input_path)
    checkpoint = load_checkpoint(input_path)
    source_info = validate_motion54_source(checkpoint)
    before_state = checkpoint["model_state"]
    before_hash = model_state_sha256(before_state)

    seq_len, seq_len_source = resolve_sequence_length(checkpoint, sequence_length, input_path)
    seq_stride, seq_stride_source = resolve_sequence_stride(checkpoint, sequence_stride)

    packaged = copy.deepcopy(checkpoint)
    model_config = dict(packaged.get("model_config") or {})
    model_config["input_size"] = KEYPOINT_MOTION54_INPUT_SIZE
    packaged["model_config"] = model_config
    packaged["input_size"] = KEYPOINT_MOTION54_INPUT_SIZE
    packaged["feature_schema_version"] = KEYPOINT_MOTION54_SCHEMA_VERSION
    packaged["feature_names"] = keypoint_motion54_feature_names()
    packaged["sequence_length"] = seq_len
    packaged["sequence_stride"] = seq_stride
    packaged["feature_type"] = packaged.get("feature_type") or "keypoints"
    packaged["packaging"] = {
        "source_path": str(input_path),
        "source_sha256": original_sha,
        "schema": KEYPOINT_MOTION54_SCHEMA_VERSION,
        "sequence_length_source": seq_len_source,
        "sequence_stride_source": seq_stride_source,
        "weights_modified": False,
    }

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "dry_run": bool(dry_run),
        "original_sha256": original_sha,
        "source": source_info,
        "sequence_length": seq_len,
        "sequence_length_source": seq_len_source,
        "sequence_stride": seq_stride,
        "sequence_stride_source": seq_stride_source,
        "feature_schema_version": KEYPOINT_MOTION54_SCHEMA_VERSION,
        "feature_names_count": len(packaged["feature_names"]),
        "model_state_keys": source_info["model_state_keys"],
        "model_state_sha256_before": before_hash,
    }

    if dry_run:
        report["status"] = "DRY_RUN"
        return report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == input_path.resolve():
        raise ValueError("output path must differ from input path (refusing overwrite)")
    torch.save(packaged, str(output_path))

    reloaded = load_checkpoint(output_path)
    after_state = reloaded["model_state"]
    after_hash = model_state_sha256(after_state)
    identical = model_states_equal(before_state, after_state) and before_hash == after_hash
    if not identical:
        raise RuntimeError("model_state changed during packaging")

    report.update(
        {
            "status": "PASS",
            "packaged_sha256": sha256_file(output_path),
            "model_state_sha256_after": after_hash,
            "model_state_identical": True,
            "reloaded_feature_schema_version": reloaded.get("feature_schema_version"),
            "reloaded_input_size": int((reloaded.get("model_config") or {}).get("input_size") or 0),
            "reloaded_sequence_length": reloaded.get("sequence_length"),
            "reloaded_sequence_stride": reloaded.get("sequence_stride"),
            "reloaded_feature_names_count": len(reloaded.get("feature_names") or []),
        }
    )
    # Ensure original bytes untouched
    if sha256_file(input_path) != original_sha:
        raise RuntimeError("original checkpoint was modified")
    report["original_unchanged"] = True
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package keypoint_motion54 checkpoint metadata without changing weights")
    parser.add_argument("--input", required=True, help="Path to original best.pt (never overwritten)")
    parser.add_argument("--output", required=True, help="Path to write packaged checkpoint")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--sequence-stride", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        print(f"[package-motion54] FAIL input not found: {input_path}", flush=True)
        return 2
    try:
        report = package_checkpoint(
            input_path,
            output_path,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"[package-motion54] FAIL {exc}", flush=True)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(
        "[package-motion54] "
        f"status={report.get('status')} "
        f"schema={report.get('feature_schema_version')} "
        f"sequence_length={report.get('sequence_length')}({report.get('sequence_length_source')}) "
        f"sequence_stride={report.get('sequence_stride')}({report.get('sequence_stride_source')}) "
        f"model_state_identical={report.get('model_state_identical', True)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

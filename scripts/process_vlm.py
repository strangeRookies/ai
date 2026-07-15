from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.vlm.contracts import (  # noqa: E402
    VlmContractError,
    validate_keyframes,
    validate_vlm_result,
)
from ai.vlm.deidentification_contracts import (  # noqa: E402
    DeidentificationFrameReport,
    DeidentificationOutcome,
    DeidentifyFrames,
)
from ai.vlm.keyframe_extractor import (  # noqa: E402
    KeyframeExtractionError,
    ExtractedKeyframe,
    extract_eight_keyframes,
    local_video_source,
)
from ai.vlm_sdk import (  # noqa: E402
    VlmAnalyzeRequest,
    VlmAnalyzeResult,
    VlmFramePayload,
    resolve_vlm_provider,
)

MetadataJson = NewType("MetadataJson", str)
_REDACTED = "[REDACTED]"
_SENSITIVE_METADATA_KEYS = {
    "access_key",
    "api_key",
    "authorization",
    "client_secret",
    "cookie",
    "credential",
    "input_url",
    "output_url",
    "output_urls",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "session_key",
    "source_url",
    "token",
}


@dataclass(frozen=True, slots=True)
class ProcessVlmArgs:
    input_url: str
    output_urls: tuple[str, ...]
    metadata: MetadataJson
    mock_mode: bool


VlmResult = VlmAnalyzeResult


class VlmProcessError(RuntimeError):
    pass


def parse_args(argv: list[str]) -> ProcessVlmArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Process a clip or snapshot with VLM-RAG contract output "
            "(direct SDK, no LangChain)."
        )
    )
    parser.add_argument("--input-url", required=True)
    parser.add_argument("--output-urls", required=True)
    parser.add_argument("--metadata", required=True)
    parsed = parser.parse_args(argv)
    output_urls = tuple(
        url.strip() for url in parsed.output_urls.split(",") if url.strip()
    )
    return ProcessVlmArgs(
        input_url=parsed.input_url,
        output_urls=output_urls,
        metadata=MetadataJson(parsed.metadata),
        mock_mode=os.getenv("VLM_MOCK_MODE", "true").lower() == "true",
    )


def process(
    args: ProcessVlmArgs,
    *,
    deidentify_frames: DeidentifyFrames | None = None,
) -> VlmResult:
    metadata = sanitize_metadata(parse_metadata(args.metadata))
    start_sec, end_sec = validate_clip_metadata(metadata)
    with local_video_source(args.input_url) as video_path:
        frames = extract_eight_keyframes(
            video_path,
            start_sec=start_sec,
            end_sec=end_sec,
        )
    validate_keyframes(frames)
    provider = resolve_vlm_provider()
    if getattr(provider, "requires_deidentified_frames", False):
        frames = _deidentify_for_provider(
            frames,
            deidentify_frames=deidentify_frames,
        )

    provider_frames = tuple(
        VlmFramePayload(
            index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            jpeg_bytes=frame.jpeg_bytes,
        )
        for frame in frames
    )
    analyzed = provider.analyze(
        VlmAnalyzeRequest(
            frames=provider_frames,
            metadata=metadata,
        )
    )
    validate_vlm_result(analyzed.to_dict())
    return analyzed


def _deidentify_for_provider(
    frames: tuple[ExtractedKeyframe, ...],
    *,
    deidentify_frames: DeidentifyFrames | None,
) -> tuple[ExtractedKeyframe, ...]:
    if deidentify_frames is None:
        raise VlmProcessError(
            "Gemini processing is blocked: no keyframe de-identification API is configured"
        )
    try:
        outcome = deidentify_frames(frames)
    except Exception as exc:
        raise VlmProcessError("keyframe de-identification failed") from exc
    if not isinstance(outcome, DeidentificationOutcome):
        raise VlmProcessError("de-identification returned an invalid outcome")
    deidentified = outcome.frames
    reports = outcome.reports
    if not isinstance(deidentified, tuple) or len(deidentified) != len(frames):
        raise VlmProcessError("de-identification returned an invalid frame batch")
    if not isinstance(reports, tuple) or len(reports) != len(frames):
        raise VlmProcessError("de-identification must return exactly eight reports")

    for original, processed in zip(frames, deidentified, strict=True):
        if not isinstance(processed, ExtractedKeyframe):
            raise VlmProcessError("de-identification returned an invalid frame")
        if (
            processed.index != original.index
            or processed.timestamp_sec != original.timestamp_sec
            or processed.frame_index != original.frame_index
        ):
            raise VlmProcessError("de-identification changed frame ordering")

    try:
        validate_keyframes(deidentified)
    except VlmContractError as exc:
        raise VlmProcessError("de-identification returned invalid keyframes") from exc

    for expected_index, report in enumerate(reports):
        if not isinstance(report, DeidentificationFrameReport):
            raise VlmProcessError("de-identification returned an invalid report")
        if report.index != expected_index:
            raise VlmProcessError("de-identification reports must be ordered")
        counts = (report.detected_person_count, report.deidentified_person_count)
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts
        ):
            raise VlmProcessError(
                "de-identification report counts must be nonnegative integers"
            )
        if report.status != "PASS":
            raise VlmProcessError("de-identification report status must be PASS")
        if report.deidentified_person_count != report.detected_person_count:
            raise VlmProcessError("de-identification report counts do not match")
    return deidentified


def parse_metadata(raw: MetadataJson) -> dict[str, object]:
    def reject_nonfinite(_value: str) -> None:
        raise VlmProcessError("metadata contains a non-finite number")

    try:
        value = json.loads(
            raw,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise VlmProcessError(f"invalid metadata JSON: {exc.msg}") from exc
    except ValueError as exc:
        raise VlmProcessError("invalid metadata JSON") from exc
    if not isinstance(value, dict):
        raise VlmProcessError("metadata must be a JSON object")
    return value


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def validate_clip_metadata(metadata: dict[str, object]) -> tuple[float, float]:
    for field_name in ("incident_id", "camera_login_id"):
        value = metadata.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise VlmProcessError(
                f"metadata.{field_name} must be a non-empty string"
            )

    start = metadata.get("clip_start_sec")
    end = metadata.get("clip_end_sec")
    if not _is_finite_number(start) or start < 0:
        raise VlmProcessError(
            "metadata.clip_start_sec must be a finite nonnegative number"
        )
    if not _is_finite_number(end) or end <= start:
        raise VlmProcessError(
            "metadata.clip_end_sec must be finite and greater than clip_start_sec"
        )
    return float(start), float(end)

def sanitize_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Copy nested incident metadata while redacting credential-bearing values."""

    def sanitize(value: object) -> object:
        if isinstance(value, dict):
            sanitized: dict[str, object] = {}
            for key, item in value.items():
                normalized = key.strip().lower().replace("-", "_")
                sensitive = (
                    normalized in _SENSITIVE_METADATA_KEYS
                    or normalized.endswith(
                        (
                            "_access_key",
                            "_api_key",
                            "_credential",
                            "_password",
                            "_private_key",
                            "_secret",
                            "_token",
                        )
                    )
                )
                sanitized[key] = _REDACTED if sensitive else sanitize(item)
            return sanitized
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    sanitized = sanitize(metadata)
    if not isinstance(sanitized, dict):  # defensive type narrowing
        raise VlmProcessError("metadata must be a JSON object")
    return sanitized


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        result = process(args)
    except (VlmProcessError, KeyframeExtractionError, VlmContractError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError):
        print("VLM processing failed", file=sys.stderr)
        return 1
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

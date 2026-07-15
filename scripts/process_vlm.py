from __future__ import annotations

import argparse
import json
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
from ai.vlm.keyframe_extractor import (  # noqa: E402
    KeyframeExtractionError,
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


def process(args: ProcessVlmArgs) -> VlmResult:
    metadata = sanitize_metadata(parse_metadata(args.metadata))
    with local_video_source(args.input_url) as video_path:
        frames = extract_eight_keyframes(video_path)
    validate_keyframes(frames)
    provider_frames = tuple(
        VlmFramePayload(
            index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            jpeg_bytes=frame.jpeg_bytes,
        )
        for frame in frames
    )

    provider = resolve_vlm_provider()
    analyzed = provider.analyze(
        VlmAnalyzeRequest(
            frames=provider_frames,
            metadata=metadata,
        )
    )
    validate_vlm_result(analyzed.to_dict())
    return analyzed


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
    if not isinstance(value, dict):
        raise VlmProcessError("metadata must be a JSON object")
    return value


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

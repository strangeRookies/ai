from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NewType
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.vlm.keyframe_extract import (  # noqa: E402
    MAX_FRAMES,
    extract_keyframes_from_bytes,
    extract_keyframes_from_path,
    parse_event_offset_sec,
    resolve_fixture_mp4,
)
from ai.vlm_sdk import VlmAnalyzeRequest, resolve_vlm_provider  # noqa: E402

MetadataJson = NewType("MetadataJson", str)


@dataclass(frozen=True, slots=True)
class ProcessVlmArgs:
    input_url: str
    output_urls: tuple[str, ...]
    metadata: MetadataJson
    mock_mode: bool


@dataclass(frozen=True, slots=True)
class VlmResult:
    visual_event_type: str
    people_count: int
    korean_search_keywords: tuple[str, ...]
    detailed_description_ko: str
    keyframe_count: int = 0
    deidentificationMode: str = "PASSTHROUGH"
    deidentified: bool = False
    safeForExternalProvider: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "visual_event_type": self.visual_event_type,
                "people_count": self.people_count,
                "korean_search_keywords": list(self.korean_search_keywords),
                "detailed_description_ko": self.detailed_description_ko,
                "keyframe_count": self.keyframe_count,
                "deidentificationMode": self.deidentificationMode,
                "deidentified": self.deidentified,
                "safeForExternalProvider": self.safeForExternalProvider,
                "uncertainty_notes": [
                    "영상만으로 신원, 얼굴 특징, 정확한 나이, 성별, 의학적 원인은 판단하지 않습니다."
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class VlmProcessError(RuntimeError):
    pass


def parse_args(argv: list[str]) -> ProcessVlmArgs:
    parser = argparse.ArgumentParser(
        description="Process a clip with real keyframe extract + VLM contract (no full-video to provider)."
    )
    parser.add_argument("--input-url", required=True)
    parser.add_argument("--output-urls", required=True)
    parser.add_argument("--metadata", required=True)
    parsed = parser.parse_args(argv)
    output_urls = tuple(url.strip() for url in parsed.output_urls.split(",") if url.strip())
    return ProcessVlmArgs(
        input_url=parsed.input_url,
        output_urls=output_urls,
        metadata=MetadataJson(parsed.metadata),
        mock_mode=os.getenv("VLM_MOCK_MODE", "true").lower() == "true",
    )


def process(args: ProcessVlmArgs) -> VlmResult:
    metadata = parse_metadata(args.metadata)
    deid_mode = (metadata.get("deidentificationMode") or os.getenv("VLM_DEID_MODE", "PASSTHROUGH")).upper()
    deidentified = deid_mode not in {"", "PASSTHROUGH", "NONE", "PASS_THROUGH"}
    safe_for_external = deidentified and metadata.get("safeForExternalProvider", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Explicit test override for synthetic fixtures with external provider scaffold
    allow_passthrough_external = os.getenv("VLM_ALLOW_PASSTHROUGH_EXTERNAL", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    provider_name = os.getenv("VLM_PROVIDER", "mock").strip().lower()
    external = (not args.mock_mode) and provider_name not in {"", "mock"}
    if external and not safe_for_external and not allow_passthrough_external:
        raise VlmProcessError(
            "external VLM provider blocked: deidentified=false / PASSTHROUGH "
            "(set VLM_ALLOW_PASSTHROUGH_EXTERNAL=true only for synthetic fixtures)"
        )

    max_frames = int(os.getenv("VLM_MAX_FRAMES", str(MAX_FRAMES)))
    event_offset = parse_event_offset_sec(metadata)

    keyframes = extract_and_upload_keyframes(
        input_url=args.input_url,
        output_urls=args.output_urls,
        metadata=metadata,
        mock_mode=args.mock_mode,
        max_frames=max_frames,
        event_offset_sec=event_offset,
    )

    # Provider receives frame metadata + optional in-memory JPEG refs, never full video bytes
    frame_meta = [
        {
            "index": kf.index,
            "timestampSec": kf.timestamp_sec,
            "frameIndex": kf.frame_index,
            "jpegSize": len(kf.jpeg_bytes),
        }
        for kf in keyframes
    ]
    provider = resolve_vlm_provider()
    analyzed = provider.analyze(
        VlmAnalyzeRequest(
            input_url=args.input_url,
            metadata={
                **metadata,
                "keyframe_count": str(len(keyframes)),
                "keyframes_json": json.dumps(frame_meta, ensure_ascii=False),
                "deidentificationMode": deid_mode if deid_mode else "PASSTHROUGH",
                "deidentified": "true" if deidentified else "false",
                "safeForExternalProvider": "true" if safe_for_external else "false",
                # Do not pass video bytes — frames only
                "media_contract": "keyframes_only",
            },
            output_urls=args.output_urls,
            keyframe_jpegs=tuple(kf.jpeg_bytes for kf in keyframes),
        )
    )
    return VlmResult(
        visual_event_type=analyzed.visual_event_type,
        people_count=analyzed.people_count,
        korean_search_keywords=analyzed.korean_search_keywords,
        detailed_description_ko=analyzed.detailed_description_ko,
        keyframe_count=len(keyframes),
        deidentificationMode="PASSTHROUGH" if not deidentified else deid_mode,
        deidentified=deidentified,
        safeForExternalProvider=safe_for_external,
    )


def extract_and_upload_keyframes(
    *,
    input_url: str,
    output_urls: tuple[str, ...],
    metadata: dict[str, str],
    mock_mode: bool,
    max_frames: int,
    event_offset_sec: float | None,
):
    if not output_urls:
        raise VlmProcessError("at least one output URL is required")

    # Prefer fixture file in mock mode (offline)
    fixture = resolve_fixture_mp4(metadata) if mock_mode else None
    if fixture is not None:
        keyframes = extract_keyframes_from_path(
            fixture,
            max_frames=min(max_frames, len(output_urls) or max_frames),
            event_offset_sec=event_offset_sec,
        )
    elif input_url.startswith("file:"):
        local = Path(input_url.removeprefix("file://").removeprefix("file:"))
        if not local.is_file():
            raise VlmProcessError(f"file input not found: {local}")
        keyframes = extract_keyframes_from_path(
            local,
            max_frames=min(max_frames, len(output_urls) or max_frames),
            event_offset_sec=event_offset_sec,
        )
    else:
        clip_bytes = download_input(input_url)
        keyframes = extract_keyframes_from_bytes(
            clip_bytes,
            max_frames=min(max_frames, len(output_urls) or max_frames),
            event_offset_sec=event_offset_sec,
        )

    if not keyframes:
        raise VlmProcessError("no keyframes extracted from clip")

    # Upload real JPEGs to put URLs (skip network when mock and URLs are dummy)
    skip_upload = mock_mode and all(
        u.startswith("http://dummy") or u.startswith("https://dummy") for u in output_urls
    )
    if not skip_upload:
        for kf, url in zip(keyframes, output_urls):
            upload_jpeg(url, kf.jpeg_bytes)
    return keyframes


def parse_metadata(raw: MetadataJson) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VlmProcessError(f"invalid metadata JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise VlmProcessError("metadata must be a JSON object")
    return {str(key): str(item) for key, item in value.items()}


def download_input(input_url: str) -> bytes:
    try:
        with urlopen(input_url, timeout=30) as response:
            return response.read()
    except (OSError, URLError) as exc:
        raise VlmProcessError(f"failed to download input media: {exc}") from exc


def upload_jpeg(output_url: str, jpeg_bytes: bytes) -> None:
    if not jpeg_bytes.startswith(b"\xff\xd8\xff"):
        raise VlmProcessError("refusing to upload non-JPEG payload")
    if len(jpeg_bytes) < 100:
        raise VlmProcessError("refusing to upload tiny/placeholder JPEG")
    request = Request(
        output_url,
        data=jpeg_bytes,
        method="PUT",
        headers={"Content-Type": "image/jpeg"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            status = getattr(response, "status", 200)
    except (OSError, URLError) as exc:
        raise VlmProcessError(f"failed to upload keyframe: {exc}") from exc
    if status >= 400:
        raise VlmProcessError(f"failed to upload keyframe: HTTP {status}")


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        result = process(args)
    except VlmProcessError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

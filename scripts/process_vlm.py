from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import NewType
from urllib.error import URLError
from urllib.request import Request, urlopen

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

    def to_json(self) -> str:
        return json.dumps(
            {
                "visual_event_type": self.visual_event_type,
                "people_count": self.people_count,
                "korean_search_keywords": list(self.korean_search_keywords),
                "detailed_description_ko": self.detailed_description_ko,
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
    parser = argparse.ArgumentParser(description="Process a clip or snapshot with VLM-RAG contract output.")
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
    if args.mock_mode:
        return mock_result(metadata)
    clip_bytes = download_input(args.input_url)
    upload_placeholder_keyframes(args.output_urls, clip_bytes)
    return mock_result(metadata)


def parse_metadata(raw: MetadataJson) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VlmProcessError(f"invalid metadata JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise VlmProcessError("metadata must be a JSON object")
    return {str(key): str(item) for key, item in value.items()}


def mock_result(metadata: dict[str, str]) -> VlmResult:
    scenario = metadata.get("scenario_type", "SAFETY_EVENT")
    description = (
        f"{scenario} 이벤트 영상에서 작업자 또는 사람이 감시 구역 바닥 근처에 있는 장면입니다. "
        "복도나 출입 구역으로 보이는 환경에서 안전모, 조끼, 바닥, 쓰러짐 여부를 검색할 수 있습니다."
    )
    return VlmResult(
        visual_event_type="person_lying_on_floor",
        people_count=1,
        korean_search_keywords=("바닥", "쓰러짐", "안전모", "조끼", "복도", scenario),
        detailed_description_ko=description,
    )


def download_input(input_url: str) -> bytes:
    try:
        with urlopen(input_url, timeout=30) as response:
            return response.read()
    except (OSError, URLError) as exc:
        raise VlmProcessError(f"failed to download input media: {exc}") from exc


def upload_placeholder_keyframes(output_urls: tuple[str, ...], clip_bytes: bytes) -> None:
    if not output_urls:
        raise VlmProcessError("at least one output URL is required")
    payload = clip_bytes[:1] or b"\xff\xd8\xff\xd9"
    for output_url in output_urls:
        request = Request(
            output_url,
            data=payload,
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

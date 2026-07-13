"""VLM provider adapters — direct SDK/HTTP only, no LangChain."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VlmAnalyzeRequest:
    input_url: str
    metadata: dict[str, str]
    output_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VlmAnalyzeResult:
    visual_event_type: str
    people_count: int
    korean_search_keywords: tuple[str, ...]
    detailed_description_ko: str

    def to_dict(self) -> dict[str, object]:
        return {
            "visual_event_type": self.visual_event_type,
            "people_count": self.people_count,
            "korean_search_keywords": list(self.korean_search_keywords),
            "detailed_description_ko": self.detailed_description_ko,
            "uncertainty_notes": [
                "영상만으로 신원, 얼굴 특징, 정확한 나이, 성별, 의학적 원인은 판단하지 않습니다."
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


class VlmProvider(Protocol):
    def analyze(self, request: VlmAnalyzeRequest) -> VlmAnalyzeResult: ...


class MockVlmProvider:
    def analyze(self, request: VlmAnalyzeRequest) -> VlmAnalyzeResult:
        scenario = request.metadata.get("scenario_type", "SAFETY_EVENT")
        description = (
            f"{scenario} 이벤트 영상에서 작업자 또는 사람이 감시 구역 바닥 근처에 있는 장면입니다. "
            "복도나 출입 구역으로 보이는 환경에서 안전모, 조끼, 바닥, 쓰러짐 여부를 검색할 수 있습니다."
        )
        return VlmAnalyzeResult(
            visual_event_type="person_lying_on_floor",
            people_count=1,
            korean_search_keywords=("바닥", "쓰러짐", "안전모", "조끼", "복도", scenario),
            detailed_description_ko=description,
        )


class GeminiVlmProvider:
    """
    Placeholder for multimodal Gemini direct SDK calls.
    Real media download + generateContent wiring is deferred until GPU/API keys are available.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def analyze(self, request: VlmAnalyzeRequest) -> VlmAnalyzeResult:
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is required for VLM_PROVIDER=gemini")
        # Intentionally not calling network in scaffold; keep contract stable for callers.
        raise RuntimeError(
            "Gemini multimodal VLM is scaffolded only. "
            "Wire generateContent with media from input_url when keys/GPU path are ready."
        )


def resolve_vlm_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
) -> VlmProvider:
    name = (provider_name or os.getenv("VLM_PROVIDER", "mock")).strip().lower()
    key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
    mock_mode = os.getenv("VLM_MOCK_MODE", "true").lower() == "true"
    if mock_mode or name in {"", "mock"}:
        return MockVlmProvider()
    if name == "gemini":
        return GeminiVlmProvider(api_key=key)
    raise ValueError(f"unsupported VLM_PROVIDER: {name}")

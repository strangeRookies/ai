"""Embedding SDK adapters — direct HTTP only, no LangChain."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Final, Protocol

EMBEDDING_DIMENSION: Final = 768


class EmbeddingProvider(Protocol):
    def model_name(self) -> str: ...

    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    model: str
    dimension: int
    embedding: list[float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "model": self.model,
                "dimension": self.dimension,
                "embedding": self.embedding,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class MockHashEmbeddingProvider:
    """Deterministic 768-d vectors for local tests (not semantic quality)."""

    def model_name(self) -> str:
        return "mock-hash-768"

    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSION
        tokens = [token for token in (text or "").lower().replace("\n", " ").split(" ") if token]
        if not tokens:
            tokens = ["empty"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, len(digest), 4):
                idx = int.from_bytes(digest[offset : offset + 4], "big") % EMBEDDING_DIMENSION
                sign = 1.0 if digest[offset] % 2 == 0 else -1.0
                vector[idx] += sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class GeminiEmbeddingProvider:
    """Gemini text-embedding-004 via REST — direct SDK-style HTTP, no LangChain."""

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        self._api_key = api_key
        self._model = model

    def model_name(self) -> str:
        return f"gemini-{self._model}"

    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def embed(self, text: str) -> list[float]:
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is required for EMBEDDING_PROVIDER=gemini")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:embedContent?key={self._api_key}"
        )
        payload = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text or ""}]},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini embed HTTP {exc.code}: {detail}") from exc
        values = body.get("embedding", {}).get("values")
        if not isinstance(values, list) or not values:
            raise RuntimeError("Gemini embed response missing embedding.values")
        return [float(value) for value in values]


def resolve_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
) -> EmbeddingProvider:
    name = (provider_name or os.getenv("EMBEDDING_PROVIDER", "mock")).strip().lower()
    key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
    if name in {"", "mock"}:
        return MockHashEmbeddingProvider()
    if name == "gemini":
        return GeminiEmbeddingProvider(api_key=key)
    raise ValueError(f"unsupported EMBEDDING_PROVIDER: {name}")


def embed_text(text: str, provider: EmbeddingProvider | None = None) -> EmbeddingResult:
    client = provider or resolve_provider()
    vector = client.embed(text)
    return EmbeddingResult(model=client.model_name(), dimension=len(vector), embedding=vector)

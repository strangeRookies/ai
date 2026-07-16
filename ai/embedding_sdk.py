"""Validated embedding providers using direct HTTP only (no LangChain)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from ai.vlm.provider_mode import resolve_embedding_provider_name, vlm_force_mock

EMBEDDING_DIMENSION: Final = 768


class EmbeddingError(RuntimeError):
    """Sanitized embedding failure safe to surface to callers."""


class EmbeddingTransportError(EmbeddingError):
    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


class EmbeddingProvider(Protocol):
    def model_name(self) -> str: ...

    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    model: str
    dimension: int
    embedding: list[float]
    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise EmbeddingError("embedding model must be a nonempty string")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise EmbeddingError("embedding dimension must be a positive integer")
        _validate_vector(self.embedding, self.dimension)

    def to_json(self) -> str:
        return json.dumps(
            {"model": self.model, "dimension": self.dimension, "embedding": self.embedding},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )


class MockHashEmbeddingProvider:
    """Deterministic 768-dimensional vectors for tests, not semantic quality."""

    def model_name(self) -> str:
        return "mock-hash-768"

    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def embed(self, text: str) -> list[float]:
        normalized = _validate_text(text)
        vector = [0.0] * EMBEDDING_DIMENSION
        for token in normalized.lower().replace("\n", " ").split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, len(digest), 4):
                index = int.from_bytes(digest[offset : offset + 4], "big") % EMBEDDING_DIMENSION
                vector[index] += 1.0 if digest[offset] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm == 0.0:
            raise EmbeddingError("embedding provider returned an invalid vector")
        return [value / norm for value in vector]


EmbeddingTransport = Callable[..., Mapping[str, Any]]


class GeminiEmbeddingProvider:
    """Gemini embedding REST provider with bounded transient-only retries."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        *,
        timeout_sec: float = 30.0,
        max_attempts: int = 3,
        transport: EmbeddingTransport | None = None,
        retry_delay_sec: float = 0.1,
    ) -> None:
        if not isinstance(timeout_sec, (int, float)) or not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("embedding timeout_sec must be finite and positive")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
            raise ValueError("embedding max_attempts must be between 1 and 5")
        self._api_key = api_key
        self._model = model
        self._timeout_sec = float(timeout_sec)
        self._max_attempts = max_attempts
        self._transport = transport or _gemini_embedding_transport
        self._retry_delay_sec = max(0.0, float(retry_delay_sec))

    def model_name(self) -> str:
        return f"gemini-{self._model}"

    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def embed(self, text: str) -> list[float]:
        normalized = _validate_text(text)
        if not isinstance(self._api_key, str) or not self._api_key:
            raise EmbeddingError("Gemini embedding credentials are not configured")
        payload = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": normalized}]},
            "outputDimensionality": self.dimension(),
        }
        for attempt in range(1, self._max_attempts + 1):
            try:
                body = self._transport(
                    api_key=self._api_key,
                    model=self._model,
                    payload=payload,
                    timeout_sec=self._timeout_sec,
                )
                return _validate_vector(_extract_embedding(body), self.dimension())
            except EmbeddingTransportError as exc:
                if not exc.transient or attempt == self._max_attempts:
                    raise
                if self._retry_delay_sec:
                    time.sleep(self._retry_delay_sec)
        raise EmbeddingError("embedding request failed")  # pragma: no cover


def _gemini_embedding_transport(
    *, api_key: str, model: str, payload: Mapping[str, object], timeout_sec: float
) -> Mapping[str, Any]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:embedContent?key={api_key}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read(2_000_001)
    except urllib.error.HTTPError as exc:
        raise EmbeddingTransportError(
            f"Gemini embedding request failed with HTTP {exc.code}",
            transient=exc.code in {408, 429} or 500 <= exc.code < 600,
        ) from None
    except (TimeoutError, socket.timeout):
        raise EmbeddingTransportError("Gemini embedding request timed out", transient=True) from None
    except urllib.error.URLError as exc:
        is_timeout = isinstance(exc.reason, (TimeoutError, socket.timeout))
        raise EmbeddingTransportError(
            "Gemini embedding request timed out" if is_timeout else "Gemini embedding network request failed",
            transient=is_timeout,
        ) from None
    if len(raw) > 2_000_000:
        raise EmbeddingTransportError("Gemini embedding response exceeded size limit", transient=False)
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EmbeddingTransportError("Gemini embedding returned malformed JSON", transient=False) from None
    if not isinstance(body, Mapping):
        raise EmbeddingTransportError("Gemini embedding response must be an object", transient=False)
    return body


def _extract_embedding(body: Mapping[str, Any]) -> list[float]:
    embedding = body.get("embedding")
    values = embedding.get("values") if isinstance(embedding, Mapping) else None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise EmbeddingTransportError("Gemini embedding response is missing vector values", transient=False)
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        raise EmbeddingTransportError("Gemini embedding response contains invalid vector values", transient=False) from None


def resolve_provider(provider_name: str | None = None, api_key: str | None = None) -> EmbeddingProvider:
    try:
        resolved = resolve_embedding_provider_name(provider_name)
    except ValueError:
        if vlm_force_mock():
            return MockHashEmbeddingProvider()
        raise
    key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
    if resolved == "gemini":
        model = os.getenv("GEMINI_EMBEDDING_MODEL", os.getenv("VLM_QUERY_EMBEDDING_MODEL", "text-embedding-004"))
        timeout = _environment_float("EMBEDDING_TIMEOUT_SEC", 30.0)
        attempts = _environment_int("EMBEDDING_MAX_ATTEMPTS", 3)
        return GeminiEmbeddingProvider(api_key=key, model=model, timeout_sec=timeout, max_attempts=attempts)
    if resolved == "mock":
        return MockHashEmbeddingProvider()
    raise ValueError(f"unsupported EMBEDDING_PROVIDER: {resolved}")


def embed_text(text: str, provider: EmbeddingProvider | None = None) -> EmbeddingResult:
    normalized = _validate_text(text)
    client = provider or resolve_provider()
    expected = client.dimension()
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise EmbeddingError("embedding provider declared an invalid dimension")
    vector = client.embed(normalized)
    validated = _validate_vector(vector, expected)
    return EmbeddingResult(model=client.model_name(), dimension=expected, embedding=validated)


def _validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingError("embedding text must be a nonempty string")
    return text.strip()


def _validate_vector(vector: object, expected_dimension: int) -> list[float]:
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence) or not vector:
        raise EmbeddingError("embedding provider returned an empty vector")
    if len(vector) != expected_dimension:
        raise EmbeddingError("embedding provider returned a vector with the wrong dimension")
    validated: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingError("embedding provider returned an invalid vector")
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingError("embedding provider returned a non-finite vector")
        validated.append(number)
    return validated


def _environment_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} must be numeric") from None


def _environment_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None

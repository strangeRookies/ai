"""Process-wide coordination, retry timing, and counters for Gemini calls."""

from __future__ import annotations

import random
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Final, Iterator, Literal

GEMINI_RETRY_BASE_DELAYS_SEC: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0)
GEMINI_RETRY_JITTER_MAX_SEC: Final = 0.25

_provider_call_semaphore = threading.BoundedSemaphore(value=1)
_metrics_lock = threading.Lock()
_rate_limit_counts: dict[str, int] = {"vlm": 0, "embedding": 0}


@contextmanager
def gemini_provider_call_slot() -> Iterator[None]:
    """Allow only one in-flight Gemini transport call in this process."""

    with _provider_call_semaphore:
        yield


def gemini_retry_delay(
    retry_index: int,
    *,
    random_value: Callable[[], float] = random.random,
) -> float:
    """Return the configured exponential delay plus bounded positive jitter."""

    if isinstance(retry_index, bool) or not isinstance(retry_index, int):
        raise TypeError("retry_index must be an integer")
    if not 0 <= retry_index < len(GEMINI_RETRY_BASE_DELAYS_SEC):
        raise ValueError("retry_index is outside the Gemini retry schedule")
    jitter_fraction = float(random_value())
    if not 0.0 <= jitter_fraction <= 1.0:
        raise ValueError("random_value must return a value between zero and one")
    return GEMINI_RETRY_BASE_DELAYS_SEC[retry_index] + jitter_fraction * GEMINI_RETRY_JITTER_MAX_SEC


def record_gemini_429(provider: Literal["vlm", "embedding"]) -> None:
    with _metrics_lock:
        _rate_limit_counts[provider] += 1


def gemini_429_metrics() -> dict[str, int]:
    """Return independent cumulative 429 counters for operational reporting."""

    with _metrics_lock:
        return {
            "gemini_vlm_429_total": _rate_limit_counts["vlm"],
            "gemini_embedding_429_total": _rate_limit_counts["embedding"],
        }


def reset_gemini_429_metrics() -> None:
    """Reset process counters. Intended for isolated tests."""

    with _metrics_lock:
        _rate_limit_counts.update(vlm=0, embedding=0)

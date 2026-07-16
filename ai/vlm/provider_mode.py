"""Shared VLM/embedding provider selection: GEMINI_API_KEY drives real mode."""

from __future__ import annotations

import os


def vlm_force_mock() -> bool:
    return os.getenv("VLM_FORCE_MOCK", "false").lower() in {"1", "true", "yes", "on"}


def gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def resolve_vlm_provider_name(explicit: str | None = None) -> str:
    if vlm_force_mock():
        return "mock"
    if gemini_api_key():
        return "gemini"
    name = (explicit or os.getenv("VLM_PROVIDER", "")).strip().lower()
    if name in {"gemini", "mock"}:
        return name
    if name:
        return name
    raise ValueError(
        "VLM provider not configured: set GEMINI_API_KEY or VLM_FORCE_MOCK=true for tests"
    )


def resolve_embedding_provider_name(explicit: str | None = None) -> str:
    if vlm_force_mock():
        return "mock"
    if gemini_api_key():
        return "gemini"
    name = (explicit or os.getenv("EMBEDDING_PROVIDER", "")).strip().lower()
    if name in {"gemini", "mock"}:
        return name
    if name:
        return name
    raise ValueError(
        "Embedding provider not configured: set GEMINI_API_KEY or VLM_FORCE_MOCK=true for tests"
    )


def snapshot_assist_should_run() -> bool:
    """Assist enabled when explicitly on or when Gemini key is present (unless force mock blocks live)."""
    flag = os.getenv("VLM_SNAPSHOT_ASSIST_ENABLED", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    return bool(gemini_api_key()) and not vlm_force_mock()
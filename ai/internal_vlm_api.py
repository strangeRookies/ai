"""Minimal stdlib HTTP server for internal VLM clip jobs (POST /internal/vlm/jobs)."""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from ai.gemini_runtime import gemini_429_metrics
from ai.vlm.provider_mode import gemini_api_key, vlm_force_mock

logger = logging.getLogger(__name__)


def _service_token() -> str:
    return (os.getenv("AI_SERVICE_TOKEN") or os.getenv("VLM_SNAPSHOT_ASSIST_SERVICE_TOKEN") or "").strip()


def _run_job(body: dict[str, Any]) -> dict[str, Any]:
    from scripts.process_vlm import MetadataJson, ProcessVlmArgs, process_index_payload  # noqa: PLC0415

    job_id = body.get("jobId")
    clip_url = body.get("clipUrl")
    if not clip_url or not isinstance(clip_url, str):
        raise ValueError("clipUrl is required")

    metadata = {
        "incident_id": body.get("incidentId") or body.get("incident_id") or f"job-{job_id}",
        "camera_login_id": body.get("cameraLoginId") or body.get("camera_login_id") or "",
        "scenario_type": body.get("scenarioType") or body.get("scenario_type") or "UNKNOWN",
        "severity": body.get("severity") or "WARNING",
        "captured_at": body.get("capturedAt") or body.get("captured_at") or "1970-01-01T00:00:00Z",
        "clip_start_sec": body.get("clipStartSec", body.get("clip_start_sec", 0)),
        "clip_end_sec": body.get("clipEndSec", body.get("clip_end_sec")),
    }
    force_mock = vlm_force_mock() or not gemini_api_key()
    if force_mock:
        from ai.embedding_sdk import MockHashEmbeddingProvider  # noqa: PLC0415
        from ai.vlm_sdk import MockVlmProvider  # noqa: PLC0415

        provider = MockVlmProvider()
        embedder = MockHashEmbeddingProvider()
    else:
        from ai.embedding_sdk import resolve_provider  # noqa: PLC0415
        from ai.vlm_sdk import resolve_vlm_provider  # noqa: PLC0415

        provider = resolve_vlm_provider()
        embedder = resolve_provider()

    args = ProcessVlmArgs(
        input_url=clip_url,
        output_urls="",
        metadata=MetadataJson(json.dumps(metadata, ensure_ascii=False)),
        output_mode="index",
        mock_mode=force_mock,
    )
    payload = process_index_payload(args, vlm_provider=provider, embedding_provider=embedder)
    return payload.to_dict()


class _VlmHandler(BaseHTTPRequestHandler):
    server_version = "StrangeInternalVlm/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        logger.info("internal-vlm %s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/internal/vlm/health":
            self._json_response(404, {"error": "not_found"})
            return
        provider = "mock" if vlm_force_mock() else ("gemini" if gemini_api_key() else "unconfigured")
        self._json_response(200, {
            "status": "UP",
            "provider": provider,
            "serviceTokenConfigured": bool(_service_token()),
            "rateLimitMetrics": gemini_429_metrics(),
        })
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/internal/vlm/jobs":
            self._json_response(404, {"error": "not_found"})
            return
        token = self.headers.get("X-Service-Token", "")
        expected = _service_token()
        if not expected or token != expected:
            self._json_response(401, {"error": "unauthorized"})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json_response(400, {"error": "invalid_json"})
            return
        try:
            result = _run_job(body)
            self._json_response(200, result)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            logger.warning("internal vlm job failed: status=%s error=%s", status_code or 500, exc)
            if status_code == 429:
                self._json_response(429, {"error": "gemini_rate_limited", "message": str(exc)[:200]})
            else:
                self._json_response(500, {"error": "job_failed", "message": str(exc)[:200]})

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve_forever(host: str = "0.0.0.0", port: int = 8091) -> None:
    httpd = ThreadingHTTPServer((host, port), _VlmHandler)
    logger.info("internal VLM API listening on %s:%s", host, port)
    httpd.serve_forever()


def start_background_server() -> threading.Thread:
    host = os.getenv("AI_INTERNAL_VLM_HOST", "0.0.0.0")
    port = int(os.getenv("AI_INTERNAL_VLM_PORT", "8091"))
    thread = threading.Thread(target=serve_forever, args=(host, port), daemon=True, name="internal-vlm-api")
    thread.start()
    return thread
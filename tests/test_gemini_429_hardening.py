from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from ai.embedding_sdk import EmbeddingTransportError, GeminiEmbeddingProvider
from ai.gemini_runtime import (
    GEMINI_RETRY_BASE_DELAYS_SEC,
    GEMINI_RETRY_JITTER_MAX_SEC,
    gemini_429_metrics,
    gemini_retry_delay,
    reset_gemini_429_metrics,
)
from ai.internal_vlm_api import _VlmHandler
from ai.vlm_sdk import GeminiTransportError, GeminiVlmProvider, VlmAnalyzeRequest, VlmFramePayload
from http.server import ThreadingHTTPServer


def _frames() -> tuple[VlmFramePayload, ...]:
    return tuple(
        VlmFramePayload(index=index, timestamp_sec=float(index + 1), jpeg_bytes=b"\xff\xd8" + bytes([index]) + b"\xff\xd9")
        for index in range(8)
    )


def _vlm_response() -> dict[str, object]:
    result = {
        "schema_version": "vlm-result-v1",
        "incident_id": "incident-1",
        "visual_event_type": "fall",
        "people_count": 1,
        "korean_search_keywords": ["쓰러짐"],
        "detailed_description_ko": "한 사람이 바닥 가까이에 있습니다.",
        "frame_count": 8,
        "provider": "gemini",
        "is_mock": False,
    }
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(result, ensure_ascii=False)}]}}]}


class Gemini429HardeningTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_gemini_429_metrics()

    def test_retry_schedule_has_exact_bases_and_bounded_jitter(self) -> None:
        self.assertEqual(
            [gemini_retry_delay(index, random_value=lambda: 0.0) for index in range(4)],
            list(GEMINI_RETRY_BASE_DELAYS_SEC),
        )
        self.assertEqual(
            [gemini_retry_delay(index, random_value=lambda: 1.0) for index in range(4)],
            [base + GEMINI_RETRY_JITTER_MAX_SEC for base in GEMINI_RETRY_BASE_DELAYS_SEC],
        )

        sleeps: list[float] = []
        attempts = 0

        def always_limited(**_kwargs: object) -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            raise GeminiTransportError("limited", transient=True, status_code=429)

        with self.assertRaises(GeminiTransportError):
            GeminiVlmProvider(
                "secret",
                transport=always_limited,
                sleep=sleeps.append,
                random_value=lambda: 1.0,
            ).analyze(VlmAnalyzeRequest(_frames(), {"incident_id": "incident-1"}))
        self.assertEqual(attempts, 5)
        self.assertEqual(sleeps, [1.25, 2.25, 4.25, 8.25])

    def test_vlm_and_embedding_share_one_provider_call_slot(self) -> None:
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def enter_call(result: dict[str, object]) -> dict[str, object]:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.04)
            with lock:
                active -= 1
            return result

        vlm = GeminiVlmProvider("secret", transport=lambda **_: enter_call(_vlm_response()))
        embedding = GeminiEmbeddingProvider(
            "secret", transport=lambda **_: enter_call({"embedding": {"values": [0.0] * 768}})
        )
        errors: list[BaseException] = []

        def run_vlm() -> None:
            try:
                barrier.wait()
                vlm.analyze(VlmAnalyzeRequest(_frames(), {"incident_id": "incident-1"}))
            except BaseException as exc:  # pragma: no cover - assertion reports thread failures
                errors.append(exc)

        def run_embedding() -> None:
            try:
                barrier.wait()
                embedding.embed("document")
            except BaseException as exc:  # pragma: no cover - assertion reports thread failures
                errors.append(exc)

        threads = [threading.Thread(target=run_vlm), threading.Thread(target=run_embedding)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)
        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(maximum_active, 1)

    def test_429_counters_are_separate(self) -> None:
        def vlm_limited(**_kwargs: object) -> dict[str, object]:
            raise GeminiTransportError("limited", transient=True, status_code=429)

        def embedding_limited(**_kwargs: object) -> dict[str, object]:
            raise EmbeddingTransportError("limited", transient=True, status_code=429)

        with self.assertRaises(GeminiTransportError):
            GeminiVlmProvider("secret", max_attempts=1, transport=vlm_limited).analyze(
                VlmAnalyzeRequest(_frames(), {"incident_id": "incident-1"})
            )
        self.assertEqual(gemini_429_metrics(), {"gemini_vlm_429_total": 1, "gemini_embedding_429_total": 0})
        with self.assertRaises(EmbeddingTransportError):
            GeminiEmbeddingProvider("secret", max_attempts=1, transport=embedding_limited).embed("document")
        self.assertEqual(gemini_429_metrics(), {"gemini_vlm_429_total": 1, "gemini_embedding_429_total": 1})

    def test_404_logs_actual_model_and_api_path(self) -> None:
        error = GeminiTransportError("missing", transient=False, status_code=404)
        with self.assertLogs("ai.vlm_sdk", level="ERROR") as captured:
            with self.assertRaises(GeminiTransportError):
                GeminiVlmProvider("secret", model="gemini-real", transport=lambda **_: (_ for _ in ()).throw(error)).analyze(
                    VlmAnalyzeRequest(_frames(), {"incident_id": "incident-1"})
                )
        self.assertIn("model=gemini-real", captured.output[0])
        self.assertIn("api_path=/v1beta/models/gemini-real:generateContent", captured.output[0])

        embedding_error = EmbeddingTransportError("missing", transient=False, status_code=404)
        with self.assertLogs("ai.embedding_sdk", level="ERROR") as captured:
            with self.assertRaises(EmbeddingTransportError):
                GeminiEmbeddingProvider(
                    "secret",
                    model="embedding-real",
                    transport=lambda **_: (_ for _ in ()).throw(embedding_error),
                ).embed("document")
        self.assertIn("model=embedding-real", captured.output[0])
        self.assertIn("api_path=/v1beta/models/embedding-real:embedContent", captured.output[0])

    def test_internal_api_preserves_gemini_429(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _VlmHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/internal/vlm/jobs",
            data=b"{}",
            headers={"Content-Type": "application/json", "X-Service-Token": "token"},
            method="POST",
        )
        try:
            with patch.dict("os.environ", {"AI_SERVICE_TOKEN": "token"}), patch(
                "ai.internal_vlm_api._run_job",
                side_effect=GeminiTransportError("Gemini request failed with HTTP 429", transient=True, status_code=429),
            ):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=2.0)
                self.assertEqual(raised.exception.code, 429)
                payload = json.loads(raised.exception.read())
                self.assertEqual(payload["error"], "gemini_rate_limited")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()

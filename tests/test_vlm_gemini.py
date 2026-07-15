from __future__ import annotations

import hashlib
import json
import unittest
import urllib.error
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from ai.vlm.keyframe_extractor import ExtractedKeyframe
from ai.vlm_sdk import (
    GeminiTransportError,
    GeminiVlmProvider,
    VlmAnalyzeRequest,
    VlmFramePayload,
)
from scripts.process_vlm import MetadataJson, ProcessVlmArgs, VlmProcessError, process


def _provider_frames(count: int = 8) -> tuple[VlmFramePayload, ...]:
    return tuple(
        VlmFramePayload(
            index=index,
            timestamp_sec=float(index + 1),
            jpeg_bytes=b"\xff\xd8" + bytes([index]) + b"\xff\xd9",
        )
        for index in range(count)
    )


def _extracted_frames() -> tuple[ExtractedKeyframe, ...]:
    frames = []
    for index, frame in enumerate(_provider_frames()):
        frames.append(
            ExtractedKeyframe(
                index=index,
                timestamp_sec=frame.timestamp_sec,
                frame_index=index + 1,
                width=1,
                height=1,
                sha256=hashlib.sha256(frame.jpeg_bytes).hexdigest(),
                jpeg_bytes=frame.jpeg_bytes,
            )
        )
    return tuple(frames)


def _gemini_response(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": "vlm-result-v1",
        "incident_id": "inc-1",
        "visual_event_type": "person_lying_on_floor",
        "people_count": 1,
        "korean_search_keywords": ["바닥", "쓰러짐"],
        "detailed_description_ko": "시간순 장면에서 한 사람이 바닥 가까이에 있는 모습이 관찰됩니다.",
        "frame_count": 8,
        "provider": "gemini",
        "is_mock": False,
    }
    result.update(overrides)
    return {
        "candidates": [
            {"content": {"parts": [{"text": json.dumps(result, ensure_ascii=False)}]}}
        ]
    }


class GeminiVlmProviderTest(unittest.TestCase):
    def test_multimodal_request_contains_eight_jpegs_and_safe_metadata(self) -> None:
        calls: list[dict[str, object]] = []

        def transport(**kwargs):
            calls.append(kwargs)
            return _gemini_response()

        provider = GeminiVlmProvider(
            "top-secret-key",
            model="gemini-test",
            timeout_sec=4.5,
            transport=transport,
            sleep=lambda _: None,
        )
        result = provider.analyze(
            VlmAnalyzeRequest(
                frames=_provider_frames(),
                metadata={
                    "incident_id": "inc-1",
                    "camera_login_id": "cam-01",
                    "api_key": "metadata-secret",
                    "subject_name": "do-not-send",
                },
            )
        )

        self.assertEqual(result.provider, "gemini")
        self.assertFalse(result.is_mock)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "gemini-test")
        self.assertEqual(calls[0]["timeout_sec"], 4.5)
        payload = calls[0]["payload"]
        parts = payload["contents"][0]["parts"]
        self.assertEqual(len(parts), 9)
        self.assertTrue(all(part["inlineData"]["mimeType"] == "image/jpeg" for part in parts[1:]))
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertIn("안전사고", serialized)
        self.assertNotIn("metadata-secret", serialized)
        self.assertNotIn("do-not-send", serialized)
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")

    def test_exactly_eight_is_enforced_before_transport(self) -> None:
        calls = 0

        def transport(**_kwargs):
            nonlocal calls
            calls += 1
            return _gemini_response()

        provider = GeminiVlmProvider("secret", transport=transport)
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            provider.analyze(
                VlmAnalyzeRequest(
                    frames=_provider_frames(7),
                    metadata={"incident_id": "inc-1"},
                )
            )
        self.assertEqual(calls, 0)

    def test_transient_failures_retry_but_malformed_response_does_not(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        def transient_then_success(**_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise GeminiTransportError("Gemini request timed out", transient=True)
            return _gemini_response()

        provider = GeminiVlmProvider(
            "secret",
            max_attempts=3,
            transport=transient_then_success,
            sleep=sleeps.append,
        )
        provider.analyze(
            VlmAnalyzeRequest(_provider_frames(), {"incident_id": "inc-1"})
        )
        self.assertEqual(attempts, 3)
        self.assertEqual(sleeps, [0.25, 0.5])

        malformed_calls = 0

        def malformed(**_kwargs):
            nonlocal malformed_calls
            malformed_calls += 1
            return {"candidates": [{"content": {"parts": [{"text": "not-json"}]}}]}

        with self.assertRaisesRegex(GeminiTransportError, "malformed"):
            GeminiVlmProvider("secret", transport=malformed).analyze(
                VlmAnalyzeRequest(_provider_frames(), {"incident_id": "inc-1"})
            )
        self.assertEqual(malformed_calls, 1)
    def test_http_failure_does_not_expose_key_or_image_bytes(self) -> None:
        api_key = "top-secret-key"
        image_marker = _provider_frames()[0].jpeg_bytes
        failure = urllib.error.HTTPError(
            f"https://example.invalid/?key={api_key}",
            503,
            "upstream body contained image bytes",
            {},
            None,
        )
        with patch("urllib.request.urlopen", side_effect=failure):
            with self.assertRaises(GeminiTransportError) as raised:
                GeminiVlmProvider(api_key, max_attempts=1).analyze(
                    VlmAnalyzeRequest(_provider_frames(), {"incident_id": "inc-1"})
                )

        error_text = str(raised.exception)
        self.assertNotIn(api_key, error_text)
        self.assertNotIn(repr(image_marker), error_text)
        self.assertEqual(error_text, "Gemini request failed with HTTP 503")

    def test_schema_and_incident_mismatch_are_rejected(self) -> None:
        for response, message in (
            (_gemini_response(provider="mock", is_mock=False), "is_mock"),
            (_gemini_response(incident_id="other"), "incident_id"),
        ):
            with self.subTest(message=message):
                provider = GeminiVlmProvider("secret", transport=lambda **_: response)
                with self.assertRaisesRegex(ValueError, message):
                    provider.analyze(
                        VlmAnalyzeRequest(_provider_frames(), {"incident_id": "inc-1"})
                    )


class ProcessDeidentificationGateTest(unittest.TestCase):
    def test_deidentification_failure_makes_zero_provider_calls(self) -> None:
        class RecordingGeminiProvider:
            requires_deidentified_frames = True
            calls = 0

            def analyze(self, _request):
                self.calls += 1
                raise AssertionError("provider must not be called")

        provider = RecordingGeminiProvider()
        args = ProcessVlmArgs(
            input_url="unused",
            output_urls=(),
            metadata=MetadataJson(
                '{"incident_id":"inc-1","camera_login_id":"cam-01",'
                '"clip_start_sec":0,"clip_end_sec":9}'
            ),
            mock_mode=False,
        )
        with (
            patch("scripts.process_vlm.local_video_source", return_value=nullcontext(Path("unused"))),
            patch("scripts.process_vlm.extract_eight_keyframes", return_value=_extracted_frames()),
            patch("scripts.process_vlm.resolve_vlm_provider", return_value=provider),
            self.assertRaisesRegex(VlmProcessError, "de-identification failed"),
        ):
            process(args, deidentify_frames=lambda _frames: (_ for _ in ()).throw(RuntimeError("secret jpeg data")))
        self.assertEqual(provider.calls, 0)

    def test_missing_or_unchanged_deidentification_blocks_gemini(self) -> None:
        class RecordingGeminiProvider:
            requires_deidentified_frames = True
            calls = 0

            def analyze(self, _request):
                self.calls += 1
                raise AssertionError("provider must not be called")

        args = ProcessVlmArgs(
            "unused",
            (),
            MetadataJson('{"incident_id":"inc-1","camera_login_id":"cam-01","clip_start_sec":0,"clip_end_sec":9}'),
            False,
        )
        for deidentifier, message in ((None, "no keyframe"), (lambda frames: frames, "unchanged")):
            provider = RecordingGeminiProvider()
            with (
                self.subTest(message=message),
                patch("scripts.process_vlm.local_video_source", return_value=nullcontext(Path("unused"))),
                patch("scripts.process_vlm.extract_eight_keyframes", return_value=_extracted_frames()),
                patch("scripts.process_vlm.resolve_vlm_provider", return_value=provider),
                self.assertRaisesRegex(VlmProcessError, message),
            ):
                process(args, deidentify_frames=deidentifier)
            self.assertEqual(provider.calls, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import io
import os
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
import json
import unittest
from unittest.mock import patch

from ai.embedding_sdk import (
    EmbeddingError,
    EmbeddingTransportError,
    GeminiEmbeddingProvider,
    embed_text,
)
from ai.vlm.index_payload import (
    INDEX_PAYLOAD_SCHEMA_VERSION,
    IndexPayloadError,
    SearchPayload,
    VlmIndexPayload,
)
from ai.vlm.keyframe_extractor import ExtractedKeyframe
from ai.vlm.search_document import SearchDocumentError, build_search_document
from ai.vlm_sdk import VlmAnalyzeResult
from scripts.process_vlm import (
    MetadataJson,
    ProcessVlmArgs,
    main,
    parse_args,
    process_index_payload,
)


def _result(*, incident_id: str = "inc-1", is_mock: bool = True) -> VlmAnalyzeResult:
    return VlmAnalyzeResult(
        schema_version="vlm-result-v1",
        incident_id=incident_id,
        visual_event_type="person_lying_on_floor",
        people_count=1,
        korean_search_keywords=("복도", "쓰러짐", "안전모"),
        detailed_description_ko="복도 바닥에 사람이 누워 있는 안전 이벤트 장면입니다.",
        frame_count=8,
        provider="mock" if is_mock else "gemini",
        is_mock=is_mock,
    )


def _search(*, keywords: tuple[str, ...] = ("복도", "쓰러짐", "안전모"), model: str = "mock-hash-768") -> SearchPayload:
    return SearchPayload(
        document="visual_event_type: person_lying_on_floor",
        keywords=keywords,
        embedding_model=model,
        embedding_dimension=768,
        embedding=(0.0,) * 768,
    )


class SearchDocumentTest(unittest.TestCase):
    def test_document_is_deterministic_ordered_and_excludes_envelope_metadata(self) -> None:
        metadata = {
            "incident_id": "inc-1",
            "camera_login_id": "lobby-1",
            "captured_at": "2026-07-15T00:00:00Z",
            "event_type": "fall_detected",
            "severity": "HIGH",
            "api_key": "must-not-appear",
        }
        first = build_search_document(_result(), metadata)
        second = build_search_document(_result(), dict(reversed(list(metadata.items()))))
        self.assertEqual(first, second)
        labels = [line.split(":", 1)[0] for line in first.splitlines()]
        self.assertEqual(
            labels,
            [
                "visual_event_type",
                "detailed_description_ko",
                "people_count",
                "korean_search_keywords",
                "event_type",
                "severity",
            ],
        )
        self.assertNotIn("inc-1", first)
        self.assertNotIn("lobby-1", first)
        self.assertNotIn("must-not-appear", first)

    def test_duplicate_oversized_and_sensitive_semantics_are_rejected(self) -> None:
        duplicate = _result().to_dict()
        duplicate["korean_search_keywords"] = ["복도", "복도"]
        with self.assertRaisesRegex(ValueError, "unique"):
            build_search_document(duplicate, {})
        oversized = _result().to_dict()
        oversized["detailed_description_ko"] = "가" * 2_001
        with self.assertRaisesRegex(SearchDocumentError, "size limit"):
            build_search_document(oversized, {})
        with self.assertRaisesRegex(SearchDocumentError, "sensitive"):
            build_search_document(_result(), {"event_type": "Bearer secret-token"})


class IndexPayloadTest(unittest.TestCase):
    def test_serialization_contract_has_no_frame_or_secret_material(self) -> None:
        payload = VlmIndexPayload(
            schema_version=INDEX_PAYLOAD_SCHEMA_VERSION,
            incident_id="inc-1",
            camera_login_id="lobby-1",
            captured_at="2026-07-15T00:00:00Z",
            vlm_result=_result(),
            search=_search(),
        )
        serialized = payload.to_json()
        parsed = json.loads(serialized)
        self.assertEqual(parsed["schema_version"], "vlm-index-payload-v1")
        self.assertEqual(parsed["search"]["embedding_dimension"], 768)
        self.assertNotIn("api_key", serialized.lower())
        self.assertNotIn("jpeg", serialized.lower())
        self.assertNotIn("output_urls", serialized)

    def test_incident_mode_and_duplicate_keyword_mismatches_are_rejected(self) -> None:
        with self.assertRaisesRegex(IndexPayloadError, "incident IDs"):
            VlmIndexPayload(
                INDEX_PAYLOAD_SCHEMA_VERSION,
                "inc-other",
                "lobby-1",
                "2026-07-15T00:00:00Z",
                _result(),
                _search(),
            )
        with self.assertRaisesRegex(IndexPayloadError, "provider modes"):
            VlmIndexPayload(
                INDEX_PAYLOAD_SCHEMA_VERSION,
                "inc-1",
                "lobby-1",
                "2026-07-15T00:00:00Z",
                _result(),
                _search(model="gemini-text-embedding-004"),
            )
        with self.assertRaisesRegex(IndexPayloadError, "keywords"):
            VlmIndexPayload(
                INDEX_PAYLOAD_SCHEMA_VERSION,
                "inc-1",
                "lobby-1",
                "2026-07-15T00:00:00Z",
                _result(),
                _search(keywords=("복도", "복도", "안전모")),
            )


class EmbeddingValidationTest(unittest.TestCase):
    def test_invalid_vectors_are_rejected(self) -> None:
        class Provider:
            def __init__(self, vector: list[float]) -> None:
                self.vector = vector

            def model_name(self) -> str:
                return "mock-test"

            def dimension(self) -> int:
                return 3

            def embed(self, text: str) -> list[float]:
                return self.vector

        for vector, message in (([], "empty"), ([1.0], "wrong dimension"), ([0.0, float("nan"), 0.0], "non-finite"), ([0.0, float("inf"), 0.0], "non-finite")):
            with self.subTest(vector=vector):
                with self.assertRaisesRegex(EmbeddingError, message):
                    embed_text("document", Provider(vector))

    def test_transient_retries_and_malformed_does_not_retry(self) -> None:
        calls = 0

        def transient_then_success(**_: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise EmbeddingTransportError("timeout", transient=True)
            return {"embedding": {"values": [0.0] * 768}}

        provider = GeminiEmbeddingProvider(
            "secret", transport=transient_then_success, max_attempts=3, retry_delay_sec=0
        )
        self.assertEqual(len(provider.embed("document")), 768)
        self.assertEqual(calls, 3)

        malformed_calls = 0

        def malformed(**_: object) -> dict[str, object]:
            nonlocal malformed_calls
            malformed_calls += 1
            return {}

        with self.assertRaises(EmbeddingTransportError):
            GeminiEmbeddingProvider(
                "secret", transport=malformed, max_attempts=3, retry_delay_sec=0
            ).embed("document")
        self.assertEqual(malformed_calls, 1)


class ProcessIndexPayloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = ProcessVlmArgs(
            input_url="unused",
            output_urls=("https://upload.example/deidentified",),
            metadata=MetadataJson("{}"),
            mock_mode=True,
        )
        self.metadata = {
            "incident_id": "inc-1",
            "camera_login_id": "lobby-1",
            "captured_at": "2026-07-15T00:00:00Z",
            "event_type": "fall_detected",
            "severity": "HIGH",
        }

    def test_vlm_failure_makes_zero_embedding_calls(self) -> None:
        class RecordingEmbedding:
            calls = 0

            def model_name(self) -> str:
                return "mock-recording"

            def dimension(self) -> int:
                return 768

            def embed(self, text: str) -> list[float]:
                self.calls += 1
                return [0.0] * 768

        embedding = RecordingEmbedding()
        with patch("scripts.process_vlm._analyze_clip", side_effect=RuntimeError("VLM failed")):
            with self.assertRaisesRegex(RuntimeError, "VLM failed"):
                process_index_payload(self.args, embedding_provider=embedding)
        self.assertEqual(embedding.calls, 0)

    def test_embedding_failure_produces_no_payload(self) -> None:
        class FailingEmbedding:
            def model_name(self) -> str:
                return "mock-failing"

            def dimension(self) -> int:
                return 768

            def embed(self, text: str) -> list[float]:
                raise EmbeddingError("embedding failed")

        with patch("scripts.process_vlm._analyze_clip", return_value=(_result(), self.metadata)):
            with self.assertRaisesRegex(EmbeddingError, "embedding failed"):
                process_index_payload(self.args, embedding_provider=FailingEmbedding())


class IndexCliModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.argv = [
            "--input-url",
            "unused",
            "--output-urls",
            "https://upload.example/deidentified",
            "--metadata",
            json.dumps(
                {
                    "incident_id": "inc-1",
                    "camera_login_id": "lobby-1",
                    "clip_start_sec": 0,
                    "clip_end_sec": 8,
                    "captured_at": "2026-07-15T00:00:00Z",
                }
            ),
        ]

    def test_default_mode_preserves_exact_vlm_stdout(self) -> None:
        stdout = io.StringIO()
        with (
            patch("scripts.process_vlm.process", return_value=_result()) as process_mock,
            patch("scripts.process_vlm.process_index_payload") as index_mock,
            redirect_stdout(stdout),
        ):
            exit_code = main(self.argv)
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), _result().to_json() + "\n")
        self.assertEqual(process_mock.call_args.args[0].output_mode, "vlm")
        index_mock.assert_not_called()

    def test_index_mode_prints_versioned_envelope_and_preserves_output_urls(self) -> None:
        payload = VlmIndexPayload(
            INDEX_PAYLOAD_SCHEMA_VERSION,
            "inc-1",
            "lobby-1",
            "2026-07-15T00:00:00Z",
            _result(),
            _search(),
        )
        stdout = io.StringIO()
        with (
            patch("scripts.process_vlm.process") as process_mock,
            patch("scripts.process_vlm.process_index_payload", return_value=payload) as index_mock,
            redirect_stdout(stdout),
        ):
            exit_code = main([*self.argv, "--output-mode", "index"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["schema_version"], INDEX_PAYLOAD_SCHEMA_VERSION)
        called_args = index_mock.call_args.args[0]
        self.assertEqual(called_args.output_mode, "index")
        self.assertEqual(called_args.output_urls, ("https://upload.example/deidentified",))
        process_mock.assert_not_called()

    def test_invalid_output_mode_is_rejected_by_parser(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args([*self.argv, "--output-mode", "unknown"])
        self.assertEqual(raised.exception.code, 2)

    def test_real_index_mode_without_deidentification_fails_closed(self) -> None:
        class RecordingGemini:
            requires_deidentified_frames = True
            calls = 0

            def analyze(self, request: object) -> VlmAnalyzeResult:
                self.calls += 1
                return _result(is_mock=False)

        frames = tuple(
            ExtractedKeyframe(
                index=index,
                timestamp_sec=float(index),
                frame_index=index,
                width=1,
                height=1,
                jpeg_bytes=(jpeg := b"\xff\xd8" + bytes([index]) + b"\xff\xd9"),
                sha256=hashlib.sha256(jpeg).hexdigest(),
            )
            for index in range(8)
        )
        provider = RecordingGemini()
        stderr = io.StringIO()

        def failing_deidentify(_frames):
            raise RuntimeError("blocked")

        with (
            patch.dict(os.environ, {"VLM_MOCK_MODE": "false"}),
            patch("scripts.process_vlm.local_video_source", return_value=nullcontext(Path("unused"))),
            patch("scripts.process_vlm.extract_eight_keyframes", return_value=frames),
            patch("scripts.process_vlm.resolve_vlm_provider", return_value=provider),
            patch("scripts.process_vlm.build_default_deidentify_frames", return_value=failing_deidentify),
            patch("scripts.process_vlm.resolve_embedding_provider") as embedding_mock,
            redirect_stderr(stderr),
        ):
            exit_code = main([*self.argv, "--output-mode", "index"])
        self.assertEqual(exit_code, 1)
        self.assertIn("keyframe de-identification failed", stderr.getvalue())
        self.assertEqual(provider.calls, 0)
        embedding_mock.assert_not_called()

if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import cv2
import numpy as np

from ai.vlm.deidentification_contracts import DeidentificationOutcome
from ai.vlm.keyframe_deidentification import deidentify_keyframes
from ai.vlm.keyframe_extractor import ExtractedKeyframe
from scripts.process_vlm import MetadataJson, ProcessVlmArgs, normalize_processing_metadata, process


def _frame_with_face() -> ExtractedKeyframe:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (70, 20), (95, 55), (0, 0, 0), thickness=-1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = encoded.tobytes()
    return ExtractedKeyframe(
        index=0,
        timestamp_sec=1.0,
        frame_index=30,
        width=160,
        height=120,
        sha256=hashlib.sha256(payload).hexdigest(),
        jpeg_bytes=payload,
    )

def _eight_unique_frames(template: ExtractedKeyframe) -> tuple[ExtractedKeyframe, ...]:
    frames: list[ExtractedKeyframe] = []
    for index in range(8):
        image = np.full((template.height, template.width, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (70, 20), (95, 55), (index * 10, 0, 0), thickness=-1)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        payload = encoded.tobytes()
        frames.append(
            ExtractedKeyframe(
                index=index,
                timestamp_sec=float(index + 1),
                frame_index=index + 1,
                width=template.width,
                height=template.height,
                sha256=hashlib.sha256(payload).hexdigest(),
                jpeg_bytes=payload,
            )
        )
    return tuple(frames)


class KeyframeDeidentificationTest(unittest.TestCase):
    def test_keypoint_mask_changes_pixels_and_reports_pass(self) -> None:
        frame = _frame_with_face()
        metadata = {
            "keypoint_data": [
                {
                    "frame_index": 30,
                    "bbox_xyxy": [40.0, 10.0, 120.0, 110.0],
                    "keypoints": [
                        [80.0, 30.0, 0.95],
                        [75.0, 28.0, 0.9],
                        [85.0, 28.0, 0.9],
                        [70.0, 32.0, 0.85],
                        [90.0, 32.0, 0.85],
                    ],
                }
            ]
        }

        outcome = deidentify_keyframes((frame,), metadata)

        self.assertIsInstance(outcome, DeidentificationOutcome)
        self.assertEqual(outcome.reports[0].status, "PASS")
        self.assertEqual(outcome.reports[0].detected_person_count, 1)
        self.assertEqual(outcome.reports[0].deidentified_person_count, 1)
        self.assertNotEqual(outcome.frames[0].jpeg_bytes, frame.jpeg_bytes)

    def test_zero_person_frames_remain_unchanged(self) -> None:
        frame = _frame_with_face()
        outcome = deidentify_keyframes((frame,), {})
        self.assertEqual(outcome.frames[0].jpeg_bytes, frame.jpeg_bytes)
        self.assertEqual(outcome.reports[0].detected_person_count, 0)
        self.assertEqual(outcome.reports[0].deidentified_person_count, 0)

    def test_normalize_processing_metadata_maps_backend_fields(self) -> None:
        normalized = normalize_processing_metadata(
            {
                "alert_event_id": 42,
                "camera_login_id": " cam-01 ",
                "detected_at": "2026-07-15T00:00:00Z",
            }
        )
        self.assertEqual(normalized["incident_id"], "42")
        self.assertEqual(normalized["camera_login_id"], "cam-01")
        self.assertEqual(normalized["clip_start_sec"], 0.0)
        self.assertEqual(normalized["captured_at"], "2026-07-15T00:00:00Z")


class ProcessGeminiDeidentificationWiringTest(unittest.TestCase):
    def test_process_uses_default_deidentification_for_gemini(self) -> None:
        class RecordingGeminiProvider:
            requires_deidentified_frames = True
            calls = 0
            request = None

            def analyze(self, request):
                self.calls += 1
                self.request = request
                from ai.vlm_sdk import MockVlmProvider

                return MockVlmProvider().analyze(request)

        frame = _frame_with_face()
        provider = RecordingGeminiProvider()
        args = ProcessVlmArgs(
            "unused",
            ("https://upload.example/0",),
            MetadataJson(
                '{"incident_id":"inc-1","camera_login_id":"cam-01",'
                '"clip_start_sec":0,"clip_end_sec":2}'
            ),
            False,
        )

        with (
            patch("scripts.process_vlm.local_video_source", return_value=nullcontext("unused")),
            patch("scripts.process_vlm.extract_eight_keyframes", return_value=_eight_unique_frames(frame)),
            patch("scripts.process_vlm.resolve_vlm_provider", return_value=provider),
            patch("scripts.process_vlm._upload_deidentified_keyframes") as upload_mock,
        ):
            result = process(args)

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(provider.request.frames), 8)
        upload_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
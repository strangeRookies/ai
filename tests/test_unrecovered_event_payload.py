"""Phase C: FAINT_SUSPECTED / FALL_UNRECOVERED MQTT payload contract."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from ai.action.faint_post_processing import (
    MEMO_UNRECOVERED,
    MEMO_UNRECOVERED_LYING,
    FaintEventPostProcessor,
)
from ai.action.fall_event_state import EVENT_TYPE_FAINT_SUSPECTED
from ai.inference.rtsp_runtime import build_inference_event_payload, lifecycle_payload_kwargs
from ai.publishers.mqtt_payloads import build_confirmed_event_payload


class UnrecoveredEventPayloadTest(unittest.TestCase):
    def test_confirmed_payload_includes_lifecycle_fields(self):
        payload = build_confirmed_event_payload(
            stream_id="cam_05",
            frame_width=640,
            frame_height=360,
            prediction={"label": "Faint", "score": 0.8, "probabilities": {"Faint": 0.8}},
            sequence={"track_id": 7, "bbox": [1, 2, 30, 40]},
            boxes=[],
            timestamp_ms=1_700_000_000_000,
            event_id="evt-unrec-1",
            event_type_override=EVENT_TYPE_FAINT_SUSPECTED,
            original_event_id="evt-fall-0",
            duration_sec=12.5,
            memo_text=MEMO_UNRECOVERED_LYING,
            posture_label="lying_like",
            movement_level="still",
            lifecycle_state="POST_FALL_LYING",
            alert_kind="unrecovered",
            track_id_override=7,
        )

        self.assertEqual(payload["type"], EVENT_TYPE_FAINT_SUSPECTED)
        self.assertEqual(payload["event_type"], EVENT_TYPE_FAINT_SUSPECTED)
        self.assertEqual(payload["cameraLoginId"], "cam_05")
        self.assertEqual(payload["originalEventId"], "evt-fall-0")
        self.assertEqual(payload["durationSec"], 12.5)
        self.assertEqual(payload["postureLabel"], "lying_like")
        self.assertEqual(payload["movementLevel"], "still")
        self.assertEqual(payload["state"], "POST_FALL_LYING")
        self.assertEqual(payload["alertKind"], "unrecovered")
        self.assertEqual(payload["trackingId"], 7)
        self.assertEqual(payload["memoText"], MEMO_UNRECOVERED_LYING)
        self.assertNotEqual(payload["memoText"], "쓰러짐 의심!")
        nested = payload["events"][0]
        self.assertEqual(nested["type"], EVENT_TYPE_FAINT_SUSPECTED)
        self.assertEqual(nested["originalEventId"], "evt-fall-0")
        self.assertEqual(nested["durationSec"], 12.5)
        self.assertEqual(nested["postureLabel"], "lying_like")
        self.assertEqual(nested["state"], "POST_FALL_LYING")

    def test_evaluate_to_inference_payload_roundtrip(self):
        processor = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=5,
            require_upright_to_lying=False,
            use_posture_estimator=False,
            unrecovered_after_seconds=5.0,
            unrecovered_repeat_seconds=30.0,
            recover_consecutive=1,
        )
        processor.evaluate("cam_01", {"label": "Faint"}, 1.0, track_id=3)
        first = processor.evaluate("cam_01", {"label": "Faint"}, 2.0, track_id=3)
        self.assertTrue(first.is_new_fall)
        original = first.event_id

        mid = processor.evaluate("cam_01", {"label": "Faint"}, 4.0, track_id=3)
        self.assertFalse(mid.emit)

        unrec = processor.evaluate(
            "cam_01",
            {"label": "Faint"},
            8.0,
            track_id=3,
            posture_label="lying_like",
        )
        # without posture estimator movement is unknown → FALL_UNRECOVERED per policy
        self.assertTrue(unrec.is_unrecovered)
        self.assertIn(unrec.event_type, {EVENT_TYPE_FAINT_SUSPECTED, "FALL_UNRECOVERED"})
        self.assertEqual(unrec.original_event_id, original)
        self.assertEqual(unrec.state, "POST_FALL_LYING")
        self.assertIn(unrec.memo_text, {MEMO_UNRECOVERED, MEMO_UNRECOVERED_LYING})

        args = SimpleNamespace(camera_id="cam_01", camera_login_id="cam_01", sequence_length=30, sequence_stride=15)
        packet = SimpleNamespace(frame=None, frame_id=10, captured_at_ms=100)
        payload = build_inference_event_payload(
            args,
            packet,
            {"label": "Faint", "score": 0.9, "probabilities": {"Faint": 0.9}},
            boxes=[],
            sequence={"track_id": 3, "bbox": [0, 0, 10, 20]},
            published_at_ms=200,
            **lifecycle_payload_kwargs(unrec),
        )
        self.assertIn(payload["type"], {EVENT_TYPE_FAINT_SUSPECTED, "FALL_UNRECOVERED"})
        self.assertEqual(payload["originalEventId"], original)
        self.assertEqual(payload["alertKind"], "unrecovered")
        self.assertEqual(payload["cameraLoginId"], "cam_01")
        self.assertEqual(payload["trackingId"], 3)


if __name__ == "__main__":
    unittest.main()

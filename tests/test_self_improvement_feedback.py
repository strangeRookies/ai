import csv
import json
import tempfile
import unittest
from pathlib import Path

from ai.learning.feedback import (
    FeedbackOutcome,
    build_feedback_record,
    candidate_type_for_outcome,
)
from ai.learning.feedback_store import append_feedback_jsonl, load_feedback_jsonl
from ai.learning.retraining_candidates import export_retraining_candidates


class SelfImprovementFeedbackTest(unittest.TestCase):
    def test_feedback_record_preserves_event_metadata_for_false_positive(self):
        event_payload = {
            "eventId": "evt-1",
            "cameraLoginId": "lobby_01",
            "frameId": 42,
            "timestampMs": 1782180000123,
            "confidence": 0.92,
            "bbox": [10, 20, 100, 200],
            "events": [{"keypoints": [{"x": 1.0, "y": 2.0, "confidence": 0.8}]}],
            "aiLatencyMs": 25,
            "publishLatencyMs": 31,
        }

        record = build_feedback_record(
            event_payload,
            FeedbackOutcome.FALSE_POSITIVE,
            operator_id="operator-a",
            feedback_text="worker sat down near wall",
            feedback_timestamp_ms=1782180000999,
        )

        self.assertEqual(record["event_id"], "evt-1")
        self.assertEqual(record["camera_login_id"], "lobby_01")
        self.assertEqual(record["frame_id"], 42)
        self.assertEqual(record["bbox"], [10, 20, 100, 200])
        self.assertEqual(record["keypoints"], [{"x": 1.0, "y": 2.0, "confidence": 0.8}])
        self.assertEqual(record["confidence"], 0.92)
        self.assertEqual(record["ai_latency_ms"], 25)
        self.assertEqual(record["publish_latency_ms"], 31)
        self.assertEqual(record["feedback_outcome"], "false_positive")
        self.assertEqual(record["candidate_type"], "hard_negative")

    def test_candidate_mapping_separates_fp_fn_tp_tn(self):
        self.assertEqual(candidate_type_for_outcome(FeedbackOutcome.FALSE_POSITIVE), "hard_negative")
        self.assertEqual(candidate_type_for_outcome(FeedbackOutcome.FALSE_NEGATIVE), "faint_fall_reinforcement")
        self.assertEqual(candidate_type_for_outcome(FeedbackOutcome.TRUE_POSITIVE), "verified_positive")
        self.assertEqual(candidate_type_for_outcome(FeedbackOutcome.TRUE_NEGATIVE), "verified_negative")

    def test_feedback_jsonl_and_candidate_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feedback_path = root / "feedback.jsonl"
            fp = build_feedback_record(
                {"eventId": "evt-fp", "camera_login_id": "cam_a", "timestampMs": 1},
                FeedbackOutcome.FALSE_POSITIVE,
                operator_id="op",
                feedback_text="FP",
                feedback_timestamp_ms=2,
            )
            fn = build_feedback_record(
                {"eventId": "evt-fn", "camera_login_id": "cam_b", "timestampMs": 3},
                FeedbackOutcome.FALSE_NEGATIVE,
                operator_id="op",
                feedback_text="FN",
                feedback_timestamp_ms=4,
            )

            append_feedback_jsonl(feedback_path, fp)
            append_feedback_jsonl(feedback_path, fn)
            summary = export_retraining_candidates(feedback_path, root / "candidates")

            self.assertEqual([row["event_id"] for row in load_feedback_jsonl(feedback_path)], ["evt-fp", "evt-fn"])
            self.assertEqual(summary["hard_negative"], 1)
            self.assertEqual(summary["faint_fall_reinforcement"], 1)
            with (root / "candidates" / "hard_negative_candidates.csv").open(encoding="utf-8", newline="") as fp_file:
                hard_negative_rows = list(csv.DictReader(fp_file))
            with (root / "candidates" / "faint_fall_reinforcement_candidates.csv").open(
                encoding="utf-8",
                newline="",
            ) as fn_file:
                reinforcement_rows = list(csv.DictReader(fn_file))
            self.assertEqual(hard_negative_rows[0]["event_id"], "evt-fp")
            self.assertEqual(reinforcement_rows[0]["event_id"], "evt-fn")
            summary_json = json.loads((root / "candidates" / "candidate_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_json["total"], 2)


if __name__ == "__main__":
    unittest.main()

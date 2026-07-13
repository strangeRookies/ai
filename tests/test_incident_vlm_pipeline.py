"""Incident VLM skeleton tests (mock only — no real Vision API)."""

from __future__ import annotations

import unittest

from ai.vlm.incident_pipeline import (
    DEFAULT_KEYFRAME_OFFSETS_SEC,
    FailingDeid,
    Incident,
    IncidentEvent,
    IncidentEventType,
    IncidentVlmPipeline,
    VlmJobPolicy,
    compute_keyframe_timestamps,
    is_analysis_eligible,
    mock_keyframe_metadata,
    validate_incident_v1,
)


def _incident(
    *,
    events: list[IncidentEvent],
    clip_start: float = 0.0,
    clip_end: float | None = 30.0,
    fps: float | None = 10.0,
    incident_id: str = "inc-1",
) -> Incident:
    return Incident(
        incident_id=incident_id,
        camera_login_id="cam_01",
        original_event_id="evt-0",
        events=tuple(events),
        clip_start_sec=clip_start,
        clip_end_sec=clip_end,
        fps=fps,
    )


class IncidentVlmPipelineTest(unittest.TestCase):
    def test_keyframe_default_offsets_and_recovery(self):
        inc = _incident(
            events=[
                IncidentEvent(IncidentEventType.NEW_FALL, 10.0, "e1"),
                IncidentEvent(IncidentEventType.RECOVERED, 14.0, "e2"),
            ]
        )
        stamps = compute_keyframe_timestamps(inc)
        self.assertIn(10.0, stamps)
        self.assertIn(14.0, stamps)  # recovery
        # T-2 clamped to clip start 0
        self.assertIn(8.0, stamps)
        self.assertEqual(len(stamps), len(set(round(s, 4) for s in stamps)))

    def test_short_clip_clamps_timestamps(self):
        inc = _incident(
            events=[IncidentEvent(IncidentEventType.NEW_FALL, 1.0, "e1")],
            clip_start=0.0,
            clip_end=2.0,
        )
        stamps = compute_keyframe_timestamps(inc)
        self.assertTrue(all(0.0 <= s <= 2.0 for s in stamps))
        self.assertIn(2.0, stamps)  # no recovery -> clip end

    def test_timestamp_range_excess_clamped(self):
        inc = _incident(
            events=[IncidentEvent(IncidentEventType.NEW_FALL, 5.0, "e1")],
            clip_start=4.0,
            clip_end=6.0,
        )
        stamps = compute_keyframe_timestamps(inc)
        self.assertTrue(all(4.0 <= s <= 6.0 for s in stamps))

    def test_no_recovery_appends_clip_end(self):
        inc = _incident(
            events=[IncidentEvent(IncidentEventType.FALL_UNRECOVERED, 12.0, "e1")],
            clip_end=20.0,
        )
        stamps = compute_keyframe_timestamps(inc)
        self.assertIn(20.0, stamps)

    def test_timestamp_dedup(self):
        # offsets that collapse after clamp
        inc = _incident(
            events=[IncidentEvent(IncidentEventType.NEW_FALL, 0.0, "e1")],
            clip_start=0.0,
            clip_end=0.5,
        )
        stamps = compute_keyframe_timestamps(inc)
        self.assertEqual(len(stamps), len(set(round(s, 4) for s in stamps)))

    def test_fps_none_frame_index_null(self):
        meta = mock_keyframe_metadata([1.0, 2.0], fps=None, incident_id="i")
        self.assertIsNone(meta[0]["frameIndex"])
        self.assertIsNone(meta[1]["frameIndex"])
        meta2 = mock_keyframe_metadata([1.0], fps=10.0, incident_id="i")
        self.assertEqual(meta2[0]["frameIndex"], 10)

    def test_duplicate_job_prevented(self):
        pipe = IncidentVlmPipeline(policy=VlmJobPolicy.FINAL_ONLY)
        inc = _incident(
            events=[
                IncidentEvent(IncidentEventType.NEW_FALL, 10.0, "e1"),
                IncidentEvent(IncidentEventType.RECOVERED, 15.0, "e2"),
            ]
        )
        job1 = pipe.process(inc)
        job2 = pipe.process(inc)
        self.assertEqual(job1.job_id, job2.job_id)
        self.assertEqual(job1.status, "SUCCESS")
        self.assertIs(job1, job2)
        self.assertEqual(pipe.vlm_calls, 1)

    def test_deid_failure_blocks_vlm(self):
        pipe = IncidentVlmPipeline(policy=VlmJobPolicy.FINAL_ONLY, deid=FailingDeid())
        inc = _incident(
            events=[
                IncidentEvent(IncidentEventType.NEW_FALL, 10.0, "e1"),
                IncidentEvent(IncidentEventType.RECOVERED, 12.0, "e2"),
            ]
        )
        job = pipe.process(inc)
        self.assertEqual(job.status, "BLOCKED_DEID")
        self.assertFalse(job.deidentified)
        self.assertIsNone(job.structured_result)
        self.assertEqual(pipe.vlm_calls, 0)

    def test_final_only_max_one_call_across_incidents(self):
        pipe = IncidentVlmPipeline(policy=VlmJobPolicy.FINAL_ONLY)
        inc1 = _incident(
            incident_id="inc-a",
            events=[
                IncidentEvent(IncidentEventType.NEW_FALL, 10.0, "e1"),
                IncidentEvent(IncidentEventType.RECOVERED, 12.0, "e2"),
            ],
        )
        inc2 = _incident(
            incident_id="inc-b",
            events=[
                IncidentEvent(IncidentEventType.NEW_FALL, 20.0, "e3"),
                IncidentEvent(IncidentEventType.FALL_UNRECOVERED, 30.0, "e4"),
            ],
        )
        j1 = pipe.process(inc1)
        j2 = pipe.process(inc2)
        self.assertEqual(j1.status, "SUCCESS")
        self.assertEqual(j2.status, "SKIPPED")
        self.assertEqual(pipe.vlm_calls, 1)

    def test_ineligible_without_terminal(self):
        pipe = IncidentVlmPipeline(policy=VlmJobPolicy.FINAL_ONLY)
        inc = _incident(
            events=[IncidentEvent(IncidentEventType.NEW_FALL, 10.0, "e1")],
            clip_end=None,
        )
        self.assertFalse(is_analysis_eligible(inc, policy=VlmJobPolicy.FINAL_ONLY))
        job = pipe.process(inc)
        self.assertEqual(job.status, "INELIGIBLE")

    def test_schema_validation_and_search_doc(self):
        pipe = IncidentVlmPipeline()
        inc = _incident(
            events=[
                IncidentEvent(IncidentEventType.NEW_FALL, 5.0, "e1"),
                IncidentEvent(IncidentEventType.RECOVERED, 8.0, "e2"),
            ]
        )
        job = pipe.process(inc)
        self.assertEqual(job.status, "SUCCESS")
        validate_incident_v1(job.structured_result)
        self.assertIn("incident:", job.search_document)
        self.assertTrue(job.deidentified)
        self.assertGreater(len(job.keyframe_timestamps_sec), 0)
        # offsets contract present
        self.assertEqual(DEFAULT_KEYFRAME_OFFSETS_SEC[0], -2.0)
        self.assertEqual(DEFAULT_KEYFRAME_OFFSETS_SEC[-1], 5.0)


if __name__ == "__main__":
    unittest.main()

import unittest

from ai.vlm.mock_adapter import MockVlmAdapter, VlmEvidence


class VlmMockAdapterTest(unittest.TestCase):
    def test_mock_vlm_describes_event_evidence_without_final_judgment(self):
        adapter = MockVlmAdapter()
        evidence = VlmEvidence(
            event_id="evt-1",
            camera_login_id="cam_01",
            timestamp_ms=1782180000123,
            snapshot_path="clips/evt-1.jpg",
            clip_path="clips/evt-1.mp4",
            detected_type="faint",
            confidence=0.88,
        )

        result = adapter.describe(evidence)

        self.assertEqual(result.event_id, "evt-1")
        self.assertEqual(result.camera_login_id, "cam_01")
        self.assertEqual(result.adapter_name, "mock-vlm")
        self.assertIn("operator-assist", result.description)
        self.assertFalse(result.final_decision)


if __name__ == "__main__":
    unittest.main()

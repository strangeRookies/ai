import unittest

from scripts.diagnose_overlay_payload_sync import (
    build_diagnostic_samples,
    classify_sync_payload,
    missing_sync_fields,
)


class OverlayPayloadSyncDiagnosisTest(unittest.TestCase):
    def test_diagnostic_samples_include_required_raw_and_normalized_fields(self):
        samples = build_diagnostic_samples("qa_lobby_01")
        raw_overlay = samples["rawOverlay"]
        normalized = samples["expectedNormalizedOverlay"]

        self.assertEqual(missing_sync_fields(raw_overlay), [])
        self.assertEqual(missing_sync_fields(normalized), [])
        self.assertEqual(samples["classification"], "sync_fields_present")
        self.assertEqual(raw_overlay["evidenceId"], "qa_lobby_01-4-1782180000100")
        self.assertIn("frameId=4", samples["expectedOverlaySyncLog"])
        self.assertIn("bufferSize=2", samples["expectedOverlaySyncLog"])

    def test_classification_separates_upstream_and_frontend_missing_fields(self):
        raw_missing = {"cameraLoginId": "qa_lobby_01"}
        raw_present = {
            "cameraLoginId": "qa_lobby_01",
            "frameId": 4,
            "timestampMs": 1782180000123,
            "capturedAtMs": 1782180000100,
            "processedAtMs": 1782180000120,
            "publishedAtMs": 1782180000123,
            "frameWidth": 640,
            "frameHeight": 360,
        }
        normalized_missing = dict(raw_present)
        del normalized_missing["frameId"]

        self.assertEqual(
            classify_sync_payload(raw_missing, raw_present),
            "upstream_raw_payload_missing_fields",
        )
        self.assertEqual(
            classify_sync_payload(raw_present, normalized_missing),
            "frontend_normalization_missing_fields",
        )


if __name__ == "__main__":
    unittest.main()

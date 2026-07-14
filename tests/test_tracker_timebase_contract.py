import unittest

from ai.inference.tracker_timebase import resolve_tracker_fps


class TrackerTimebaseContractTest(unittest.TestCase):
    def test_source_fps_is_preferred_when_valid(self):
        self.assertEqual(resolve_tracker_fps(15.0, 30.0), 15.0)

    def test_invalid_source_fps_uses_configured_fallback(self):
        self.assertEqual(resolve_tracker_fps(0.0, 30.0), 30.0)
        self.assertEqual(resolve_tracker_fps(float("nan"), 30.0), 30.0)

    def test_invalid_source_and_configured_fps_uses_safe_default(self):
        self.assertEqual(resolve_tracker_fps(0.0, 0.0), 30.0)


if __name__ == "__main__":
    unittest.main()

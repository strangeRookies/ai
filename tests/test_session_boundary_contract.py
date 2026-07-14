import unittest

from ai.inference.session_boundary import session_reset_reason


class SessionBoundaryContractTest(unittest.TestCase):
    def test_video_start_keeps_clean_session(self):
        self.assertIsNone(session_reset_reason(None, 0, None, 1000))

    def test_reconnect_and_boundary_conditions_reset_session(self):
        self.assertEqual(session_reset_reason(10, 1, 1000, 1100), "FRAME_ID_RESET")
        self.assertEqual(session_reset_reason(1, 2, 1000, 5001), "LARGE_TIME_GAP")
        self.assertEqual(session_reset_reason(1, 92, 1000, 1100), "LARGE_FRAME_GAP")


if __name__ == "__main__":
    unittest.main()
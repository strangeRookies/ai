import unittest
from argparse import Namespace

from scripts.serve_ai_overlay import mjpeg_debug_enabled, mjpeg_server_enabled


class MjpegDebugModeTest(unittest.TestCase):
    def test_mjpeg_output_is_disabled_when_both_flags_off(self):
        self.assertFalse(mjpeg_debug_enabled(Namespace(mjpeg_debug=False, mjpeg_enabled=False)))

    def test_mjpeg_output_is_enabled_with_debug_flag(self):
        self.assertTrue(mjpeg_debug_enabled(Namespace(mjpeg_debug=True, mjpeg_enabled=False)))

    def test_mjpeg_server_enabled_with_mjpeg_enabled_flag(self):
        """mjpeg_enabled=True 이면 mjpeg_server_enabled()/mjpeg_debug_enabled()가 True 여야 한다."""
        self.assertTrue(mjpeg_server_enabled(Namespace(mjpeg_enabled=True, mjpeg_debug=False)))
        self.assertTrue(mjpeg_debug_enabled(Namespace(mjpeg_enabled=True, mjpeg_debug=False)))

    def test_mjpeg_server_disabled_without_any_flag(self):
        """mjpeg_debug 속성 없이 False 기본값만 있으면 비활성화."""
        self.assertFalse(mjpeg_debug_enabled(Namespace(mjpeg_debug=False)))


if __name__ == "__main__":
    unittest.main()

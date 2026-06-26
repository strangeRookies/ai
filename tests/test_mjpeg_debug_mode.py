import unittest
from argparse import Namespace

from scripts.serve_ai_overlay import mjpeg_debug_enabled


class MjpegDebugModeTest(unittest.TestCase):
    def test_mjpeg_output_is_disabled_without_explicit_debug_flag(self):
        self.assertFalse(mjpeg_debug_enabled(Namespace(mjpeg_debug=False)))

    def test_mjpeg_output_is_enabled_only_with_explicit_debug_flag(self):
        self.assertTrue(mjpeg_debug_enabled(Namespace(mjpeg_debug=True)))


if __name__ == "__main__":
    unittest.main()

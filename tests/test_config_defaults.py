import os
import unittest
from unittest.mock import patch

from config import load_settings


class ConfigDefaultsTest(unittest.TestCase):
    def test_load_settings_defaults_sequence_length_to_30(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()

        self.assertEqual(settings.sequence_length, 30)


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest.mock import patch

from config import load_settings


class ConfigDefaultsTest(unittest.TestCase):
    def test_load_settings_defaults_sequence_contract_to_30_15(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()

        self.assertEqual(settings.sequence_length, 30)
        self.assertEqual(settings.sequence_stride, 15)


if __name__ == "__main__":
    unittest.main()

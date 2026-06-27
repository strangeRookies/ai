import json
import tempfile
import unittest
from pathlib import Path

from ai.labels.event_label_loader import load_event_label


class EventLabelLoaderTest(unittest.TestCase):
    def test_event_frame_active_check(self):
        payload = {
            "metadata": {"file_name": "E03_001.mp4", "frame_count": 100},
            "annotations": {"event_class": "Fight", "event_frame": [[10, 20], [40, 50]]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            label = load_event_label(path)

        self.assertEqual(label.event_class, "Fight")
        self.assertTrue(label.is_active(10))
        self.assertTrue(label.is_active(45))
        self.assertFalse(label.is_active(9))
        self.assertFalse(label.is_active(51))


if __name__ == "__main__":
    unittest.main()

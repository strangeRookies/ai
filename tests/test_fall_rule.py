import unittest

from rules.fall_rule import FallRuleEngine


class FallRuleTest(unittest.TestCase):
    def test_fall_rule_requires_duration_before_event(self):
        rule = FallRuleEngine(min_duration_seconds=1.0, debounce_seconds=10)
        detection = {
            "track_id": 1,
            "bbox": [100, 200, 380, 330],
            "confidence": 0.9,
            "pose_horizontal": True,
            "rule_hint": 0.9,
        }

        self.assertEqual(rule.evaluate([detection], now=100.0), [])
        self.assertEqual(rule.evaluate([detection], now=100.5), [])

        events = rule.evaluate([detection], now=101.1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][2], "LYING")

    def test_fall_rule_debounces_repeated_events(self):
        rule = FallRuleEngine(min_duration_seconds=0.1, debounce_seconds=10)
        detection = {
            "track_id": 1,
            "bbox": [100, 200, 380, 330],
            "confidence": 0.9,
            "pose_horizontal": True,
            "rule_hint": 0.9,
        }

        rule.evaluate([detection], now=100.0)
        self.assertEqual(len(rule.evaluate([detection], now=100.2)), 1)
        self.assertEqual(rule.evaluate([detection], now=101.0), [])


if __name__ == "__main__":
    unittest.main()

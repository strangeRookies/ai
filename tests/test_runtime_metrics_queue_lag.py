import unittest

from ai.runtime_metrics import RuntimeMetrics


class RuntimeMetricsQueueLagTest(unittest.TestCase):
    def test_summary_includes_queue_lag_percentiles(self):
        metrics = RuntimeMetrics()
        for lag in (5, 10, 15, 40, 20):
            metrics.add_queue_lag_ms(lag)
        summary = metrics.summary(10, 0, 0, 0, 0)
        self.assertEqual(summary["avg_queue_lag_ms"], 18.0)
        self.assertEqual(summary["p50_queue_lag_ms"], 15.0)
        self.assertEqual(summary["p95_queue_lag_ms"], 40.0)
        self.assertEqual(summary["max_queue_lag_ms"], 40.0)

    def test_queue_lag_empty_is_null(self):
        summary = RuntimeMetrics().summary(0, 0, 0, 0, 0)
        self.assertIsNone(summary["avg_queue_lag_ms"])
        self.assertIsNone(summary["max_queue_lag_ms"])


if __name__ == "__main__":
    unittest.main()
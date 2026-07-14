import unittest
from ai.action.per_track_sequence_buffer import aggregate_sequence_completion


class SequenceCompletionContractTest(unittest.TestCase):
    def test_completion_summary_separates_completed_and_incomplete_tracks(self):
        summary = aggregate_sequence_completion(
            {1: {"buffer_length": 30, "reason": "sequence_ready"}, 2: {"buffer_length": 3, "reason": "buffer_not_full"}},
            {1: 2},
        )
        self.assertEqual(summary["total_observed_tracks"], 2)
        self.assertEqual(summary["tracks_with_completed_sequence"], 1)
        self.assertEqual(summary["incomplete_buffer_not_full"], 1)


if __name__ == "__main__":
    unittest.main()

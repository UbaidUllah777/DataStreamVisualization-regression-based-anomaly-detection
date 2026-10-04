import unittest

import pandas as pd

from analysis.streaming_detector import StreamingAnomalyDetector


class TestStreamingAnomalyDetector(unittest.TestCase):
    """Check rule boundaries using simple, synthetic residual values."""

    def setUp(self):
        self.origin = pd.Timestamp("2026-01-01T00:00:00Z")
        self.thresholds = pd.DataFrame(
            {
                "candidate_MinC": [2.0],
                "candidate_MaxC": [5.0],
            },
            index=["j1"],
        )

    def detector(self):
        return StreamingAnomalyDetector(
            thresholds=self.thresholds,
            duration_seconds=8.0,
            max_gap_seconds=5.0,
        )

    def feed(self, detector, seconds, residual):
        return detector.update(
            self.origin + pd.Timedelta(seconds=seconds),
            {"j1": residual},
        )

    def test_exact_minimum_and_duration(self):
        detector = self.detector()

        for seconds in (0, 2, 4, 6):
            status = self.feed(detector, seconds, 2.0)
            self.assertEqual(status["j1"], "Normal")

        self.assertTrue(detector.event_frame().empty)

        status = self.feed(detector, 8, 2.0)
        events = detector.event_frame()

        self.assertEqual(status["j1"], "Alert")
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["duration_seconds"], 8.0)

    def test_exact_maximum_triggers_both_rules(self):
        detector = self.detector()

        for seconds in (0, 2, 4, 6, 8):
            status = self.feed(detector, seconds, 5.0)

        events = detector.event_frame()
        self.assertEqual(status["j1"], "Error")
        self.assertEqual(set(events["severity"]), {"Alert", "Error"})
        self.assertEqual(len(events), 2)

    def test_below_threshold_resets_timer(self):
        detector = self.detector()

        for seconds in (0, 2, 4, 6):
            self.feed(detector, seconds, 3.0)

        self.feed(detector, 8, 1.0)

        for seconds in (10, 12, 14, 16):
            self.feed(detector, seconds, 3.0)

        self.assertTrue(detector.event_frame().empty)

        self.feed(detector, 18, 3.0)
        event = detector.event_frame().iloc[0]

        self.assertEqual(
            event["start"], self.origin + pd.Timedelta(seconds=10)
        )
        self.assertEqual(
            event["triggered_at"], self.origin + pd.Timedelta(seconds=18)
        )

    def test_staggered_escalation_and_downgrade(self):
        detector = self.detector()

        for seconds in (0, 2, 4, 6, 8):
            self.feed(detector, seconds, 3.0)

        for seconds in (10, 12, 14, 16):
            status = self.feed(detector, seconds, 6.0)
            self.assertEqual(status["j1"], "Alert")

        status = self.feed(detector, 18, 6.0)
        self.assertEqual(status["j1"], "Error")

        status = self.feed(detector, 20, 3.0)
        self.assertEqual(status["j1"], "Alert")

        status = self.feed(detector, 22, 0.0)
        self.assertEqual(status["j1"], "Normal")

        events = detector.event_frame().set_index("severity")

        self.assertEqual(len(events), 2)
        self.assertEqual(events.loc["Alert", "duration_seconds"], 20.0)
        self.assertEqual(events.loc["Error", "duration_seconds"], 8.0)
        self.assertEqual(
            events.loc["Error", "triggered_at"],
            self.origin + pd.Timedelta(seconds=18),
        )
        self.assertTrue(events["close_reason"].eq("below_threshold").all())

    def test_gap_closes_event_and_restarts_timer(self):
        detector = self.detector()

        for seconds in (0, 2, 4, 6, 8):
            self.feed(detector, seconds, 3.0)

        status = self.feed(detector, 14, 3.0)
        self.assertEqual(status["j1"], "Normal")

        for seconds in (16, 18, 20, 22):
            self.feed(detector, seconds, 3.0)

        events = detector.event_frame()

        self.assertEqual(len(events), 2)
        self.assertEqual(events.iloc[0]["close_reason"], "recording_gap")
        self.assertEqual(events.iloc[0]["duration_seconds"], 8.0)
        self.assertEqual(
            events.iloc[1]["start"],
            self.origin + pd.Timedelta(seconds=14),
        )

    def test_gap_equal_to_limit_is_permitted(self):
        detector = self.detector()

        self.feed(detector, 0, 3.0)
        self.feed(detector, 5, 3.0)
        status = self.feed(detector, 8, 3.0)

        self.assertEqual(status["j1"], "Alert")
        self.assertEqual(len(detector.event_frame()), 1)

    def test_stream_end_does_not_claim_recovery(self):
        detector = self.detector()

        for seconds in (0, 2, 4, 6, 8):
            self.feed(detector, seconds, 3.0)

        detector.finish()
        event = detector.event_frame().iloc[0]

        self.assertEqual(event["close_reason"], "stream_end")
        self.assertEqual(event["duration_seconds"], 8.0)

        with self.assertRaises(RuntimeError):
            self.feed(detector, 10, 3.0)

    def test_rejects_duplicate_and_out_of_order_timestamps(self):
        detector = self.detector()
        self.feed(detector, 2, 3.0)

        with self.assertRaises(ValueError):
            self.feed(detector, 2, 3.0)

        with self.assertRaises(ValueError):
            self.feed(detector, 1, 3.0)


if __name__ == "__main__":
    unittest.main()
from __future__ import annotations

import math
import unittest

from inference.benchmark_l4_runtime_acceleration import (
    TensorRTRunner,
    decide_benefit,
    percentile,
    summarize_latencies,
)


class RuntimeAccelerationHelpersTest(unittest.TestCase):
    def test_runner_exposes_graph_parity_gate(self) -> None:
        self.assertTrue(hasattr(TensorRTRunner, "parity"))

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)

    def test_percentile_rejects_empty_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            percentile([], 50)

    def test_summarize_latencies_reports_expected_fields(self) -> None:
        summary = summarize_latencies([1.0, 2.0, 3.0], input_seconds=16.0)
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["mean_ms"], 2.0)
        self.assertEqual(summary["p50_ms"], 2.0)
        self.assertEqual(summary["p95_ms"], 2.9)
        self.assertEqual(summary["requests_per_second"], 500.0)
        self.assertEqual(summary["audio_seconds_per_second"], 8000.0)

    def test_summarize_latencies_rejects_non_positive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite"):
            summarize_latencies([1.0, math.inf], input_seconds=16.0)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            summarize_latencies([1.0, 0.0], input_seconds=16.0)

    def test_decide_benefit_requires_mean_and_p95_improvement(self) -> None:
        baseline = {"mean_ms": 10.0, "p95_ms": 11.0}
        accepted = decide_benefit(baseline, {"mean_ms": 9.5, "p95_ms": 10.4})
        self.assertTrue(accepted["accepted"])
        self.assertAlmostEqual(accepted["mean_speedup"], 10.0 / 9.5)

        noisy = decide_benefit(baseline, {"mean_ms": 9.0, "p95_ms": 11.2})
        self.assertFalse(noisy["accepted"])
        self.assertIn("p95", noisy["reason"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from inference.benchmark_trt_mixed_duration_burst import (
    RequestSpec,
    expand_manifest,
    greedy_lpt_assign,
    manifest_stats,
    percentile,
    route_duration,
    run_one_burst,
    summarize_aggregate,
    validate_worker_count,
)


class MixedDurationBurstHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.required_routes = {
            "short_dynamic",
            "mid_dynamic",
            "fixed10",
            "long_dynamic",
        }

    def test_builtin_manifest_a_matches_contract(self) -> None:
        durations = expand_manifest("A")
        stats = manifest_stats(durations)
        self.assertEqual(len(durations), 50)
        self.assertAlmostEqual(stats["mean_seconds"], 5.50)
        self.assertEqual(stats["above_16_requests"], 2)
        self.assertAlmostEqual(stats["above_16_percent"], 4.0)

    def test_builtin_manifest_b_matches_contract(self) -> None:
        durations = expand_manifest("B")
        stats = manifest_stats(durations)
        self.assertEqual(len(durations), 50)
        self.assertAlmostEqual(stats["mean_seconds"], 5.86)
        self.assertEqual(stats["above_16_requests"], 3)
        self.assertAlmostEqual(stats["above_16_percent"], 6.0)

    def test_route_selection_and_padding(self) -> None:
        self.assertEqual(route_duration(1.0, self.required_routes), ("short_dynamic", 2.0))
        self.assertEqual(route_duration(6.0, self.required_routes), ("short_dynamic", 6.0))
        self.assertEqual(route_duration(7.0, self.required_routes), ("short_dynamic", 7.0))
        self.assertEqual(route_duration(7.01, self.required_routes), ("mid_dynamic", 7.01))
        self.assertEqual(route_duration(8.0, self.required_routes), ("mid_dynamic", 8.0))
        self.assertEqual(route_duration(9.0, self.required_routes), ("fixed10", 10.0))
        self.assertEqual(route_duration(10.0, self.required_routes), ("fixed10", 10.0))
        self.assertEqual(route_duration(12.0, self.required_routes), ("mid_dynamic", 12.0))
        self.assertEqual(route_duration(18.0, self.required_routes), ("long_dynamic", 18.0))
        self.assertEqual(route_duration(30.0, self.required_routes), ("long_dynamic", 30.0))
        with self.assertRaisesRegex(ValueError, "overflow single-forward fallback"):
            route_duration(30.0001, self.required_routes)

    def test_fixed16_is_optional_with_mid_dynamic_fallback(self) -> None:
        self.assertEqual(route_duration(15.0, self.required_routes), ("mid_dynamic", 15.0))
        with_fixed16 = self.required_routes | {"fixed16"}
        self.assertEqual(route_duration(15.0, with_fixed16), ("fixed16", 16.0))

    def test_greedy_lpt_assigns_every_request_once_for_1_2_4_workers(self) -> None:
        requests = [
            RequestSpec(f"r{index}", float(index), float(index), "short_dynamic", estimate)
            for index, estimate in enumerate((9.0, 8.0, 7.0, 6.0, 5.0), start=1)
        ]
        for worker_count in (1, 2, 4):
            assignments, loads = greedy_lpt_assign(requests, worker_count)
            assigned_ids = [request.request_id for worker in assignments for request in worker]
            self.assertCountEqual(assigned_ids, [request.request_id for request in requests])
            self.assertEqual(len(assignments), worker_count)
            self.assertEqual(len(loads), worker_count)
        assignments, loads = greedy_lpt_assign(requests, 2)
        self.assertEqual(loads, [20.0, 15.0])
        self.assertEqual(assignments[0][0].estimated_gpu_ms, 9.0)
        self.assertEqual(assignments[1][0].estimated_gpu_ms, 8.0)

    def test_cli_worker_count_helper_accepts_1_2_4_and_rejects_zero(self) -> None:
        for worker_count in (1, 2, 4):
            self.assertEqual(validate_worker_count(worker_count), worker_count)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_worker_count(0)

    def test_run_one_burst_rejects_assignment_worker_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "assignment/worker count mismatch"):
            run_one_burst("A", [[], []], [], None)

    def test_aggregate_requests_per_second_uses_sum_of_burst_seconds(self) -> None:
        runs = [
            {
                "burst_ms": 100.0,
                "host_wall_ms": 110.0,
                "request_completion_ms": [40.0, 80.0],
            },
            {
                "burst_ms": 300.0,
                "host_wall_ms": 310.0,
                "request_completion_ms": [100.0, 250.0],
            },
        ]
        summary = summarize_aggregate(runs)
        self.assertEqual(summary["requests"], 4)
        self.assertAlmostEqual(summary["requests_per_second"], 10.0)

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50.0), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 95.0), 3.85)

    def test_percentile_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            percentile([], 50.0)


if __name__ == "__main__":
    unittest.main()

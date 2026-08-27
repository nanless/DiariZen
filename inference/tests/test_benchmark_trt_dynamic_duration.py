from __future__ import annotations

import unittest

from inference.benchmark_trt_dynamic_duration import (
    engine_device_memory_bytes,
    profile_metadata,
    profile_shapes_for_seconds,
    shape_for_seconds,
    validate_benchmark_shapes,
    validate_output_shapes,
    validate_profile_shapes,
)


class DynamicDurationHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = ((1, 1, 32000), (1, 1, 96000), (1, 1, 480000))

    def test_profile_shapes_are_derived_from_seconds(self) -> None:
        self.assertEqual(profile_shapes_for_seconds([2.0, 6.0, 30.0], 16000), self.profile)
        self.assertEqual(shape_for_seconds(16.1, 16000), (1, 1, 257600))

    def test_profile_shape_requires_at_least_one_sample(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            shape_for_seconds(0.00001, 1)

    def test_requested_profile_must_match_engine_exactly(self) -> None:
        validate_profile_shapes(self.profile, self.profile)
        mismatched = (self.profile[0], (1, 1, 160000), self.profile[2])
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_profile_shapes(self.profile, mismatched)

    def test_benchmark_shapes_accept_profile_boundaries(self) -> None:
        validate_benchmark_shapes(
            [(1, 1, 32000), (1, 1, 256000), (1, 1, 480000)], self.profile
        )

    def test_benchmark_shapes_reject_out_of_profile_and_wrong_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside engine profile"):
            validate_benchmark_shapes([(1, 1, 480001)], self.profile)
        with self.assertRaisesRegex(ValueError, "rank"):
            validate_benchmark_shapes([(1, 480000)], self.profile)

    def test_output_shapes_must_be_resolved_and_positive(self) -> None:
        validate_output_shapes({"multilabel": (1, 1499, 4)})
        with self.assertRaisesRegex(ValueError, "non-positive"):
            validate_output_shapes({"multilabel": (1, -1, 4)})
        with self.assertRaisesRegex(ValueError, "no output"):
            validate_output_shapes({})

    def test_profile_metadata_reports_shapes_and_seconds(self) -> None:
        metadata = profile_metadata(self.profile, 16000)
        self.assertEqual(metadata["min"], {"shape": [1, 1, 32000], "seconds": 2.0})
        self.assertEqual(metadata["max"], {"shape": [1, 1, 480000], "seconds": 30.0})

    def test_engine_device_memory_prefers_v2_and_supports_legacy(self) -> None:
        with_v2 = type("Engine", (), {"device_memory_size_v2": 123, "device_memory_size": 99})()
        legacy = type("Engine", (), {"device_memory_size": 77})()
        self.assertEqual(engine_device_memory_bytes(with_v2), 123)
        self.assertEqual(engine_device_memory_bytes(legacy), 77)


if __name__ == "__main__":
    unittest.main()

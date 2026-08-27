from __future__ import annotations

import unittest
from pathlib import Path

from inference.benchmark_trt_fixed10s_parity import (
    normalize_profile_shapes,
    resolve_engine_path,
    validate_shape_in_profile,
)


class TensorRTParityHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dynamic_profile = (
            (1, 1, 32000),
            (1, 1, 96000),
            (1, 1, 480000),
        )

    def test_explicit_engine_overrides_legacy_resolution(self) -> None:
        explicit = Path("/tmp/dynamic_2s-30s.plan")
        resolved = resolve_engine_path(
            explicit,
            Path("/tmp/legacy"),
            "30s",
            1,
            "fp16",
        )
        self.assertEqual(resolved, explicit)

    def test_legacy_engine_path_is_unchanged(self) -> None:
        resolved = resolve_engine_path(
            None,
            Path("/tmp/legacy"),
            "10s",
            8,
            "fp16",
        )
        self.assertEqual(
            resolved,
            Path("/tmp/legacy/segmentation_10s_bs8_fp16.plan"),
        )

    def test_engine_resolution_requires_one_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "either --engine or --engine-dir"):
            resolve_engine_path(None, None, "30s", 1, "fp16")

    def test_dynamic_profile_accepts_batch1_30_seconds(self) -> None:
        shape = validate_shape_in_profile((1, 1, 480000), self.dynamic_profile)
        self.assertEqual(shape, (1, 1, 480000))

    def test_dynamic_profile_rejects_shape_above_max_and_wrong_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside engine profile"):
            validate_shape_in_profile((1, 1, 480001), self.dynamic_profile)
        with self.assertRaisesRegex(ValueError, "outside engine profile"):
            validate_shape_in_profile((2, 1, 480000), self.dynamic_profile)

    def test_profile_shapes_must_be_positive_and_ordered(self) -> None:
        self.assertEqual(normalize_profile_shapes(self.dynamic_profile), self.dynamic_profile)
        with self.assertRaisesRegex(ValueError, "non-positive"):
            normalize_profile_shapes(((1, 1, 0), (1, 1, 1), (1, 1, 2)))
        with self.assertRaisesRegex(ValueError, "min <= opt <= max"):
            normalize_profile_shapes(
                ((1, 1, 32000), (1, 1, 480000), (1, 1, 96000))
            )


if __name__ == "__main__":
    unittest.main()

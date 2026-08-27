#!/usr/bin/env python3
"""Create explicit-Q/DQ TensorRT ONNX variants using NVIDIA ModelOpt."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from modelopt.onnx.quantization import quantize


def calibration_data(waveform_npy: Path, input_name: str) -> dict[str, np.ndarray]:
    base = np.load(waveform_npy).astype(np.float32)
    if base.shape != (1, 1, 160_000):
        raise ValueError(f"Expected (1, 1, 160000), got {base.shape}")
    rng = np.random.default_rng(20_260_826)
    variants = [
        base,
        base * 0.7,
        np.roll(base, 8_000, axis=-1),
        np.clip(base + rng.normal(0.0, 0.005, size=base.shape), -1.0, 1.0).astype(np.float32),
    ]
    return {input_name: np.concatenate(variants, axis=0)}


def summarize(path: Path) -> dict:
    model = onnx.load(path, load_external_data=False)
    op_counts = Counter(node.op_type for node in model.graph.node)
    initializer_types = Counter(
        onnx.TensorProto.DataType.Name(initializer.data_type)
        for initializer in model.graph.initializer
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "op_counts": dict(op_counts),
        "initializer_types": dict(initializer_types),
    }


def int4_nodes_with_unaligned_weights(model: onnx.ModelProto, block_size: int) -> list[str]:
    """Exclude weight matrices TensorRT cannot represent with exact INT4 blocks."""
    initializer_shapes = {
        initializer.name: tuple(initializer.dims)
        for initializer in model.graph.initializer
    }
    excluded = []
    for node in model.graph.node:
        if node.op_type not in {"MatMul", "Gemm"} or len(node.input) < 2:
            continue
        shape = initializer_shapes.get(node.input[1])
        if shape and shape[0] % block_size:
            excluded.append(node.name)
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["fp8", "int8", "int4"], required=True)
    parser.add_argument("--waveform-npy", type=Path)
    parser.add_argument("--high-precision-dtype", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=False)
    input_name = model.graph.input[0].name
    kwargs = {
        "onnx_path": str(args.input),
        "quantize_mode": args.mode,
        "output_path": str(args.output),
        "op_types_to_quantize": ["MatMul", "Gemm"],
        "high_precision_dtype": args.high_precision_dtype,
        "use_external_data_format": False,
        "log_level": "INFO",
    }
    if args.mode in {"fp8", "int8"}:
        if args.waveform_npy is None:
            parser.error("--waveform-npy is required for FP8/INT8 calibration")
        kwargs.update(
            calibration_data=calibration_data(args.waveform_npy, input_name),
            calibration_method="max",
            calibration_eps=["cpu"],
        )
    else:
        block_size = 128
        excluded = int4_nodes_with_unaligned_weights(model, block_size)
        kwargs.update(
            calibration_method="rtn_dq",
            block_size=block_size,
            nodes_to_exclude=excluded,
        )
        print(
            json.dumps(
                {
                    "int4_block_size": block_size,
                    "excluded_unaligned_weight_nodes": len(excluded),
                    "excluded_node_names": excluded,
                },
                indent=2,
            ),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize(**kwargs)
    print(json.dumps(summarize(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()

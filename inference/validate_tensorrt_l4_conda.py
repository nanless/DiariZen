#!/usr/bin/env python3
"""Validate every fixed-10s TensorRT engine in the managed Conda environment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


def execute_engine(engine_path: Path, waveforms: np.ndarray) -> tuple[np.ndarray, float]:
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {engine_path}")
    context = engine.create_execution_context()
    trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "bfloat16"):
        trt_to_torch[trt.bfloat16] = torch.bfloat16

    tensors: dict[str, torch.Tensor] = {}
    output_names: list[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        dtype = trt_to_torch[engine.get_tensor_dtype(name)]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, waveforms.shape)
            tensor = torch.from_numpy(waveforms).to(device="cuda", dtype=dtype)
        else:
            output_names.append(name)
            tensor = torch.empty(
                tuple(context.get_tensor_shape(name)), device="cuda", dtype=dtype
            )
        tensors[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.Stream()
    # Input H2D copies use PyTorch's current stream; TensorRT uses a separate
    # stream. Synchronize once to avoid a cross-stream input race.
    torch.cuda.synchronize()
    repeated_outputs = []
    for _ in range(2):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution returned false for {engine_path}")
        stream.synchronize()
        repeated_outputs.append(tensors[output_names[0]].cpu().numpy().copy())
    repeat_equal = float((repeated_outputs[0] == repeated_outputs[1]).mean()) * 100.0
    actual = repeated_outputs[1]
    del tensors, context, engine
    gc.collect()
    torch.cuda.empty_cache()
    return actual, repeat_equal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-engine-dir", type=Path, required=True)
    parser.add_argument("--precision-engine-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,4,8,32")
    parser.add_argument(
        "--precisions", default="fp16,fp32_tf32,fp32,bf16,fp8,int8,int4"
    )
    parser.add_argument("--expected-conda-env", default="diarizen-trt1010")
    args = parser.parse_args()

    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if active_env != args.expected_conda_env:
        raise RuntimeError(
            f"Expected CONDA_DEFAULT_ENV={args.expected_conda_env}, got {active_env!r}"
        )

    batches = [int(value) for value in args.batch_sizes.split(",")]
    precisions = args.precisions.split(",")
    report = {
        "environment": {
            "conda_env": active_env,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "python": sys.version.split()[0],
            "tensorrt": trt.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "scope": "all fixed-10s TensorRT plans; adversarial synthetic only",
        "results": [],
    }

    for precision in precisions:
        engine_dir = args.fp16_engine_dir if precision == "fp16" else args.precision_engine_dir
        for batch_size in batches:
            engine_path = engine_dir / f"segmentation_10s_bs{batch_size}_{precision}.plan"
            if not engine_path.is_file():
                raise FileNotFoundError(engine_path)
            artifact = np.load(
                args.artifact_dir / f"ort_fp32_synthetic_10s_bs{batch_size}.npz"
            )
            waveforms = artifact["waveforms"]
            reference = artifact["reference"]
            actual, repeat_exact_match_pct = execute_engine(engine_path, waveforms)
            equal = actual == reference
            frame_equal = np.all(equal, axis=-1)
            result = {
                "precision": precision,
                "batch_size": batch_size,
                "engine": str(engine_path),
                "engine_bytes": engine_path.stat().st_size,
                "output_shape": list(actual.shape),
                "cell_exact_match_pct": round(float(equal.mean()) * 100.0, 6),
                "frame_exact_match_pct": round(float(frame_equal.mean()) * 100.0, 6),
                "mismatched_cells": int((~equal).sum()),
                "mismatched_frames": int((~frame_equal).sum()),
                "repeat_exact_match_pct": round(repeat_exact_match_pct, 6),
            }
            report["results"].append(result)
            print(json.dumps(result), flush=True)

    report["engine_count"] = len(report["results"])
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

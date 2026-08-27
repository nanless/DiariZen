#!/usr/bin/env python3
"""Generate ORT FP32 references and compare fixed-duration TensorRT outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
SAMPLES = 10 * SAMPLE_RATE
DURATION_LABEL = "10s"


def resolve_engine_path(
    explicit_engine: Path | None,
    engine_dir: Path | None,
    duration_label: str,
    batch_size: int,
    precision: str,
) -> Path:
    """Resolve an explicit dynamic plan or the legacy fixed-engine filename."""
    if explicit_engine is not None:
        return explicit_engine
    if engine_dir is None:
        raise ValueError("either --engine or --engine-dir is required for compare mode")
    return engine_dir / f"segmentation_{duration_label}_bs{batch_size}_{precision}.plan"


def normalize_profile_shapes(
    shapes,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    normalized = tuple(tuple(int(dim) for dim in shape) for shape in shapes)
    if len(normalized) != 3:
        raise ValueError(f"profile must contain min,opt,max shapes, got {normalized}")
    if len({len(shape) for shape in normalized}) != 1:
        raise ValueError(f"profile ranks differ: {normalized}")
    if any(dim <= 0 for shape in normalized for dim in shape):
        raise ValueError(f"profile contains non-positive dimensions: {normalized}")
    minimum, optimum, maximum = normalized
    if not all(
        minimum[axis] <= optimum[axis] <= maximum[axis]
        for axis in range(len(minimum))
    ):
        raise ValueError(f"profile must satisfy min <= opt <= max: {normalized}")
    return normalized


def validate_shape_in_profile(
    input_shape, profile_shapes
) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in input_shape)
    minimum, _, maximum = normalize_profile_shapes(profile_shapes)
    if len(shape) != len(minimum):
        raise ValueError(
            f"input rank {len(shape)} does not match profile rank {len(minimum)}: {shape}"
        )
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"input shape contains non-positive dimensions: {shape}")
    if any(dim < lo or dim > hi for dim, lo, hi in zip(shape, minimum, maximum)):
        raise ValueError(
            f"input shape is outside engine profile: shape={shape}, min={minimum}, max={maximum}"
        )
    return shape


def profile_metadata(profile_shapes) -> dict[str, list[int]]:
    normalized = normalize_profile_shapes(profile_shapes)
    return {
        key: list(shape)
        for key, shape in zip(("min", "opt", "max"), normalized)
    }


def make_synthetic_batch(batch_size: int) -> np.ndarray:
    """Create deterministic harmonic pseudo-speech without using real audio."""
    time_axis = np.arange(SAMPLES, dtype=np.float32) / SAMPLE_RATE
    waveforms = np.empty((batch_size, 1, SAMPLES), dtype=np.float32)
    for item in range(batch_size):
        rng = np.random.default_rng(20_260_826 + batch_size * 100 + item)
        fundamental = 95.0 + (item * 17) % 95
        vibrato = 1.0 + 0.025 * np.sin(2.0 * np.pi * (4.2 + item % 3) * time_axis)
        phase = 2.0 * np.pi * np.cumsum(fundamental * vibrato) / SAMPLE_RATE
        formants = (520.0 + item % 5 * 35.0, 1_420.0 + item % 7 * 45.0, 2_450.0)
        bandwidths = (90.0, 140.0, 220.0)
        signal = np.zeros(SAMPLES, dtype=np.float32)
        for harmonic in range(1, 41):
            frequency = harmonic * fundamental
            resonance = sum(
                np.exp(-0.5 * ((frequency - formant) / bandwidth) ** 2)
                for formant, bandwidth in zip(formants, bandwidths)
            )
            weight = (0.12 + resonance) / harmonic
            signal += weight * np.sin(harmonic * phase + rng.uniform(0.0, 2.0 * np.pi))
        syllable_envelope = np.clip(
            0.25 + 0.85 * np.sin(2.0 * np.pi * (1.7 + 0.03 * item) * time_axis) ** 2,
            0.0,
            1.0,
        )
        signal *= syllable_envelope
        signal += rng.normal(0.0, 0.012, size=SAMPLES).astype(np.float32)
        signal /= max(float(np.max(np.abs(signal))), 1e-6)
        signal *= 0.35
        signal[: SAMPLE_RATE // 2] = 0.0
        signal[5 * SAMPLE_RATE : 6 * SAMPLE_RATE] *= 0.05
        waveforms[item, 0] = signal.astype(np.float32)
    return waveforms


def generate_references(
    onnx_path: Path,
    artifact_dir: Path,
    batch_sizes: list[int],
    waveform_npy: Path | None,
) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    active_provider = session.get_providers()[0]
    report = {
        "reference_backend": f"ORT {active_provider} FP32",
        "onnxruntime_version": ort.__version__,
        "session_providers": session.get_providers(),
        "batches": {},
    }
    for batch_size in batch_sizes:
        if waveform_npy is None:
            waveforms = make_synthetic_batch(batch_size)
        else:
            base_waveform = np.load(waveform_npy).astype(np.float32)
            if base_waveform.shape != (1, 1, SAMPLES):
                raise ValueError(f"Expected waveform shape (1, 1, {SAMPLES}), got {base_waveform.shape}")
            waveforms = np.repeat(base_waveform, batch_size, axis=0)
        reference = session.run([output_name], {input_name: waveforms})[0]
        artifact_path = artifact_dir / f"ort_fp32_synthetic_{DURATION_LABEL}_bs{batch_size}.npz"
        np.savez(artifact_path, waveforms=waveforms, reference=reference)
        values, counts = np.unique(reference, return_counts=True)
        report["batches"][str(batch_size)] = {
            "artifact": str(artifact_path),
            "output_shape": list(reference.shape),
            "output_value_counts": {str(float(v)): int(c) for v, c in zip(values, counts)},
        }
        print(json.dumps({"batch_size": batch_size, **report["batches"][str(batch_size)]}), flush=True)
    return report


def compare_engines(
    engine_dir: Path | None,
    artifact_dir: Path,
    batch_sizes: list[int],
    precision: str,
    explicit_engine: Path | None = None,
) -> dict:
    import tensorrt as trt
    import torch

    trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "bfloat16"):
        trt_to_torch[trt.bfloat16] = torch.bfloat16
    logger = trt.Logger(trt.Logger.WARNING)
    report = {
        "backend": f"TensorRT {precision}",
        "tensorrt_version": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "batches": {},
    }
    for batch_size in batch_sizes:
        artifact = np.load(
            artifact_dir / f"ort_fp32_synthetic_{DURATION_LABEL}_bs{batch_size}.npz"
        )
        waveforms = artifact["waveforms"]
        reference = artifact["reference"]
        engine_path = resolve_engine_path(
            explicit_engine,
            engine_dir,
            DURATION_LABEL,
            batch_size,
            precision,
        )
        if not engine_path.is_file():
            raise FileNotFoundError(f"Engine does not exist: {engine_path}")
        engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}")
        if engine.num_optimization_profiles != 1:
            raise RuntimeError(
                f"Expected one optimization profile in {engine_path}, got "
                f"{engine.num_optimization_profiles}"
            )
        input_names = [
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        ]
        output_names = [
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        ]
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"Expected one input/output in {engine_path}, got "
                f"inputs={input_names}, outputs={output_names}"
            )
        input_name = input_names[0]
        output_name = output_names[0]
        actual_profile = normalize_profile_shapes(
            engine.get_tensor_profile_shape(input_name, 0)
        )
        input_shape = validate_shape_in_profile(waveforms.shape, actual_profile)
        context = engine.create_execution_context()
        if not context.set_input_shape(input_name, input_shape):
            raise RuntimeError(f"Could not set {input_name} to {input_shape}")
        unresolved = list(context.infer_shapes())
        if unresolved:
            raise RuntimeError(f"Could not infer all shapes for {engine_path}: {unresolved}")
        output_shape = tuple(int(dim) for dim in context.get_tensor_shape(output_name))
        if not output_shape or any(dim <= 0 for dim in output_shape):
            raise RuntimeError(f"Invalid output shape from {engine_path}: {output_shape}")
        tensors: dict[str, torch.Tensor] = {}
        tensors[input_name] = torch.from_numpy(waveforms).to(
            device="cuda", dtype=trt_to_torch[engine.get_tensor_dtype(input_name)]
        )
        tensors[output_name] = torch.empty(
            output_shape,
            device="cuda",
            dtype=trt_to_torch[engine.get_tensor_dtype(output_name)],
        )
        for name, tensor in tensors.items():
            if not context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Could not bind tensor address for {name}")
        stream = torch.cuda.Stream()
        # Tensor creation/copy above happens on PyTorch's current stream,
        # while TensorRT executes on this dedicated stream. Ensure the input
        # copy and output allocation are visible before launching TensorRT.
        torch.cuda.synchronize()
        # One untimed warmup also avoids treating lazy context initialization
        # as part of the parity observation.
        for _ in range(2):
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execution returned false")
            stream.synchronize()
        actual = tensors[output_name].cpu().numpy()
        if actual.shape != reference.shape:
            raise RuntimeError(
                f"Output/reference shape mismatch for {engine_path}: "
                f"actual={actual.shape}, reference={reference.shape}"
            )
        equal = actual == reference
        frame_equal = np.all(equal, axis=-1)
        diff = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
        actual_values, actual_counts = np.unique(actual, return_counts=True)
        reference_values, reference_counts = np.unique(reference, return_counts=True)
        result = {
            "engine": str(engine_path),
            "engine_profile": profile_metadata(actual_profile),
            "input_shape": list(input_shape),
            "output_shape": list(actual.shape),
            "cell_exact_match_pct": round(float(equal.mean()) * 100.0, 6),
            "frame_exact_match_pct": round(float(frame_equal.mean()) * 100.0, 6),
            "mismatched_cells": int((~equal).sum()),
            "mismatched_frames": int((~frame_equal).sum()),
            "max_abs_diff": float(diff.max()),
            "actual_value_counts": {
                str(float(value)): int(count)
                for value, count in zip(actual_values, actual_counts)
            },
            "reference_value_counts": {
                str(float(value)): int(count)
                for value, count in zip(reference_values, reference_counts)
            },
        }
        report["batches"][str(batch_size)] = result
        print(json.dumps({"batch_size": batch_size, **result}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["reference", "compare"], required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument(
        "--engine",
        type=Path,
        help="explicit TensorRT plan for compare mode; overrides legacy --engine-dir lookup",
    )
    parser.add_argument("--waveform-npy", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,4,8,32")
    parser.add_argument("--input-seconds", type=float, default=10.0)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    global SAMPLES, DURATION_LABEL
    SAMPLES = int(round(args.input_seconds * SAMPLE_RATE))
    if SAMPLES <= 0:
        parser.error("--input-seconds must be positive")
    DURATION_LABEL = f"{args.input_seconds:g}s".replace(".", "p")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if not batch_sizes or min(batch_sizes) <= 0:
        parser.error("--batch-sizes must contain positive integers")
    if args.mode == "reference":
        if args.onnx is None:
            parser.error("--onnx is required for reference mode")
        report = generate_references(args.onnx, args.artifact_dir, batch_sizes, args.waveform_npy)
    else:
        if args.engine is None and args.engine_dir is None:
            parser.error("--engine or --engine-dir is required for compare mode")
        if args.engine is not None and not args.engine.is_file():
            parser.error(f"--engine does not exist: {args.engine}")
        report = compare_engines(
            args.engine_dir,
            args.artifact_dir,
            batch_sizes,
            args.precision,
            explicit_engine=args.engine,
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build and benchmark a batch-1 FP16 engine with a dynamic audio duration profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROFILE_KEYS = ("min", "opt", "max")


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_for_seconds(seconds: float, sample_rate: int) -> tuple[int, int, int]:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    samples = int(round(seconds * sample_rate))
    if samples <= 0:
        raise ValueError("seconds and sample_rate must produce at least one sample")
    return (1, 1, samples)


def profile_shapes_for_seconds(
    profile_seconds: Sequence[float], sample_rate: int
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    if len(profile_seconds) != 3:
        raise ValueError("profile must contain min,opt,max")
    shapes = tuple(shape_for_seconds(value, sample_rate) for value in profile_seconds)
    if not all(
        shapes[0][axis] <= shapes[1][axis] <= shapes[2][axis]
        for axis in range(len(shapes[0]))
    ):
        raise ValueError("profile shapes must satisfy min <= opt <= max")
    return shapes


def normalize_profile_shapes(
    shapes: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    normalized = tuple(tuple(int(dim) for dim in shape) for shape in shapes)
    if len(normalized) != 3:
        raise ValueError("engine profile must contain min,opt,max shapes")
    if not normalized[0] or any(dim <= 0 for shape in normalized for dim in shape):
        raise ValueError(f"engine profile contains non-positive dimensions: {normalized}")
    if len({len(shape) for shape in normalized}) != 1:
        raise ValueError(f"engine profile ranks differ: {normalized}")
    return normalized


def validate_profile_shapes(
    requested: Sequence[Sequence[int]], actual: Sequence[Sequence[int]]
) -> None:
    requested_normalized = normalize_profile_shapes(requested)
    actual_normalized = normalize_profile_shapes(actual)
    if requested_normalized != actual_normalized:
        raise ValueError(
            "engine profile does not match requested CLI profile: "
            f"requested={requested_normalized}, actual={actual_normalized}"
        )


def validate_benchmark_shapes(
    benchmark_shapes: Iterable[Sequence[int]], actual_profile: Sequence[Sequence[int]]
) -> None:
    profile = normalize_profile_shapes(actual_profile)
    minimum, _, maximum = profile
    for shape in benchmark_shapes:
        normalized = tuple(int(dim) for dim in shape)
        if len(normalized) != len(minimum):
            raise ValueError(
                f"benchmark shape rank {len(normalized)} does not match profile rank {len(minimum)}: {normalized}"
            )
        if any(dim <= 0 for dim in normalized):
            raise ValueError(f"benchmark shape contains non-positive dimensions: {normalized}")
        if any(dim < lo or dim > hi for dim, lo, hi in zip(normalized, minimum, maximum)):
            raise ValueError(
                f"benchmark shape is outside engine profile: shape={normalized}, min={minimum}, max={maximum}"
            )


def validate_output_shapes(output_shapes: dict[str, Sequence[int]]) -> None:
    if not output_shapes:
        raise ValueError("engine exposes no output shapes")
    for name, shape in output_shapes.items():
        normalized = tuple(int(dim) for dim in shape)
        if not normalized or any(dim <= 0 for dim in normalized):
            raise ValueError(f"output {name!r} has unresolved or non-positive shape {normalized}")


def profile_metadata(
    shapes: Sequence[Sequence[int]], sample_rate: int
) -> dict[str, dict[str, object]]:
    normalized = normalize_profile_shapes(shapes)
    return {
        key: {"shape": list(shape), "seconds": shape[-1] / sample_rate}
        for key, shape in zip(PROFILE_KEYS, normalized)
    }


def file_metadata(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def engine_device_memory_bytes(engine) -> int:
    value = getattr(engine, "device_memory_size_v2", None)
    if value is None:
        value = getattr(engine, "device_memory_size")
    return int(value)


def collect_environment(trt_version: str) -> dict[str, object]:
    import torch

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    fields = [field.strip() for field in gpu.split(",")]
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "gpu": fields[0],
        "gpu_uuid": fields[1],
        "driver_version": fields[2],
        "gpu_memory_total_mib": int(fields[3]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "tensorrt": trt_version,
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    sample_rate: int,
    min_seconds: float,
    opt_seconds: float,
    max_seconds: float,
    workspace_gb: int,
    timing_cache_path: Path,
) -> float:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.builder_optimization_level = 3
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.FP16)
    timing_data = timing_cache_path.read_bytes() if timing_cache_path.is_file() else b""
    config.set_timing_cache(config.create_timing_cache(timing_data), ignore_mismatch=False)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    shape = lambda seconds: shape_for_seconds(seconds, sample_rate)
    profile.set_shape(input_name, shape(min_seconds), shape(opt_seconds), shape(max_seconds))
    config.add_optimization_profile(profile)

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    timing_cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    return elapsed


def benchmark(engine, input_samples: int, input_seconds: float, warmup: int, repeats: int):
    import tensorrt as trt
    import torch

    trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    tensors = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = (1, 1, input_samples)
            if not context.set_input_shape(name, shape):
                raise RuntimeError(f"Could not set {name} to {shape}")
            tensors[name] = torch.zeros(
                shape,
                dtype=trt_to_torch[engine.get_tensor_dtype(name)],
                device="cuda",
            )
    unresolved = list(context.infer_shapes())
    if unresolved:
        raise RuntimeError(f"TensorRT could not infer all tensor shapes: {unresolved}")
    output_shapes = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_shapes[name] = list(context.get_tensor_shape(name))
    validate_output_shapes(output_shapes)
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            tensors[name] = torch.empty(
                tuple(output_shapes[name]),
                dtype=trt_to_torch[engine.get_tensor_dtype(name)],
                device="cuda",
            )
    for name, tensor in tensors.items():
        context.set_tensor_address(name, tensor.data_ptr())
    torch.cuda.synchronize()

    def execute():
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")

    for _ in range(warmup):
        execute()
    stream.synchronize()
    latencies = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        execute()
        end.record(stream)
        end.synchronize()
        latencies.append(float(start.elapsed_time(end)))
    mean_ms = statistics.mean(latencies)
    return {
        "input_seconds": input_seconds,
        "input_samples": input_samples,
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(percentile(latencies, 50), 3),
        "p95_ms": round(percentile(latencies, 95), 3),
        "requests_per_second": round(1000.0 / mean_ms, 2),
        "audio_seconds_per_second": round(input_seconds * 1000.0 / mean_ms, 1),
        "output_shapes": output_shapes,
    }


def main() -> None:
    import tensorrt as trt

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--profile-seconds", default="2,10,16", help="min,opt,max")
    parser.add_argument("--benchmark-seconds", default="2,4,8,10,12,16")
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    try:
        profile = [float(value) for value in args.profile_seconds.split(",")]
        requested_profile_shapes = profile_shapes_for_seconds(profile, args.sample_rate)
        benchmark_seconds = [float(value) for value in args.benchmark_seconds.split(",")]
        benchmark_shapes = [shape_for_seconds(value, args.sample_rate) for value in benchmark_seconds]
        validate_benchmark_shapes(benchmark_shapes, requested_profile_shapes)
    except ValueError as exc:
        parser.error(str(exc))
    if len(profile) != 3 or not 0 < profile[0] <= profile[1] <= profile[2]:
        parser.error("--profile-seconds must be positive min,opt,max")
    timing_cache_path = args.engine.parent / "fp16_dynamic_duration.timing.cache"
    if args.reuse_existing and args.engine.is_file():
        build_seconds = None
    else:
        build_seconds = build_engine(
            args.onnx,
            args.engine,
            args.sample_rate,
            *profile,
            args.workspace_gb,
            timing_cache_path,
        )

    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {args.engine}")
    input_names = [
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
    ]
    if len(input_names) != 1:
        raise RuntimeError(f"Expected exactly one engine input, got {input_names}")
    if engine.num_optimization_profiles != 1:
        raise RuntimeError(
            f"Expected exactly one optimization profile, got {engine.num_optimization_profiles}"
        )
    actual_profile_shapes = normalize_profile_shapes(
        engine.get_tensor_profile_shape(input_names[0], 0)
    )
    validate_profile_shapes(requested_profile_shapes, actual_profile_shapes)
    validate_benchmark_shapes(benchmark_shapes, actual_profile_shapes)
    results = []
    for seconds in benchmark_seconds:
        result = benchmark(
            engine,
            int(round(seconds * args.sample_rate)),
            seconds,
            args.warmup,
            args.repeats,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {
        "backend": "TensorRT FP16 dynamic duration, fixed batch=1",
        "tensorrt_version": trt.__version__,
        "profile_seconds": {"min": profile[0], "opt": profile[1], "max": profile[2]},
        "actual_profile": profile_metadata(actual_profile_shapes, args.sample_rate),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "build_seconds": None if build_seconds is None else round(build_seconds, 3),
        "engine_bytes": args.engine.stat().st_size,
        "engine_device_memory_bytes": engine_device_memory_bytes(engine),
        "onnx": file_metadata(args.onnx),
        "engine": file_metadata(args.engine),
        "environment": collect_environment(trt.__version__),
        "results": results,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(encoded)
    print(encoded, flush=True)


if __name__ == "__main__":
    main()

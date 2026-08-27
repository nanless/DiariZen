#!/usr/bin/env python3
"""Build and benchmark a batch-1 FP16 engine with a dynamic audio duration profile."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import tensorrt as trt


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
    shape = lambda seconds: (1, 1, int(round(seconds * sample_rate)))
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
    output_shapes = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_shapes[name] = list(context.get_tensor_shape(name))
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

    profile = [float(value) for value in args.profile_seconds.split(",")]
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
    results = []
    for seconds in [float(value) for value in args.benchmark_seconds.split(",")]:
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
        "build_seconds": None if build_seconds is None else round(build_seconds, 3),
        "engine_bytes": args.engine.stat().st_size,
        "results": results,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.out_json.write_text(encoded)
    print(encoded, flush=True)


if __name__ == "__main__":
    main()

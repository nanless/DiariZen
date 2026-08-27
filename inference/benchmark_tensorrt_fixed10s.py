#!/usr/bin/env python3
"""Build and benchmark fixed-shape TensorRT precision engines for synthetic audio."""

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
    batch_size: int,
    workspace_gb: int,
    optimization_level: int,
    precision: str,
    timing_cache_path: Path,
    input_samples: int,
    min_batch_size: int | None = None,
    opt_batch_size: int | None = None,
    max_batch_size: int | None = None,
    max_aux_streams: int | None = None,
) -> float:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if precision in {"fp8", "int8", "int4"}:
        # Explicit Q/DQ networks using low-precision types should be strongly
        # typed so TensorRT does not have to infer quantized tensor types.
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.builder_optimization_level = optimization_level
    if max_aux_streams is not None:
        if max_aux_streams < 0:
            raise ValueError("max_aux_streams must be >= 0")
        config.max_aux_streams = max_aux_streams
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision in {"fp8", "int8", "int4"}:
        # Q/DQ and ONNX tensor types determine layer precision. Builder
        # precision flags are invalid for strongly typed networks.
        pass
    elif precision != "fp32_tf32":
        raise ValueError(f"Unsupported precision: {precision}")

    timing_cache_data = timing_cache_path.read_bytes() if timing_cache_path.is_file() else b""
    timing_cache = config.create_timing_cache(timing_cache_data)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    min_shape = (min_batch_size or batch_size, 1, input_samples)
    opt_shape = (opt_batch_size or batch_size, 1, input_samples)
    max_shape = (max_batch_size or batch_size, 1, input_samples)
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    timing_cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    return build_seconds


def benchmark_engine(
    engine_path: Path,
    batch_size: int,
    input_samples: int,
    input_seconds: float,
    warmup: int,
    repeats: int,
) -> dict:
    import torch

    trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {engine_path}")
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    inputs: dict[str, torch.Tensor] = {}
    outputs: dict[str, torch.Tensor] = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            shape = (batch_size, 1, input_samples)
            context.set_input_shape(name, shape)
            tensor = torch.zeros(shape, dtype=trt_to_torch[engine.get_tensor_dtype(name)], device="cuda")
            inputs[name] = tensor

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(context.get_tensor_shape(name))
            tensor = torch.empty(shape, dtype=trt_to_torch[engine.get_tensor_dtype(name)], device="cuda")
            outputs[name] = tensor

    for name, tensor in {**inputs, **outputs}.items():
        context.set_tensor_address(name, tensor.data_ptr())

    # Tensor allocation/initialization uses PyTorch's current stream whereas
    # inference uses a dedicated TensorRT stream below.
    torch.cuda.synchronize()

    def execute() -> None:
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")

    for _ in range(warmup):
        execute()
    stream.synchronize()

    latencies_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        execute()
        end.record(stream)
        end.synchronize()
        latencies_ms.append(float(start.elapsed_time(end)))

    mean_ms = statistics.mean(latencies_ms)
    inspector = engine.create_engine_inspector()
    inspector_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    inspector_path = engine_path.with_suffix(".inspector.json")
    inspector_path.write_text(inspector_text)
    try:
        inspector_json = json.loads(inspector_text)
        layers = inspector_json.get("Layers", []) if isinstance(inspector_json, dict) else inspector_json
    except json.JSONDecodeError:
        layers = []
    precision_tokens = ("fp8", "int8", "int4", "bf16", "half", "float")
    inspector_layer_counts = {
        token: sum(token in json.dumps(layer).lower() for layer in layers)
        for token in precision_tokens
    }
    return {
        "batch_size": batch_size,
        "input_seconds": input_seconds,
        "input_samples": input_samples,
        "warmup": warmup,
        "repeats": repeats,
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(percentile(latencies_ms, 50), 3),
        "p95_ms": round(percentile(latencies_ms, 95), 3),
        "per_item_ms": round(mean_ms / batch_size, 3),
        "audio_seconds_per_second": round(batch_size * input_seconds * 1000.0 / mean_ms, 1),
        "output_shapes": {name: list(tensor.shape) for name, tensor in outputs.items()},
        "engine_bytes": engine_path.stat().st_size,
        "inspector_layer_counts": inspector_layer_counts,
        "inspector_path": str(inspector_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,4,8,32")
    parser.add_argument("--input-seconds", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Build one engine whose batch profile spans min..max --batch-sizes.",
    )
    parser.add_argument(
        "--opt-batch-size",
        type=int,
        default=8,
        help="Optimization point used with --dynamic-batch.",
    )
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument(
        "--precision",
        choices=["fp32_tf32", "fp32", "fp16", "bf16", "fp8", "int8", "int4"],
        default="fp16",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    input_samples = int(round(args.input_seconds * args.sample_rate))
    if input_samples <= 0:
        parser.error("--input-seconds and --sample-rate must produce at least one sample")
    duration_label = f"{args.input_seconds:g}s".replace(".", "p")

    report = {
        "backend": f"TensorRT {args.precision}",
        "tensorrt_version": trt.__version__,
        "input_seconds": args.input_seconds,
        "input_samples": input_samples,
        "results": [],
    }
    timing_cache_path = args.engine_dir / f"{args.precision}.timing.cache"
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if not batch_sizes or min(batch_sizes) <= 0:
        parser.error("--batch-sizes must contain positive integers")
    if args.dynamic_batch and not min(batch_sizes) <= args.opt_batch_size <= max(batch_sizes):
        parser.error("--opt-batch-size must be within the --batch-sizes range")

    if args.dynamic_batch:
        engine_path = args.engine_dir / (
            f"segmentation_{duration_label}_bs{min(batch_sizes)}-{max(batch_sizes)}"
            f"_opt{args.opt_batch_size}_{args.precision}.plan"
        )
        if args.reuse_existing and engine_path.is_file():
            dynamic_build_seconds = None
        else:
            dynamic_build_seconds = build_engine(
                args.onnx,
                engine_path,
                args.opt_batch_size,
                args.workspace_gb,
                args.optimization_level,
                args.precision,
                timing_cache_path,
                input_samples,
                min_batch_size=min(batch_sizes),
                opt_batch_size=args.opt_batch_size,
                max_batch_size=max(batch_sizes),
            )

    for batch_size in batch_sizes:
        if args.dynamic_batch:
            build_seconds = dynamic_build_seconds
        else:
            engine_path = args.engine_dir / f"segmentation_{duration_label}_bs{batch_size}_{args.precision}.plan"
            if args.reuse_existing and engine_path.is_file():
                build_seconds = None
            else:
                build_seconds = build_engine(
                    args.onnx,
                    engine_path,
                    batch_size,
                    args.workspace_gb,
                    args.optimization_level,
                    args.precision,
                    timing_cache_path,
                    input_samples,
                )
        result = benchmark_engine(
            engine_path,
            batch_size,
            input_samples,
            args.input_seconds,
            args.warmup,
            args.repeats,
        )
        # Report the one-time dynamic build only once.
        reported_build_seconds = build_seconds if not args.dynamic_batch or batch_size == batch_sizes[0] else None
        result["build_seconds"] = (
            None if reported_build_seconds is None else round(reported_build_seconds, 3)
        )
        result["reused_existing_engine"] = build_seconds is None
        result["dynamic_batch_profile"] = (
            {
                "min": min(batch_sizes),
                "opt": args.opt_batch_size,
                "max": max(batch_sizes),
            }
            if args.dynamic_batch
            else None
        )
        report["results"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

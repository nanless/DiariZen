#!/usr/bin/env python3
"""Benchmark a burst of independent requests against one fixed-batch TensorRT engine."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.bfloat16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def make_worker(engine: trt.ICudaEngine, input_samples: int):
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
                dtype=TRT_TO_TORCH[engine.get_tensor_dtype(name)],
                device="cuda",
            )
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            tensors[name] = torch.empty(
                tuple(context.get_tensor_shape(name)),
                dtype=TRT_TO_TORCH[engine.get_tensor_dtype(name)],
                device="cuda",
            )
    for name, tensor in tensors.items():
        context.set_tensor_address(name, tensor.data_ptr())
    return context, stream, tensors


def run_burst(workers, requests: int) -> list[float]:
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    completion_events = []
    for request_index in range(requests):
        context, stream, _ = workers[request_index % len(workers)]
        if request_index < len(workers):
            stream.wait_event(start)
        with torch.cuda.stream(stream):
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            done = torch.cuda.Event(enable_timing=True)
            done.record(stream)
            completion_events.append(done)
    for event in completion_events:
        event.synchronize()
    return [float(start.elapsed_time(event)) for event in completion_events]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--input-seconds", type=float, default=16.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--streams", default="1,2,4,8")
    parser.add_argument("--warmup-bursts", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {args.engine}")
    input_samples = int(round(args.input_seconds * args.sample_rate))
    report = {
        "engine": str(args.engine),
        "input_seconds": args.input_seconds,
        "requests_per_burst": args.requests,
        "results": [],
    }
    for stream_count in [int(value) for value in args.streams.split(",")]:
        workers = [make_worker(engine, input_samples) for _ in range(stream_count)]
        torch.cuda.synchronize()
        for _ in range(args.warmup_bursts):
            run_burst(workers, args.requests)

        burst_maxima = []
        aggregate_p50 = []
        aggregate_p95 = []
        for _ in range(args.repeats):
            completion_ms = run_burst(workers, args.requests)
            burst_maxima.append(max(completion_ms))
            aggregate_p50.append(percentile(completion_ms, 50))
            aggregate_p95.append(percentile(completion_ms, 95))
        result = {
            "streams": stream_count,
            "burst_mean_ms": round(statistics.mean(burst_maxima), 3),
            "burst_p95_ms": round(percentile(burst_maxima, 95), 3),
            "request_completion_p50_mean_ms": round(statistics.mean(aggregate_p50), 3),
            "request_completion_p95_mean_ms": round(statistics.mean(aggregate_p95), 3),
            "requests_per_second": round(args.requests * 1000.0 / statistics.mean(burst_maxima), 2),
        }
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        result["device_memory_used_mib"] = round((total_bytes - free_bytes) / (1024**2), 1)
        report["results"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del workers
        torch.cuda.empty_cache()

    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out_json:
        args.out_json.write_text(encoded)
    print(encoded, flush=True)


if __name__ == "__main__":
    main()

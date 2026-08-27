#!/usr/bin/env python3
"""Benchmark multi-worker TensorRT serving with representative mixed-duration bursts.

Each request executes exactly one segmentation forward pass. Requests are never
windowed or split. Each persistent worker owns one CUDA stream and one execution
context plus fixed-address pinned/device buffers for every configured route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
LOCK_CONTRACT = "/tmp/diarizen_l4_benchmark.lock"
ROUTE_KEYS = ("short_dynamic", "mid_dynamic", "fixed10", "fixed16", "long_dynamic")
REQUIRED_ROUTE_KEYS = ("short_dynamic", "mid_dynamic", "fixed10", "long_dynamic")
MANIFEST_DEFINITIONS = {
    "A": ((1.0, 4), (2.0, 6), (3.0, 8), (4.5, 8), (5.0, 7), (6.0, 5),
          (8.0, 5), (10.0, 3), (12.0, 2), (18.0, 1), (22.0, 1)),
    "B": ((1.0, 4), (2.0, 6), (3.0, 8), (4.5, 8), (5.0, 7), (6.0, 5),
          (8.0, 4), (10.0, 3), (12.0, 2), (18.0, 1), (22.0, 1), (26.0, 1)),
}


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    duration_seconds: float
    padded_seconds: float
    route: str
    estimated_gpu_ms: float


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * q / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def expand_manifest(name: str) -> list[float]:
    try:
        definition = MANIFEST_DEFINITIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown manifest {name!r}") from exc
    durations = [duration for duration, count in definition for _ in range(count)]
    if len(durations) != 50:
        raise ValueError(f"manifest {name} must contain 50 requests, got {len(durations)}")
    return durations


def manifest_stats(durations: Sequence[float]) -> dict[str, object]:
    if not durations:
        raise ValueError("manifest must not be empty")
    values = [float(value) for value in durations]
    above_16 = sum(value > 16.0 for value in values)
    return {
        "requests": len(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": percentile(values, 95.0),
        "max_seconds": max(values),
        "above_16_requests": above_16,
        "above_16_percent": above_16 * 100.0 / len(values),
        "duration_counts": {f"{key:g}": count for key, count in sorted(Counter(values).items())},
    }


def validate_worker_count(value: int) -> int:
    worker_count = int(value)
    if worker_count < 1:
        raise ValueError("workers must be a positive integer")
    return worker_count


def route_duration(
    duration_seconds: float,
    available_routes: Iterable[str],
    max_seconds: float = 30.0,
) -> tuple[str, float]:
    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")
    if duration > max_seconds:
        raise ValueError(
            f"duration {duration:g}s exceeds max_seconds={max_seconds:g}s; "
            "use an overflow single-forward fallback instead of routing to the <=max engine"
        )
    routes = set(available_routes)
    missing = set(REQUIRED_ROUTE_KEYS) - routes
    if missing:
        raise ValueError(f"missing required routes: {sorted(missing)}")
    if duration <= 7.0:
        return "short_dynamic", max(2.0, duration)
    if duration <= 8.0:
        return "mid_dynamic", duration
    if duration <= 10.0:
        return "fixed10", 10.0
    if duration <= 12.0:
        return "mid_dynamic", duration
    if duration <= 16.0:
        if "fixed16" in routes:
            return "fixed16", 16.0
        return "mid_dynamic", duration
    return "long_dynamic", duration


def canonical_duration(value: float) -> str:
    return f"{float(value):g}"


def default_estimated_gpu_ms(padded_seconds: float) -> float:
    """Conservative monotonic fallback; measured values should be supplied when available."""
    return 2.0 + 0.04 * padded_seconds * padded_seconds


def build_requests(
    manifest_name: str,
    available_routes: Iterable[str],
    estimate_by_duration: Mapping[str, float] | None = None,
    max_seconds: float = 30.0,
) -> list[RequestSpec]:
    estimates = estimate_by_duration or {}
    requests = []
    for index, duration in enumerate(expand_manifest(manifest_name)):
        route, padded = route_duration(duration, available_routes, max_seconds=max_seconds)
        estimate = float(
            estimates.get(canonical_duration(duration), default_estimated_gpu_ms(padded))
        )
        if not math.isfinite(estimate) or estimate <= 0:
            raise ValueError(f"invalid estimated GPU ms for {duration:g}s: {estimate}")
        requests.append(
            RequestSpec(
                request_id=f"{manifest_name}-{index:02d}",
                duration_seconds=duration,
                padded_seconds=padded,
                route=route,
                estimated_gpu_ms=estimate,
            )
        )
    return requests


def greedy_lpt_assign(
    requests: Sequence[RequestSpec], worker_count: int
) -> tuple[list[list[RequestSpec]], list[float]]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    assignments = [[] for _ in range(worker_count)]
    loads = [0.0 for _ in range(worker_count)]
    ordered = sorted(requests, key=lambda item: (-item.estimated_gpu_ms, item.request_id))
    for request in ordered:
        worker_index = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker_index].append(request)
        loads[worker_index] += request.estimated_gpu_ms
    return assignments, loads


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_object(value: str | None) -> dict[str, object]:
    if not value:
        return {}
    if value.lstrip().startswith("{"):
        encoded = value
    else:
        candidate = Path(value)
        encoded = candidate.read_text() if candidate.is_file() else value
    parsed = json.loads(encoded)
    if not isinstance(parsed, dict):
        raise ValueError("JSON value must be an object")
    return parsed


def normalize_profile_shapes(shapes: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(int(dim) for dim in shape) for shape in shapes)
    if len(normalized) != 3 or len({len(shape) for shape in normalized}) != 1:
        raise ValueError(f"invalid min/opt/max profile shapes: {normalized}")
    if any(dim <= 0 for shape in normalized for dim in shape):
        raise ValueError(f"profile has non-positive dimensions: {normalized}")
    return normalized


def profile_contains_samples(profile: Sequence[Sequence[int]], samples: int) -> bool:
    minimum, _, maximum = normalize_profile_shapes(profile)
    shape = (1, 1, int(samples))
    return all(lo <= dim <= hi for dim, lo, hi in zip(shape, minimum, maximum))


def summarize_samples(values: Sequence[float], prefix: str) -> dict[str, float]:
    if not values:
        raise ValueError(f"{prefix} samples must not be empty")
    return {
        f"{prefix}_mean_ms": statistics.mean(values),
        f"{prefix}_p50_ms": percentile(values, 50.0),
        f"{prefix}_p95_ms": percentile(values, 95.0),
    }


class EngineBundle:
    def __init__(self, path: Path, trt_module):
        self.path = path
        self.trt = trt_module
        self.logger = trt_module.Logger(trt_module.Logger.WARNING)
        self.runtime = trt_module.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {path}")
        if self.engine.num_optimization_profiles != 1:
            raise RuntimeError(
                f"{path} must contain exactly one optimization profile, got "
                f"{self.engine.num_optimization_profiles}"
            )
        self.input_names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt_module.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt_module.TensorIOMode.OUTPUT
        ]
        if len(self.input_names) != 1 or len(self.output_names) != 1:
            raise RuntimeError(
                f"{path} must expose one input and one output, got "
                f"inputs={self.input_names}, outputs={self.output_names}"
            )
        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]
        self.profile = normalize_profile_shapes(
            self.engine.get_tensor_profile_shape(self.input_name, 0)
        )
        if any(len(shape) != 3 or shape[:2] != (1, 1) for shape in self.profile):
            raise RuntimeError(f"{path} is not batch=1 mono: {self.profile}")

    @property
    def device_memory_bytes(self) -> int:
        value = getattr(self.engine, "device_memory_size_v2", None)
        if value is None:
            value = self.engine.device_memory_size
        return int(value)

    def metadata(self, sample_rate: int) -> dict[str, object]:
        return {
            "path": str(self.path),
            "bytes": self.path.stat().st_size,
            "sha256": sha256_file(self.path),
            "input_name": self.input_name,
            "output_name": self.output_name,
            "profile": {
                key: {"shape": list(shape), "seconds": shape[-1] / sample_rate}
                for key, shape in zip(("min", "opt", "max"), self.profile)
            },
            "device_memory_bytes_per_context": self.device_memory_bytes,
        }


def torch_dtype_for_trt(dtype, trt_module, torch_module):
    mapping = {
        trt_module.float32: torch_module.float32,
        trt_module.float16: torch_module.float16,
        trt_module.bfloat16: torch_module.bfloat16,
        trt_module.int32: torch_module.int32,
        trt_module.int8: torch_module.int8,
        trt_module.bool: torch_module.bool,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported TensorRT dtype {dtype}") from exc


class RouteRuntime:
    def __init__(
        self,
        route: str,
        bundle: EngineBundle,
        stream,
        torch_module,
        trt_module,
        seed: int,
    ):
        self.route = route
        self.bundle = bundle
        self.stream = stream
        self.torch = torch_module
        self.trt = trt_module
        self.context = bundle.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"could not create context for route {route}")
        self.max_input_shape = tuple(bundle.profile[2])
        if not self.context.set_input_shape(bundle.input_name, self.max_input_shape):
            raise RuntimeError(f"could not set max input shape for route {route}")
        unresolved = list(self.context.infer_shapes())
        if unresolved:
            raise RuntimeError(f"unresolved shapes for route {route}: {unresolved}")
        self.max_output_shape = tuple(
            int(dim) for dim in self.context.get_tensor_shape(bundle.output_name)
        )
        if not self.max_output_shape or any(dim <= 0 for dim in self.max_output_shape):
            raise RuntimeError(
                f"invalid max output shape for route {route}: {self.max_output_shape}"
            )
        input_dtype = torch_dtype_for_trt(
            bundle.engine.get_tensor_dtype(bundle.input_name), trt_module, torch_module
        )
        output_dtype = torch_dtype_for_trt(
            bundle.engine.get_tensor_dtype(bundle.output_name), trt_module, torch_module
        )
        self.host_input = torch_module.empty(
            self.max_input_shape, dtype=input_dtype, pin_memory=True
        )
        generator = torch_module.Generator(device="cpu")
        generator.manual_seed(seed)
        self.host_input.normal_(mean=0.0, std=0.01, generator=generator)
        self.device_input = torch_module.empty(
            self.max_input_shape, dtype=input_dtype, device="cuda"
        )
        self.host_output = torch_module.empty(
            self.max_output_shape, dtype=output_dtype, pin_memory=True
        )
        self.device_output = torch_module.empty(
            self.max_output_shape, dtype=output_dtype, device="cuda"
        )
        if not self.context.set_tensor_address(bundle.input_name, self.device_input.data_ptr()):
            raise RuntimeError(f"could not bind input address for route {route}")
        if not self.context.set_tensor_address(bundle.output_name, self.device_output.data_ptr()):
            raise RuntimeError(f"could not bind output address for route {route}")

    def enqueue_request(self, request: RequestSpec, sample_rate: int):
        samples = int(round(request.padded_seconds * sample_rate))
        if not profile_contains_samples(self.bundle.profile, samples):
            raise RuntimeError(
                f"request {request.request_id} shape {(1, 1, samples)} is outside "
                f"route {self.route} profile {self.bundle.profile}"
            )
        # TensorRT requires all work using the context to finish before its
        # dynamic shape is changed. The second worker remains independent.
        self.stream.synchronize()
        shape = (1, 1, samples)
        if not self.context.set_input_shape(self.bundle.input_name, shape):
            raise RuntimeError(f"could not set {self.route} input shape to {shape}")
        unresolved = list(self.context.infer_shapes())
        if unresolved:
            raise RuntimeError(
                f"could not infer {self.route} shapes for {request.request_id}: {unresolved}"
            )
        output_shape = tuple(
            int(dim) for dim in self.context.get_tensor_shape(self.bundle.output_name)
        )
        if (
            not output_shape
            or len(output_shape) != len(self.max_output_shape)
            or any(dim <= 0 for dim in output_shape)
            or any(dim > maximum for dim, maximum in zip(output_shape, self.max_output_shape))
        ):
            raise RuntimeError(
                f"invalid output shape for {request.request_id}: {output_shape}, "
                f"max={self.max_output_shape}"
            )
        input_elements = math.prod(shape)
        output_elements = math.prod(output_shape)
        with self.torch.cuda.stream(self.stream):
            self.device_input.view(-1)[:input_elements].copy_(
                self.host_input.view(-1)[:input_elements], non_blocking=True
            )
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError(f"TensorRT enqueue failed for {request.request_id}")
            self.host_output.view(-1)[:output_elements].copy_(
                self.device_output.view(-1)[:output_elements], non_blocking=True
            )
            done = self.torch.cuda.Event(enable_timing=True)
            done.record(self.stream)
        return done


class WorkerRuntime:
    def __init__(
        self,
        worker_id: int,
        route_bundles: Mapping[str, EngineBundle],
        torch_module,
        trt_module,
        seed: int,
    ):
        self.worker_id = worker_id
        self.torch = torch_module
        self.stream = torch_module.cuda.Stream()
        self.routes = {
            route: RouteRuntime(
                route,
                bundle,
                self.stream,
                torch_module,
                trt_module,
                seed + worker_id * 100 + route_index,
            )
            for route_index, (route, bundle) in enumerate(route_bundles.items())
        }

    def run_wave(
        self, requests: Sequence[RequestSpec], start_event, sample_rate: int
    ) -> list[tuple[str, object]]:
        self.stream.wait_event(start_event)
        completions = []
        for request in requests:
            done = self.routes[request.route].enqueue_request(request, sample_rate)
            completions.append((request.request_id, done))
        self.stream.synchronize()
        return completions


class PersistentWorkerThread:
    def __init__(self, runtime: WorkerRuntime, sample_rate: int):
        self.runtime = runtime
        self.sample_rate = sample_rate
        self.tasks: queue.Queue = queue.Queue()
        self.thread = threading.Thread(
            target=self._loop, name=f"trt-worker-{runtime.worker_id}", daemon=True
        )
        self.thread.start()

    def _loop(self) -> None:
        while True:
            task = self.tasks.get()
            if task is None:
                return
            start_event, requests, future = task
            try:
                future.set_result(
                    self.runtime.run_wave(requests, start_event, self.sample_rate)
                )
            except BaseException as exc:
                future.set_exception(exc)

    def submit(self, start_event, requests: Sequence[RequestSpec]) -> Future:
        future = Future()
        self.tasks.put((start_event, requests, future))
        return future

    def close(self) -> None:
        self.tasks.put(None)
        self.thread.join()


def gpu_snapshot(torch_module, label: str) -> dict[str, object]:
    free_bytes, total_bytes = torch_module.cuda.mem_get_info()
    gpu_line = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    return {
        "label": label,
        "cuda_free_mib": free_bytes / 2**20,
        "cuda_total_mib": total_bytes / 2**20,
        "torch_allocated_mib": torch_module.cuda.memory_allocated() / 2**20,
        "torch_reserved_mib": torch_module.cuda.memory_reserved() / 2**20,
        "nvidia_smi": gpu_line,
    }


def preflight_gpu_idle() -> dict[str, str]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    if processes:
        raise RuntimeError(
            f"GPU has compute processes before benchmark; acquire {LOCK_CONTRACT}: {processes}"
        )
    return {"gpu": gpu, "compute_processes": processes}


def run_one_burst(
    manifest_name: str,
    assignments: Sequence[Sequence[RequestSpec]],
    workers: Sequence[PersistentWorkerThread],
    torch_module,
) -> dict[str, object]:
    if len(assignments) != len(workers):
        raise ValueError(
            f"assignment/worker count mismatch: assignments={len(assignments)}, "
            f"workers={len(workers)}"
        )
    start_event = torch_module.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    futures = [
        worker.submit(start_event, worker_requests)
        for worker, worker_requests in zip(workers, assignments)
    ]
    completions = [item for future in futures for item in future.result()]
    host_wall_ms = (time.perf_counter() - started) * 1000.0
    completion_ms = [float(start_event.elapsed_time(event)) for _, event in completions]
    return {
        "manifest": manifest_name,
        "burst_ms": max(completion_ms),
        "host_wall_ms": host_wall_ms,
        "request_completion_ms": completion_ms,
    }


def summarize_manifest_runs(
    manifest_name: str,
    durations: Sequence[float],
    runs: Sequence[dict[str, object]],
    assignment_loads: Sequence[float],
    assignments: Sequence[Sequence[RequestSpec]],
) -> dict[str, object]:
    burst = [float(run["burst_ms"]) for run in runs]
    host = [float(run["host_wall_ms"]) for run in runs]
    completions = [
        float(value) for run in runs for value in run["request_completion_ms"]
    ]
    return {
        "manifest": manifest_name,
        "manifest_stats": manifest_stats(durations),
        "measured_bursts": len(runs),
        "measured_requests": len(completions),
        **summarize_samples(burst, "burst"),
        **summarize_samples(host, "host_wall"),
        "request_completion_p50_ms": percentile(completions, 50.0),
        "request_completion_p95_ms": percentile(completions, 95.0),
        "requests_per_second": len(durations) * 1000.0 / statistics.mean(burst),
        "burst_samples_ms": burst,
        "worker_estimated_gpu_ms": list(assignment_loads),
        "worker_assignments": [
            [asdict(request) for request in worker_requests]
            for worker_requests in assignments
        ],
    }


def summarize_aggregate(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    burst = [float(run["burst_ms"]) for run in runs]
    host = [float(run["host_wall_ms"]) for run in runs]
    completions = [
        float(value) for run in runs for value in run["request_completion_ms"]
    ]
    return {
        "bursts": len(runs),
        "requests": len(completions),
        "requests_per_second": len(completions) * 1000.0 / sum(burst),
        **summarize_samples(burst, "burst"),
        **summarize_samples(host, "host_wall"),
        "request_completion_p50_ms": percentile(completions, 50.0),
        "request_completion_p95_ms": percentile(completions, 95.0),
    }


def resolve_engine_map(args, parser: argparse.ArgumentParser) -> dict[str, Path]:
    try:
        raw_map = parse_json_object(args.engine_map_json)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"invalid --engine-map-json: {exc}")
    mapping = {key: Path(str(value)) for key, value in raw_map.items() if key in ROUTE_KEYS}
    explicit = {
        "short_dynamic": args.short_dynamic_engine,
        "mid_dynamic": args.mid_dynamic_engine,
        "fixed10": args.fixed10_engine,
        "fixed16": args.fixed16_engine,
        "long_dynamic": args.long_dynamic_engine,
    }
    mapping.update({key: value for key, value in explicit.items() if value is not None})
    missing = set(REQUIRED_ROUTE_KEYS) - set(mapping)
    if missing:
        parser.error(f"missing engine routes: {sorted(missing)}")
    for route, path in mapping.items():
        if not path.is_file():
            parser.error(f"{route} engine does not exist: {path}")
    return {key: mapping[key] for key in ROUTE_KEYS if key in mapping}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine-map-json",
        help="JSON object or path mapping route names to engine paths; explicit args override",
    )
    parser.add_argument("--short-dynamic-engine", type=Path)
    parser.add_argument("--mid-dynamic-engine", type=Path)
    parser.add_argument("--fixed10-engine", type=Path)
    parser.add_argument("--fixed16-engine", type=Path)
    parser.add_argument("--long-dynamic-engine", type=Path)
    parser.add_argument(
        "--estimate-ms-json",
        help="optional JSON object/path mapping duration strings to estimated GPU milliseconds",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--warmup-waves", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        worker_count = validate_worker_count(args.workers)
    except ValueError as exc:
        parser.error(str(exc))
    if (
        args.sample_rate <= 0
        or not math.isfinite(args.max_seconds)
        or args.max_seconds <= 0
        or args.warmup_waves < 1
        or args.repeats < 2
    ):
        parser.error(
            "sample-rate/max-seconds must be positive, warmup-waves >=1, and repeats >=2"
        )
    engine_paths = resolve_engine_map(args, parser)
    try:
        estimate_raw = parse_json_object(args.estimate_ms_json)
        estimate_map = {str(key): float(value) for key, value in estimate_raw.items()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(f"invalid --estimate-ms-json: {exc}")
    manifests = {name: expand_manifest(name) for name in ("A", "B")}
    try:
        requests = {
            name: build_requests(
                name,
                engine_paths,
                estimate_map,
                max_seconds=args.max_seconds,
            )
            for name in manifests
        }
    except ValueError as exc:
        parser.error(str(exc))

    # Delayed imports keep helper tests independent of TensorRT and CUDA.
    import tensorrt as trt
    import torch

    preflight = preflight_gpu_idle()
    torch.cuda.set_device(0)
    snapshots = [gpu_snapshot(torch, "before_engine_deserialize")]
    bundles_by_path: dict[Path, EngineBundle] = {}
    route_bundles = {}
    for route, path in engine_paths.items():
        resolved = path.resolve()
        if resolved not in bundles_by_path:
            bundles_by_path[resolved] = EngineBundle(resolved, trt)
        route_bundles[route] = bundles_by_path[resolved]
    snapshots.append(gpu_snapshot(torch, "after_engine_deserialize"))

    for request_list in requests.values():
        for request in request_list:
            samples = int(round(request.padded_seconds * args.sample_rate))
            bundle = route_bundles[request.route]
            if not profile_contains_samples(bundle.profile, samples):
                parser.error(
                    f"{request.request_id} ({request.padded_seconds:g}s) is outside "
                    f"{request.route} profile {bundle.profile}"
                )
    assignments = {}
    assignment_loads = {}
    for name, request_list in requests.items():
        assignments[name], assignment_loads[name] = greedy_lpt_assign(
            request_list, worker_count
        )

    worker_runtimes = [
        WorkerRuntime(index, route_bundles, torch, trt, args.seed)
        for index in range(worker_count)
    ]
    workers = [
        PersistentWorkerThread(runtime, args.sample_rate) for runtime in worker_runtimes
    ]
    snapshots.append(gpu_snapshot(torch, "after_worker_route_preallocation"))
    measured = {"A": [], "B": []}
    all_runs = []
    try:
        for wave in range(args.warmup_waves):
            order = ("A", "B") if wave % 2 == 0 else ("B", "A")
            for name in order:
                run_one_burst(name, assignments[name], workers, torch)
        snapshots.append(gpu_snapshot(torch, "after_warmup"))
        for repeat in range(args.repeats):
            order = ("A", "B") if repeat % 2 == 0 else ("B", "A")
            for name in order:
                result = run_one_burst(name, assignments[name], workers, torch)
                measured[name].append(result)
                all_runs.append(result)
        snapshots.append(gpu_snapshot(torch, "after_measurement"))
    finally:
        for worker in workers:
            worker.close()

    report = {
        "scope": (
            "mixed-duration batch=1; exactly one segmentation forward per request; "
            f"no windowing; {worker_count} persistent worker threads and independent CUDA streams"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_lock_contract": LOCK_CONTRACT,
        "sample_rate": args.sample_rate,
        "max_seconds": args.max_seconds,
        "worker_count": worker_count,
        "warmup_waves": args.warmup_waves,
        "repeats_per_manifest": args.repeats,
        "measurement_order": "A/B forward-reverse alternating by warmup wave and measured round",
        "assignment": "LPT descending estimated GPU ms, greedily assigned to lower cumulative worker load",
        "estimate_source": (
            "--estimate-ms-json with quadratic GPU-ms fallback"
            if estimate_map
            else "quadratic GPU-ms heuristic: 2 + 0.04 * padded_seconds^2"
        ),
        "preflight": preflight,
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensorrt": trt.__version__,
        },
        "route_engines": {
            route: bundle.metadata(args.sample_rate)
            for route, bundle in route_bundles.items()
        },
        "memory_snapshots": snapshots,
        "results": {
            name: summarize_manifest_runs(
                name,
                manifests[name],
                measured[name],
                assignment_loads[name],
                assignments[name],
            )
            for name in ("A", "B")
        },
        "aggregate_requests": summarize_aggregate(all_runs),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

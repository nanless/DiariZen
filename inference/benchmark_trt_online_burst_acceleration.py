#!/usr/bin/env python3
"""Compare TensorRT enqueue and CUDA Graph for a 50-request 16s burst."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from inference.benchmark_l4_runtime_acceleration import TensorRTRunner, percentile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_burst(workers, graphs, requests: int, mode: str) -> list[float]:
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    completion_events = []
    for request_index in range(requests):
        worker_index = request_index % len(workers)
        worker = workers[worker_index]
        graph = graphs[worker_index]
        if request_index < len(workers):
            worker.stream.wait_event(start)
        use_graph = mode.startswith("cudagraph") or (
            mode.startswith("hybrid") and worker_index == 0
        )
        include_transfers = mode.endswith("pinned_e2e")
        if include_transfers:
            with torch.cuda.stream(worker.stream):
                worker.input_tensor.copy_(worker.host_input, non_blocking=True)
        if use_graph:
            graph.replay()
        else:
            worker.enqueue()
        with torch.cuda.stream(worker.stream):
            if include_transfers:
                worker.host_output.copy_(worker.output_tensor, non_blocking=True)
            done = torch.cuda.Event(enable_timing=True)
            done.record(worker.stream)
        completion_events.append(done)
    for event in completion_events:
        event.synchronize()
    return [float(start.elapsed_time(event)) for event in completion_events]


def benchmark_combinations(
    engine: Path,
    stream_counts: list[int],
    requests: int,
    warmup_bursts: int,
    repeats: int,
    modes: list[str],
) -> list[dict]:
    groups = {}
    first_runner = TensorRTRunner(engine)
    shared_engine = first_runner.engine
    pending_runners = [first_runner]
    for streams in stream_counts:
        workers = []
        while pending_runners and len(workers) < streams:
            workers.append(pending_runners.pop())
        workers.extend(
            TensorRTRunner(engine, shared_engine=shared_engine)
            for _ in range(streams - len(workers))
        )
        graphs = [worker.capture_inference_graph() for worker in workers]
        groups[streams] = {
            "workers": workers,
            "graphs": graphs,
            "parity": [worker.parity(graph) for worker, graph in zip(workers, graphs)],
            "graph_nodes": [graph.node_count for graph in graphs],
        }
    torch.cuda.synchronize()
    combinations = [(streams, mode) for streams in stream_counts for mode in modes]
    for _ in range(warmup_bursts):
        for streams, mode in combinations:
            group = groups[streams]
            run_burst(group["workers"], group["graphs"], requests, mode)
    measurements = {
        combination: {"burst": [], "request_p50": [], "request_p95": []}
        for combination in combinations
    }
    for repeat in range(repeats):
        # Reverse every other round so later modes do not systematically see a
        # different clock, temperature, or power state.
        order = combinations if repeat % 2 == 0 else list(reversed(combinations))
        for streams, mode in order:
            group = groups[streams]
            completion = run_burst(group["workers"], group["graphs"], requests, mode)
            sample = measurements[(streams, mode)]
            sample["burst"].append(max(completion))
            sample["request_p50"].append(percentile(completion, 50.0))
            sample["request_p95"].append(percentile(completion, 95.0))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    results = []
    for streams, mode in combinations:
        sample = measurements[(streams, mode)]
        mean_burst = statistics.mean(sample["burst"])
        group = groups[streams]
        results.append(
            {
                "mode": mode,
                "streams": streams,
                "requests": requests,
                "burst_mean_ms": round(mean_burst, 6),
                "burst_p50_ms": round(percentile(sample["burst"], 50.0), 6),
                "burst_p95_ms": round(percentile(sample["burst"], 95.0), 6),
                "request_completion_p50_mean_ms": round(
                    statistics.mean(sample["request_p50"]), 6
                ),
                "request_completion_p95_mean_ms": round(
                    statistics.mean(sample["request_p95"]), 6
                ),
                "requests_per_second": round(requests * 1000.0 / mean_burst, 3),
                "graph_node_counts": group["graph_nodes"],
                "graph_parity": group["parity"],
                "device_memory_used_mib_all_groups": round(
                    (total_bytes - free_bytes) / (1024**2), 1
                ),
            }
        )
    for group in groups.values():
        for graph in group["graphs"]:
            graph.close()
    del groups
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--streams", default="1,2")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--warmup-bursts", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--modes",
        default="enqueue_device,cudagraph_device,enqueue_pinned_e2e,cudagraph_pinned_e2e",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if not args.engine.is_file():
        parser.error(f"engine does not exist: {args.engine}")
    if args.requests < 1 or args.warmup_bursts < 1 or args.repeats < 2:
        parser.error("requests/warmup-bursts must be >= 1 and repeats must be >= 2")
    valid_modes = {
        "enqueue_device",
        "cudagraph_device",
        "enqueue_pinned_e2e",
        "cudagraph_pinned_e2e",
        "hybrid_device",
        "hybrid_pinned_e2e",
    }
    modes = args.modes.split(",")
    if not set(modes) <= valid_modes:
        parser.error(f"--modes must be a subset of {sorted(valid_modes)}")
    report = {
        "scope": "fixed 16-second synthetic input, batch=1, 50-request burst",
        "engine": str(args.engine),
        "engine_metadata": {
            "bytes": args.engine.stat().st_size,
            "sha256": sha256_file(args.engine),
        },
        "input_shape": [1, 1, 256000],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensorrt": trt.__version__,
        },
        "benchmark_lock_contract": "/tmp/diarizen_l4_benchmark.lock",
        "warmup_bursts": args.warmup_bursts,
        "repeats": args.repeats,
        "results": [],
    }
    stream_counts = [int(value) for value in args.streams.split(",")]
    report["method"] = (
        "all workers/graphs allocated once; combinations measured in forward/reverse "
        "alternating order to reduce thermal and clock-order bias"
    )
    report["device_memory_note"] = (
        "device_memory_used_mib_all_groups is process-wide memory after allocating every "
        "requested stream-count group, not per-context memory"
    )
    report["results"] = benchmark_combinations(
        args.engine,
        stream_counts,
        args.requests,
        args.warmup_bursts,
        args.repeats,
        modes,
    )
    for result in report["results"]:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark a full fixed-16s TensorRT CUDA Graph serving pipeline.

CUDA Runtime is called directly through ctypes: ``torch.cuda.CUDAGraph`` is
deliberately unused.  This is a benchmark/reference implementation; a service
must give every concurrent slot its own context, buffers, stream, and graph.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorrt as trt


CUDA_SUCCESS = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_STREAM_NON_BLOCKING = 1
CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
CUDA_HOST_ALLOC_DEFAULT = 0


class CudaRuntime:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so.12")
        self._bind("cudaGetErrorString", [ctypes.c_int], ctypes.c_char_p)
        v = ctypes.c_int()
        self._bind("cudaRuntimeGetVersion", [ctypes.POINTER(ctypes.c_int)])
        self.check(self.lib.cudaRuntimeGetVersion(ctypes.byref(v)), "cudaRuntimeGetVersion")
        self.runtime_version = v.value

        self._bind("cudaMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t])
        self._bind("cudaFree", [ctypes.c_void_p])
        self._bind(
            "cudaHostAlloc",
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
        )
        self._bind("cudaFreeHost", [ctypes.c_void_p])
        self._bind(
            "cudaMemcpyAsync",
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
        )
        self._bind(
            "cudaStreamCreateWithFlags",
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
        )
        self._bind("cudaStreamDestroy", [ctypes.c_void_p])
        self._bind("cudaStreamSynchronize", [ctypes.c_void_p])
        self._bind(
            "cudaStreamBeginCapture", [ctypes.c_void_p, ctypes.c_int]
        )
        self._bind(
            "cudaStreamEndCapture", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        )
        self._bind("cudaGraphGetNodes", [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)])
        self._bind("cudaGraphNodeGetType", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)])
        self._bind(
            "cudaGraphInstantiateWithFlags",
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_ulonglong],
        )
        self._bind("cudaGraphLaunch", [ctypes.c_void_p, ctypes.c_void_p])
        self._bind("cudaGraphExecDestroy", [ctypes.c_void_p])
        self._bind("cudaGraphDestroy", [ctypes.c_void_p])
        self._bind("cudaEventCreate", [ctypes.POINTER(ctypes.c_void_p)])
        self._bind("cudaEventDestroy", [ctypes.c_void_p])
        self._bind("cudaEventRecord", [ctypes.c_void_p, ctypes.c_void_p])
        self._bind("cudaEventSynchronize", [ctypes.c_void_p])
        self._bind(
            "cudaEventElapsedTime",
            [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p],
        )

    def _bind(self, name: str, argtypes: list, restype=ctypes.c_int) -> None:
        fn = getattr(self.lib, name)
        fn.argtypes = argtypes
        fn.restype = restype

    def check(self, code: int, where: str) -> None:
        if code != CUDA_SUCCESS:
            try:
                message = self.lib.cudaGetErrorString(code).decode()
            except Exception:
                message = "unknown CUDA error"
            raise RuntimeError(f"{where}: CUDA error {code}: {message}")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr

    def host_alloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(
            self.lib.cudaHostAlloc(ctypes.byref(ptr), nbytes, CUDA_HOST_ALLOC_DEFAULT),
            "cudaHostAlloc",
        )
        return ptr

    def stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self.check(
            self.lib.cudaStreamCreateWithFlags(ctypes.byref(stream), CUDA_STREAM_NON_BLOCKING),
            "cudaStreamCreateWithFlags",
        )
        return stream

    def event(self) -> ctypes.c_void_p:
        event = ctypes.c_void_p()
        self.check(self.lib.cudaEventCreate(ctypes.byref(event)), "cudaEventCreate")
        return event


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def parity(actual: np.ndarray, reference: np.ndarray) -> dict:
    diff = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
    return {
        "shape": list(actual.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "exact_match_pct": float((actual == reference).mean() * 100.0),
        "mismatched_cells": int(np.count_nonzero(actual != reference)),
        "actual_value_counts": {
            str(float(k)): int(v) for k, v in zip(*np.unique(actual, return_counts=True))
        },
        "reference_value_counts": {
            str(float(k)): int(v) for k, v in zip(*np.unique(reference, return_counts=True))
        },
    }


GPU_OWNER_HOST_PID: int | None = None


def gpu_snapshot(label: str, establish_owner: bool = False) -> dict:
    global GPU_OWNER_HOST_PID
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.used,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    proc_text = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    rows = []
    for line in proc_text.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        rows.append({"pid": int(fields[0]), "name": fields[1], "memory_mib": fields[2]})
    # nvidia-smi runs in the host PID namespace while this Python process may
    # run in a container PID namespace.  The lock launcher verifies that there
    # are zero GPU processes immediately before Python starts.  Therefore the
    # sole process appearing after TensorRT context creation is this runner's
    # host PID; pin it once and reject every other PID thereafter.
    if establish_owner:
        if len(rows) != 1:
            raise RuntimeError(
                f"Expected exactly one GPU process while establishing runner owner, got {rows}"
            )
        GPU_OWNER_HOST_PID = rows[0]["pid"]
    external = [row for row in rows if row["pid"] != GPU_OWNER_HOST_PID]
    snap = {
        "label": label,
        "utc": utc_now(),
        "gpu": query,
        "compute_processes": rows,
        "runner_host_gpu_pid": GPU_OWNER_HOST_PID,
        "external_processes": external,
    }
    print("GPU_SNAPSHOT " + json.dumps(snap, ensure_ascii=False), flush=True)
    if external:
        raise RuntimeError(f"External GPU processes found before {label}: {external}")
    return snap


def host_array(ptr: ctypes.c_void_p, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    if dtype == np.float32:
        ctype = ctypes.c_float
    elif dtype == np.float16:
        ctype = ctypes.c_uint16
    else:
        raise TypeError(dtype)
    raw = (ctype * n).from_address(ptr.value)
    return np.ctypeslib.as_array(raw).view(dtype).reshape(shape)


def graph_node_types(cuda: CudaRuntime, graph: ctypes.c_void_p) -> tuple[int, dict[str, int]]:
    count = ctypes.c_size_t()
    cuda.check(cuda.lib.cudaGraphGetNodes(graph, None, ctypes.byref(count)), "cudaGraphGetNodes(count)")
    nodes = (ctypes.c_void_p * count.value)()
    cuda.check(
        cuda.lib.cudaGraphGetNodes(graph, ctypes.cast(nodes, ctypes.c_void_p), ctypes.byref(count)),
        "cudaGraphGetNodes(nodes)",
    )
    names = {
        0: "kernel",
        1: "memcpy",
        2: "memset",
        3: "host",
        4: "child_graph",
        5: "empty",
        6: "event_wait",
        7: "event_record",
        8: "ext_sem_signal",
        9: "ext_sem_wait",
        10: "mem_alloc",
        11: "mem_free",
        12: "conditional",
    }
    result: Counter[str] = Counter()
    for i in range(count.value):
        kind = ctypes.c_int()
        cuda.check(cuda.lib.cudaGraphNodeGetType(nodes[i], ctypes.byref(kind)), "cudaGraphNodeGetType")
        result[names.get(kind.value, f"unknown_{kind.value}")] += 1
    return int(count.value), dict(result)


def timed_gpu_iteration(cuda: CudaRuntime, stream, submit) -> tuple[float, float, float]:
    start = cuda.event()
    end = cuda.event()
    try:
        cuda.check(cuda.lib.cudaEventRecord(start, stream), "cudaEventRecord(start)")
        wall_start = time.perf_counter_ns()
        enqueue_start = time.perf_counter_ns()
        submit()
        enqueue_end = time.perf_counter_ns()
        cuda.check(cuda.lib.cudaEventRecord(end, stream), "cudaEventRecord(end)")
        cuda.check(cuda.lib.cudaEventSynchronize(end), "cudaEventSynchronize(end)")
        wall_end = time.perf_counter_ns()
        elapsed = ctypes.c_float()
        cuda.check(
            cuda.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start, end),
            "cudaEventElapsedTime",
        )
        return (
            float(elapsed.value),
            (enqueue_end - enqueue_start) / 1e6,
            (wall_end - wall_start) / 1e6,
        )
    finally:
        cuda.check(cuda.lib.cudaEventDestroy(start), "cudaEventDestroy(start)")
        cuda.check(cuda.lib.cudaEventDestroy(end), "cudaEventDestroy(end)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        parser.error("--warmup must be >= 1 and --repeats must be >= 2")
    for path in (args.engine, args.artifact):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")

    started_utc = utc_now()
    cuda = CudaRuntime()
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT engine deserialization failed")
    context = engine.create_execution_context()
    artifact = np.load(args.artifact)
    waveforms = np.ascontiguousarray(artifact["waveforms"], dtype=np.float32)
    reference = np.ascontiguousarray(artifact["reference"], dtype=np.float32)
    expected_shape = (1, 1, 256000)
    if waveforms.shape != expected_shape:
        raise ValueError(f"Expected {expected_shape}, got {waveforms.shape}")

    input_name = next(
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
    )
    output_name = next(
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
    )
    if not context.set_input_shape(input_name, waveforms.shape):
        raise RuntimeError("set_input_shape failed")
    output_shape = tuple(context.get_tensor_shape(output_name))
    if output_shape != reference.shape:
        raise ValueError(f"Engine output {output_shape} != reference {reference.shape}")
    if engine.get_tensor_dtype(input_name) != trt.float32 or engine.get_tensor_dtype(output_name) != trt.float32:
        raise TypeError("This runner expects float32 engine I/O")

    in_nbytes = waveforms.nbytes
    out_nbytes = reference.nbytes
    h_in = cuda.host_alloc(in_nbytes)
    h_out = cuda.host_alloc(out_nbytes)
    pinned_in = host_array(h_in, waveforms.shape, np.float32)
    pinned_out = host_array(h_out, output_shape, np.float32)
    np.copyto(pinned_in, waveforms)
    pageable_out = np.empty(output_shape, dtype=np.float32)
    d_in = cuda.malloc(in_nbytes)
    d_out = cuda.malloc(out_nbytes)
    stream = cuda.stream()
    context.set_tensor_address(input_name, d_in.value)
    context.set_tensor_address(output_name, d_out.value)

    snapshots = [gpu_snapshot("before_runner_warmup", establish_owner=True)]

    def execute() -> None:
        if not context.execute_async_v3(stream.value):
            raise RuntimeError("execute_async_v3 returned false")

    def pinned_pipeline() -> None:
        cuda.check(
            cuda.lib.cudaMemcpyAsync(d_in, h_in, in_nbytes, CUDA_MEMCPY_HOST_TO_DEVICE, stream),
            "cudaMemcpyAsync(H2D,pinned)",
        )
        execute()
        cuda.check(
            cuda.lib.cudaMemcpyAsync(h_out, d_out, out_nbytes, CUDA_MEMCPY_DEVICE_TO_HOST, stream),
            "cudaMemcpyAsync(D2H,pinned)",
        )

    def pageable_pipeline() -> None:
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                d_in,
                ctypes.c_void_p(waveforms.ctypes.data),
                in_nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            ),
            "cudaMemcpyAsync(H2D,pageable)",
        )
        execute()
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                ctypes.c_void_p(pageable_out.ctypes.data),
                d_out,
                out_nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            ),
            "cudaMemcpyAsync(D2H,pageable)",
        )

    for _ in range(args.warmup):
        pinned_pipeline()
    cuda.check(cuda.lib.cudaStreamSynchronize(stream), "warmup synchronize")

    results: dict[str, dict] = {}

    # 1. Fixed engine, fixed preallocated device buffers, inference only.
    snapshots.append(gpu_snapshot("before_preallocated_compute_only"))
    gpu_ms: list[float] = []
    host_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(args.repeats):
        a, b, c = timed_gpu_iteration(cuda, stream, execute)
        gpu_ms.append(a); host_ms.append(b); wall_ms.append(c)
    results["preallocated_compute_only"] = {
        "gpu_latency": summarize(gpu_ms),
        "host_enqueue": summarize(host_ms),
        "wall_latency": summarize(wall_ms),
        "audio_seconds_per_second": 16000.0 / statistics.mean(gpu_ms),
    }

    # 2. No device preallocation: cudaMalloc/cudaFree on every request.
    snapshots.append(gpu_snapshot("before_per_request_device_alloc_pinned_async"))
    gpu_ms = []; host_ms = []; wall_ms = []
    for _ in range(args.repeats):
        wall_start = time.perf_counter_ns()
        local_d_in = cuda.malloc(in_nbytes)
        local_d_out = cuda.malloc(out_nbytes)
        context.set_tensor_address(input_name, local_d_in.value)
        context.set_tensor_address(output_name, local_d_out.value)

        def one_request() -> None:
            cuda.check(cuda.lib.cudaMemcpyAsync(local_d_in, h_in, in_nbytes, CUDA_MEMCPY_HOST_TO_DEVICE, stream), "alloc variant H2D")
            execute()
            cuda.check(cuda.lib.cudaMemcpyAsync(h_out, local_d_out, out_nbytes, CUDA_MEMCPY_DEVICE_TO_HOST, stream), "alloc variant D2H")

        a, b, _ = timed_gpu_iteration(cuda, stream, one_request)
        cuda.check(cuda.lib.cudaFree(local_d_in), "cudaFree(local input)")
        cuda.check(cuda.lib.cudaFree(local_d_out), "cudaFree(local output)")
        wall_end = time.perf_counter_ns()
        gpu_ms.append(a); host_ms.append(b); wall_ms.append((wall_end - wall_start) / 1e6)
    context.set_tensor_address(input_name, d_in.value)
    context.set_tensor_address(output_name, d_out.value)
    results["per_request_device_alloc_pinned_async_e2e"] = {
        "gpu_pipeline_latency": summarize(gpu_ms),
        "host_pipeline_enqueue_excludes_alloc_free": summarize(host_ms),
        "wall_latency_includes_alloc_free": summarize(wall_ms),
        "parity": parity(pinned_out.copy(), reference),
    }

    # 3. Device preallocation, but ordinary pageable NumPy host buffers.
    snapshots.append(gpu_snapshot("before_preallocated_pageable_async_e2e"))
    gpu_ms = []; host_ms = []; wall_ms = []
    for _ in range(args.repeats):
        a, b, c = timed_gpu_iteration(cuda, stream, pageable_pipeline)
        gpu_ms.append(a); host_ms.append(b); wall_ms.append(c)
    results["preallocated_pageable_async_e2e"] = {
        "gpu_pipeline_latency": summarize(gpu_ms),
        "host_enqueue": summarize(host_ms),
        "wall_latency": summarize(wall_ms),
        "parity": parity(pageable_out.copy(), reference),
    }

    # 4. Device preallocation plus explicitly pinned async H2D/D2H.
    snapshots.append(gpu_snapshot("before_preallocated_pinned_async_e2e"))
    gpu_ms = []; host_ms = []; wall_ms = []
    for _ in range(args.repeats):
        a, b, c = timed_gpu_iteration(cuda, stream, pinned_pipeline)
        gpu_ms.append(a); host_ms.append(b); wall_ms.append(c)
    results["preallocated_pinned_async_e2e"] = {
        "gpu_pipeline_latency": summarize(gpu_ms),
        "host_enqueue": summarize(host_ms),
        "wall_latency": summarize(wall_ms),
        "parity": parity(pinned_out.copy(), reference),
    }

    # 5. Capture the exact pinned H2D -> TensorRT -> pinned D2H pipeline.
    snapshots.append(gpu_snapshot("before_cuda_runtime_graph_capture"))
    cuda.check(
        cuda.lib.cudaStreamBeginCapture(stream, CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL),
        "cudaStreamBeginCapture",
    )
    pinned_pipeline()
    graph = ctypes.c_void_p()
    cuda.check(cuda.lib.cudaStreamEndCapture(stream, ctypes.byref(graph)), "cudaStreamEndCapture")
    node_count, node_types = graph_node_types(cuda, graph)
    if node_count <= 0:
        raise RuntimeError("CUDA Runtime captured an empty graph")
    graph_exec = ctypes.c_void_p()
    cuda.check(
        cuda.lib.cudaGraphInstantiateWithFlags(ctypes.byref(graph_exec), graph, 0),
        "cudaGraphInstantiateWithFlags",
    )

    def graph_launch() -> None:
        cuda.check(cuda.lib.cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch")

    for _ in range(args.warmup):
        graph_launch()
    cuda.check(cuda.lib.cudaStreamSynchronize(stream), "graph warmup synchronize")
    snapshots.append(gpu_snapshot("before_cuda_runtime_graph_benchmark"))
    gpu_ms = []; host_ms = []; wall_ms = []
    for _ in range(args.repeats):
        a, b, c = timed_gpu_iteration(cuda, stream, graph_launch)
        gpu_ms.append(a); host_ms.append(b); wall_ms.append(c)
    results["cuda_runtime_graph_pinned_async_e2e"] = {
        "capture_api": "cudaStreamBeginCapture/cudaStreamEndCapture + cudaGraphInstantiateWithFlags + cudaGraphLaunch",
        "torch_cuda_cudagraph_used": False,
        "graph_node_count": node_count,
        "graph_node_type_counts": node_types,
        "gpu_pipeline_latency": summarize(gpu_ms),
        "host_enqueue": summarize(host_ms),
        "wall_latency": summarize(wall_ms),
        "parity": parity(pinned_out.copy(), reference),
    }

    cuda.check(cuda.lib.cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy")
    cuda.check(cuda.lib.cudaGraphDestroy(graph), "cudaGraphDestroy")
    snapshots.append(gpu_snapshot("after_all_benchmarks"))

    report = {
        "scope": "fixed16s synthetic only; batch=1; shape=[1,1,256000]",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "pid": os.getpid(),
        "flock_required_by_launcher": "/tmp/diarizen_l4_benchmark.lock",
        "versions": {
            "tensorrt": trt.__version__,
            "cuda_runtime_integer": cuda.runtime_version,
        },
        "engine": {
            "path": str(args.engine),
            "bytes": args.engine.stat().st_size,
            "sha256": sha256(args.engine),
            "input_name": input_name,
            "input_shape": list(waveforms.shape),
            "output_name": output_name,
            "output_shape": list(output_shape),
        },
        "artifact": {
            "path": str(args.artifact),
            "bytes": args.artifact.stat().st_size,
            "sha256": sha256(args.artifact),
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "snapshots": snapshots,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("FINAL_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)

    cuda.check(cuda.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")
    cuda.check(cuda.lib.cudaFree(d_in), "cudaFree(input)")
    cuda.check(cuda.lib.cudaFree(d_out), "cudaFree(output)")
    cuda.check(cuda.lib.cudaFreeHost(h_in), "cudaFreeHost(input)")
    cuda.check(cuda.lib.cudaFreeHost(h_out), "cudaFreeHost(output)")


if __name__ == "__main__":
    main()

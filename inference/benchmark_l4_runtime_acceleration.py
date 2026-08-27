#!/usr/bin/env python3
"""Benchmark production-style TensorRT execution for fixed 16-second audio.

The benchmark separates device execution, host API enqueue overhead, and an
end-to-end path with preallocated pinned host buffers.  It intentionally uses
batch=1 and fixed ``[1, 1, 256000]`` inputs to match the L4 serving decision.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable


INPUT_SECONDS = 16.0
SAMPLE_RATE = 16_000
INPUT_SAMPLES = int(INPUT_SECONDS * SAMPLE_RATE)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_latencies(values_ms: list[float], input_seconds: float) -> dict:
    if not values_ms:
        raise ValueError("latencies must not be empty")
    if input_seconds <= 0.0 or not math.isfinite(input_seconds):
        raise ValueError("input_seconds must be positive finite")
    if any(value <= 0.0 or not math.isfinite(value) for value in values_ms):
        raise ValueError("latencies must be positive finite")
    mean_ms = statistics.mean(values_ms)
    return {
        "samples": len(values_ms),
        "mean_ms": round(mean_ms, 6),
        "p50_ms": round(percentile(values_ms, 50.0), 6),
        "p95_ms": round(percentile(values_ms, 95.0), 6),
        "min_ms": round(min(values_ms), 6),
        "max_ms": round(max(values_ms), 6),
        "stdev_ms": round(statistics.pstdev(values_ms), 6),
        "requests_per_second": round(1000.0 / mean_ms, 3),
        "audio_seconds_per_second": round(input_seconds * 1000.0 / mean_ms, 3),
    }


def decide_benefit(baseline: dict, candidate: dict, minimum_pct: float = 1.0) -> dict:
    mean_speedup = float(baseline["mean_ms"]) / float(candidate["mean_ms"])
    p95_speedup = float(baseline["p95_ms"]) / float(candidate["p95_ms"])
    threshold = 1.0 + minimum_pct / 100.0
    accepted = mean_speedup >= threshold and p95_speedup >= threshold
    if accepted:
        reason = f"mean and p95 improve by at least {minimum_pct:g}%"
    elif mean_speedup < threshold and p95_speedup < threshold:
        reason = "mean and p95 improvements are below the acceptance threshold"
    elif mean_speedup < threshold:
        reason = "mean improvement is below the acceptance threshold"
    else:
        reason = "p95 improvement is below the acceptance threshold"
    return {
        "accepted": accepted,
        "minimum_improvement_pct": minimum_pct,
        "mean_speedup": mean_speedup,
        "p95_speedup": p95_speedup,
        "reason": reason,
    }


class TensorRTRunner:
    def __init__(self, engine_path: Path, shared_engine=None) -> None:
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = shared_engine
        if self.engine is None:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        trt_to_torch = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = (1, 1, INPUT_SAMPLES)
                if not self.context.set_input_shape(name, shape):
                    raise RuntimeError(f"TensorRT rejected input shape {shape} for {name}")
                self.inputs[name] = torch.empty(
                    shape, dtype=trt_to_torch[self.engine.get_tensor_dtype(name)], device="cuda"
                )
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(f"Unresolved output shape for {name}: {shape}")
                self.outputs[name] = torch.empty(
                    shape, dtype=trt_to_torch[self.engine.get_tensor_dtype(name)], device="cuda"
                )
        for name, tensor in {**self.inputs, **self.outputs}.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT rejected address for {name}")
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise RuntimeError(
                f"Expected one input and one output, got {list(self.inputs)} and {list(self.outputs)}"
            )
        self.input_name, self.input_tensor = next(iter(self.inputs.items()))
        self.output_name, self.output_tensor = next(iter(self.outputs.items()))
        self.host_input = torch.empty_like(self.input_tensor, device="cpu", pin_memory=True)
        self.host_output = torch.empty_like(self.output_tensor, device="cpu", pin_memory=True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260827)
        self.host_input.normal_(mean=0.0, std=0.05, generator=generator).clamp_(-1.0, 1.0)
        with torch.cuda.stream(self.stream):
            self.input_tensor.copy_(self.host_input, non_blocking=True)
        self.stream.synchronize()

    def enqueue(self) -> None:
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")

    def capture_inference_graph(self):
        for _ in range(5):
            self.enqueue()
        self.stream.synchronize()
        graph = CudaRuntimeGraph(self.torch, self.stream.cuda_stream)
        graph.begin_capture()
        self.enqueue()
        graph.end_capture()
        self.stream.synchronize()
        return graph

    def pinned_e2e(self, inference: Callable[[], None]) -> None:
        torch = self.torch
        with torch.cuda.stream(self.stream):
            self.input_tensor.copy_(self.host_input, non_blocking=True)
        inference()
        with torch.cuda.stream(self.stream):
            self.host_output.copy_(self.output_tensor, non_blocking=True)

    def measure(
        self,
        operation: Callable[[], None],
        warmup: int,
        repeats: int,
    ) -> dict:
        torch = self.torch
        for _ in range(warmup):
            operation()
        self.stream.synchronize()
        gpu_ms: list[float] = []
        wall_ms: list[float] = []
        api_us: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_started = time.perf_counter_ns()
            start.record(self.stream)
            api_started = time.perf_counter_ns()
            operation()
            api_us.append((time.perf_counter_ns() - api_started) / 1_000.0)
            end.record(self.stream)
            end.synchronize()
            wall_ms.append((time.perf_counter_ns() - wall_started) / 1_000_000.0)
            gpu_ms.append(float(start.elapsed_time(end)))
        return {
            "gpu": summarize_latencies(gpu_ms, INPUT_SECONDS),
            "wall": summarize_latencies(wall_ms, INPUT_SECONDS),
            "host_api_us": {
                "samples": len(api_us),
                "mean_us": round(statistics.mean(api_us), 6),
                "p50_us": round(percentile(api_us, 50.0), 6),
                "p95_us": round(percentile(api_us, 95.0), 6),
            },
        }

    def parity(self, graph) -> dict:
        self.enqueue()
        self.stream.synchronize()
        enqueue_output = self.output_tensor.detach().clone()
        graph.replay()
        self.stream.synchronize()
        graph_output = self.output_tensor.detach().clone()
        difference = (enqueue_output.float() - graph_output.float()).abs()
        return {
            "exact": bool(self.torch.equal(enqueue_output, graph_output)),
            "max_abs_diff": float(difference.max().item()),
            "mean_abs_diff": float(difference.mean().item()),
            "output_shape": list(graph_output.shape),
            "output_dtype": str(graph_output.dtype),
        }


class CudaRuntimeGraph:
    """Capture external TensorRT launches with the CUDA Runtime API.

    ``torch.cuda.CUDAGraph`` only tracks work issued through PyTorch and can
    produce an apparently successful but empty graph for external runtimes.
    Direct stream capture makes the TensorRT kernels part of the graph and the
    node-count guard prevents accepting an empty capture.
    """

    def __init__(self, torch_module, stream_handle: int) -> None:
        self.stream = ctypes.c_void_p(stream_handle)
        site_packages = Path(torch_module.__file__).resolve().parents[1]
        candidates = sorted(
            (site_packages / "nvidia" / "cuda_runtime" / "lib").glob("libcudart.so*")
        )
        if not candidates:
            raise RuntimeError("Could not locate libcudart in the TensorRT conda environment")
        self.lib = ctypes.CDLL(str(candidates[-1]))
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.cudaStreamBeginCapture.restype = ctypes.c_int
        self.lib.cudaStreamEndCapture.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.cudaStreamEndCapture.restype = ctypes.c_int
        self.lib.cudaGraphGetNodes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.cudaGraphGetNodes.restype = ctypes.c_int
        self.lib.cudaGraphInstantiateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ]
        self.lib.cudaGraphInstantiateWithFlags.restype = ctypes.c_int
        self.lib.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaGraphLaunch.restype = ctypes.c_int
        self.lib.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphExecDestroy.restype = ctypes.c_int
        self.lib.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphDestroy.restype = ctypes.c_int
        self.graph = ctypes.c_void_p()
        self.instance = ctypes.c_void_p()
        self.node_count = 0

    def _check(self, code: int, operation: str) -> None:
        if code != 0:
            message = self.lib.cudaGetErrorString(code)
            detail = message.decode("utf-8", errors="replace") if message else "unknown"
            raise RuntimeError(f"{operation} failed with CUDA error {code}: {detail}")

    def begin_capture(self) -> None:
        self._check(
            self.lib.cudaStreamBeginCapture(self.stream, 0),
            "cudaStreamBeginCapture",
        )

    def end_capture(self) -> None:
        self._check(
            self.lib.cudaStreamEndCapture(self.stream, ctypes.byref(self.graph)),
            "cudaStreamEndCapture",
        )
        node_count = ctypes.c_size_t()
        self._check(
            self.lib.cudaGraphGetNodes(self.graph, None, ctypes.byref(node_count)),
            "cudaGraphGetNodes",
        )
        self.node_count = int(node_count.value)
        if self.node_count == 0:
            raise RuntimeError("CUDA graph capture produced zero nodes")
        self._check(
            self.lib.cudaGraphInstantiateWithFlags(
                ctypes.byref(self.instance), self.graph, ctypes.c_ulonglong(0)
            ),
            "cudaGraphInstantiateWithFlags",
        )

    def replay(self) -> None:
        self._check(self.lib.cudaGraphLaunch(self.instance, self.stream), "cudaGraphLaunch")

    def close(self) -> None:
        if self.instance.value:
            self.lib.cudaGraphExecDestroy(self.instance)
            self.instance = ctypes.c_void_p()
        if self.graph.value:
            self.lib.cudaGraphDestroy(self.graph)
            self.graph = ctypes.c_void_p()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--minimum-improvement-pct", type=float, default=1.0)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        parser.error("--warmup must be >= 1 and --repeats must be >= 2")
    if not args.engine.is_file():
        parser.error(f"engine does not exist: {args.engine}")

    import tensorrt as trt
    import torch

    runner = TensorRTRunner(args.engine)
    report = {
        "scope": "fixed 16-second synthetic input only",
        "input_shape": [1, 1, INPUT_SAMPLES],
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "engine": str(args.engine),
        "engine_bytes": args.engine.stat().st_size,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensorrt": trt.__version__,
        },
        "results": {},
    }
    report["results"]["enqueue_device"] = runner.measure(
        runner.enqueue, args.warmup, args.repeats
    )
    report["results"]["enqueue_pinned_e2e"] = runner.measure(
        lambda: runner.pinned_e2e(runner.enqueue), args.warmup, args.repeats
    )
    try:
        graph = runner.capture_inference_graph()
        report["cuda_graph_capture"] = {
            "success": True,
            "implementation": "CUDA Runtime stream capture",
            "node_count": graph.node_count,
        }
        report["cuda_graph_parity"] = runner.parity(graph)
        report["results"]["cudagraph_device"] = runner.measure(
            graph.replay, args.warmup, args.repeats
        )
        report["results"]["cudagraph_pinned_e2e"] = runner.measure(
            lambda: runner.pinned_e2e(graph.replay), args.warmup, args.repeats
        )
        for metric in ("gpu", "wall"):
            report.setdefault("decisions", {})[f"cudagraph_device_{metric}"] = decide_benefit(
                report["results"]["enqueue_device"][metric],
                report["results"]["cudagraph_device"][metric],
                args.minimum_improvement_pct,
            )
            report["decisions"][f"cudagraph_pinned_e2e_{metric}"] = decide_benefit(
                report["results"]["enqueue_pinned_e2e"][metric],
                report["results"]["cudagraph_pinned_e2e"][metric],
                args.minimum_improvement_pct,
            )
    except Exception as error:
        report["cuda_graph_capture"] = {
            "success": False,
            "error": f"{type(error).__name__}: {error}",
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

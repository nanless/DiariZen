#!/usr/bin/env python3
"""Benchmark ORT CUDA I/O Binding and CUDA Graph on fixed 16-second input."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
import torch


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / (
    "inference/models/kaldi_merged_1219_all_ft_large/"
    "epoch_0016_multilabel_hard.onnx"
)
SHAPE = (1, 1, 256000)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def stats_ms(values_s: list[float]) -> dict[str, float]:
    mean_s = statistics.mean(values_s)
    return {
        "mean_ms": mean_s * 1000.0,
        "p50_ms": percentile(values_s, 50) * 1000.0,
        "p95_ms": percentile(values_s, 95) * 1000.0,
        "min_ms": min(values_s) * 1000.0,
        "max_ms": max(values_s) * 1000.0,
        "audio_sec_per_s": 16.0 / mean_s,
        "rtf": mean_s / 16.0,
    }


def error_stats(ref: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    delta = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    return {
        "shape": list(got.shape),
        "dtype": str(got.dtype),
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "exact_equal": bool(np.array_equal(ref, got)),
        "mismatch_cells": int(np.count_nonzero(ref != got)),
    }


def gpu_snapshot(label: str) -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ]
    value = subprocess.check_output(cmd, text=True).strip()
    print(f"GPU_SNAPSHOT {label}: {value}", flush=True)
    return value


def make_session(
    model: Path, *, cuda_graph: bool, profiling: bool = False
) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 2
    if profiling:
        so.enable_profiling = True
        so.profile_file_prefix = "/tmp/diarizen_ort_fixed16s_profile"
    cuda_options = {
        "device_id": "0",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": "1",
        "enable_cuda_graph": "1" if cuda_graph else "0",
    }
    return ort.InferenceSession(
        model.as_posix(),
        sess_options=so,
        providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
    )


def time_call(
    fn: Callable[[], None], *, synchronize: bool, warmup: int, repeats: int
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    if synchronize:
        torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        if synchronize:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if synchronize:
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
    return stats_ms(timings)


def bind_fixed_gpu(
    sess: ort.InferenceSession,
    x_gpu: torch.Tensor,
    y_gpu: torch.Tensor,
) -> ort.IOBinding:
    io = sess.io_binding()
    io.bind_input(
        name=sess.get_inputs()[0].name,
        device_type="cuda",
        device_id=0,
        element_type=np.float32,
        shape=tuple(x_gpu.shape),
        buffer_ptr=x_gpu.data_ptr(),
    )
    io.bind_output(
        name=sess.get_outputs()[0].name,
        device_type="cuda",
        device_id=0,
        element_type=np.float32,
        shape=tuple(y_gpu.shape),
        buffer_ptr=y_gpu.data_ptr(),
    )
    return io


def node_assignment_diagnostic(model: Path, x_np: np.ndarray) -> dict[str, Any]:
    sess = make_session(model, cuda_graph=False, profiling=True)
    sess.run(None, {sess.get_inputs()[0].name: x_np})
    profile_path = Path(sess.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    provider_counts: dict[str, int] = {}
    node_names: dict[str, list[str]] = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = str(event.get("args", {}).get("provider", "unknown"))
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        node_names.setdefault(provider, []).append(str(event.get("name", "")))
    return {
        "profile_path": profile_path.as_posix(),
        "provider_event_counts": provider_counts,
        "sample_node_names": {k: v[:20] for k, v in node_names.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument(
        "--artifact-npz",
        type=Path,
        help="Optionally save generated waveforms and ORT FP32 reference for TensorRT parity.",
    )
    args = parser.parse_args()
    if not args.onnx.is_file():
        parser.error(f"ONNX model does not exist: {args.onnx}")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("--warmup must be >= 1 and --repeats must be >= 2")
    if args.noise_std <= 0:
        parser.error("--noise-std must be positive")

    gpu_start = gpu_snapshot("before_sessions")
    assert torch.cuda.is_available(), "torch CUDA unavailable"

    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_start": gpu_start,
        "model": args.onnx.as_posix(),
        "input_shape": list(SHAPE),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "noise_std": args.noise_std,
    }
    report: dict[str, Any] = {"environment": env, "results": {}, "diagnostics": {}, "failures": {}}

    rng = np.random.default_rng(args.seed)
    x_np = rng.standard_normal(SHAPE, dtype=np.float32) * np.float32(args.noise_std)

    # 1. session.run baseline: host NumPy input and host NumPy output.
    sess = make_session(args.onnx, cuda_graph=False)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    y_ref = sess.run([output_name], {input_name: x_np})[0]
    if args.artifact_npz:
        args.artifact_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.artifact_npz, waveforms=x_np, reference=y_ref)
        report["environment"]["artifact_npz"] = args.artifact_npz.as_posix()
    gpu_snapshot("before_session_run")
    report["results"]["session_run_host_io"] = {
        **time_call(
            lambda: sess.run([output_name], {input_name: x_np}),
            synchronize=True,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "io": "host numpy input -> CUDA EP -> host numpy output",
        "provider_options": sess.get_provider_options(),
        "numeric_vs_session_run": error_stats(y_ref, y_ref),
    }

    # 2. I/O Binding with persistent device input/output buffers.
    x_gpu = torch.from_numpy(x_np).to("cuda")
    y_gpu = torch.empty(y_ref.shape, device="cuda", dtype=torch.float32)
    io = bind_fixed_gpu(sess, x_gpu, y_gpu)
    sess.run_with_iobinding(io)
    torch.cuda.synchronize()
    y_iob = y_gpu.cpu().numpy().copy()
    gpu_snapshot("before_iobinding_fixed_gpu")
    report["results"]["iobinding_fixed_gpu_buffers"] = {
        **time_call(
            lambda: sess.run_with_iobinding(io),
            synchronize=True,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "io": "persistent torch CUDA input/output; no H2D or D2H in timed region",
        "input_ptr": int(x_gpu.data_ptr()),
        "output_ptr": int(y_gpu.data_ptr()),
        "numeric_vs_session_run": error_stats(y_ref, y_iob),
    }

    # 3. CUDA Graph session + the same fixed-address I/O Binding contract.
    try:
        sess_graph = make_session(args.onnx, cuda_graph=True)
        graph_options = sess_graph.get_provider_options()
        y_graph_gpu = torch.empty(y_ref.shape, device="cuda", dtype=torch.float32)
        io_graph = bind_fixed_gpu(sess_graph, x_gpu, y_graph_gpu)
        # First run captures; subsequent runs replay. Warmup is intentionally outside timing.
        sess_graph.run_with_iobinding(io_graph)
        torch.cuda.synchronize()
        y_graph = y_graph_gpu.cpu().numpy().copy()
        gpu_snapshot("before_cuda_graph_iobinding")
        report["results"]["cuda_graph_iobinding_fixed_gpu_buffers"] = {
            **time_call(
                lambda: sess_graph.run_with_iobinding(io_graph),
                synchronize=True,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "io": "enable_cuda_graph=1; persistent CUDA input/output; capture excluded from timing",
            "provider_options": graph_options,
            "input_ptr": int(x_gpu.data_ptr()),
            "output_ptr": int(y_graph_gpu.data_ptr()),
            "numeric_vs_session_run": error_stats(y_ref, y_graph),
        }
    except Exception as exc:
        report["failures"]["cuda_graph_iobinding_fixed_gpu_buffers"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    try:
        report["diagnostics"]["node_assignment"] = node_assignment_diagnostic(
            args.onnx, x_np
        )
    except Exception as exc:
        report["failures"]["node_assignment"] = {"type": type(exc).__name__, "message": str(exc)}

    report["environment"]["gpu_end"] = gpu_snapshot("end")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"RESULT={args.out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark ONNX Runtime model variants on fixed 10-second synthetic inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort


SAMPLES = 160_000


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def create_session(path: Path, profiling: bool = False) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = profiling
    if profiling:
        options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ort_{path.stem}")
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def benchmark(path: Path, batch_sizes: list[int], warmup: int, repeats: int) -> dict:
    session = create_session(path, profiling=True)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = {
        "path": str(path),
        "session_providers": session.get_providers(),
        "batches": {},
    }
    for batch_size in batch_sizes:
        waveforms = np.zeros((batch_size, 1, SAMPLES), dtype=np.float32)
        try:
            for _ in range(warmup):
                session.run([output_name], {input_name: waveforms})
            latencies: list[float] = []
            for _ in range(repeats):
                started = time.perf_counter()
                session.run([output_name], {input_name: waveforms})
                latencies.append((time.perf_counter() - started) * 1000.0)
            mean_ms = statistics.mean(latencies)
            result["batches"][str(batch_size)] = {
                "wall_ms_mean": round(mean_ms, 2),
                "wall_ms_p50": round(percentile(latencies, 50), 2),
                "wall_ms_p95": round(percentile(latencies, 95), 2),
                "per_item_ms": round(mean_ms / batch_size, 3),
                "audio_sec_per_s": round(batch_size * 10_000.0 / mean_ms, 1),
                "rtf": round(mean_ms / (batch_size * 10_000.0), 5),
            }
        except Exception as error:
            result["batches"][str(batch_size)] = {
                "error": f"{type(error).__name__}: {error}"
            }
        print(json.dumps({"model": path.name, "batch": batch_size, **result["batches"][str(batch_size)]}), flush=True)

    profile_path = Path(session.end_profiling())
    profile = json.loads(profile_path.read_text())
    assignments: set[tuple[str, str]] = set()
    for event in profile:
        provider = event.get("args", {}).get("provider")
        if event.get("cat") == "Node" and provider:
            assignments.add((event.get("name", "unknown"), provider))
    result["unique_profiled_nodes_by_provider"] = dict(
        Counter(provider for _, provider in assignments)
    )
    return result


def synthetic_parity(reference: Path, variant: Path, seeds: list[int]) -> dict:
    ref_session = create_session(reference)
    variant_session = create_session(variant)
    ref_input = ref_session.get_inputs()[0].name
    ref_output = ref_session.get_outputs()[0].name
    variant_input = variant_session.get_inputs()[0].name
    variant_output = variant_session.get_outputs()[0].name
    matched_cells = 0
    matched_frames = 0
    total_cells = 0
    total_frames = 0
    max_abs_diff = 0.0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        waveforms = np.clip(
            rng.normal(0.0, 0.05, size=(1, 1, SAMPLES)), -1.0, 1.0
        ).astype(np.float32)
        reference_output = ref_session.run([ref_output], {ref_input: waveforms})[0]
        variant_output_value = variant_session.run([variant_output], {variant_input: waveforms})[0]
        equal = reference_output == variant_output_value
        matched_cells += int(equal.sum())
        total_cells += int(equal.size)
        frame_equal = np.all(equal, axis=-1)
        matched_frames += int(frame_equal.sum())
        total_frames += int(frame_equal.size)
        max_abs_diff = max(
            max_abs_diff,
            float(np.max(np.abs(reference_output.astype(np.float32) - variant_output_value.astype(np.float32)))),
        )
    return {
        "input": "10s seeded Gaussian noise, mean=0, std=0.05, clipped to [-1, 1]",
        "seeds": seeds,
        "cell_exact_match_pct": round(100.0 * matched_cells / total_cells, 4),
        "frame_exact_match_pct": round(100.0 * matched_frames / total_frames, 4),
        "max_abs_diff": max_abs_diff,
        "total_frames": total_frames,
    }


def parse_model(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("model must be LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument("--batch-sizes", default="1,8,32,50")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seeds", default="1001,1002,1003")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    report = {
        "scope": "fixed 10-second synthetic inputs only",
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "models": {},
    }
    for label, path in args.model:
        speed = benchmark(path, batch_sizes, args.warmup, args.repeats)
        parity = synthetic_parity(args.reference, path, seeds)
        report["models"][label] = {"speed": speed, "synthetic_parity_vs_fp32": parity}
        print(json.dumps({"model": label, "parity": parity}, ensure_ascii=False), flush=True)

    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

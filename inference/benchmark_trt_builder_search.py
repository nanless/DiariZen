#!/usr/bin/env python3
"""Search TensorRT FP16 builder settings for fixed 16-second L4 inference."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from inference.benchmark_l4_runtime_acceleration import (
    INPUT_SAMPLES,
    TensorRTRunner,
    decide_benefit,
)
from inference.benchmark_tensorrt_fixed10s import build_engine


DEFAULT_CANDIDATES = (
    ("opt4_ws8_auxauto", 4, 8, None),
    ("opt5_ws8_auxauto", 5, 8, None),
    ("opt5_ws8_aux0", 5, 8, 0),
    ("opt5_ws8_aux2", 5, 8, 2),
    ("opt5_ws16_auxauto", 5, 16, None),
)


def benchmark_engine(path: Path, warmup: int, repeats: int) -> dict:
    runner = TensorRTRunner(path)
    result = {
        "engine_bytes": path.stat().st_size,
        "engine_num_aux_streams": runner.engine.num_aux_streams,
        "engine_device_memory_bytes": runner.engine.device_memory_size_v2,
        "enqueue_device": runner.measure(runner.enqueue, warmup, repeats),
    }
    graph = runner.capture_inference_graph()
    result["cuda_graph_node_count"] = graph.node_count
    result["cuda_graph_parity"] = runner.parity(graph)
    result["cudagraph_device"] = runner.measure(graph.replay, warmup, repeats)
    graph.close()
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--baseline-engine", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--seed-timing-cache", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--minimum-improvement-pct", type=float, default=1.0)
    args = parser.parse_args()
    for path in (args.onnx, args.baseline_engine):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    args.engine_dir.mkdir(parents=True, exist_ok=True)
    timing_cache = args.engine_dir / "fp16.timing.cache"
    if not timing_cache.exists() and args.seed_timing_cache:
        shutil.copy2(args.seed_timing_cache, timing_cache)

    report = {
        "scope": "fixed 16-second synthetic input only",
        "input_shape": [1, 1, INPUT_SAMPLES],
        "precision": "TensorRT FP16 mixed precision",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "baseline": {
            "name": "opt3_existing",
            "path": str(args.baseline_engine),
            "build": {"optimization_level": 3, "reused": True},
            "benchmark": benchmark_engine(args.baseline_engine, args.warmup, args.repeats),
        },
        "candidates": {},
    }
    baseline_graph_gpu = report["baseline"]["benchmark"]["cudagraph_device"]["gpu"]
    for name, optimization_level, workspace_gb, max_aux_streams in DEFAULT_CANDIDATES:
        engine_path = args.engine_dir / f"segmentation_16s_bs1_fp16_{name}.plan"
        candidate = {
            "path": str(engine_path),
            "build": {
                "optimization_level": optimization_level,
                "workspace_gb": workspace_gb,
                "requested_max_aux_streams": max_aux_streams,
            },
        }
        try:
            if args.reuse_existing and engine_path.is_file():
                build_seconds = None
            else:
                started = time.perf_counter()
                build_seconds = build_engine(
                    args.onnx,
                    engine_path,
                    batch_size=1,
                    workspace_gb=workspace_gb,
                    optimization_level=optimization_level,
                    precision="fp16",
                    timing_cache_path=timing_cache,
                    input_samples=INPUT_SAMPLES,
                    max_aux_streams=max_aux_streams,
                )
                candidate["build"]["wrapper_wall_seconds"] = round(
                    time.perf_counter() - started, 3
                )
            candidate["build"]["build_seconds"] = (
                None if build_seconds is None else round(build_seconds, 3)
            )
            candidate["build"]["reused"] = build_seconds is None
            candidate["benchmark"] = benchmark_engine(engine_path, args.warmup, args.repeats)
            candidate["decision_vs_opt3_cudagraph_gpu"] = decide_benefit(
                baseline_graph_gpu,
                candidate["benchmark"]["cudagraph_device"]["gpu"],
                args.minimum_improvement_pct,
            )
        except Exception as error:
            candidate["error"] = f"{type(error).__name__}: {error}"
        report["candidates"][name] = candidate
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"candidate": name, **candidate}, ensure_ascii=False), flush=True)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

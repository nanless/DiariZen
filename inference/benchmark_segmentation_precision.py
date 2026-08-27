#!/usr/bin/env python3
"""Benchmark segmentation-only model: speed + accuracy vs FP16/FP32 reference.

Designed for fixed 10s @ 16kHz mono input [B, 1, 160000].
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import toml
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diarizen.utils import instantiate
from inference.utils import dump_json, frames_to_segments, list_audio_files, load_audio_mono_16k, write_rttm

SEC_10 = 10
SAMPLES_10S = SEC_10 * 16000


def _percentile(xs: List[float], q: float) -> float:
    xs_sorted = sorted(xs)
    k = int(round((len(xs_sorted) - 1) * q))
    return float(xs_sorted[k])


class PowersetToMultilabelHard(torch.nn.Module):
    """Match the hard powerset decoding embedded in the exported ONNX model."""

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model
        assert getattr(base_model.specifications, "powerset", False), "expected powerset model"
        self.register_buffer("mapping", base_model.powerset.mapping.float(), persistent=False)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        powerset_logprobs = self.base_model(waveforms)
        indices = torch.argmax(powerset_logprobs, dim=-1)
        return self.mapping[indices]


def load_pytorch_model(config: Path, ckpt: Path, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    cfg = toml.load(config)
    model = instantiate(cfg["model"]["path"], args=cfg["model"]["args"].copy())
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().to(device=device, dtype=dtype)
    return PowersetToMultilabelHard(model).to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def bench_pytorch(
    model: torch.nn.Module,
    device: torch.device,
    batch_sizes: List[int],
    *,
    warmup: int = 5,
    repeats: int = 20,
    use_fp16: bool = True,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"backend": "pytorch_fp16" if use_fp16 else "pytorch_fp32", "batches": {}}
    for bs in batch_sizes:
        x = torch.zeros(bs, 1, SAMPLES_10S, device=device, dtype=torch.float16 if use_fp16 else torch.float32)
        # warmup
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16 and device.type == "cuda"):
                _ = model(x.float() if not use_fp16 else x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies: List[float] = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16 and device.type == "cuda"):
                _ = model(x.float() if not use_fp16 else x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        wall = statistics.mean(latencies)
        results["batches"][str(bs)] = {
            "wall_ms_mean": round(wall * 1000, 2),
            "wall_ms_p50": round(_percentile(latencies, 0.5) * 1000, 2),
            "wall_ms_p95": round(_percentile(latencies, 0.95) * 1000, 2),
            "per_item_ms": round(wall / bs * 1000, 3),
            "audio_sec_per_s": round(bs * SEC_10 / wall, 1),
            "rtf": round(wall / (bs * SEC_10), 5),
        }
    return results


def bench_ort(
    onnx_path: Path,
    batch_sizes: List[int],
    *,
    providers: List[str],
    warmup: int = 5,
    repeats: int = 20,
) -> Dict[str, Any]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path.as_posix(), sess_options=so, providers=providers)
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    results: Dict[str, Any] = {"backend": "ort_" + providers[0].replace("ExecutionProvider", "").lower(), "batches": {}}
    for bs in batch_sizes:
        x = np.zeros((bs, 1, SAMPLES_10S), np.float32)
        for _ in range(warmup):
            sess.run([out_name], {inp_name: x})
        latencies: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            sess.run([out_name], {inp_name: x})
            latencies.append(time.perf_counter() - t0)
        wall = statistics.mean(latencies)
        results["batches"][str(bs)] = {
            "wall_ms_mean": round(wall * 1000, 2),
            "wall_ms_p50": round(_percentile(latencies, 0.5) * 1000, 2),
            "wall_ms_p95": round(_percentile(latencies, 0.95) * 1000, 2),
            "per_item_ms": round(wall / bs * 1000, 3),
            "audio_sec_per_s": round(bs * SEC_10 / wall, 1),
            "rtf": round(wall / (bs * SEC_10), 5),
        }
    return results


def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    batch_size: int,
    *,
    fp16: bool = True,
    workspace_gb: int = 8,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed for TensorRT")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    shape = (batch_size, 1, SAMPLES_10S)
    profile.set_shape(inp.name, shape, shape, shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for batch={batch_size}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)


def bench_trt(engine_path: Path, batch_size: int, *, warmup: int = 5, repeats: int = 20) -> Dict[str, float]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    context.set_input_shape(inp_name, (batch_size, 1, SAMPLES_10S))

    import torch

    x = torch.zeros(batch_size, 1, SAMPLES_10S, device="cuda", dtype=torch.float32)
    out_shape = context.get_tensor_shape(out_name)
    output_dtype = engine.get_tensor_dtype(out_name)
    trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if output_dtype not in trt_to_torch:
        raise TypeError(f"Unsupported TensorRT output dtype: {output_dtype}")
    y = torch.empty(tuple(out_shape), device="cuda", dtype=trt_to_torch[output_dtype])
    context.set_tensor_address(inp_name, int(x.data_ptr()))
    context.set_tensor_address(out_name, int(y.data_ptr()))
    stream = torch.cuda.Stream()
    for _ in range(warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    latencies: List[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        latencies.append(time.perf_counter() - t0)
    wall = statistics.mean(latencies)
    return {
        "wall_ms_mean": round(wall * 1000, 2),
        "wall_ms_p50": round(_percentile(latencies, 0.5) * 1000, 2),
        "wall_ms_p95": round(_percentile(latencies, 0.95) * 1000, 2),
        "per_item_ms": round(wall / batch_size * 1000, 3),
        "audio_sec_per_s": round(batch_size * SEC_10 / wall, 1),
        "rtf": round(wall / (batch_size * SEC_10), 5),
    }


def eval_accuracy_vs_ref(
    ref_runner,
    sys_runner,
    audio_roots: List[Path],
    *,
    label: str,
) -> Dict[str, Any]:
    audios = sorted({a for r in audio_roots for a in list_audio_files(str(r))})
    per_file = []
    total_frames = 0
    total_mismatch = 0
    total_cells = 0
    total_cell_match = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ref_rttms, sys_rttms = [], []
        for ap in audios:
            p = Path(ap)
            x, dur = load_audio_mono_16k(ap)
            x_b = x[None, ...].astype(np.float32)
            y_ref = ref_runner(x_b)[0].astype(np.uint8)
            y_sys = sys_runner(x_b)[0].astype(np.uint8)
            n = min(y_ref.shape[0], y_sys.shape[0])
            y_ref, y_sys = y_ref[:n], y_sys[:n]
            ns = max(y_ref.shape[1], y_sys.shape[1])
            if y_ref.shape[1] < ns:
                y_ref = np.concatenate([y_ref, np.zeros((n, ns - y_ref.shape[1]), np.uint8)], 1)
            if y_sys.shape[1] < ns:
                y_sys = np.concatenate([y_sys, np.zeros((n, ns - y_sys.shape[1]), np.uint8)], 1)
            exact = (y_ref == y_sys).all(axis=1)
            mismatch = int((~exact).sum())
            cell_match = int((y_ref == y_sys).sum())
            total_frames += n
            total_mismatch += mismatch
            total_cell_match += cell_match
            total_cells += n * ns
            sid = p.stem
            r_ref = tmp_p / f"{sid}_ref.rttm"
            r_sys = tmp_p / f"{sid}_sys.rttm"
            write_rttm(frames_to_segments(y_ref, 0.02), sid, str(r_ref))
            write_rttm(frames_to_segments(y_sys, 0.02), sid, str(r_sys))
            ref_rttms.append(r_ref)
            sys_rttms.append(r_sys)
            per_file.append(
                {
                    "audio": p.name,
                    "duration_s": round(dur, 2),
                    "frames": n,
                    "frame_exact_match_rate": round(float(exact.mean()), 6),
                    "frame_mismatch_frames": mismatch,
                    "speaker_cell_match_rate": round(cell_match / (n * ns), 6),
                }
            )

        sys.path.insert(0, str(REPO / "dscore"))
        from scorelib.rttm import load_rttm
        from scorelib.score import score
        from scorelib.turn import merge_turns, trim_turns
        from scorelib.uem import gen_uem

        ref_turns, sys_turns = [], []
        for r, s in zip(ref_rttms, sys_rttms):
            t, _, _ = load_rttm(str(r))
            ref_turns.extend(t)
            t, _, _ = load_rttm(str(s))
            sys_turns.extend(t)
        uem = gen_uem(ref_turns, sys_turns)
        ref_turns = merge_turns(trim_turns(ref_turns, uem))
        sys_turns = merge_turns(trim_turns(sys_turns, uem))
        _, global_scores = score(ref_turns, sys_turns, uem, step=0.01, collar=0.0, ignore_overlaps=False)

    return {
        "label": label,
        "summary": {
            "num_files": len(audios),
            "total_frames": total_frames,
            "frame_exact_match_rate": 1 - total_mismatch / total_frames,
            "speaker_cell_match_rate": total_cell_match / total_cells,
            "cross_der_percent": global_scores.der,
            "cross_jer_percent": global_scores.jer,
        },
        "per_file": per_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-sizes", type=str, default="1,8,32,50")
    parser.add_argument("--calibration-root", action="append", default=[])
    parser.add_argument("--mode", type=str, default="all", choices=["all", "speed", "accuracy"])
    parser.add_argument("--skip-trt", action="store_true")
    parser.add_argument(
        "--engine-dir",
        type=str,
        default=str(REPO / "inference" / "models" / "kaldi_merged_1219_all_ft_large" / "trt_l4"),
    )
    parser.add_argument(
        "--tensorrt-python-path",
        type=str,
        default="",
        help="Optional site-packages directory containing TensorRT bindings.",
    )
    parser.add_argument("--trt-workspace-gb", type=int, default=8)
    parser.add_argument("--accuracy-ref", type=str, default="pytorch_fp32_cpu", choices=["pytorch_fp32_cpu", "pytorch_fp16_gpu"])
    args = parser.parse_args()

    config = Path(args.config)
    ckpt = Path(args.ckpt)
    onnx_path = Path(args.onnx)
    out_json = Path(args.out_json)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    report: Dict[str, Any] = {
        "input_shape": f"[B, 1, {SAMPLES_10S}]",
        "audio_sec": SEC_10,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "speed": {},
        "accuracy": {},
        "notes": [],
    }

    # Speed benchmarks
    if args.mode in ("all", "speed"):
        model_fp32 = load_pytorch_model(config, ckpt, device, torch.float32)
        report["speed"]["pytorch_fp32"] = bench_pytorch(model_fp32, device, batch_sizes, use_fp16=False)
        del model_fp32
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model_fp16 = load_pytorch_model(config, ckpt, device, torch.float32)
        report["speed"]["pytorch_fp16"] = bench_pytorch(model_fp16, device, batch_sizes, use_fp16=True)
        del model_fp16
        if device.type == "cuda":
            torch.cuda.empty_cache()

        report["speed"]["ort_cuda_fp32"] = bench_ort(
            onnx_path, batch_sizes, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        if not args.skip_trt and args.mode in ("all", "speed"):
            try:
                if args.tensorrt_python_path:
                    sys.path.append(args.tensorrt_python_path)
                import tensorrt  # noqa: F401

                engine_dir = Path(args.engine_dir)
                trt_results: Dict[str, Any] = {"backend": "tensorrt_fp16", "batches": {}}
                for bs in batch_sizes:
                    eng = engine_dir / f"seg_fp16_bs{bs}.engine"
                    if not eng.is_file():
                        build_trt_engine(
                            onnx_path,
                            eng,
                            bs,
                            fp16=True,
                            workspace_gb=args.trt_workspace_gb,
                        )
                    trt_results["batches"][str(bs)] = bench_trt(eng, bs)
                report["speed"]["tensorrt_fp16"] = trt_results
            except Exception as e:
                report["notes"].append(f"TensorRT skipped: {e}")

    # Accuracy (separate process recommended for GPU memory hygiene)
    if args.mode in ("all", "accuracy"):
        roots = [Path(p) for p in args.calibration_root] if args.calibration_root else [
            REPO / "example",
            Path("/root/code/gitlab_repos/speaker_diarize_infer/test_audios/bench_largescale_20"),
        ]
        if args.accuracy_ref == "pytorch_fp16_gpu":
            ref_device = device
            model_ref = load_pytorch_model(config, ckpt, ref_device, torch.float32)

            @torch.inference_mode()
            def ref_runner(x_np: np.ndarray) -> np.ndarray:
                x = torch.from_numpy(x_np).to(device=ref_device, dtype=torch.float32)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=ref_device.type == "cuda"):
                    y = model_ref(x)
                return y.detach().float().cpu().numpy()

            ref_label = "pytorch_fp16_gpu"
        else:
            model_ref = load_pytorch_model(config, ckpt, torch.device("cpu"), torch.float32)

            @torch.inference_mode()
            def ref_runner(x_np: np.ndarray) -> np.ndarray:
                x = torch.from_numpy(x_np)
                y = model_ref(x)
                return y.detach().float().numpy()

            ref_label = "pytorch_fp32_cpu"

        import onnxruntime as ort

        so = ort.SessionOptions()
        ort_sess = ort.InferenceSession(
            onnx_path.as_posix(), sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        inp_n, out_n = ort_sess.get_inputs()[0].name, ort_sess.get_outputs()[0].name

        def ort_runner(x_np: np.ndarray) -> np.ndarray:
            return ort_sess.run([out_n], {inp_n: x_np})[0]

        report["accuracy"]["ort_cuda_vs_" + ref_label] = eval_accuracy_vs_ref(
            ref_runner, ort_runner, roots, label="ort_cuda"
        )

    if args.mode == "speed" and out_json.is_file():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        prev.setdefault("speed", {}).update(report["speed"])
        prev.setdefault("notes", []).extend(report["notes"])
        report = prev

    dump_json(report, out_json.as_posix())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

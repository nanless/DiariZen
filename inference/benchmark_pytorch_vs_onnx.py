#!/usr/bin/env python3
"""
Benchmark PyTorch vs ONNX on the same audio set:
- output difference metrics
- detailed timing and RTF stats

NOTE: Run in the `diarizen` conda env.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List

import toml

from diarizen.utils import instantiate

from inference.utils import dump_json, list_audio_files, load_audio_mono_16k

from inference.cpu_runtime import configure_env_single_thread

# Must happen before importing numpy/torch/onnxruntime
configure_env_single_thread()


def _percentile(xs: List[float], q: float) -> float:
    assert 0.0 <= q <= 1.0
    assert xs
    xs_sorted = sorted(xs)
    k = int(round((len(xs_sorted) - 1) * q))
    return float(xs_sorted[k])


def main():
    parser = argparse.ArgumentParser("PyTorch vs ONNX benchmark for DiariZen powerset model")
    parser.add_argument("in_root", type=str, help="Input audio root directory")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment dir containing checkpoints/")
    parser.add_argument("--config", type=str, required=True, help="Resolved config__*.toml")
    parser.add_argument("--ckpt-name", type=str, required=True, help="best or epoch_0010")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX path exported by export_to_onnx.py")
    parser.add_argument("--out-json", type=str, default="", help="Output JSON report path (optional)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Torch device")
    parser.add_argument("--providers", type=str, default="cuda,cpu", help="Comma-separated: cuda,cpu")
    parser.add_argument(
        "--torch-tf32",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to allow TF32 in torch matmul/cudnn. Set 0 for maximum numerical parity.",
    )
    parser.add_argument(
        "--ort-tf32",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to allow TF32 in onnxruntime CUDA EP (if supported). Set 0 for maximum numerical parity.",
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use deterministic torch algorithms when possible (may impact performance).",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Expected sample rate")
    parser.add_argument("--max-files", type=int, default=0, help="If >0, only benchmark first N files")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs for each backend")
    parser.add_argument(
        "--quantize-logprobs",
        type=float,
        default=0.0,
        help="If >0, quantize powerset logprobs by this step before argmax to reduce hard output flips.",
    )
    args = parser.parse_args()

    # Import heavy deps after env is configured.
    import numpy as np
    import torch

    from inference.cpu_runtime import configure_torch_single_thread

    in_root = Path(args.in_root)
    exp_dir = Path(args.exp_dir)
    config_path = Path(args.config)
    ckpt_path = exp_dir / "checkpoints" / args.ckpt_name / "pytorch_model.bin"
    onnx_path = Path(args.onnx)
    assert in_root.is_dir(), f"in_root not found: {in_root}"
    assert config_path.is_file(), f"config not found: {config_path}"
    assert ckpt_path.is_file(), f"checkpoint not found: {ckpt_path}"
    assert onnx_path.is_file(), f"onnx not found: {onnx_path}"

    audios = list_audio_files(in_root.as_posix())
    if args.max_files and args.max_files > 0:
        audios = audios[: args.max_files]
    assert audios, f"no audio files found under: {in_root}"

    # Force single-thread torch CPU behavior regardless of requested device.
    configure_torch_single_thread(torch)

    # Torch numerical controls (must happen before heavy CUDA kernels run)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.torch_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.torch_tf32)
        try:
            # 2.0+: "highest" avoids TF32 fast paths.
            torch.set_float32_matmul_precision("highest" if not args.torch_tf32 else "high")
        except Exception:
            pass
    if args.deterministic:
        # cuBLAS reproducibility requirement when deterministic algos are enabled.
        # See: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        if args.device == "cuda" and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # ONNXRuntime session (respect --providers)
    import onnxruntime as ort

    # Parse providers
    providers = []
    for p in [s.strip().lower() for s in args.providers.split(",") if s.strip()]:
        if p == "cuda":
            providers.append("CUDAExecutionProvider")
        elif p == "cpu":
            providers.append("CPUExecutionProvider")
        else:
            raise ValueError(f"unknown provider key: {p}")
    if not providers:
        providers = ["CPUExecutionProvider"]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    so.enable_mem_reuse = True

    # ORT TF32 control for CUDA EP (if supported by installed ORT build)
    if "CUDAExecutionProvider" in providers:
        try:
            # Available in newer ORT builds; ignore if not supported.
            so.add_session_config_entry("session.set_denormal_as_zero", "1")
            so.add_session_config_entry("session.use_deterministic_compute", "1" if args.deterministic else "0")
            so.add_session_config_entry(
                "ep.cuda.allow_tf32",
                "1" if args.ort_tf32 else "0",
            )
        except Exception:
            pass

    ort_sess = ort.InferenceSession(onnx_path.as_posix(), sess_options=so, providers=providers)
    ort_in = ort_sess.get_inputs()[0].name
    ort_out = ort_sess.get_outputs()[0].name

    # Torch model
    config = toml.load(config_path)
    model = instantiate(config["model"]["path"], args=config["model"]["args"].copy())
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "device=cuda but torch.cuda.is_available() is False"
    model = model.to(device)

    class TorchWrapperHard(torch.nn.Module):
        def __init__(self, base_model: torch.nn.Module):
            super().__init__()
            self.base_model = base_model
            assert base_model.specifications.powerset, "expected powerset model"
            self.register_buffer("mapping", base_model.powerset.mapping.float(), persistent=True)

        def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
            y = self.base_model(waveforms)  # (B,T,P) log_softmax
            if args.quantize_logprobs and args.quantize_logprobs > 0:
                q = float(args.quantize_logprobs)
                y = torch.round(y / q) * q
            idx = torch.argmax(y, dim=-1)  # (B,T)
            return self.mapping[idx]  # (B,T,S) 0/1

    torch_wrapped = TorchWrapperHard(model).to(device).eval()

    # warmup using a short dummy input
    dummy = torch.zeros(1, 1, args.sample_rate, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = torch_wrapped(dummy)
    dummy_np = np.zeros((1, 1, args.sample_rate), dtype=np.float32)
    for _ in range(args.warmup):
        _ = ort_sess.run([ort_out], {ort_in: dummy_np})[0]

    per_audio: List[Dict] = []
    diff_abs_all = []
    diff_max_all = []
    torch_elapsed = []
    onnx_elapsed = []
    rtfs_torch = []
    rtfs_onnx = []

    for audio_path in audios:
        x_ch1, duration = load_audio_mono_16k(audio_path, sample_rate=args.sample_rate)
        x_np = x_ch1[None, ...].astype(np.float32)  # (1,1,S)
        x_t = torch.from_numpy(x_np).to(device)

        # torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            y_t = torch_wrapped(x_t).cpu().numpy()
        t1 = time.perf_counter()
        te = t1 - t0

        # onnx
        t2 = time.perf_counter()
        y_o = ort_sess.run([ort_out], {ort_in: x_np})[0]
        t3 = time.perf_counter()
        oe = t3 - t2

        assert y_t.shape == y_o.shape, f"shape mismatch: torch={y_t.shape} onnx={y_o.shape} ({audio_path})"
        d = np.abs(y_t - y_o)
        diff_abs = float(d.mean())
        diff_max = float(d.max())

        diff_abs_all.append(diff_abs)
        diff_max_all.append(diff_max)
        torch_elapsed.append(te)
        onnx_elapsed.append(oe)
        rtfs_torch.append(te / max(1e-9, duration))
        rtfs_onnx.append(oe / max(1e-9, duration))

        per_audio.append(
            {
                "audio": audio_path,
                "duration": duration,
                "torch_elapsed_s": te,
                "onnx_elapsed_s": oe,
                "torch_rtf": rtfs_torch[-1],
                "onnx_rtf": rtfs_onnx[-1],
                "diff_abs_mean": diff_abs,
                "diff_abs_max": diff_max,
                "frames": int(y_t.shape[1]),
                "speakers": int(y_t.shape[2]),
            }
        )

    report = {
        "count": len(per_audio),
        "onnx_providers": providers,
        "torch_device": str(device),
        "summary": {
            "diff_abs_mean__mean": float(statistics.mean(diff_abs_all)),
            "diff_abs_mean__p50": _percentile(diff_abs_all, 0.50),
            "diff_abs_mean__p90": _percentile(diff_abs_all, 0.90),
            "diff_abs_max__max": float(max(diff_max_all)),
            "torch_elapsed_s__mean": float(statistics.mean(torch_elapsed)),
            "onnx_elapsed_s__mean": float(statistics.mean(onnx_elapsed)),
            "torch_rtf__mean": float(statistics.mean(rtfs_torch)),
            "onnx_rtf__mean": float(statistics.mean(rtfs_onnx)),
            "torch_rtf__p50": _percentile(rtfs_torch, 0.50),
            "torch_rtf__p90": _percentile(rtfs_torch, 0.90),
            "torch_rtf__p99": _percentile(rtfs_torch, 0.99),
            "onnx_rtf__p50": _percentile(rtfs_onnx, 0.50),
            "onnx_rtf__p90": _percentile(rtfs_onnx, 0.90),
            "onnx_rtf__p99": _percentile(rtfs_onnx, 0.99),
        },
        "per_audio": per_audio,
    }

    if args.out_json:
        dump_json(report, args.out_json)
        print(f"Wrote report: {args.out_json}")

    print("Summary:")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Benchmark PyTorch compile, AMP, and SDPA on fixed 16-second input."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large/config__2025_12_26--11_44_15.toml"
DEFAULT_CKPT = REPO / "recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large/checkpoints/epoch_0016/pytorch_model.bin"
SAMPLES = 256000
AUDIO_SECONDS = 16.0


def run_text(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def gpu_snapshot() -> dict:
    gpu = run_text([
        "nvidia-smi", "--query-gpu=timestamp,index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,pstate",
        "--format=csv,noheader,nounits",
    ])
    procs = run_text([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits",
    ])
    fields = [x.strip() for x in gpu.split(",")]
    snap = {
        "raw_gpu": gpu,
        "raw_compute_processes": procs,
        "timestamp": fields[0],
        "index": int(fields[1]),
        "name": fields[2],
        "uuid": fields[3],
        "driver_version": fields[4],
        "memory_total_mib": int(fields[5]),
        "memory_used_mib": int(fields[6]),
        "utilization_gpu_percent": int(fields[7]),
        "temperature_c": int(fields[8]),
        "pstate": fields[9],
    }
    if procs or snap["memory_used_mib"] > 256 or snap["utilization_gpu_percent"] > 10:
        raise RuntimeError(f"GPU is not idle before run: {snap}")
    return snap


def percentile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    return ys[int(round((len(ys) - 1) * q))]


def summary_ms(xs: list[float]) -> dict:
    mean = statistics.mean(xs) * 1000.0
    return {
        "repeats": len(xs),
        "mean_ms": mean,
        "p50_ms": percentile(xs, 0.50) * 1000.0,
        "p95_ms": percentile(xs, 0.95) * 1000.0,
        "min_ms": min(xs) * 1000.0,
        "max_ms": max(xs) * 1000.0,
        "audio_seconds_per_second": AUDIO_SECONDS / statistics.mean(xs),
        "rtf": statistics.mean(xs) / AUDIO_SECONDS,
    }


def tensor_diff(a, b) -> dict:
    import torch
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "allclose_rtol_1e-3_atol_1e-3": bool(torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-3)),
    }


def hard_diff(a, b) -> dict:
    import torch
    neq = a != b
    return {
        "mismatch_cells": int(neq.sum().item()),
        "total_cells": int(neq.numel()),
        "mismatch_rate": float(neq.float().mean().item()),
        "exact": bool(not neq.any().item()),
    }


def autocast_ctx(torch, amp: bool):
    if amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def install_sdpa_patches(torch):
    """Patch the two attention cores in memory, preserving dense WavLM relative bias."""
    import torch.nn.functional as F
    from diarizen.models.module import conformer as conformer_mod
    from diarizen.models.module.wav2vec2 import components as wav_components

    original_self_forward = wav_components.SelfAttention.forward
    original_conf_call = conformer_mod.MultiHeadSelfAttention.__call__

    def self_attention_sdpa(self, x, attention_mask=None, position_bias=None, key_padding_mask=None):
        if self.hard_concrete_for_heads is not None or self.hard_concrete_for_layer is not None:
            raise RuntimeError("SDPA temporary patch does not support active hard-concrete pruning")
        b, length, _ = x.shape
        shape = (b, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)
        k = self.k_proj(x).view(*shape).transpose(2, 1)
        v = self.v_proj(x).view(*shape).transpose(2, 1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(2, 1).reshape(b, length, self.num_heads * self.head_dim)
        return self.out_proj(out), None

    def conformer_sdpa(self, x, batch_size, pos_k=None):
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        additive = None
        if pos_k is not None:
            rq = q.reshape(batch_size * self.h, -1, self.d_k).transpose(0, 1)
            ps = torch.matmul(rq, pos_k.transpose(-2, -1))
            ps = ps.transpose(0, 1).reshape(batch_size, self.h, pos_k.size(0), pos_k.size(1))
            additive = ps / (self.d_k ** 0.5)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=additive, dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        self.att = None
        out = out.permute(0, 2, 1, 3).reshape(-1, self.h * self.d_k)
        return self.linearO(out)

    wav_components.SelfAttention.forward = self_attention_sdpa
    conformer_mod.MultiHeadSelfAttention.__call__ = conformer_sdpa
    return original_self_forward, original_conf_call


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=[
        "eager_fp32", "eager_amp_fp16", "compile_default_amp_fp16",
        "compile_reduce_overhead_amp_fp16", "compile_max_autotune_amp_fp16", "sdpa_amp_fp16",
        "compile_reduce_overhead_sdpa_amp_fp16", "compile_max_autotune_sdpa_amp_fp16",
    ])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        ap.error("--warmup must be >= 1 and --repeats must be >= 2")
    for path in (args.config, args.ckpt):
        if not path.is_file():
            ap.error(f"file does not exist: {path}")

    started = dt.datetime.now(dt.timezone.utc).astimezone()
    result = {
        "status": "started", "mode": args.mode, "fixed_input": {"shape": [1, 1, SAMPLES], "seconds": 16, "sample_rate": 16000, "batch": 1, "seed": args.seed, "distribution": "torch.randn"},
        "warmup": args.warmup, "repeats": args.repeats, "started_at": started.isoformat(),
        "repo": str(REPO), "config": str(args.config), "checkpoint": str(args.ckpt),
        "pid": os.getpid(), "flock_contract": "/tmp/diarizen_l4_benchmark.lock",
    }
    try:
        result["gpu_preflight_after_lock"] = gpu_snapshot()
        os.chdir(REPO)
        sys.path.insert(0, str(REPO))
        import toml
        import torch
        from diarizen.utils import instantiate

        result["environment"] = {
            "python": sys.version.replace("\n", " "), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "torch_compile_available": hasattr(torch, "compile"),
        }
        cfg = toml.load(args.config)
        result["model_config"] = {"path": cfg["model"]["path"], "args": cfg["model"]["args"]}
        model = instantiate(cfg["model"]["path"], args=cfg["model"]["args"].copy())
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().cuda()
        mapping = model.powerset.mapping.float().cuda()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        x = torch.randn(1, 1, SAMPLES, device="cuda", dtype=torch.float32)

        @torch.inference_mode()
        def run(mod, amp):
            with autocast_ctx(torch, amp):
                raw = mod(x)
            hard = mapping[raw.argmax(dim=-1)]
            return raw, hard

        # Stable numeric references are computed before temporary patching/compilation.
        ref_fp32_raw, ref_fp32_hard = run(model, False)
        ref_amp_raw, ref_amp_hard = run(model, True)
        torch.cuda.synchronize()

        amp = args.mode != "eager_fp32"
        candidate = model
        compile_info = None
        use_sdpa_patch = args.mode == "sdpa_amp_fp16" or "_sdpa_amp_fp16" in args.mode
        if use_sdpa_patch:
            install_sdpa_patches(torch)
        if args.mode.startswith("compile_"):
            mode = {
                "compile_default_amp_fp16": "default",
                "compile_reduce_overhead_amp_fp16": "reduce-overhead",
                "compile_max_autotune_amp_fp16": "max-autotune",
                "compile_reduce_overhead_sdpa_amp_fp16": "reduce-overhead",
                "compile_max_autotune_sdpa_amp_fp16": "max-autotune",
            }[args.mode]
            import torch._dynamo
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            t_compile = time.perf_counter()
            candidate = torch.compile(model, mode=mode, fullgraph=False, dynamic=False)
            first_raw, first_hard = run(candidate, amp)
            torch.cuda.synchronize()
            compile_info = {
                "torch_compile_mode": mode,
                "sdpa_patch_installed_before_compile": use_sdpa_patch,
                "first_call_seconds_including_compile": time.perf_counter() - t_compile,
            }

        # One output after candidate transformation for parity.
        cand_raw, cand_hard = run(candidate, amp)
        torch.cuda.synchronize()
        result["output"] = {
            "raw_kind": "powerset log probabilities", "raw_shape": list(cand_raw.shape), "raw_dtype": str(cand_raw.dtype),
            "hard_kind": "powerset argmax mapped to hard multilabel", "hard_shape": list(cand_hard.shape), "hard_dtype": str(cand_hard.dtype),
            "candidate_vs_eager_fp32_raw": tensor_diff(cand_raw, ref_fp32_raw),
            "candidate_vs_eager_amp_fp16_raw": tensor_diff(cand_raw, ref_amp_raw),
            "candidate_vs_eager_fp32_hard": hard_diff(cand_hard, ref_fp32_hard),
            "candidate_vs_eager_amp_fp16_hard": hard_diff(cand_hard, ref_amp_hard),
            "eager_amp_fp16_vs_eager_fp32_raw": tensor_diff(ref_amp_raw, ref_fp32_raw),
            "eager_amp_fp16_vs_eager_fp32_hard": hard_diff(ref_amp_hard, ref_fp32_hard),
        }

        for _ in range(args.warmup):
            run(candidate, amp)
        torch.cuda.synchronize()
        result["gpu_before_timed_loop_after_lock"] = {
            "raw": run_text(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,pstate", "--format=csv,noheader"]),
            "compute_processes": run_text(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
        }
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run(candidate, amp)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        result["timing"] = summary_ms(latencies)
        result["timing"]["measurement"] = "host perf_counter around forward with cuda synchronize before and after"
        result["memory"] = {
            "peak_allocated_mib_timed_loop": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib_timed_loop": torch.cuda.max_memory_reserved() / 2**20,
            "allocated_mib_after": torch.cuda.memory_allocated() / 2**20,
            "reserved_mib_after": torch.cuda.memory_reserved() / 2**20,
        }
        if compile_info is not None:
            def clean_counter(x):
                return {str(k): int(v) for k, v in x.items()}
            counters = torch._dynamo.utils.counters
            compile_info["dynamo_counters"] = {str(k): clean_counter(v) for k, v in counters.items()}
            result["compile"] = compile_info
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    result["finished_at"] = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

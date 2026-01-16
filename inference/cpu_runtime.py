"""
CPU runtime configuration helpers.

Goals:
- Force single-thread execution (torch + onnxruntime + common BLAS/OpenMP stacks)
- Enable common CPU inference optimizations (ORT graph opts, mem arena/pattern)

Important: environment variables should be set BEFORE importing heavy libs that
load OpenMP/BLAS runtimes. Therefore scripts should call `configure_env_single_thread()`
at the very top (before importing torch/numpy/onnxruntime).
"""

from __future__ import annotations

import os


def configure_env_single_thread() -> None:
    # Common thread env vars used by OpenMP / MKL / OpenBLAS / NumExpr.
    # We set unconditionally to guarantee single-thread behavior.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"

    # Optional knobs (safe even if ignored)
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "0"


def configure_torch_single_thread(torch_module) -> None:
    # torch_module is imported torch
    torch_module.set_num_threads(1)
    try:
        torch_module.set_num_interop_threads(1)
    except Exception:
        pass

    # Enable mkldnn fast path on CPU (usually default True, but keep explicit)
    try:
        torch_module.backends.mkldnn.enabled = True
    except Exception:
        pass

    # Flush denormals can speed up some CPU kernels.
    try:
        torch_module.set_flush_denormal(True)
    except Exception:
        pass


def make_ort_session(onnxruntime_module, onnx_path: str):
    # onnxruntime_module is imported onnxruntime as ort
    so = onnxruntime_module.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = onnxruntime_module.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = onnxruntime_module.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    so.enable_mem_reuse = True
    # Force CPU provider only for consistent CPU benchmarking.
    return onnxruntime_module.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio

from diarizen.pipelines.inference import DiariZenPipeline


MODEL_REPO_IDS: Dict[str, str] = {
    "base": "BUT-FIT/diarizen-wavlm-base-s80-md",
    "large": "BUT-FIT/diarizen-wavlm-large-s80-md",
}


def get_audio_duration_seconds(audio_path: str) -> float:
    info = torchaudio.info(audio_path)
    # duration = num_frames / sample_rate
    return float(info.num_frames) / float(info.sample_rate)


def run_single(model_key: str, audio_path: str) -> Tuple[float, float]:
    """
    Returns: (elapsed_seconds, rtf)
    """
    repo_id = MODEL_REPO_IDS[model_key]
    # Load pipeline (download if needed)
    pipeline = DiariZenPipeline.from_pretrained(repo_id)

    # Warmup: short no-op by running a 0.01s slice if possible, else skip
    # We avoid additional file IO; the pipeline will anyway compile kernels on first call.
    # So the first real call is acceptable as warm start might not fully help.

    # Time the diarization
    audio_dur = get_audio_duration_seconds(audio_path)
    t0 = time.perf_counter()
    _ = pipeline(audio_path)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    rtf = elapsed / max(1e-9, audio_dur)
    return elapsed, rtf


def main():
    parser = argparse.ArgumentParser("Benchmark RTF for DiariZen models")
    parser.add_argument(
        "--audio",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "example" / "EN2002a_30s.wav"),
        help="Path to input WAV file.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="base,large",
        help="Comma-separated list of models to test: base,large",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Device hint. 'cpu' will error if CUDA is visible. Prefer using launcher to set CUDA_VISIBLE_DEVICES.",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().absolute().as_posix()
    assert Path(audio_path).exists(), f"Audio not found: {audio_path}"

    # Device enforcement: this script prefers that caller controls CUDA visibility.
    # We add a safety check to reduce surprises.
    cuda_visible = torch.cuda.is_available()
    if args.device == "cpu" and cuda_visible:
        raise RuntimeError(
            "CUDA is available but device=cpu requested. Launch with CUDA_VISIBLE_DEVICES='' to force CPU."
        )
    if args.device == "gpu" and not cuda_visible:
        raise RuntimeError("GPU requested but CUDA not available.")

    selected_models: List[str] = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in selected_models:
        if m not in MODEL_REPO_IDS:
            raise ValueError(f"Unknown model key '{m}'. Valid: {list(MODEL_REPO_IDS.keys())}")

    # Print header
    print(f"Audio: {audio_path}")
    print(f"Audio duration (s): {get_audio_duration_seconds(audio_path):.3f}")
    print(f"CUDA available: {cuda_visible}")
    print(f"Device hint: {args.device}")
    print("")

    results: Dict[str, Tuple[float, float]] = {}
    for model_key in selected_models:
        print(f"Running model: {model_key} ({MODEL_REPO_IDS[model_key]})")
        elapsed, rtf = run_single(model_key, audio_path)
        print(f"  Elapsed: {elapsed:.3f} s | RTF: {rtf:.3f}")
        results[model_key] = (elapsed, rtf)

    print("\nSummary:")
    for model_key, (elapsed, rtf) in results.items():
        print(f"- {model_key:>5s}: elapsed={elapsed:.3f}s  rtf={rtf:.3f}")


if __name__ == "__main__":
    main()



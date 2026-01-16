#!/usr/bin/env python3
"""
Minimal ONNX inference script (no diarizen/pyannote training dependencies).

Inputs:
- a directory of audio files (recursively scanned)
- an exported ONNX model from `export_to_onnx.py`

Outputs:
- RTTM files per audio
- summary.json with basic stats
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from inference.cpu_runtime import configure_env_single_thread

# Must happen before importing numpy/onnxruntime
configure_env_single_thread()


def main():
    parser = argparse.ArgumentParser("DiariZen ONNX diarization (segmentation-only)")
    parser.add_argument("in_root", type=str, help="Input audio root directory")
    parser.add_argument("--onnx", type=str, required=True, help="Path to exported ONNX model")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Expected sample rate")
    parser.add_argument("--frame-step", type=float, default=0.02, help="Seconds per frame (model receptive field step)")
    parser.add_argument("--min-duration", type=float, default=0.0, help="Minimum segment duration in seconds")
    parser.add_argument("--max-files", type=int, default=0, help="If >0, only process first N files")
    parser.add_argument("--providers", type=str, default="cuda,cpu", help="Comma-separated: cuda,cpu")
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort

    from inference.cpu_runtime import make_ort_session
    from inference.utils import dump_json, frames_to_segments, list_audio_files, load_audio_mono_16k, write_rttm

    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    onnx_path = Path(args.onnx)
    assert in_root.is_dir(), f"in_root not found: {in_root}"
    assert onnx_path.is_file(), f"onnx not found: {onnx_path}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Force CPU single-thread session regardless of args.providers (by request)
    sess = make_ort_session(ort, onnx_path.as_posix())
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    audios = list_audio_files(in_root.as_posix())
    if args.max_files and args.max_files > 0:
        audios = audios[: args.max_files]
    assert audios, f"no audio files found under: {in_root}"

    summary = []
    for audio_path in audios:
        rel = Path(audio_path).relative_to(in_root)
        sess_name = rel.with_suffix("").as_posix().replace("/", "__")

        x_ch1, duration = load_audio_mono_16k(audio_path, sample_rate=args.sample_rate)
        x = x_ch1[None, ...].astype(np.float32)  # (1, 1, samples)

        t0 = time.perf_counter()
        y = sess.run([output_name], {input_name: x})[0]  # (1, frames, speakers)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        rtf = elapsed / max(1e-9, duration)

        frame_labels = y[0].astype(np.uint8)
        # drop inactive speakers
        active = frame_labels.sum(axis=0) > 0
        frame_labels = frame_labels[:, active]

        segments = frames_to_segments(frame_labels, frame_step=args.frame_step, min_duration=args.min_duration)
        rttm_path = out_dir / f"{sess_name}.rttm"
        write_rttm(segments, sess_name, rttm_path.as_posix())

        summary.append(
            {
                "audio": audio_path,
                "session": sess_name,
                "duration": duration,
                "frames": int(y.shape[1]),
                "num_speakers_active": int(frame_labels.shape[1]),
                "num_segments": int(len(segments)),
                "elapsed_s": elapsed,
                "rtf": rtf,
                "rttm": rttm_path.as_posix(),
            }
        )

    dump_json(summary, (out_dir / "summary.json").as_posix())
    print(f"Done. audios={len(summary)}  summary={out_dir/'summary.json'}")


if __name__ == "__main__":
    main()


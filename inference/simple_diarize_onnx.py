#!/usr/bin/env python3
"""
ONNX version of `simple_diarize.py` (segmentation-only).

Produces the same style outputs:
- one RTTM per audio
- one PNG visualization per audio
- summary.json

Run example:
conda run --no-capture-output -n diarizen python inference/simple_diarize_onnx.py \
  /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios \
  --onnx /root/code/github_repos/DiariZen/inference/models/kaldi_merged_1219_all_ft_base/epoch_0010_multilabel_hard.onnx \
  --out-dir /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios_Diarizen_simple_base_1219_all/epoch_0010_onnx \
  --providers cpu
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path
from typing import Optional

from inference.cpu_runtime import configure_env_single_thread

# Must happen before importing numpy/onnxruntime/matplotlib (OpenMP/BLAS stacks)
configure_env_single_thread()

def _preload_conda_libstdcxx() -> None:
    """
    Workaround for runtime error:
      /usr/lib/.../libstdc++.so.6: version `GLIBCXX_3.4.29' not found

    Some wheels (e.g. Pillow -> libLerc) require a newer libstdc++ than the system one.
    When the dynamic loader resolves libstdc++ from /usr/lib first, importing PIL/matplotlib
    may fail. Preloading conda's libstdc++ with RTLD_GLOBAL usually fixes it.
    """
    prefix = Path(sys.prefix)
    candidates = [
        prefix / "lib" / "libstdc++.so.6",
        prefix / "x86_64-conda-linux-gnu" / "lib" / "libstdc++.so.6",
    ]
    for p in candidates:
        if p.is_file():
            try:
                ctypes.CDLL(p.as_posix(), mode=ctypes.RTLD_GLOBAL)
                return
            except Exception:
                # If preload fails, we'll fall back to non-plot behavior later.
                return


def plot_diarization(
    audio_path: str,
    segments,
    duration: float,
    session_name: str,
    output_path: str,
    output_format: str = "png",
) -> Optional[str]:
    try:
        _preload_conda_libstdcxx()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import soundfile as sf
        from pathlib import Path as _Path

        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        t = np.linspace(0, len(data) / sr, len(data))

        speakers = sorted(set(spk for _, _, spk in segments))
        colors = plt.cm.get_cmap("tab20", max(len(speakers), 1))
        spk_color = {spk: colors(i) for i, spk in enumerate(speakers)}
        spk_pos = {spk: idx for idx, spk in enumerate(speakers)}

        fig, (ax_wave, ax_diar) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12, 4),
            gridspec_kw={"height_ratios": [1.4, 1]},
        )

        ax_wave.plot(t, data, color="gray", alpha=0.6, linewidth=0.5)
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_title(session_name)
        ax_wave.set_xlim(0, duration)

        if speakers:
            for start, end, spk in segments:
                y = spk_pos[spk]
                ax_diar.broken_barh(
                    [(start, end - start)],
                    (y - 0.4, 0.8),
                    facecolors=spk_color[spk],
                    alpha=0.8,
                )
            ax_diar.set_yticks(list(spk_pos.values()))
            ax_diar.set_yticklabels([f"speaker_{s:02d}" for s in spk_pos.keys()])
            ax_diar.set_ylim(-0.5, len(speakers) - 0.5)
        else:
            ax_diar.text(0.5, 0.5, "no speech", ha="center", va="center", transform=ax_diar.transAxes)
            ax_diar.set_ylim(-0.5, 0.5)

        ax_diar.set_xlabel("Time (s)")
        ax_diar.set_ylabel("Speaker")
        ax_diar.grid(True, axis="x", linestyle="--", alpha=0.3)

        fig.tight_layout()
        out_path = _Path(output_path)
        fmt = output_format.lower().strip()
        # Prefer requested format, but if PNG fails (PIL/libstdc++ mismatch),
        # fallback to SVG which does not require Pillow.
        try:
            if fmt == "svg":
                out_path = out_path.with_suffix(".svg")
                fig.savefig(out_path.as_posix(), format="svg")
            else:
                out_path = out_path.with_suffix(".png")
                fig.savefig(out_path.as_posix(), dpi=150)
        except Exception as e:
            out_path = out_path.with_suffix(".svg")
            fig.savefig(out_path.as_posix(), format="svg")
            print(f"plot: png failed ({e}); wrote svg instead: {out_path}")
        plt.close(fig)
        return out_path.as_posix()
    except Exception as e:
        print(f"plot failed: {e}")
        return None


def _providers_from_arg(arg: str):
    providers = []
    for p in [s.strip().lower() for s in arg.split(",") if s.strip()]:
        if p == "cuda":
            providers.append("CUDAExecutionProvider")
        elif p == "cpu":
            providers.append("CPUExecutionProvider")
        else:
            raise ValueError(f"unknown provider key: {p}")
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers


def main():
    parser = argparse.ArgumentParser("Simple diarize using ONNX (DiariZen segmentation-only)")
    parser.add_argument("in_root", type=str, help="Input audio root directory")
    parser.add_argument("--onnx", type=str, required=True, help="Path to exported ONNX model")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--providers", type=str, default="cpu", help="Comma-separated: cpu,cuda")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Expected sample rate")
    parser.add_argument(
        "--frame-step",
        type=float,
        default=0.02,
        help="Seconds per frame (model receptive field step). For this experiment it is 0.02.",
    )
    parser.add_argument("--min-duration", type=float, default=0.0, help="Minimum segment duration (seconds)")
    parser.add_argument("--max-files", type=int, default=0, help="If >0, only process first N files")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate visualization PNG")
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="Disable visualization PNG")
    parser.add_argument(
        "--plot-format",
        type=str,
        default="png",
        choices=["png", "svg"],
        help="Plot output format. Use svg to avoid Pillow/libstdc++ issues.",
    )
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
    if not audios:
        print(f"no audios found: {in_root}")
        return

    summary = []
    for audio_path in audios:
        rel = Path(audio_path).relative_to(in_root)
        sess_name = rel.with_suffix("").as_posix().replace("/", "__")
        print(f"\nprocessing: {sess_name}")

        x_ch1, duration = load_audio_mono_16k(audio_path, sample_rate=args.sample_rate)
        x = x_ch1[None, ...].astype(np.float32)  # (1,1,samples)

        t0 = time.perf_counter()
        y = sess.run([output_name], {input_name: x})[0]  # (1, frames, speakers) 0/1
        t1 = time.perf_counter()
        print(f"  onnx out: {y.shape} elapsed={t1-t0:.3f}s rtf={(t1-t0)/max(1e-9,duration):.3f}")

        frame_labels = y[0].astype(np.uint8)
        active = frame_labels.sum(axis=0) > 0
        frame_labels = frame_labels[:, active]
        num_speakers = int(frame_labels.shape[1])
        print(f"  active speakers: {num_speakers}")

        segments = frames_to_segments(frame_labels, frame_step=args.frame_step, min_duration=args.min_duration)
        print(f"  segments: {len(segments)}")

        rttm_path = out_dir / f"{sess_name}.rttm"
        write_rttm(segments, sess_name, rttm_path.as_posix())

        plot_path = None
        if args.plot:
            plot_path = out_dir / f"{sess_name}.{args.plot_format}"
            actual = plot_diarization(
                audio_path,
                segments,
                duration,
                sess_name,
                plot_path.as_posix(),
                output_format=args.plot_format,
            )
            plot_path = Path(actual) if actual else None

        summary.append(
            {
                "audio": audio_path,
                "session": sess_name,
                "duration": duration,
                "num_speakers": num_speakers,
                "num_segments": len(segments),
                "rttm": rttm_path.as_posix(),
                "plot": plot_path.as_posix() if plot_path else None,
            }
        )

    dump_json(summary, (out_dir / "summary.json").as_posix())
    print(f"\nDone. processed={len(summary)} summary={(out_dir/'summary.json').as_posix()}")


if __name__ == "__main__":
    main()


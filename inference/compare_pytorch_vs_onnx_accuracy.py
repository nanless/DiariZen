#!/usr/bin/env python3
"""
Compare PyTorch vs ONNX diarization accuracy (DER/JER/...) against reference RTTM.

This script:
1) runs PyTorch checkpoint -> frame-level multilabel (0/1) using the SAME hard powerset decoding as ONNX exporter
2) runs ONNX model -> frame-level multilabel (0/1)
3) converts both to RTTM
4) scores both vs reference RTTM using dscore (DER, JER, ...), and writes a JSON report

Note:
- File-id in RTTM must match between reference and system RTTMs.
- Use --session-id stem for typical RTTM setups (file_id == basename w/o extension),
  or --session-id relative to match DiariZen's "relative path with '__'" convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import toml

from inference.cpu_runtime import configure_env_single_thread

# Must happen before importing numpy/torch/onnxruntime (OpenMP/BLAS stacks)
configure_env_single_thread()


@dataclass(frozen=True)
class ScoreSummary:
    der: float
    jer: float
    bcubed_precision: float
    bcubed_recall: float
    bcubed_f1: float
    tau_ref_sys: float
    tau_sys_ref: float
    ce_ref_sys: float
    ce_sys_ref: float
    mi: float
    nmi: float


def _dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _providers_from_arg(arg: str) -> List[str]:
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


def _session_id_for_audio(audio_path: Path, in_root: Path, mode: str) -> str:
    if mode == "stem":
        return audio_path.stem
    if mode == "relative":
        rel = audio_path.relative_to(in_root)
        return rel.with_suffix("").as_posix().replace("/", "__")
    raise ValueError(f"unknown session-id mode: {mode}")


def _collect_ref_rttm_paths(
    ref_rttm_dir: Optional[Path],
    ref_rttm_scp: Optional[Path],
    ref_rttm_files: Sequence[Path],
) -> List[Path]:
    if ref_rttm_dir:
        assert ref_rttm_dir.is_dir(), f"--ref-rttm-dir not found: {ref_rttm_dir}"
        rttms = sorted(ref_rttm_dir.glob("*.rttm"))
        assert rttms, f"no *.rttm under: {ref_rttm_dir}"
        return rttms
    if ref_rttm_scp:
        assert ref_rttm_scp.is_file(), f"--ref-rttm-scp not found: {ref_rttm_scp}"
        paths = []
        for line in ref_rttm_scp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line))
        assert paths, f"empty ref rttm scp: {ref_rttm_scp}"
        return paths
    if ref_rttm_files:
        return [Path(p) for p in ref_rttm_files]
    raise ValueError("must provide one of: --ref-rttm-dir, --ref-rttm-scp, --ref-rttm-files")


def _score_with_dscore(
    ref_rttms: Sequence[Path],
    sys_rttms: Sequence[Path],
    uem: Optional[Path],
    collar: float,
    ignore_overlaps: bool,
    step: float,
) -> Tuple[Dict[str, dict], dict]:
    # dscore isn't a normal installed package; add repo's dscore/ to sys.path.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, (repo_root / "dscore").as_posix())

    from scorelib.rttm import load_rttm  # type: ignore
    from scorelib.score import score  # type: ignore
    from scorelib.turn import merge_turns, trim_turns  # type: ignore
    from scorelib.uem import gen_uem, load_uem  # type: ignore

    ref_turns = []
    for p in ref_rttms:
        turns, _, _ = load_rttm(p.as_posix())
        ref_turns.extend(turns)
    sys_turns = []
    for p in sys_rttms:
        turns, _, _ = load_rttm(p.as_posix())
        sys_turns.extend(turns)

    if uem is not None:
        assert uem.is_file(), f"uem not found: {uem}"
        uem_obj = load_uem(uem.as_posix())
    else:
        uem_obj = gen_uem(ref_turns, sys_turns)

    # Mirror dscore/score.py behavior: trim + merge overlaps.
    ref_turns = trim_turns(ref_turns, uem_obj)
    sys_turns = trim_turns(sys_turns, uem_obj)
    ref_turns = merge_turns(ref_turns)
    sys_turns = merge_turns(sys_turns)

    file_scores, global_scores = score(
        ref_turns,
        sys_turns,
        uem_obj,
        step=step,
        collar=collar,
        ignore_overlaps=ignore_overlaps,
    )

    per_file = {s.file_id: s._asdict() for s in file_scores}
    overall = global_scores._asdict()
    return per_file, overall


def main() -> None:
    parser = argparse.ArgumentParser("Compare PyTorch vs ONNX diarization accuracy (dscore)")
    parser.add_argument("in_root", type=str, help="Input audio root directory")

    # Reference RTTMs (one of these)
    parser.add_argument("--ref-rttm-dir", type=str, default="", help="Directory containing reference *.rttm")
    parser.add_argument("--ref-rttm-scp", type=str, default="", help="Script file containing reference rttm paths")
    parser.add_argument("--ref-rttm-files", type=str, nargs="*", default=[], help="Reference RTTM paths")
    parser.add_argument("--uem", type=str, default="", help="Optional UEM file path")

    # PyTorch side
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment dir containing checkpoints/")
    parser.add_argument("--config", type=str, required=True, help="Resolved config__*.toml")
    parser.add_argument("--ckpt-name", type=str, required=True, help="best or epoch_0002")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Torch device for PyTorch run")

    # ONNX side
    parser.add_argument("--onnx", type=str, required=True, help="ONNX path exported by export_to_onnx.py")
    parser.add_argument("--providers", type=str, default="cpu", help="Comma-separated: cpu,cuda (onnxruntime)")

    # Decoding -> RTTM
    parser.add_argument(
        "--session-id",
        type=str,
        default="stem",
        choices=["stem", "relative"],
        help="How to derive RTTM file_id from audio path",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Expected sample rate")
    parser.add_argument("--frame-step", type=float, default=0.02, help="Seconds per frame (model receptive field step)")
    parser.add_argument("--min-duration", type=float, default=0.0, help="Minimum segment duration in seconds")
    parser.add_argument("--max-files", type=int, default=0, help="If >0, only process first N files")

    # Scoring config
    parser.add_argument("--collar", type=float, default=0.0, help="DER collar in seconds (dscore/md-eval)")
    parser.add_argument("--ignore-overlaps", action="store_true", default=False, help="Ignore overlaps when computing DER")
    parser.add_argument("--step", type=float, default=0.01, help="Frame step for clustering metrics (dscore)")

    # Outputs
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory (RTTMs + report.json)")
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    import torch

    from diarizen.utils import instantiate
    from inference.cpu_runtime import configure_torch_single_thread
    from inference.utils import frames_to_segments, list_audio_files, load_audio_mono_16k, write_rttm

    from inference.cpu_runtime import make_ort_session

    in_root = Path(args.in_root)
    exp_dir = Path(args.exp_dir)
    config_path = Path(args.config)
    ckpt_path = exp_dir / "checkpoints" / args.ckpt_name / "pytorch_model.bin"
    onnx_path = Path(args.onnx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert in_root.is_dir(), f"in_root not found: {in_root}"
    assert config_path.is_file(), f"config not found: {config_path}"
    assert ckpt_path.is_file(), f"checkpoint not found: {ckpt_path}"
    assert onnx_path.is_file(), f"onnx not found: {onnx_path}"

    # List audios
    audios = list_audio_files(in_root.as_posix())
    if args.max_files and args.max_files > 0:
        audios = audios[: args.max_files]
    assert audios, f"no audio files found under: {in_root}"

    # Derive file_ids for this run (used for RTTM naming and ref filtering)
    audio_paths = [Path(p) for p in audios]
    file_ids = [_session_id_for_audio(p, in_root, args.session_id) for p in audio_paths]

    # Collect reference RTTMs
    # - if ref-rttm-dir is provided, we REQUIRE one RTTM per file_id: <file_id>.rttm
    # - otherwise, we accept arbitrary RTTM lists (scp/files), but you must ensure file_id matches
    if args.ref_rttm_dir:
        ref_dir = Path(args.ref_rttm_dir)
        assert ref_dir.is_dir(), f"--ref-rttm-dir not found: {ref_dir}"
        missing = []
        ref_rttms = []
        for fid in file_ids:
            p = ref_dir / f"{fid}.rttm"
            if not p.is_file():
                missing.append(p.as_posix())
            else:
                ref_rttms.append(p)
        if missing:
            raise FileNotFoundError(
                "missing reference RTTM(s) for current audio set (check --session-id):\n  "
                + "\n  ".join(missing[:50])
                + ("\n  ... (truncated)" if len(missing) > 50 else "")
            )
    else:
        ref_rttms = _collect_ref_rttm_paths(
            Path(args.ref_rttm_dir) if args.ref_rttm_dir else None,
            Path(args.ref_rttm_scp) if args.ref_rttm_scp else None,
            [Path(p) for p in args.ref_rttm_files],
        )

    # Torch controls (force single-thread CPU even when device=cuda; keeps parity)
    configure_torch_single_thread(torch)

    # Build torch model and hard-decoder wrapper (same as exporter/benchmark)
    cfg = toml.load(config_path)
    model = instantiate(cfg["model"]["path"], args=cfg["model"]["args"].copy())
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
            idx = torch.argmax(y, dim=-1)  # (B,T)
            return self.mapping[idx]  # (B,T,S) 0/1

    torch_model = TorchWrapperHard(model).to(device).eval()

    # ONNX session
    providers = _providers_from_arg(args.providers)
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path.as_posix(), sess_options=sess_opts, providers=providers)
    ort_in = sess.get_inputs()[0].name
    ort_out = sess.get_outputs()[0].name

    # Output folders
    out_torch = out_dir / "pytorch_rttm"
    out_onnx = out_dir / "onnx_rttm"
    out_torch.mkdir(parents=True, exist_ok=True)
    out_onnx.mkdir(parents=True, exist_ok=True)

    # Generate RTTMs
    sys_rttms_torch: List[Path] = []
    sys_rttms_onnx: List[Path] = []
    for ap, file_id in zip(audio_paths, file_ids):

        x_ch1, _duration = load_audio_mono_16k(ap.as_posix(), sample_rate=args.sample_rate)
        x_np = x_ch1[None, ...].astype(np.float32)  # (1,1,samples)

        # torch
        with torch.inference_mode():
            y_t = torch_model(torch.from_numpy(x_np).to(device)).cpu().numpy()[0].astype(np.uint8)  # (T,S)

        # onnx
        y_o = sess.run([ort_out], {ort_in: x_np})[0][0].astype(np.uint8)  # (T,S)

        # segments + rttm
        for tag, y, out_base, out_list in [
            ("pytorch", y_t, out_torch, sys_rttms_torch),
            ("onnx", y_o, out_onnx, sys_rttms_onnx),
        ]:
            # drop inactive speakers
            active = y.sum(axis=0) > 0
            y_active = y[:, active]
            segments = frames_to_segments(y_active, frame_step=args.frame_step, min_duration=args.min_duration)
            rttm_path = out_base / f"{file_id}.rttm"
            write_rttm(segments, file_id, rttm_path.as_posix())
            out_list.append(rttm_path)

    # Score both systems
    uem_path = Path(args.uem) if args.uem else None
    per_file_t, overall_t = _score_with_dscore(
        ref_rttms=ref_rttms,
        sys_rttms=sys_rttms_torch,
        uem=uem_path,
        collar=float(args.collar),
        ignore_overlaps=bool(args.ignore_overlaps),
        step=float(args.step),
    )
    per_file_o, overall_o = _score_with_dscore(
        ref_rttms=ref_rttms,
        sys_rttms=sys_rttms_onnx,
        uem=uem_path,
        collar=float(args.collar),
        ignore_overlaps=bool(args.ignore_overlaps),
        step=float(args.step),
    )

    # Summaries + deltas (torch - onnx)
    summary_t = ScoreSummary(**{k: float(overall_t[k]) for k in ScoreSummary.__annotations__.keys()})
    summary_o = ScoreSummary(**{k: float(overall_o[k]) for k in ScoreSummary.__annotations__.keys()})
    delta = {k: float(getattr(summary_t, k) - getattr(summary_o, k)) for k in ScoreSummary.__annotations__.keys()}

    report = {
        "args": vars(args),
        "paths": {
            "out_dir": out_dir.as_posix(),
            "pytorch_rttm_dir": out_torch.as_posix(),
            "onnx_rttm_dir": out_onnx.as_posix(),
        },
        "pytorch": {
            "overall": overall_t,
            "overall_summary": asdict(summary_t),
            "per_file": per_file_t,
        },
        "onnx": {
            "overall": overall_o,
            "overall_summary": asdict(summary_o),
            "per_file": per_file_o,
        },
        "delta__pytorch_minus_onnx": delta,
    }

    report_path = out_dir / "report.json"
    _dump_json(report, report_path)

    # Print high-signal summary
    print("Wrote:", report_path)
    print("Overall (PyTorch):", asdict(summary_t))
    print("Overall (ONNX):   ", asdict(summary_o))
    print("Delta (T - O):    ", delta)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
随机从训练数据采样若干条（默认 100），分别用「微调前基线」与「微调后模型」做推理，
同时可视化波形、GT、基线预测、微调预测四行对比。

示例：
python sample_infer_viz.py \
  --wav-scp data/kaldi_merged_1205_1207_ft/train/wav.scp \
  --rttm data/kaldi_merged_1205_1207_ft/train/rttm \
  --ckpt-dir exp/kaldi_merged_1205_1207_ft/checkpoints/epoch_0008 \
  --config exp/kaldi_merged_1205_1207_ft/config__2025_12_08--20_44_28.toml \
  --out-dir exp/kaldi_merged_1205_1207_ft/sample_viz
"""

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torchaudio

from diarizen.pipelines.inference import DiariZenPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SNAPSHOT_DIR = REPO_ROOT / "cache/models--BUT-FIT--diarizen-wavlm-base-s80-md/snapshots/a9857fc34908197fb5336d9d0562f291834a04b2"
DEFAULT_BASE_CONFIG = BASE_SNAPSHOT_DIR / "config.toml"
DEFAULT_BASE_MODEL = "BUT-FIT/diarizen-wavlm-base-s80-md"
DEFAULT_HF_CACHE = REPO_ROOT / "cache"


def load_scp(scp_path: Path) -> Dict[str, str]:
    return dict(line.strip().split(None, 1) for line in scp_path.read_text().splitlines() if line.strip())


def load_rttm(rttm_path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    out: Dict[str, List[Tuple[float, float, str]]] = {}
    with rttm_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            session = parts[1]
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[-2] if parts[-2] != "<NA>" else parts[-3]
            out.setdefault(session, []).append((start, start + dur, spk))
    return out


def _merge_inference_config(train_config_path: Path, base_config_path: Path) -> Path:
    import toml

    train_cfg = toml.load(train_config_path)
    base_cfg = toml.load(base_config_path)
    merged = dict(train_cfg)
    if "inference" not in merged and "inference" in base_cfg:
        merged["inference"] = base_cfg["inference"]
    if "clustering" not in merged and "clustering" in base_cfg:
        merged["clustering"] = base_cfg["clustering"]
    tmp_cfg = Path(tempfile.mktemp(prefix="diarizen_cfg_", suffix=".toml"))
    with open(tmp_cfg, "w") as f:
        toml.dump(merged, f)
    return tmp_cfg


def prepare_diarizen_dir(ckpt_dir: Path, config_path: Path, base_config_path: Path, plda_dir: Path = None) -> Path:
    staging = Path(tempfile.mkdtemp(prefix="diarizen_finetune_sample_"))
    model_path = ckpt_dir / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {model_path}")
    shutil.copy(model_path, staging / "pytorch_model.bin")
    merged_cfg = _merge_inference_config(config_path, base_config_path)
    shutil.copy(merged_cfg, staging / "config.toml")
    target_plda = staging / "plda"
    if target_plda.exists():
        shutil.rmtree(target_plda)
    if plda_dir and plda_dir.is_dir():
        shutil.copytree(plda_dir, target_plda)
    else:
        candidates = [
            ckpt_dir.parent / "plda",
            BASE_SNAPSHOT_DIR / "plda",
            base_config_path.parent / "plda",
        ]
        for cand in candidates:
            if cand.is_dir():
                shutil.copytree(cand, target_plda)
                break
    return staging


def plot_diar(
    wav_path: Path,
    gt_segments: List[Tuple[float, float, str]],
    pred_segments_ft,
    pred_segments_base,
    out_png: Path,
):
    """
    绘制四行对比：波形 / GT / 基线预测 / 微调预测。
    pred_segments_ft / pred_segments_base 期待为 pyannote 的 diarization 对象或 None。
    """

    def diar_to_segments(diar) -> Tuple[List[Tuple[float, float, str]], List[str]]:
        if diar is None:
            return [], []
        segs: List[Tuple[float, float, str]] = []
        speakers = set()
        for segment, _, label in diar.itertracks(yield_label=True):
            speakers.add(label)
            segs.append((segment.start, segment.end, label))
        return segs, sorted(speakers)

    waveform, sr = torchaudio.load(wav_path)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    t = torch.linspace(0, waveform.shape[-1] / sr, steps=waveform.shape[-1])

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1]}
    )
    axes[0].plot(t, waveform[0].numpy(), color="gray", alpha=0.5, linewidth=0.5)
    axes[0].set_title("Waveform")

    gt_speakers = sorted({spk for _, _, spk in gt_segments}) if gt_segments else []
    segs_base, base_speakers = diar_to_segments(pred_segments_base)
    segs_ft, ft_speakers = diar_to_segments(pred_segments_ft)

    colors_gt = cm.get_cmap("tab20", max(len(gt_speakers), 1))
    colors_base = cm.get_cmap("tab20", max(len(base_speakers), 1))
    colors_ft = cm.get_cmap("tab20", max(len(ft_speakers), 1))
    spk_color_gt = {spk: colors_gt(i) for i, spk in enumerate(gt_speakers)}
    spk_color_base = {spk: colors_base(i) for i, spk in enumerate(base_speakers)}
    spk_color_ft = {spk: colors_ft(i) for i, spk in enumerate(ft_speakers)}

    # GT
    for spk in gt_speakers:
        y = gt_speakers.index(spk)
        for st, ed, s in gt_segments:
            if s != spk:
                continue
            axes[1].broken_barh([(st, ed - st)], (y, 0.8), facecolors=spk_color_gt.get(spk, "C0"), alpha=0.6)
    if gt_speakers:
        axes[1].set_ylim(0, len(gt_speakers) + 1)
        axes[1].set_yticks([i + 0.4 for i in range(len(gt_speakers))])
        axes[1].set_yticklabels(gt_speakers, fontsize=8)
    axes[1].set_ylabel("GT")
    axes[1].set_title("Ground Truth")

    # Base (pre-finetune)
    for spk in base_speakers:
        y = base_speakers.index(spk)
        for st, ed, s in segs_base:
            if s != spk:
                continue
            axes[2].broken_barh([(st, ed - st)], (y, 0.8), facecolors=spk_color_base.get(spk, "C1"), alpha=0.6)
    if base_speakers:
        axes[2].set_ylim(0, len(base_speakers) + 1)
        axes[2].set_yticks([i + 0.4 for i in range(len(base_speakers))])
        axes[2].set_yticklabels(base_speakers, fontsize=8)
    axes[2].set_ylabel("Base")
    axes[2].set_title("Pre-finetune Prediction")

    # Finetune
    for spk in ft_speakers:
        y = ft_speakers.index(spk)
        for st, ed, s in segs_ft:
            if s != spk:
                continue
            axes[3].broken_barh([(st, ed - st)], (y, 0.8), facecolors=spk_color_ft.get(spk, "C2"), alpha=0.6)
    if ft_speakers:
        axes[3].set_ylim(0, len(ft_speakers) + 1)
        axes[3].set_yticks([i + 0.4 for i in range(len(ft_speakers))])
        axes[3].set_yticklabels(ft_speakers, fontsize=8)
    axes[3].set_ylabel("Finetune")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Post-finetune Prediction")

    axes[3].set_xlim(0, t[-1].item() if len(t) else None)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="采样若干条并对比微调前/后的推理结果（GT vs Base vs Finetune）"
    )
    parser.add_argument("--wav-scp", type=str, required=True)
    parser.add_argument("--rttm", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-config", type=str, default=None, help="用于补齐 inference 的 base config（默认使用仓库缓存）")
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"基线模型 repo_id 或本地路径（与 batch_diarize_any_audio.py 保持一致，默认 {DEFAULT_BASE_MODEL}）",
    )
    parser.add_argument(
        "--base-ckpt-dir",
        type=str,
        default=None,
        help="可选：使用本地基线 checkpoint 目录（若提供则优先于 base-model，并按 config/PLDA 补齐）",
    )
    parser.add_argument("--plda-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu"])
    parser.add_argument("--segmentation-only", action="store_true", default=True, help="只跑 segmentation（默认开启）")
    parser.add_argument("--full-pipeline", action="store_false", dest="segmentation_only", help="关闭 segmentation-only，启用 embedding+clustering")
    parser.add_argument("--binarize-onset", type=float, default=0.5, help="与 batch_diarize_any_audio.py 保持一致，降低可提高召回")
    parser.add_argument("--binarize-offset", type=float, default=None, help="默认等于 onset")
    parser.add_argument("--binarize-min-duration-on", type=float, default=0.0, help="同 batch_diarize_any_audio.py")
    parser.add_argument("--binarize-min-duration-off", type=float, default=0.0, help="同 batch_diarize_any_audio.py")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--hf-cache",
        type=str,
        default=str(DEFAULT_HF_CACHE),
        help=f"指定 Hugging Face 缓存目录（默认 {DEFAULT_HF_CACHE}，已有文件可复用，避免重复下载）",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config_path = Path(args.base_config) if args.base_config else DEFAULT_BASE_CONFIG
    base_ckpt_dir = Path(args.base_ckpt_dir) if args.base_ckpt_dir else None
    base_model = args.base_model.strip() if args.base_model else DEFAULT_BASE_MODEL
    ckpt_dir = Path(args.ckpt_dir)
    config_path = Path(args.config)
    plda_dir = Path(args.plda_dir) if args.plda_dir else None
    hf_cache = Path(args.hf_cache).expanduser() if args.hf_cache else DEFAULT_HF_CACHE
    cache_ft = hf_cache if hf_cache else out_dir / "cache_finetune"
    cache_base = hf_cache if hf_cache else out_dir / "cache_base"
    cache_ft.mkdir(parents=True, exist_ok=True)
    cache_base.mkdir(parents=True, exist_ok=True)

    wav_scp = load_scp(Path(args.wav_scp))
    rttm_dict = load_rttm(Path(args.rttm))

    sessions = list(wav_scp.keys())
    if len(sessions) == 0:
        raise ValueError("wav.scp 为空")
    random.shuffle(sessions)
    sessions = sessions[: args.num_samples]

    diarizen_hub_ft = prepare_diarizen_dir(ckpt_dir, config_path, base_config_path, plda_dir=plda_dir)
    if base_ckpt_dir:
        # 若提供本地基线 checkpoint，则按微调同样方式补齐 config/PLDA
        diarizen_hub_base = prepare_diarizen_dir(base_ckpt_dir, base_config_path, base_config_path, plda_dir=plda_dir)
        base_repo_id = str(diarizen_hub_base)
    else:
        # 否则直接使用 repo_id 或本地模型路径（与 batch_diarize_any_audio.py 一致）
        base_repo_id = base_model

    # 设备
    device = None
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device in ["cuda", "gpu"]:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    pipeline_ft = DiariZenPipeline.from_pretrained(
        repo_id=str(diarizen_hub_ft),
        cache_dir=str(cache_ft),
        rttm_out_dir=str(out_dir / "rttm_pred_finetune"),
        device=device,
        segmentation_only=args.segmentation_only,
        binarize_onset=args.binarize_onset,
        binarize_offset=args.binarize_offset,
        binarize_min_duration_on=args.binarize_min_duration_on,
        binarize_min_duration_off=args.binarize_min_duration_off,
    )
    pipeline_base = DiariZenPipeline.from_pretrained(
        repo_id=base_repo_id,
        cache_dir=str(cache_base),
        rttm_out_dir=str(out_dir / "rttm_pred_base"),
        device=device,
        segmentation_only=args.segmentation_only,
        binarize_onset=args.binarize_onset,
        binarize_offset=args.binarize_offset,
        binarize_min_duration_on=args.binarize_min_duration_on,
        binarize_min_duration_off=args.binarize_min_duration_off,
    )
    rttm_pred_ft_dir = out_dir / "rttm_pred_finetune"
    rttm_pred_base_dir = out_dir / "rttm_pred_base"
    plots_dir = out_dir / "plots"
    wavs_dir = out_dir / "wavs"
    rttm_gt_dir = out_dir / "rttm_gt"

    rttm_pred_ft_dir.mkdir(parents=True, exist_ok=True)
    rttm_pred_base_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(parents=True, exist_ok=True)
    rttm_gt_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for idx, sess in enumerate(sessions, 1):
        wav_path = Path(wav_scp[sess])
        gt_segments = rttm_dict.get(sess, [])

        info = torchaudio.info(wav_path)
        duration = float(info.num_frames) / float(info.sample_rate) if info.sample_rate else 0.0
        print(f"[{idx}/{len(sessions)}] {sess}")

        # 拷贝原始音频和 GT RTTM
        wav_copy = wavs_dir / wav_path.name
        try:
            shutil.copy2(wav_path, wav_copy)
        except Exception as e_copy_wav:
            print(f"拷贝音频失败 {sess}: {e_copy_wav}")
        rttm_gt_copy = None
        if gt_segments:
            rttm_gt_copy = rttm_gt_dir / f"{sess}.rttm"
            try:
                with open(rttm_gt_copy, "w") as f:
                    for st, ed, spk in gt_segments:
                        dur = ed - st
                        f.write(f"SPEAKER {sess} 1 {st:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")
            except Exception as e_gt:
                print(f"写 GT RTTM 失败 {sess}: {e_gt}")
                rttm_gt_copy = None

        diar_base = None
        diar_ft = None

        # 基线模型推理
        try:
            diar_base = pipeline_base(str(wav_path), sess_name=sess)
        except Exception as e_infer_base:
            print(f"基线推理失败 {sess}: {e_infer_base}")

        # 微调模型推理
        try:
            diar_ft = pipeline_ft(str(wav_path), sess_name=sess)
        except Exception as e_infer_ft:
            print(f"微调推理失败 {sess}: {e_infer_ft}")

        pred_rttm_base = rttm_pred_base_dir / f"{sess}.rttm"
        pred_rttm_ft = rttm_pred_ft_dir / f"{sess}.rttm"
        plot_path = None

        if diar_base is not None:
            try:
                with open(pred_rttm_base, "w") as f:
                    diar_base.write_rttm(f)
            except Exception as e_write_base:
                print(f"写基线 RTTM 失败 {sess}: {e_write_base}")
                diar_base = None
        else:
            try:
                with open(pred_rttm_base, "w") as f:
                    pass
            except Exception:
                pass

        if diar_ft is not None:
            try:
                with open(pred_rttm_ft, "w") as f:
                    diar_ft.write_rttm(f)
            except Exception as e_write_ft:
                print(f"写微调 RTTM 失败 {sess}: {e_write_ft}")
                diar_ft = None
        else:
            try:
                with open(pred_rttm_ft, "w") as f:
                    pass
            except Exception:
                pass

        if diar_base is not None or diar_ft is not None:
            try:
                plot_path = plots_dir / f"{sess}.png"
                plot_diar(wav_path, gt_segments, diar_ft, diar_base, plot_path)
            except Exception as e_plot:
                print(f"绘图失败 {sess}: {e_plot}")
                plot_path = None

        summary.append(
            {
                "session": sess,
                "wav": str(wav_path),
                "wav_copied": str(wav_copy),
                "duration": duration,
                "rttm_pred_base": str(pred_rttm_base),
                "rttm_pred_finetune": str(pred_rttm_ft),
                "rttm_gt": str(rttm_gt_copy) if rttm_gt_copy else None,
                "plot": str(plot_path) if plot_path else None,
            }
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"完成推理与可视化，结果见 {out_dir}")


if __name__ == "__main__":
    main()


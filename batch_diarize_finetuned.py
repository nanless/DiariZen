#!/usr/bin/env python3
"""
批量对目录内音频进行 diarization，使用自训练/finetune 的 DiariZen checkpoint。

用法示例：
python batch_diarize_finetuned.py \
  /path/to/audios \
  --ckpt-dir /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1205_1207_ft/checkpoints/epoch_0008 \
  --config /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1205_1207_ft/config__2025_12_08--20_44_28.toml \
  --out-dir /path/to/out \
  --device auto
"""

import argparse
import glob
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import torch
import torchaudio

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diarizen.pipelines.inference import DiariZenPipeline

# 默认基座配置（用于从训练 config 中补齐 inference/clustering 部分）
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = REPO_ROOT / "cache/models--BUT-FIT--diarizen-wavlm-base-s80-md/snapshots/a9857fc34908197fb5336d9d0562f291834a04b2/config.toml"


def list_audio_files(root_dir: str) -> List[str]:
    exts = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac", "*.wma"]
    files = set()
    for ext in exts:
        files.update(Path(root_dir).rglob(ext))
    files = [f for f in files if f.is_file() and f.stat().st_size > 1024]
    return sorted(str(f) for f in files)


def _merge_inference_config(train_config_path: Path, base_config_path: Path) -> Path:
    """
    Merge training config (no inference section) with base inference config.
    Priority: training config overrides base for model args; inference/clustering taken from base when missing.
    Returns path to a temporary merged config.
    """
    import toml
    train_cfg = toml.load(train_config_path)
    base_cfg = toml.load(base_config_path)

    merged = dict(train_cfg)
    # fill inference / clustering if absent
    if "inference" not in merged and "inference" in base_cfg:
        merged["inference"] = base_cfg["inference"]
    if "clustering" not in merged and "clustering" in base_cfg:
        merged["clustering"] = base_cfg["clustering"]

    tmp_cfg = Path(tempfile.mktemp(prefix="diarizen_cfg_", suffix=".toml"))
    with open(tmp_cfg, "w") as f:
        toml.dump(merged, f)
    return tmp_cfg


def prepare_diarizen_dir(
    ckpt_dir: Path,
    config_path: Path = None,
    plda_dir: Path = None,
    base_config_path: Path = None,
) -> Path:
    """
    Build a minimal "diarizen hub" directory containing:
    - config.toml
    - pytorch_model.bin
    - optional plda/ directory (for VBx)
    """
    staging = Path(tempfile.mkdtemp(prefix="diarizen_finetune_"))

    # copy checkpoint
    src_model = ckpt_dir / "pytorch_model.bin"
    if not src_model.exists():
        raise FileNotFoundError(f"checkpoint not found: {src_model}")
    shutil.copy(src_model, staging / "pytorch_model.bin")

    # config
    if config_path is None:
        candidates = sorted((ckpt_dir.parent.glob("config__*.toml")), reverse=True)
        if not candidates:
            raise FileNotFoundError("config not specified and no config__*.toml found near checkpoint")
        config_path = candidates[0]

    if base_config_path is None:
        base_config_path = DEFAULT_BASE_CONFIG
    if not base_config_path.exists():
        raise FileNotFoundError(f"base config not found: {base_config_path}")

    # merge to ensure inference/clustering exist
    merged_cfg = _merge_inference_config(config_path, base_config_path)
    shutil.copy(merged_cfg, staging / "config.toml")

    # plda (optional but needed if config uses VBx)
    if plda_dir and plda_dir.is_dir():
        shutil.copytree(plda_dir, staging / "plda")
    else:
        # try inherit from checkpoint parent if exists
        parent_plda = ckpt_dir.parent / "plda"
        if parent_plda.is_dir():
            shutil.copytree(parent_plda, staging / "plda")

    return staging


def main():
    parser = argparse.ArgumentParser(description="Batch diarization with finetuned DiariZen checkpoint")
    parser.add_argument("in_root", type=str, help="输入音频根目录")
    parser.add_argument("--out-dir", type=str, required=True, help="RTTM 输出目录")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="finetune checkpoint 目录，含 pytorch_model.bin")
    parser.add_argument("--config", type=str, default=None, help="config.toml 路径；若缺省自动取 exp 下最新 config__*.toml")
    parser.add_argument("--plda-dir", type=str, default=None, help="VBx 所需的 plda 目录，可选")
    parser.add_argument("--base-config", type=str, default=None, help="用于补齐 inference/clustering 的 base config（默认使用仓库内缓存的 base 配置）")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="模型缓存目录（用于嵌入模型下载）")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu"], help="设备选择")
    parser.add_argument("--segmentation-only", action="store_true", default=True, help="只跑 segmentation，跳过 embedding+clustering（默认开启）")
    parser.add_argument("--full-pipeline", action="store_false", dest="segmentation_only", help="关闭 segmentation-only，启用 embedding+clustering")
    parser.add_argument("--binarize-onset", type=float, default=0.5)
    parser.add_argument("--binarize-offset", type=float, default=None)
    parser.add_argument("--binarize-min-duration-on", type=float, default=0.0)
    parser.add_argument("--binarize-min-duration-off", type=float, default=0.0)
    parser.add_argument("--seg-duration", type=float, default=None, help="Inference segment duration (seconds); if None, uses config or model default")
    parser.add_argument("--seg-step", type=float, default=None, help="Inference segment step ratio (0-1); if None, uses config default")
    parser.add_argument("--full-utterance", action="store_true", help="Use full utterance inference (no sliding window)")
    parser.add_argument("--plot", action="store_true", default=True, help="为每个样本生成可视化 png（默认开启）")
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    ckpt_dir = Path(args.ckpt_dir)
    config_path = Path(args.config) if args.config else None
    plda_dir = Path(args.plda_dir) if args.plda_dir else None
    base_config_path = Path(args.base_config) if args.base_config else DEFAULT_BASE_CONFIG

    audios = list_audio_files(str(in_root))
    if not audios:
        print(f"未找到音频: {in_root}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    diarizen_hub = prepare_diarizen_dir(
        ckpt_dir,
        config_path=config_path,
        plda_dir=plda_dir,
        base_config_path=base_config_path,
    )
    print(f"使用 checkpoint: {ckpt_dir}")
    print(f"使用 config: {config_path if config_path else 'auto-detected'}")
    print(f"base inference config: {base_config_path}")
    print(f"临时模型目录: {diarizen_hub}")

    # 设备选择
    device = None
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device in ["cuda", "gpu"]:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("警告: CUDA 不可用，退回 CPU")
            device = torch.device("cpu")

    # 构建 config_parse 以覆盖默认推理配置
    config_parse = None
    if args.seg_duration is not None or args.seg_step is not None or args.full_utterance:
        config_parse = {
            "inference": {"args": {}},
            "clustering": {"args": {}}
        }
        if args.seg_duration is not None:
            config_parse["inference"]["args"]["seg_duration"] = args.seg_duration
        if args.seg_step is not None:
            config_parse["inference"]["args"]["segmentation_step"] = args.seg_step
        if args.full_utterance:
            # 在 DiariZenPipeline 中，通常通过设置大的 seg_duration 来模拟整句推理
            # 或者在 pipeline 内部根据 full_utterance 标志位处理
            # 这里的逻辑取决于 DiariZenPipeline 如何响应 config
            config_parse["inference"]["args"]["seg_duration"] = 1000000.0 # 极大值模拟整句
            config_parse["inference"]["args"]["segmentation_step"] = 1.0

    pipeline = DiariZenPipeline.from_pretrained(
        repo_id=str(diarizen_hub),
        cache_dir=args.cache_dir,
        rttm_out_dir=str(out_dir),
        device=device,
        segmentation_only=args.segmentation_only,
        binarize_onset=args.binarize_onset,
        binarize_offset=args.binarize_offset,
        binarize_min_duration_on=args.binarize_min_duration_on,
        binarize_min_duration_off=args.binarize_min_duration_off,
        config_parse=config_parse,
    )

    summary = []
    for wav in audios:
        rel = Path(wav).relative_to(in_root)
        sess_name = rel.with_suffix("").as_posix().replace("/", "__")
        info = torchaudio.info(wav)
        duration = float(info.num_frames) / float(info.sample_rate) if info.sample_rate else 0.0

        print(f"[Diar] {wav}")
        diar = pipeline(wav, sess_name=sess_name)
        rttm_path = out_dir / f"{sess_name}.rttm"
        # pipeline 已写 RTTM；保守再确认
        if not rttm_path.exists():
            diar.write_rttm(str(rttm_path))

        plot_path = None
        if args.plot:
            plot_path = out_dir / f"{sess_name}.png"
            try:
                # 读取波形用于背景参考（单通道）
                waveform, sr = torchaudio.load(wav)
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                t = torch.linspace(0, waveform.shape[-1] / sr, steps=waveform.shape[-1])

                speakers = sorted(set(label for _, _, label in diar.itertracks(yield_label=True)))
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

                ax_wave.plot(t, waveform[0].numpy(), color="gray", alpha=0.6, linewidth=0.5)
                ax_wave.set_ylabel("Amplitude")
                ax_wave.set_title(sess_name)
                ax_wave.set_xlim(0, duration if duration > 0 else None)

                if speakers:
                    for segment, _, spk in diar.itertracks(yield_label=True):
                        y = spk_pos[spk]
                        ax_diar.broken_barh(
                            [(segment.start, segment.end - segment.start)],
                            (y - 0.4, 0.8),
                            facecolors=spk_color.get(spk, "C0"),
                            alpha=0.8,
                        )
                    ax_diar.set_yticks(list(spk_pos.values()))
                    ax_diar.set_yticklabels(list(spk_pos.keys()))
                    ax_diar.set_ylim(-0.5, len(speakers) - 0.5)
                else:
                    ax_diar.text(
                        0.5,
                        0.5,
                        "no speech",
                        ha="center",
                        va="center",
                        transform=ax_diar.transAxes,
                    )
                    ax_diar.set_ylim(-0.5, 0.5)

                ax_diar.set_xlabel("Time (s)")
                ax_diar.set_ylabel("Speaker")
                ax_diar.grid(True, axis="x", linestyle="--", alpha=0.3)

                fig.tight_layout()
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"绘图失败 {wav}: {e}")
                plot_path = None

        summary.append(
            {
                "wav": wav,
                "session": sess_name,
                "duration": duration,
                "rttm": str(rttm_path),
                "plot": str(plot_path) if plot_path else None,
            }
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"完成，共处理 {len(summary)} 条音频，汇总写入 {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()


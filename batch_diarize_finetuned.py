#!/usr/bin/env python3
"""
批量说话人分离脚本

该脚本用于批量处理目录内的音频文件，使用自训练或微调的DiariZen模型进行说话人分离。
主要功能：
1. 扫描目录中的所有音频文件
2. 准备模型checkpoint和配置文件
3. 对每个音频文件进行说话人分离
4. 生成RTTM标注文件和可视化图表
5. 生成处理汇总报告

用法示例：
python batch_diarize_finetuned.py \
  /path/to/audios \
  --ckpt-dir /path/to/checkpoints/epoch_0008 \
  --config /path/to/config.toml \
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

matplotlib.use("Agg")  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt

from diarizen.pipelines.inference import DiariZenPipeline

# 默认基座配置路径（用于从训练config中补齐inference/clustering部分）
# 训练时的config可能不包含推理和聚类配置，需要从基础配置中补充
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = REPO_ROOT / "cache/models--BUT-FIT--diarizen-wavlm-base-s80-md/snapshots/a9857fc34908197fb5336d9d0562f291834a04b2/config.toml"


def list_audio_files(root_dir: str) -> List[str]:
    """递归扫描目录，列出所有音频文件

    支持的音频格式：wav, mp3, m4a, flac, ogg, aac, wma
    只返回文件大小大于1KB的文件（过滤掉空文件或损坏文件）

    参数
    ----------
    root_dir : str
        要扫描的根目录路径

    返回
    -------
    List[str]
        音频文件路径列表（已排序）
    """
    # 支持的音频文件扩展名
    exts = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac", "*.wma"]
    files = set()
    # 递归搜索所有匹配的文件
    for ext in exts:
        files.update(Path(root_dir).rglob(ext))
    # 过滤：只保留文件且大小大于1KB
    files = [f for f in files if f.is_file() and f.stat().st_size > 1024]
    # 返回排序后的文件路径列表
    return sorted(str(f) for f in files)


def _merge_inference_config(train_config_path: Path, base_config_path: Path) -> Path:
    """合并训练配置和基础推理配置

    训练时的配置文件可能不包含inference和clustering部分，
    需要从基础配置中补充这些部分。

    合并策略：
    - 训练配置优先：模型参数使用训练配置
    - 基础配置补充：如果训练配置缺少inference/clustering部分，从基础配置中补充

    参数
    ----------
    train_config_path : Path
        训练配置文件路径（可能缺少inference/clustering部分）
    base_config_path : Path
        基础配置文件路径（包含完整的inference/clustering配置）

    返回
    -------
    Path
        临时合并后的配置文件路径
    """
    import toml
    # 加载两个配置文件
    train_cfg = toml.load(train_config_path)
    base_cfg = toml.load(base_config_path)

    # 从训练配置开始
    merged = dict(train_cfg)
    
    # 如果训练配置缺少inference部分，从基础配置补充
    if "inference" not in merged and "inference" in base_cfg:
        merged["inference"] = base_cfg["inference"]
    # 如果训练配置缺少clustering部分，从基础配置补充
    if "clustering" not in merged and "clustering" in base_cfg:
        merged["clustering"] = base_cfg["clustering"]

    # 创建临时配置文件
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
    """准备DiariZen模型目录（hub格式）

    构建一个最小化的DiariZen hub目录，包含：
    - config.toml（合并后的配置文件）
    - pytorch_model.bin（模型权重）
    - plda/目录（可选，VBx聚类所需）

    参数
    ----------
    ckpt_dir : Path
        checkpoint目录路径，应包含pytorch_model.bin
    config_path : Path, 可选
        训练配置文件路径；如果为None，自动查找checkpoint父目录下的config__*.toml
    plda_dir : Path, 可选
        PLDA目录路径（用于VBx聚类）；如果为None，尝试从checkpoint父目录继承
    base_config_path : Path, 可选
        基础配置文件路径（用于补充inference/clustering配置）

    返回
    -------
    Path
        临时staging目录路径，包含完整的模型文件

    异常
    -----
    FileNotFoundError
        - checkpoint文件不存在
        - 配置文件未找到
        - 基础配置文件不存在
    """
    # 创建临时staging目录
    staging = Path(tempfile.mkdtemp(prefix="diarizen_finetune_"))

    # 复制checkpoint文件
    src_model = ckpt_dir / "pytorch_model.bin"
    if not src_model.exists():
        raise FileNotFoundError(f"checkpoint not found: {src_model}")
    shutil.copy(src_model, staging / "pytorch_model.bin")

    # 处理配置文件
    if config_path is None:
        # 自动查找checkpoint父目录下最新的config__*.toml文件
        candidates = sorted((ckpt_dir.parent.glob("config__*.toml")), reverse=True)
        if not candidates:
            raise FileNotFoundError("config not specified and no config__*.toml found near checkpoint")
        config_path = candidates[0]

    # 设置基础配置路径
    if base_config_path is None:
        base_config_path = DEFAULT_BASE_CONFIG
    if not base_config_path.exists():
        raise FileNotFoundError(f"base config not found: {base_config_path}")

    # 合并配置，确保inference/clustering部分存在
    merged_cfg = _merge_inference_config(config_path, base_config_path)
    shutil.copy(merged_cfg, staging / "config.toml")

    # 处理PLDA目录（VBx聚类所需）
    if plda_dir and plda_dir.is_dir():
        # 如果指定了plda_dir，直接复制
        shutil.copytree(plda_dir, staging / "plda")
    else:
        # 否则尝试从checkpoint父目录继承
        parent_plda = ckpt_dir.parent / "plda"
        if parent_plda.is_dir():
            shutil.copytree(parent_plda, staging / "plda")

    return staging


def main():
    """主函数：批量处理音频文件进行说话人分离
    
    流程：
    1. 解析命令行参数
    2. 扫描音频文件
    3. 准备模型目录和配置
    4. 初始化推理管道
    5. 逐个处理音频文件
    6. 生成RTTM文件和可视化图表
    7. 生成处理汇总报告
    """
    parser = argparse.ArgumentParser(description="Batch diarization with finetuned DiariZen checkpoint")
    
    # 必需参数
    parser.add_argument("in_root", type=str, help="输入音频根目录")
    parser.add_argument("--out-dir", type=str, required=True, help="RTTM输出目录")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="finetune checkpoint目录，需包含pytorch_model.bin")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, default=None, help="config.toml路径；若缺省自动取exp下最新config__*.toml")
    parser.add_argument("--plda-dir", type=str, default=None, help="VBx所需的plda目录，可选")
    parser.add_argument("--base-config", type=str, default=None, help="用于补齐inference/clustering的base config（默认使用仓库内缓存的base配置）")
    
    # 模型和系统参数
    parser.add_argument("--cache-dir", type=str, default="./cache", help="模型缓存目录（用于嵌入模型下载）")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu"], help="设备选择：auto自动检测，cpu使用CPU，cuda/gpu使用GPU")
    
    # 推理模式参数
    parser.add_argument("--segmentation-only", action="store_true", default=True, help="只跑segmentation，跳过embedding+clustering（默认开启）")
    parser.add_argument("--full-pipeline", action="store_false", dest="segmentation_only", help="关闭segmentation-only，启用embedding+clustering完整流程")
    
    # Binarize参数（用于将segmentation转换为最终diarization）
    parser.add_argument("--binarize-onset", type=float, default=0.5, help="说话开始阈值（0-1），高于此值认为说话开始")
    parser.add_argument("--binarize-offset", type=float, default=None, help="说话结束阈值（0-1），低于此值认为说话结束；默认与onset相同")
    parser.add_argument("--binarize-min-duration-on", type=float, default=0.0, help="说话段最小持续时间（秒）")
    parser.add_argument("--binarize-min-duration-off", type=float, default=0.0, help="静音段最小持续时间（秒）")
    
    # 分段参数
    parser.add_argument("--seg-duration", type=float, default=None, help="推理分段持续时间（秒）；如果为None，使用config或模型默认值")
    parser.add_argument("--seg-step", type=float, default=None, help="推理分段步长比例（0-1）；如果为None，使用config默认值")
    parser.add_argument("--full-utterance", action="store_true", help="使用整句推理模式（不使用滑动窗口）")
    
    # 可视化参数
    parser.add_argument("--plot", action="store_true", default=True, help="为每个样本生成可视化png（默认开启）")
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="关闭可视化")
    
    args = parser.parse_args()

    # 转换路径参数
    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    ckpt_dir = Path(args.ckpt_dir)
    config_path = Path(args.config) if args.config else None
    plda_dir = Path(args.plda_dir) if args.plda_dir else None
    base_config_path = Path(args.base_config) if args.base_config else DEFAULT_BASE_CONFIG

    # 扫描音频文件
    audios = list_audio_files(str(in_root))
    if not audios:
        print(f"未找到音频: {in_root}")
        return
    # 创建输出目录
    out_dir.mkdir(parents=True, exist_ok=True)

    # 准备模型目录（hub格式）
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

    # 设备选择逻辑
    device = None
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device in ["cuda", "gpu"]:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("警告: CUDA 不可用，退回 CPU")
            device = torch.device("cpu")
    # 如果device为None（auto模式），将在pipeline初始化时自动检测

    # 构建config_parse以覆盖默认推理配置
    # 如果命令行指定了分段参数，则覆盖配置文件中的设置
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
            # 整句推理模式：设置极大的seg_duration来模拟整句推理
            # 在DiariZenPipeline中会检测到该值并切换到整句模式
            config_parse["inference"]["args"]["seg_duration"] = 1000000.0  # 极大值模拟整句
            config_parse["inference"]["args"]["segmentation_step"] = 1.0

    # 初始化推理管道
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

    # 处理每个音频文件
    summary = []
    for wav in audios:
        # 生成会话名称：使用相对路径，将路径分隔符替换为双下划线
        rel = Path(wav).relative_to(in_root)
        sess_name = rel.with_suffix("").as_posix().replace("/", "__")
        
        # 获取音频信息
        info = torchaudio.info(wav)
        duration = float(info.num_frames) / float(info.sample_rate) if info.sample_rate else 0.0

        print(f"[Diar] {wav}")
        # 执行说话人分离
        diar = pipeline(wav, sess_name=sess_name)
        
        # 确认RTTM文件已写入（pipeline应该已经写入，这里保守检查）
        rttm_path = out_dir / f"{sess_name}.rttm"
        if not rttm_path.exists():
            diar.write_rttm(str(rttm_path))

        # 生成可视化图表（如果启用）
        plot_path = None
        if args.plot:
            plot_path = out_dir / f"{sess_name}.png"
            try:
                # 读取波形用于背景参考（转换为单通道）
                waveform, sr = torchaudio.load(wav)
                if waveform.dim() > 1:
                    # 多通道音频：取平均值
                    waveform = waveform.mean(dim=0, keepdim=True)
                # 生成时间轴
                t = torch.linspace(0, waveform.shape[-1] / sr, steps=waveform.shape[-1])

                # 获取所有说话人标签并排序
                speakers = sorted(set(label for _, _, label in diar.itertracks(yield_label=True)))
                # 为每个说话人分配颜色（使用tab20调色板）
                colors = plt.cm.get_cmap("tab20", max(len(speakers), 1))
                spk_color = {spk: colors(i) for i, spk in enumerate(speakers)}
                spk_pos = {spk: idx for idx, spk in enumerate(speakers)}  # 说话人在y轴上的位置

                # 创建双子图：上方显示波形，下方显示说话人分离结果
                fig, (ax_wave, ax_diar) = plt.subplots(
                    2,
                    1,
                    sharex=True,  # 共享x轴
                    figsize=(12, 4),
                    gridspec_kw={"height_ratios": [1.4, 1]},  # 波形图占更多空间
                )

                # 绘制波形图
                ax_wave.plot(t, waveform[0].numpy(), color="gray", alpha=0.6, linewidth=0.5)
                ax_wave.set_ylabel("Amplitude")
                ax_wave.set_title(sess_name)
                ax_wave.set_xlim(0, duration if duration > 0 else None)

                # 绘制说话人分离结果
                if speakers:
                    # 为每个说话人段绘制水平条形图
                    for segment, _, spk in diar.itertracks(yield_label=True):
                        y = spk_pos[spk]
                        ax_diar.broken_barh(
                            [(segment.start, segment.end - segment.start)],  # (x位置, 宽度)
                            (y - 0.4, 0.8),  # (y位置, 高度)
                            facecolors=spk_color.get(spk, "C0"),
                            alpha=0.8,
                        )
                    # 设置y轴刻度和标签
                    ax_diar.set_yticks(list(spk_pos.values()))
                    ax_diar.set_yticklabels(list(spk_pos.keys()))
                    ax_diar.set_ylim(-0.5, len(speakers) - 0.5)
                else:
                    # 如果没有检测到说话人，显示提示信息
                    ax_diar.text(
                        0.5,
                        0.5,
                        "no speech",
                        ha="center",
                        va="center",
                        transform=ax_diar.transAxes,
                    )
                    ax_diar.set_ylim(-0.5, 0.5)

                # 设置标签和网格
                ax_diar.set_xlabel("Time (s)")
                ax_diar.set_ylabel("Speaker")
                ax_diar.grid(True, axis="x", linestyle="--", alpha=0.3)

                # 保存图表
                fig.tight_layout()
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"绘图失败 {wav}: {e}")
                plot_path = None

        # 记录处理结果
        summary.append(
            {
                "wav": wav,  # 音频文件路径
                "session": sess_name,  # 会话名称
                "duration": duration,  # 音频时长（秒）
                "rttm": str(rttm_path),  # RTTM文件路径
                "plot": str(plot_path) if plot_path else None,  # 可视化图表路径（如果有）
            }
        )

    # 生成处理汇总报告
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"完成，共处理 {len(summary)} 条音频，汇总写入 {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()


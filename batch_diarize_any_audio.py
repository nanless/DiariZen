import os
import sys
import glob
import json
import time
import argparse
from typing import List

import torch
import torchaudio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diarizen.pipelines.inference import DiariZenPipeline

# CPU 优化设置（全局标志，确保只设置一次）
_cpu_optimizations_setup = False

def setup_cpu_optimizations(force=False):
    """设置 CPU 优化参数（单线程模式）
    
    Args:
        force: 是否强制重新设置（不推荐，可能导致错误）
    """
    global _cpu_optimizations_setup
    
    # 如果已经设置过且不强制，则跳过
    if _cpu_optimizations_setup and not force:
        return
    
    # 使用单线程模式
    num_threads = 1
    
    # 只在第一次调用时设置 interop_threads（避免 "cannot set after parallel work started" 错误）
    if not _cpu_optimizations_setup:
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError as e:
            # 如果已经启动并行工作，忽略这个错误
            print(f"警告: 无法设置 interop_threads（可能已启动并行工作）: {e}")
    
    # 设置线程数（这个可以多次调用）
    torch.set_num_threads(num_threads)
    
    # 启用 MKL 优化
    if hasattr(torch.backends, 'mkl'):
        torch.backends.mkl.enabled = True
    
    # 设置环境变量优化（只在第一次设置）
    if not _cpu_optimizations_setup:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
        os.environ['KMP_BLOCKTIME'] = '0'  # 减少线程阻塞时间
        os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'  # 线程亲和性优化
    
    # 启用 CPU 的 JIT 优化
    torch.set_flush_denormal(True)
    
    _cpu_optimizations_setup = True
    
    if not force:
        print(f"CPU 优化设置（单线程模式）: 线程数={num_threads}, MKL={torch.backends.mkl.is_available() if hasattr(torch.backends, 'mkl') else 'N/A'}")


DEFAULT_DIR = "/root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios"
DEFAULT_OUT_DIR = "/root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios_Diarizen_segmentation_only"
DEFAULT_JSON_PATH = None  # 若未指定，则使用 <out_dir>/DiariZen_diar_summary.json
DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"  # 默认模型

# 预定义的模型快捷名称
MODEL_PRESETS = {
    "base": "BUT-FIT/diarizen-wavlm-base-s80-md",
    "large": "BUT-FIT/diarizen-wavlm-large-s80-md",
}


def list_audio_files(root_dir: str) -> List[str]:
    # 递归匹配常见音频格式
    exts = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac", "*.wma"]
    files = set()
    for ext in exts:
        pattern = os.path.join(root_dir, "**", ext)
        files.update(glob.glob(pattern, recursive=True))
    # 排除隐藏文件或非常小的占位文件
    files = [f for f in files if os.path.getsize(f) > 1024]
    return sorted(files)


def count_unique_speakers(diarization_result) -> int:
    speakers = set()
    for _, _, speaker in diarization_result.itertracks(yield_label=True):
        speakers.add(speaker)
    return len(speakers)


def make_session_name(path: str, root_dir: str) -> str:
    rel = os.path.relpath(path, root_dir)
    base_no_ext = os.path.splitext(rel)[0]
    return base_no_ext.replace(os.sep, "__")


def main():
    parser = argparse.ArgumentParser(
        description="批量处理音频文件进行说话人分离",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "in_root",
        type=str,
        nargs="?",
        default=DEFAULT_DIR,
        help=f"输入音频文件根目录（默认: {DEFAULT_DIR}）"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUT_DIR}）"
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="汇总JSON文件路径（默认: <out_dir>/DiariZen_diar_summary.json）"
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="只使用 segmentation 模型，跳过 embedding 和 clustering（加快速度但可能降低精度）"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="模型缓存目录（默认: ./cache）"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "gpu"],
        default="cpu",
        help="计算设备：auto（自动检测，默认）、cpu、cuda/gpu"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"DiariZen 模型名称或路径。可以是：\n"
             f"  - Hugging Face repo_id（如 'BUT-FIT/diarizen-wavlm-large-s80-md'）\n"
             f"  - 本地路径\n"
             f"  - 预设名称：{', '.join(MODEL_PRESETS.keys())}（如 'base' 或 'large'）\n"
             f"默认: {DEFAULT_MODEL}"
    )
    # DiariZen Binarize参数（用于提高多人检测召回率）
    parser.add_argument(
        "--binarize-onset",
        type=float,
        default=0.5,
        help="Binarize开启阈值，降低可提高召回率（默认: 0.5，推荐范围: 0.3-0.6）"
    )
    parser.add_argument(
        "--binarize-offset",
        type=float,
        default=None,
        help="Binarize关闭阈值，默认等于onset（默认: None，即等于onset）"
    )
    parser.add_argument(
        "--binarize-min-duration-on",
        type=float,
        default=0.0,
        help="Binarize最小语音持续时间（秒，默认: 0.0）"
    )
    parser.add_argument(
        "--binarize-min-duration-off",
        type=float,
        default=0.0,
        help="Binarize最小静音持续时间（秒，默认: 0.0）"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="为每个样本生成可视化 PNG（默认关闭）"
    )
    
    args = parser.parse_args()
    
    in_root = args.in_root
    out_dir = args.out_dir
    json_path = args.json_path or os.path.join(out_dir, "DiariZen_diar_summary.json")
    segmentation_only = args.segmentation_only
    cache_dir = args.cache_dir
    device_arg = args.device.lower()
    model_arg = args.model.strip()
    
    # 处理模型名称：如果是预设名称，转换为完整的 repo_id
    if model_arg.lower() in MODEL_PRESETS:
        model_name = MODEL_PRESETS[model_arg.lower()]
        model_display = f"{model_arg} ({model_name})"
    else:
        model_name = model_arg
        model_display = model_name

    # 确定设备
    if device_arg == "auto":
        device = None  # 让 pipeline 自动检测
        cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        if cuda_available:
            device_name = "GPU (cuda:0)"
        else:
            device_name = "CPU"
            # 自动检测到 CPU，应用优化
            setup_cpu_optimizations()
    elif device_arg in ["cuda", "gpu"]:
        if not torch.cuda.is_available():
            print("警告: GPU 不可用，将使用 CPU")
            device = torch.device("cpu")
            device_name = "CPU (GPU不可用)"
            # 使用 CPU，应用优化
            setup_cpu_optimizations()
        else:
            device = torch.device("cuda:0")
            device_name = f"GPU (cuda:0)"
    else:  # cpu
        device = torch.device("cpu")
        device_name = "CPU"
        # 明确使用 CPU，应用优化
        setup_cpu_optimizations()

    # 如果使用 segmentation_only 模式，在输出目录名称中添加标识
    if segmentation_only:
        if json_path == (args.json_path or os.path.join(args.out_dir, "DiariZen_diar_summary.json")):
            json_path = os.path.join(out_dir, "DiariZen_diar_summary.json")

    # 设置 HF 镜像（如无需可忽略）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    os.makedirs(out_dir, exist_ok=True)

    audio_paths = list_audio_files(in_root)
    if not audio_paths:
        print(f"未找到音频文件: {in_root}")
        return

    print(f"共找到 {len(audio_paths)} 个待处理音频文件。")
    print(f"使用设备: {device_name}")
    print(f"使用模型: {model_display}")
    if segmentation_only:
        print("使用 segmentation-only 模式（跳过 embedding 和 clustering）")
    print(f"Binarize参数: onset={args.binarize_onset}, offset={args.binarize_offset if args.binarize_offset is not None else args.binarize_onset}")

    diar_pipeline = DiariZenPipeline.from_pretrained(
        model_name,
        rttm_out_dir=out_dir,
        cache_dir=cache_dir,
        device=device,
        segmentation_only=segmentation_only,
        binarize_onset=args.binarize_onset,
        binarize_offset=args.binarize_offset,
        binarize_min_duration_on=args.binarize_min_duration_on,
        binarize_min_duration_off=args.binarize_min_duration_off,
    )

    summary = []

    for audio_path in audio_paths:
        try:
            session_name = make_session_name(audio_path, in_root)

            # 读取音频以获取时长
            info = torchaudio.info(audio_path)
            duration = float(info.num_frames) / float(info.sample_rate) if info.sample_rate and info.sample_rate > 0 else 0.0

            t0 = time.time()
            try:
                diar = diar_pipeline(audio_path, sess_name=session_name)
                elapsed = time.time() - t0
                rtf = (elapsed / duration) if duration > 0 else None
                num_speakers = count_unique_speakers(diar)
                is_multispeaker = bool(num_speakers and num_speakers > 1)
                # 与 pipeline 写入的 RTTM 名称保持一致：<session_name>.rttm
                rttm_path = os.path.join(out_dir, f"{session_name}.rttm")
                plot_path = None

                if args.plot:
                    plot_path = os.path.join(out_dir, f"{session_name}.png")
                    try:
                        waveform, sr = torchaudio.load(audio_path)
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

                        # 波形参考
                        ax_wave.plot(t, waveform[0].numpy(), color="gray", alpha=0.6, linewidth=0.5)
                        ax_wave.set_ylabel("Amplitude")
                        ax_wave.set_title(session_name)
                        ax_wave.set_xlim(0, duration if duration > 0 else None)

                        # 按说话人分行绘制区间
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
                    except Exception as e_plot:
                        print(f"绘图失败 {audio_path}: {e_plot}")
                        plot_path = None

                summary.append({
                    "wav_path": os.path.abspath(audio_path),
                    "session_name": session_name,
                    "num_speakers": num_speakers,
                    "is_multispeaker": is_multispeaker,
                    "rttm_path": rttm_path,
                    "duration_sec": duration,
                    "elapsed_sec": elapsed,
                    "rtf": rtf,
                    "plot_path": plot_path,
                })
                rtf_str = f"{rtf:.3f}" if rtf is not None else "NA"
                extra_plot = f"\tPNG={plot_path}" if plot_path else ""
                print(f"{audio_path}\t说话人数量={num_speakers}\t多说话人={is_multispeaker}\tRTF={rtf_str}\tRTTM={rttm_path}{extra_plot}")
            except Exception as e_inner:
                # 特判：模型内部报错 "negative dimensions are not allowed" 视为 0 说话人
                msg = str(e_inner).lower()
                if "negative dimensions are not allowed" in msg:
                    elapsed = time.time() - t0
                    rtf = (elapsed / duration) if duration > 0 else None
                    num_speakers = 0
                    is_multispeaker = False
                    rttm_path = os.path.join(out_dir, f"{session_name}.rttm")
                    # 写入空 RTTM 占位文件，表示无说话人
                    try:
                        with open(rttm_path, "w", encoding="utf-8"):
                            pass
                    except Exception:
                        # 即便 RTTM 占位写失败，也不要中断流程
                        pass
                    summary.append({
                        "wav_path": os.path.abspath(audio_path),
                        "session_name": session_name,
                        "num_speakers": num_speakers,
                        "is_multispeaker": is_multispeaker,
                        "rttm_path": rttm_path,
                        "duration_sec": duration,
                        "elapsed_sec": elapsed,
                        "rtf": rtf,
                        "plot_path": None,
                    })
                    rtf_str = f"{rtf:.3f}" if rtf is not None else "NA"
                    print(f"{audio_path}\t说话人数量=0\t多说话人=False\tRTF={rtf_str}\tRTTM={rttm_path}\t(异常推断)")
                else:
                    # 其它异常仍按失败处理
                    raise
        except Exception as e:
            print(f"处理失败: {audio_path}\t错误: {e}")

    # 简单 RTF 统计
    rtf_values = [it["rtf"] for it in summary if isinstance(it.get("rtf"), (int, float, float))]
    rtf_stats = None
    if rtf_values:
        rtf_stats = {
            "count": len(rtf_values),
            "mean": sum(rtf_values) / len(rtf_values),
            "min": min(rtf_values),
            "max": max(rtf_values),
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "root_dir": os.path.abspath(in_root),
            "out_dir": os.path.abspath(out_dir),
            "model": model_name,
            "device": device_name,
            "segmentation_only": segmentation_only,
            "binarize_params": {
                "onset": args.binarize_onset,
                "offset": args.binarize_offset if args.binarize_offset is not None else args.binarize_onset,
                "min_duration_on": args.binarize_min_duration_on,
                "min_duration_off": args.binarize_min_duration_off,
            },
            "rtf_stats": rtf_stats,
            "items": summary,
        }, f, ensure_ascii=False, indent=2)

    print(f"已写入JSON汇总: {json_path}")


if __name__ == "__main__":
    main()



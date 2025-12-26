#!/usr/bin/env python3
"""
简洁的说话人分离推理脚本

直接使用训练好的DiariZen模型进行推理，输出RTTM和可视化图表。
不依赖复杂的pipeline，只做segmentation模型的前向推理。

用法示例:
export PYTHONPATH=/root/code/github_repos/DiariZen/pyannote-audio:/root/code/github_repos/DiariZen:$PYTHONPATH
python simple_diarize.py \
  /path/to/audios \
  --ckpt-dir /path/to/checkpoints/epoch_0004 \
  --config /path/to/config.toml \
  --out-dir /path/to/output
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import toml

from diarizen.utils import instantiate


def list_audio_files(root_dir: str) -> List[str]:
    """递归扫描目录，列出所有音频文件"""
    exts = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac", "*.wma"]
    files = set()
    for ext in exts:
        files.update(Path(root_dir).rglob(ext))
    files = [f for f in files if f.is_file() and f.stat().st_size > 1024]
    return sorted(str(f) for f in files)


def load_audio(audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, float]:
    """加载音频文件并转换为模型输入格式
    
    返回:
        audio: (channels, samples) 的numpy数组
        duration: 音频时长（秒）
    """
    data, sr = sf.read(audio_path)
    assert sr == sample_rate, f"采样率不匹配: {sr} vs {sample_rate}"
    
    # 处理单声道和多声道
    if data.ndim == 1:
        data = data.reshape(1, -1)
    else:
        data = np.einsum('tc->ct', data)
    
    duration = data.shape[-1] / sample_rate
    return data, duration


def frames_to_segments(
    frame_labels: np.ndarray,
    frame_step: float = 0.02,
    min_duration: float = 0.0,
) -> List[Tuple[float, float, int]]:
    """将帧级别的标签转换为时间段
    
    参数:
        frame_labels: (num_frames, num_speakers) 的二值数组
        frame_step: 每帧的时间步长（秒）
        min_duration: 最小段持续时间（秒）
    
    返回:
        List[(start, end, speaker_id)]
    """
    segments = []
    num_frames, num_speakers = frame_labels.shape
    
    for spk_id in range(num_speakers):
        spk_frames = frame_labels[:, spk_id]
        
        # 找到说话段的起始和结束
        in_segment = False
        start_frame = 0
        
        for frame_idx in range(num_frames):
            if spk_frames[frame_idx] > 0.5:  # 说话
                if not in_segment:
                    start_frame = frame_idx
                    in_segment = True
            else:  # 静音
                if in_segment:
                    # 段结束
                    end_frame = frame_idx
                    start_time = start_frame * frame_step
                    end_time = end_frame * frame_step
                    duration = end_time - start_time
                    
                    if duration >= min_duration:
                        segments.append((start_time, end_time, spk_id))
                    
                    in_segment = False
        
        # 处理最后一个段
        if in_segment:
            end_frame = num_frames
            start_time = start_frame * frame_step
            end_time = end_frame * frame_step
            duration = end_time - start_time
            
            if duration >= min_duration:
                segments.append((start_time, end_time, spk_id))
    
    # 按时间排序
    segments.sort(key=lambda x: x[0])
    return segments


def write_rttm(segments: List[Tuple[float, float, int]], session_name: str, output_path: str):
    """写RTTM文件
    
    RTTM格式: SPEAKER <session> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    with open(output_path, 'w') as f:
        for start, end, spk_id in segments:
            duration = end - start
            line = f"SPEAKER {session_name} 1 {start:.3f} {duration:.3f} <NA> <NA> speaker_{spk_id:02d} <NA> <NA>\n"
            f.write(line)


def plot_diarization(
    audio_path: str,
    segments: List[Tuple[float, float, int]],
    duration: float,
    session_name: str,
    output_path: str,
):
    """生成可视化图表"""
    try:
        # 读取波形
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        t = np.linspace(0, len(data) / sr, len(data))
        
        # 获取所有说话人
        speakers = sorted(set(spk for _, _, spk in segments))
        colors = plt.cm.get_cmap("tab20", max(len(speakers), 1))
        spk_color = {spk: colors(i) for i, spk in enumerate(speakers)}
        spk_pos = {spk: idx for idx, spk in enumerate(speakers)}
        
        # 创建双子图
        fig, (ax_wave, ax_diar) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 4),
            gridspec_kw={"height_ratios": [1.4, 1]}
        )
        
        # 波形图
        ax_wave.plot(t, data, color="gray", alpha=0.6, linewidth=0.5)
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_title(session_name)
        ax_wave.set_xlim(0, duration)
        
        # 说话人分离结果
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
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"绘图失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="简洁的说话人分离推理脚本")
    
    parser.add_argument("in_root", type=str, help="输入音频根目录")
    parser.add_argument("--out-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="checkpoint目录")
    parser.add_argument("--config", type=str, required=True, help="config.toml路径")
    
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="设备")
    parser.add_argument("--sample-rate", type=int, default=16000, help="采样率")
    parser.add_argument("--chunk-size", type=float, default=None, help="分块大小（秒），None表示整句推理")
    parser.add_argument("--min-duration", type=float, default=0.0, help="最小段持续时间（秒）")
    parser.add_argument("--plot", action="store_true", default=True, help="生成可视化")
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="关闭可视化")
    
    args = parser.parse_args()
    
    # 路径
    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    ckpt_dir = Path(args.ckpt_dir)
    config_path = Path(args.config)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    print(f"加载配置: {config_path}")
    config = toml.load(config_path)
    
    # 设备
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # 打印 GPU 内存信息
            print(f"CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
            print(f"当前 GPU: {torch.cuda.current_device()}")
            print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU 内存已分配: {mem_allocated:.2f} GB, 已保留: {mem_reserved:.2f} GB")
        else:
            print("CUDA 不可用，使用 CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    print(f"使用设备: {device}")
    
    # 实例化模型
    print(f"加载模型: {ckpt_dir}")
    model_args = config["model"]["args"].copy()
    
    model = instantiate(config["model"]["path"], args=model_args)
    
    # 加载checkpoint
    ckpt_path = ckpt_dir / "pytorch_model.bin"
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 获取模型感受野信息
    model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info
    print(f"模型感受野: duration={model_rf_duration:.3f}s, step={model_rf_step:.3f}s, num_frames={model_num_frames}")
    
    # 扫描音频文件
    audios = list_audio_files(str(in_root))
    if not audios:
        print(f"未找到音频: {in_root}")
        return
    print(f"找到 {len(audios)} 个音频文件")
    
    # 处理每个音频
    summary = []
    
    for audio_path in audios:
        # 生成会话名称
        rel = Path(audio_path).relative_to(in_root)
        sess_name = rel.with_suffix("").as_posix().replace("/", "__")
        
        print(f"\n处理: {sess_name}")
        
        # 加载音频
        audio_data, duration = load_audio(audio_path, args.sample_rate)
        print(f"  时长: {duration:.2f}s")
        
        # 准备输入
        x = torch.from_numpy(audio_data).float().unsqueeze(0).to(device)  # (1, channels, samples)
        
        # 推理
        with torch.no_grad():
            y_pred = model(x)  # (1, num_frames, num_classes)
        
        print(f"  预测形状: {y_pred.shape}")
        
        # 转换为multilabel
        if hasattr(model, 'specifications') and hasattr(model.specifications, 'powerset') and model.specifications.powerset:
            # Powerset模式：转换为multilabel
            print(f"  使用powerset模式")
            # model.powerset 是 Powerset 对象，而 model.specifications.powerset 是布尔值
            multilabel = model.powerset.to_multilabel(y_pred)  # (1, num_frames, max_speakers)
        else:
            # 已经是multilabel - 使用sigmoid激活
            print(f"  使用multilabel模式")
            multilabel = torch.sigmoid(y_pred)
        
        # 转换为numpy
        frame_labels = multilabel[0].cpu().numpy()  # (num_frames, max_speakers)
        frame_labels = (frame_labels > 0.5).astype(np.uint8)  # 二值化
        
        print(f"  帧标签形状: {frame_labels.shape}")
        
        # 移除全零的说话人
        active_speakers = frame_labels.sum(axis=0) > 0
        frame_labels = frame_labels[:, active_speakers]
        
        num_speakers = frame_labels.shape[1]
        print(f"  检测到说话人数: {num_speakers}")
        
        # 转换为时间段
        segments = frames_to_segments(
            frame_labels,
            frame_step=model_rf_step,
            min_duration=args.min_duration,
        )
        
        print(f"  生成段数: {len(segments)}")
        
        # 写RTTM
        rttm_path = out_dir / f"{sess_name}.rttm"
        write_rttm(segments, sess_name, str(rttm_path))
        print(f"  RTTM: {rttm_path}")
        
        # 画图
        plot_path = None
        if args.plot:
            plot_path = out_dir / f"{sess_name}.png"
            plot_diarization(audio_path, segments, duration, sess_name, str(plot_path))
            print(f"  图表: {plot_path}")
        
        # 记录
        summary.append({
            "audio": audio_path,
            "session": sess_name,
            "duration": duration,
            "num_speakers": int(num_speakers),
            "num_segments": len(segments),
            "rttm": str(rttm_path),
            "plot": str(plot_path) if plot_path else None,
        })
    
    # 写汇总
    summary_path = out_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！共处理 {len(summary)} 个音频")
    print(f"汇总: {summary_path}")


if __name__ == "__main__":
    main()


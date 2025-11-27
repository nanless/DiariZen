#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理音频文件：先用VAD处理降噪音频，根据VAD时间戳裁剪原始音频并拼接，然后进行说话人分离

流程：
1. 使用ten-vad处理降噪后的音频（在 denoised_audio_dir），得到VAD时间戳
2. 根据VAD时间戳，从原始音频（original_audio_dir）中截掉长段且连续的无语音段
3. 将截取后剩下的音频拼接
4. 在拼接后的音频上做diarization（使用DiariZen）

Usage:
    python batch_diarize_with_vad_preprocessing.py \\
        --original-audio-dir /path/to/original_audios \\
        --denoised-audio-dir /path/to/original_audios_SC_CausalMelBandRNN_... \\
        --out-dir /path/to/output \\
        [其他DiariZen参数]
"""

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from diarizen.pipelines.inference import DiariZenPipeline

# 添加3D-Speaker路径以使用VAD相关函数
sys.path.insert(0, '/root/code/github_repos/3D-Speaker')
from speakerlab.utils.fileio import load_audio

# CPU 优化设置（从 batch_diarize_any_audio.py 复制）
_cpu_optimizations_setup = False

def setup_cpu_optimizations(force=False):
    """设置 CPU 优化参数（单线程模式）"""
    global _cpu_optimizations_setup
    
    if _cpu_optimizations_setup and not force:
        return
    
    num_threads = 1
    
    if not _cpu_optimizations_setup:
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError as e:
            print(f"警告: 无法设置 interop_threads（可能已启动并行工作）: {e}")
    
    torch.set_num_threads(num_threads)
    
    if hasattr(torch.backends, 'mkl'):
        torch.backends.mkl.enabled = True
    
    if not _cpu_optimizations_setup:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
        os.environ['KMP_BLOCKTIME'] = '0'
        os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
    
    torch.set_flush_denormal(True)
    
    _cpu_optimizations_setup = True
    
    if not force:
        print(f"CPU 优化设置（单线程模式）: 线程数={num_threads}, MKL={torch.backends.mkl.is_available() if hasattr(torch.backends, 'mkl') else 'N/A'}")


# 默认值
DEFAULT_ORIGINAL_DIR = "/root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios"
DEFAULT_DENOISED_DIR = "/root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios_SC_CausalMelBandRNN_EDA_16k_resume3_variable_length_narrowgap_E0001_B030000"
DEFAULT_OUT_DIR = "/root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios_VAD_preprocessed_DiariZen"
DEFAULT_JSON_PATH = None
DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"

MODEL_PRESETS = {
    "base": "BUT-FIT/diarizen-wavlm-base-s80-md",
    "large": "BUT-FIT/diarizen-wavlm-large-s80-md",
}


def get_voice_activity_detection_model(device: torch.device=None, cache_dir:str = None, threshold: float=0.5):
    """使用TenVad进行VAD处理"""
    try:
        from ten_vad import TenVad
    except ImportError:
        try:
            sys.path.append('/root/code/gitlab_repos/se_train')
            from ten_vad import TenVad
        except Exception as e:
            raise ImportError('ten_vad is required for VAD. Please install/ensure it is available.') from e

    class TenVadWrapper:
        def __init__(self, sample_rate: int = 16000, frame_ms: float = 16.0, threshold: float = 0.5):
            self.sample_rate = sample_rate
            self.hop_size = int(frame_ms * sample_rate / 1000)
            self.engine = TenVad(self.hop_size, threshold)

        def __call__(self, wav_1d):
            # to numpy float32 in [-1, 1]
            if hasattr(wav_1d, 'detach'):
                x = wav_1d.detach().cpu().numpy().astype(np.float32)
            else:
                x = np.asarray(wav_1d).astype(np.float32)
            if x.size == 0:
                return [], x
            x = np.clip(x, -1.0, 1.0)
            x_i16 = (x * 32767).astype(np.int16)

            num_frames = len(x_i16) // self.hop_size
            flags = []
            for i in range(num_frames):
                frame = x_i16[i*self.hop_size:(i+1)*self.hop_size]
                if len(frame) == self.hop_size:
                    _, f = self.engine.process(frame)
                    flags.append(int(f))
                else:
                    flags.append(0)
            
            return flags, x

    return TenVadWrapper(sample_rate=16000, frame_ms=16.0, threshold=threshold)


class VADProcessor:
    """VAD处理器，包含后处理和refine逻辑"""
    
    def __init__(self, 
                 vad_threshold: float = 0.5,
                 vad_min_speech_ms: float = 200.0,
                 vad_max_silence_ms: float = 300.0,
                 vad_energy_threshold: float = 0.05,
                 vad_boundary_expansion_ms: float = 10.0,
                 vad_boundary_energy_percentile: float = 10.0,
                 sample_rate: int = 16000):
        self.vad_model = get_voice_activity_detection_model(threshold=vad_threshold)
        self.fs = sample_rate
        self.vad_frame_size_ms = 16.0
        self.vad_min_speech_ms = float(vad_min_speech_ms)
        self.vad_max_silence_ms = float(vad_max_silence_ms)
        self.vad_energy_threshold = float(vad_energy_threshold)
        self.vad_boundary_expansion_ms = float(vad_boundary_expansion_ms)
        self.vad_boundary_energy_percentile = float(vad_boundary_energy_percentile)
    
    def process(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        处理音频文件，返回VAD时间戳列表 [(start_sec, end_sec), ...]
        """
        wav_data = load_audio(audio_path, None, self.fs)
        
        # 执行VAD
        speech_flags, wav_data_for_vad = self._do_vad(wav_data)
        
        # 后处理
        vad_processed_mask, vad_refined_mask, vad_time = self._postprocess_vad(speech_flags, wav_data_for_vad)
        
        return vad_time
    
    def _do_vad(self, wav):
        """执行VAD"""
        speech_flags, wav_data = self.vad_model(wav[0])
        return speech_flags, wav_data
    
    def _postprocess_vad(self, speech_flags, wav_data):
        """VAD后处理：平滑、形态学填充、能量边界细化"""
        # 转换为处理后的flags
        processed_flags = self._post_process_speech_flags(speech_flags)
        
        # 转换为mask
        hop_size = int(self.vad_frame_size_ms * self.fs / 1000)
        processed_mask = np.zeros(len(wav_data), dtype=np.float32)
        for i, flag in enumerate(processed_flags):
            s = i * hop_size
            e = min((i + 1) * hop_size, len(wav_data))
            processed_mask[s:e] = flag
        
        # 能量边界细化
        refined_mask = self._refine_vad_boundaries_with_energy(wav_data, processed_mask)
        
        # 转换为时间间隔
        vad_time = self._mask_to_intervals(refined_mask)
        return processed_mask, refined_mask, vad_time
    
    def _post_process_speech_flags(self, flags):
        """平滑和形态学填充"""
        flags = np.array(flags, dtype=np.float32)
        
        # 简单移动平均平滑
        win = 3
        pad = np.pad(flags, (win // 2, win // 2), mode='edge')
        smooth = np.convolve(pad, np.ones(win) / win, mode='valid')
        smooth = (smooth > 0.5).astype(np.float32)

        # 最小语音段/最大静音段约束
        min_speech_frames = max(1, int(self.vad_min_speech_ms / self.vad_frame_size_ms))
        max_silence_frames = max(1, int(self.vad_max_silence_ms / self.vad_frame_size_ms))

        res = smooth.copy()
        # 填充短静音间隙
        count0 = 0
        for i in range(len(res)):
            if res[i] == 0:
                count0 += 1
            else:
                if 0 < count0 <= max_silence_frames:
                    res[i - count0 : i] = 1
                count0 = 0
        # 移除过短语音段
        count1 = 0
        for i in range(len(res)):
            if res[i] == 1:
                count1 += 1
            else:
                if 0 < count1 < min_speech_frames:
                    res[i - count1 : i] = 0
                count1 = 0
        return res.astype(np.float32)
    
    def _refine_vad_boundaries_with_energy(self, audio_data, vad_mask):
        """使用能量方法细化VAD边界"""
        refined_mask = vad_mask.copy()
        window_size = int(0.02 * self.fs)  # 20ms
        hop_length = int(0.01 * self.fs)   # 10ms
        n_frames = (len(audio_data) - window_size) // hop_length + 1
        if n_frames <= 0:
            return refined_mask

        frame_energy = np.zeros(len(audio_data), dtype=np.float32)
        for i in range(n_frames):
            s = i * hop_length
            e = min(s + window_size, len(audio_data))
            en = float(np.mean(audio_data[s:e] ** 2))
            frame_energy[s:e] = max(frame_energy[s:e].max(), en)

        vad_diff = np.diff(np.concatenate(([0], vad_mask, [0])))
        speech_starts = np.where(vad_diff > 0)[0]
        speech_ends = np.where(vad_diff < 0)[0]
        if len(speech_starts) == 0 or len(speech_ends) == 0:
            return refined_mask

        lookahead_frames = 10
        lookahead_samples = lookahead_frames * hop_length
        energy_floor = float(self.vad_energy_threshold)
        energy_percentile = float(self.vad_boundary_energy_percentile)
        boundary_expand_ms = float(self.vad_boundary_expansion_ms)
        boundary_expand_samples = int(boundary_expand_ms * self.fs / 1000.0)

        for start, end in zip(speech_starts, speech_ends):
            seg_energy = frame_energy[start:end]
            if len(seg_energy) == 0:
                continue
            dynamic_th = max(np.percentile(seg_energy, energy_percentile), energy_floor)
            
            # 前向收缩
            new_start = start
            for i in range(start, min(end, start + lookahead_samples)):
                if frame_energy[i] < dynamic_th:
                    refined_mask[start:i] = 0
                    new_start = i
                    break
            
            # 后向收缩
            new_end = end
            for i in range(end - 1, max(new_start, end - lookahead_samples), -1):
                if frame_energy[i] < dynamic_th:
                    refined_mask[i:end] = 0
                    new_end = i + 1
                    break
            
            # 边界扩展
            if boundary_expand_samples > 0:
                expand_start_begin = max(start, new_start - boundary_expand_samples)
                expand_start_end = new_start
                refined_mask[expand_start_begin:expand_start_end] = 1
                
                expand_end_begin = new_end
                expand_end_end = end
                refined_mask[expand_end_begin:expand_end_end] = 1
        return refined_mask.astype(np.float32)
    
    def _mask_to_intervals(self, mask):
        """将VAD mask转换为时间间隔（秒）"""
        if len(mask) == 0:
            return []
        
        diff = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.where(diff > 0)[0]
        ends = np.where(diff < 0)[0]
        
        if len(starts) == 0:
            return []
        
        intervals = []
        for s, e in zip(starts, ends):
            start_sec = float(s) / self.fs
            end_sec = float(e) / self.fs
            if end_sec > start_sec:
                intervals.append([start_sec, end_sec])
        
        return intervals


def list_audio_files(root_dir: str) -> List[str]:
    """递归查找音频文件"""
    exts = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac", "*.wma"]
    files = set()
    for ext in exts:
        pattern = os.path.join(root_dir, "**", ext)
        files.update(glob.glob(pattern, recursive=True))
    files = [f for f in files if os.path.getsize(f) > 1024]
    return sorted(files)


def find_denoised_audio(original_path: str, original_dir: str, denoised_dir: str) -> Optional[str]:
    """
    根据原始音频路径，找到对应的降噪音频文件
    命名规则：去掉原始文件扩展名 + "_speech_estimate.wav"
    例如: "file.wav" -> "file_speech_estimate.wav"
    """
    original_basename = os.path.basename(original_path)
    original_name_no_ext = os.path.splitext(original_basename)[0]
    
    # 构造降噪文件名：原始文件名（无扩展名）+ "_speech_estimate.wav"
    denoised_filename = original_name_no_ext + "_speech_estimate.wav"
    
    # 策略1: 在降噪目录根目录查找
    candidate = os.path.join(denoised_dir, denoised_filename)
    if os.path.exists(candidate):
        return candidate
    
    # 策略2: 保持相对路径结构查找
    # 如果原始文件在子目录中，尝试在降噪目录的相同相对路径查找
    try:
        original_rel = os.path.relpath(original_path, original_dir)
        original_rel_dir = os.path.dirname(original_rel)
        if original_rel_dir:
            # 原始文件在子目录中，在降噪目录的相同子目录中查找
            candidate = os.path.join(denoised_dir, original_rel_dir, denoised_filename)
            if os.path.exists(candidate):
                return candidate
    except ValueError:
        # 如果路径不在同一根目录下，忽略此策略
        pass
    
    # 策略3: 递归查找所有匹配的文件
    pattern = os.path.join(denoised_dir, "**", denoised_filename)
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    
    return None


def filter_long_silence_gaps(vad_intervals: List[Tuple[float, float]], 
                            max_silence_duration: float = 3.0) -> List[Tuple[float, float]]:
    """
    过滤VAD间隔，只保留间隔之间静音段不超过阈值的部分
    如果两个间隔之间的静音段超过阈值，则只保留前面的间隔，后面的间隔被丢弃
    
    Args:
        vad_intervals: VAD间隔列表 [(start_sec, end_sec), ...]
        max_silence_duration: 最大允许的连续无语音段时长（秒），超过此时长的会被去掉
    
    Returns:
        过滤后的VAD间隔列表
    """
    if not vad_intervals:
        return []
    
    # 按开始时间排序
    sorted_intervals = sorted(vad_intervals, key=lambda x: x[0])
    
    filtered_intervals = []
    prev_end = None
    
    for start_sec, end_sec in sorted_intervals:
        if prev_end is None:
            # 第一个间隔，直接添加
            filtered_intervals.append([start_sec, end_sec])
            prev_end = end_sec
        else:
            # 计算与上一个间隔之间的静音时长
            silence_duration = start_sec - prev_end
            
            if silence_duration <= max_silence_duration:
                # 静音段不超过阈值，保留这个间隔
                # 如果与上一个间隔相邻或重叠，合并它们
                if start_sec <= prev_end:
                    # 重叠或相邻，合并
                    filtered_intervals[-1][1] = max(filtered_intervals[-1][1], end_sec)
                else:
                    # 有间隔但不超过阈值，添加新间隔（静音段会被保留在拼接中）
                    filtered_intervals.append([start_sec, end_sec])
                prev_end = max(prev_end, end_sec)
            else:
                # 静音段超过阈值，不保留这个间隔（跳过）
                # 不更新prev_end，这样后续间隔如果与第一个间隔组之间的静音段不超过阈值，仍可被保留
                continue
    
    return filtered_intervals


def extract_and_concatenate_audio_segments(original_audio_path: str, 
                                          vad_intervals: List[Tuple[float, float]],
                                          output_path: str,
                                          sample_rate: int = 16000,
                                          max_silence_duration: float = 3.0) -> Tuple[float, float]:
    """
    根据VAD时间戳从原始音频中提取语音段并拼接
    只去掉超过max_silence_duration的连续无语音段
    
    Args:
        original_audio_path: 原始音频路径
        vad_intervals: VAD间隔列表
        output_path: 输出路径
        sample_rate: 采样率
        max_silence_duration: 最大允许的连续无语音段时长（秒）
    
    Returns:
        (original_duration, processed_duration)
    """
    # 先过滤掉超过阈值的静音段
    filtered_intervals = filter_long_silence_gaps(vad_intervals, max_silence_duration)
    
    # 加载原始音频
    wav_data, orig_sr = torchaudio.load(original_audio_path)
    if wav_data.shape[0] > 1:
        wav_data = wav_data.mean(dim=0, keepdim=True)
    
    # 重采样到目标采样率
    if orig_sr != sample_rate:
        wav_data = torchaudio.functional.resample(wav_data, orig_freq=orig_sr, new_freq=sample_rate)
    
    wav_data = wav_data[0].numpy()  # [T]
    original_duration = len(wav_data) / sample_rate
    
    # 提取语音段并拼接
    # 只拼接间隔之间静音段不超过阈值的部分
    segments = []
    prev_end = None
    
    for start_sec, end_sec in filtered_intervals:
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        start_sample = max(0, min(start_sample, len(wav_data)))
        end_sample = max(0, min(end_sample, len(wav_data)))
        
        if end_sample > start_sample:
            segment = wav_data[start_sample:end_sample]
            segments.append(segment)
            prev_end = end_sec
    
    if not segments:
        # 如果没有语音段，创建一个空音频
        concatenated = np.array([], dtype=np.float32)
    else:
        # 拼接所有语音段（静音段已经被过滤掉了）
        concatenated = np.concatenate(segments)
    
    processed_duration = len(concatenated) / sample_rate
    
    # 保存拼接后的音频
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, concatenated, sample_rate)
    
    return original_duration, processed_duration


def count_unique_speakers(diarization_result) -> int:
    """统计说话人数量"""
    speakers = set()
    for _, _, speaker in diarization_result.itertracks(yield_label=True):
        speakers.add(speaker)
    return len(speakers)


def make_session_name(path: str, root_dir: str) -> str:
    """生成session名称"""
    rel = os.path.relpath(path, root_dir)
    base_no_ext = os.path.splitext(rel)[0]
    return base_no_ext.replace(os.sep, "__")


def main():
    parser = argparse.ArgumentParser(
        description="批量处理音频：VAD预处理 + DiariZen说话人分离",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--original-audio-dir",
        type=str,
        default=DEFAULT_ORIGINAL_DIR,
        help=f"原始音频文件目录（默认: {DEFAULT_ORIGINAL_DIR}）"
    )
    parser.add_argument(
        "--denoised-audio-dir",
        type=str,
        default=DEFAULT_DENOISED_DIR,
        help=f"降噪音频文件目录（默认: {DEFAULT_DENOISED_DIR}）"
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
        help="只使用 segmentation 模型，跳过 embedding 和 clustering"
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
        help="计算设备：auto（自动检测）、cpu、cuda/gpu"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"DiariZen 模型名称或路径。可以是：\n"
             f"  - Hugging Face repo_id\n"
             f"  - 本地路径\n"
             f"  - 预设名称：{', '.join(MODEL_PRESETS.keys())}（如 'base' 或 'large'）\n"
             f"默认: {DEFAULT_MODEL}"
    )
    # VAD参数
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD阈值（默认: 0.5）"
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=float,
        default=200.0,
        help="VAD最小语音段时长（毫秒，默认: 200.0）"
    )
    parser.add_argument(
        "--vad-max-silence-ms",
        type=float,
        default=300.0,
        help="VAD最大静音间隙（毫秒，默认: 300.0）"
    )
    parser.add_argument(
        "--vad-energy-threshold",
        type=float,
        default=0.05,
        help="VAD能量阈值（默认: 0.05）"
    )
    parser.add_argument(
        "--vad-boundary-expansion-ms",
        type=float,
        default=10.0,
        help="VAD边界扩展（毫秒，默认: 10.0）"
    )
    parser.add_argument(
        "--vad-boundary-energy-percentile",
        type=float,
        default=10.0,
        help="VAD边界能量百分位（默认: 10.0）"
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
    
    args = parser.parse_args()
    
    original_dir = args.original_audio_dir
    denoised_dir = args.denoised_audio_dir
    out_dir = args.out_dir
    json_path = args.json_path or os.path.join(out_dir, "DiariZen_diar_summary.json")
    segmentation_only = args.segmentation_only
    cache_dir = args.cache_dir
    device_arg = args.device.lower()
    model_arg = args.model.strip()
    
    # 处理模型名称
    if model_arg.lower() in MODEL_PRESETS:
        model_name = MODEL_PRESETS[model_arg.lower()]
        model_display = f"{model_arg} ({model_name})"
    else:
        model_name = model_arg
        model_display = model_name

    # 确定设备
    if device_arg == "auto":
        device = None
        cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        if cuda_available:
            device_name = "GPU (cuda:0)"
        else:
            device_name = "CPU"
            setup_cpu_optimizations()
    elif device_arg in ["cuda", "gpu"]:
        if not torch.cuda.is_available():
            print("警告: GPU 不可用，将使用 CPU")
            device = torch.device("cpu")
            device_name = "CPU (GPU不可用)"
            setup_cpu_optimizations()
        else:
            device = torch.device("cuda:0")
            device_name = f"GPU (cuda:0)"
    else:  # cpu
        device = torch.device("cpu")
        device_name = "CPU"
        setup_cpu_optimizations()

    # 设置 HF 镜像
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    os.makedirs(out_dir, exist_ok=True)
    
    # 创建中间目录用于存放预处理后的音频
    preprocessed_audio_dir = os.path.join(out_dir, "preprocessed_audios")
    os.makedirs(preprocessed_audio_dir, exist_ok=True)

    # 查找原始音频文件（从original_dir遍历）
    original_audio_paths = list_audio_files(original_dir)
    if not original_audio_paths:
        print(f"未找到原始音频文件: {original_dir}")
        return

    print(f"共找到 {len(original_audio_paths)} 个原始音频文件。")
    print(f"使用设备: {device_name}")
    print(f"使用模型: {model_display}")
    if segmentation_only:
        print("使用 segmentation-only 模式（跳过 embedding 和 clustering）")
    print(f"Binarize参数: onset={args.binarize_onset}, offset={args.binarize_offset if args.binarize_offset is not None else args.binarize_onset}")
    print()

    # 初始化VAD处理器
    vad_processor = VADProcessor(
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_max_silence_ms=args.vad_max_silence_ms,
        vad_energy_threshold=args.vad_energy_threshold,
        vad_boundary_expansion_ms=args.vad_boundary_expansion_ms,
        vad_boundary_energy_percentile=args.vad_boundary_energy_percentile,
    )

    # 初始化DiariZen pipeline
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
    processed_count = 0
    skipped_count = 0

    for original_path in original_audio_paths:
        try:
            # 查找对应的降噪音频文件
            denoised_path = find_denoised_audio(original_path, original_dir, denoised_dir)
            if denoised_path is None:
                print(f"警告: 未找到降噪音频对应文件: {original_path}")
                skipped_count += 1
                continue

            print(f"处理: {os.path.basename(original_path)}")
            
            # 步骤1: VAD处理降噪音频
            t0_vad = time.time()
            try:
                vad_intervals = vad_processor.process(denoised_path)
                elapsed_vad = time.time() - t0_vad
            except Exception as e:
                print(f"  VAD处理失败: {e}")
                skipped_count += 1
                continue

            if not vad_intervals:
                print(f"  警告: 未检测到语音段，将输出空RTTM文件")
                # 继续处理，输出空RTTM文件

            print(f"  VAD完成: 检测到 {len(vad_intervals)} 个语音段，耗时 {elapsed_vad:.2f}s")

            # 获取原始音频时长（用于RTF计算）
            info_original = torchaudio.info(original_path)
            original_duration = float(info_original.num_frames) / float(info_original.sample_rate) if info_original.sample_rate and info_original.sample_rate > 0 else 0.0

            # 步骤2: 从原始音频提取并拼接语音段
            # 预处理后的音频文件名基于原始文件名
            original_basename_no_ext = os.path.splitext(os.path.basename(original_path))[0]
            preprocessed_filename = original_basename_no_ext + "_preprocessed.wav"
            preprocessed_path = os.path.join(preprocessed_audio_dir, preprocessed_filename)
            
            t0_extract = time.time()
            try:
                _, processed_duration = extract_and_concatenate_audio_segments(
                    original_path, vad_intervals, preprocessed_path, sample_rate=16000, max_silence_duration=3.0
                )
                elapsed_extract = time.time() - t0_extract
            except Exception as e:
                print(f"  音频提取失败: {e}")
                skipped_count += 1
                continue

            print(f"  音频提取完成: 原始时长 {original_duration:.2f}s -> 处理后 {processed_duration:.2f}s "
                  f"(保留 {processed_duration/original_duration*100:.1f}%)，耗时 {elapsed_extract:.2f}s")
            print(f"  预处理后音频已保存: {preprocessed_path}")

            # 步骤3: Diarization处理
            # 使用原始文件名作为session名称（保持一致性）
            session_name = make_session_name(original_path, original_dir)
            rttm_path = os.path.join(out_dir, f"{session_name}.rttm")
            
            # 如果没有语音段，直接输出空RTTM文件
            if not vad_intervals or processed_duration == 0:
                print(f"  无语音段，输出空RTTM文件: {rttm_path}")
                try:
                    with open(rttm_path, "w", encoding="utf-8"):
                        pass
                except Exception:
                    pass
                summary.append({
                    "original_wav_path": os.path.abspath(original_path),
                    "denoised_wav_path": os.path.abspath(denoised_path),
                    "preprocessed_wav_path": os.path.abspath(preprocessed_path),
                    "session_name": session_name,
                    "num_speakers": 0,
                    "is_multispeaker": False,
                    "rttm_path": rttm_path,
                    "original_duration_sec": original_duration,
                    "preprocessed_duration_sec": processed_duration,
                    "duration_sec": processed_duration,
                    "vad_intervals": vad_intervals,
                    "num_vad_segments": len(vad_intervals),
                    "elapsed_vad_sec": elapsed_vad,
                    "elapsed_extract_sec": elapsed_extract,
                    "elapsed_diar_sec": 0.0,
                    "elapsed_total_sec": elapsed_vad + elapsed_extract,
                    "rtf": None,
                })
                processed_count += 1
                print()
                continue
            
            info = torchaudio.info(preprocessed_path)
            duration = float(info.num_frames) / float(info.sample_rate) if info.sample_rate and info.sample_rate > 0 else 0.0

            t0_diar = time.time()
            try:
                diar = diar_pipeline(preprocessed_path, sess_name=session_name)
                elapsed_diar = time.time() - t0_diar
                # RTF使用原始音频时长计算
                rtf = (elapsed_diar / original_duration) if original_duration > 0 else None
                num_speakers = count_unique_speakers(diar)
                is_multispeaker = bool(num_speakers and num_speakers > 1)
                
                # 如果没有检测到说话人，确保输出空RTTM文件
                if num_speakers == 0:
                    try:
                        with open(rttm_path, "w", encoding="utf-8"):
                            pass
                    except Exception:
                        pass

                summary.append({
                    "original_wav_path": os.path.abspath(original_path),
                    "denoised_wav_path": os.path.abspath(denoised_path),
                    "preprocessed_wav_path": os.path.abspath(preprocessed_path),
                    "session_name": session_name,
                    "num_speakers": num_speakers,
                    "is_multispeaker": is_multispeaker,
                    "rttm_path": rttm_path,
                    "original_duration_sec": original_duration,
                    "preprocessed_duration_sec": processed_duration,
                    "duration_sec": duration,
                    "vad_intervals": vad_intervals,
                    "num_vad_segments": len(vad_intervals),
                    "elapsed_vad_sec": elapsed_vad,
                    "elapsed_extract_sec": elapsed_extract,
                    "elapsed_diar_sec": elapsed_diar,
                    "elapsed_total_sec": elapsed_vad + elapsed_extract + elapsed_diar,
                    "rtf": rtf,
                })
                rtf_str = f"{rtf:.3f}" if rtf is not None else "NA"
                print(f"  Diarization完成: 说话人数量={num_speakers}, 多说话人={is_multispeaker}, RTF={rtf_str}")
                processed_count += 1
            except Exception as e_inner:
                msg = str(e_inner).lower()
                if "negative dimensions are not allowed" in msg:
                    elapsed_diar = time.time() - t0_diar
                    # RTF使用原始音频时长计算
                    rtf = (elapsed_diar / original_duration) if original_duration > 0 else None
                    num_speakers = 0
                    is_multispeaker = False
                    rttm_path = os.path.join(out_dir, f"{session_name}.rttm")
                    try:
                        with open(rttm_path, "w", encoding="utf-8"):
                            pass
                    except Exception:
                        pass
                    summary.append({
                        "original_wav_path": os.path.abspath(original_path),
                        "denoised_wav_path": os.path.abspath(denoised_path),
                        "preprocessed_wav_path": os.path.abspath(preprocessed_path),
                        "session_name": session_name,
                        "num_speakers": num_speakers,
                        "is_multispeaker": is_multispeaker,
                        "rttm_path": rttm_path,
                        "original_duration_sec": original_duration,
                        "preprocessed_duration_sec": processed_duration,
                        "duration_sec": duration,
                        "vad_intervals": vad_intervals,
                        "num_vad_segments": len(vad_intervals),
                        "elapsed_vad_sec": elapsed_vad,
                        "elapsed_extract_sec": elapsed_extract,
                        "elapsed_diar_sec": elapsed_diar,
                        "elapsed_total_sec": elapsed_vad + elapsed_extract + elapsed_diar,
                        "rtf": rtf,
                    })
                    rtf_str = f"{rtf:.3f}" if rtf is not None else "NA"
                    print(f"  Diarization完成（异常推断）: 说话人数量=0, RTF={rtf_str}")
                    processed_count += 1
                else:
                    print(f"  Diarization失败: {e_inner}")
                    skipped_count += 1
            print()
        except Exception as e:
            print(f"处理失败: {denoised_path}\t错误: {e}")
            skipped_count += 1
            print()

    # RTF统计
    rtf_values = [it["rtf"] for it in summary if isinstance(it.get("rtf"), (int, float))]
    rtf_stats = None
    if rtf_values:
        rtf_stats = {
            "count": len(rtf_values),
            "mean": sum(rtf_values) / len(rtf_values),
            "min": min(rtf_values),
            "max": max(rtf_values),
        }

    # 保存汇总JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "original_audio_dir": os.path.abspath(original_dir),
            "denoised_audio_dir": os.path.abspath(denoised_dir),
            "out_dir": os.path.abspath(out_dir),
            "preprocessed_audio_dir": os.path.abspath(preprocessed_audio_dir),
            "model": model_name,
            "device": device_name,
            "segmentation_only": segmentation_only,
            "vad_params": {
                "threshold": args.vad_threshold,
                "min_speech_ms": args.vad_min_speech_ms,
                "max_silence_ms": args.vad_max_silence_ms,
                "energy_threshold": args.vad_energy_threshold,
                "boundary_expansion_ms": args.vad_boundary_expansion_ms,
                "boundary_energy_percentile": args.vad_boundary_energy_percentile,
            },
            "binarize_params": {
                "onset": args.binarize_onset,
                "offset": args.binarize_offset if args.binarize_offset is not None else args.binarize_onset,
                "min_duration_on": args.binarize_min_duration_on,
                "min_duration_off": args.binarize_min_duration_off,
            },
            "rtf_stats": rtf_stats,
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "items": summary,
        }, f, ensure_ascii=False, indent=2)

    print(f"处理完成: 成功 {processed_count} 个，跳过 {skipped_count} 个")
    print(f"已写入JSON汇总: {json_path}")


if __name__ == "__main__":
    main()


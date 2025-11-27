# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
音频I/O处理模块

本模块提供音频文件的读取、重采样、裁剪等功能。
pyannote.audio依赖torchaudio进行音频读取和重采样操作。

主要类：
- Audio: 音频I/O处理类，提供统一的音频文件接口
"""

import math
import random
import warnings
from io import IOBase
from pathlib import Path
from typing import Mapping, Optional, Text, Tuple, Union

import numpy as np
import torch.nn.functional as F
import torchaudio
from pyannote.core import Segment
from torch import Tensor

AudioFile = Union[Text, Path, IOBase, Mapping]

AudioFileDocString = """
Audio files can be provided to the Audio class using different types:
    - a "str" or "Path" instance: "audio.wav" or Path("audio.wav")
    - a "IOBase" instance with "read" and "seek" support: open("audio.wav", "rb")
    - a "Mapping" with any of the above as "audio" key: {"audio": ...}
    - a "Mapping" with both "waveform" and "sample_rate" key:
        {"waveform": (channel, time) numpy.ndarray or torch.Tensor, "sample_rate": 44100}

For last two options, an additional "channel" key can be provided as a zero-indexed
integer to load a specific channel: {"audio": "stereo.wav", "channel": 0}
"""


def get_torchaudio_info(file: AudioFile):
    """Protocol preprocessor used to cache output of torchaudio.info

    This is useful to speed future random access to this file, e.g.
    in dataloaders using Audio.crop a lot....
    """

    info = torchaudio.info(file["audio"])

    # rewind if needed
    if isinstance(file["audio"], IOBase):
        file["audio"].seek(0)

    return info


class Audio:
    """音频I/O处理类
    
    提供统一的音频文件读取、重采样、裁剪等功能。
    支持多种输入格式：文件路径、文件对象、内存中的波形数据。
    
    参数
    ----------
    sample_rate : int, 可选
        目标采样率（Hz）
        如果为None，使用音频文件的原始采样率
        如果指定，会自动重采样到目标采样率
    mono : {'random', 'downmix'}, 可选
        多声道转单声道的策略：
        - 'random': 随机选择一个声道
        - 'downmix': 对所有声道求平均（推荐）
        如果为None，保持原始声道数
    
    使用示例
    -----
    >>> # 创建Audio对象，目标采样率16kHz，下混为单声道
    >>> audio = Audio(sample_rate=16000, mono='downmix')
    >>> # 从文件读取音频
    >>> waveform, sample_rate = audio({"audio": "/path/to/audio.wav"})
    >>> assert sample_rate == 16000
    >>> # 从内存中的波形数据读取
    >>> sample_rate = 44100
    >>> two_seconds_stereo = torch.rand(2, 2 * sample_rate)
    >>> waveform, sample_rate = audio({"waveform": two_seconds_stereo, "sample_rate": sample_rate})
    >>> assert sample_rate == 16000
    >>> assert waveform.shape[0] == 1  # 单声道
    """

    # 时间精度常量（秒），用于边界检查
    PRECISION = 0.001

    @staticmethod
    def power_normalize(waveform: Tensor) -> Tensor:
        """功率归一化波形
        
        将波形归一化到单位功率（RMS=1），使不同音频的音量一致。
        这对于模型训练和推理很重要，可以避免音量差异影响模型性能。
        
        参数
        ----------
        waveform : (..., time) Tensor
            输入波形，最后一个维度是时间
            可以是任意形状，例如：
            - (time,): 单声道单样本
            - (channel, time): 多声道单样本
            - (batch, channel, time): 批次数据
        
        返回
        -------
        Tensor
            功率归一化后的波形，形状与输入相同
            RMS（均方根）值被归一化为1
        
        注意
        -----
        使用1e-8作为分母的小值，避免除零错误
        """
        # 计算RMS（均方根）：sqrt(mean(square(waveform)))
        rms = waveform.square().mean(dim=-1, keepdim=True).sqrt()
        # 归一化：waveform / RMS
        return waveform / (rms + 1e-8)

    @staticmethod
    def validate_file(file: AudioFile) -> Mapping:
        """验证和规范化音频文件输入
        
        将各种格式的音频输入转换为统一的字典格式，便于后续处理。
        支持多种输入类型：
        1. 文件路径（str或Path）
        2. 文件对象（IOBase）
        3. 字典（包含"audio"或"waveform"键）
        4. 内存中的波形数据（包含"waveform"和"sample_rate"）
        
        参数
        ----------
        file : AudioFile
            音频文件输入，可以是：
            - str或Path: 文件路径
            - IOBase: 文件对象
            - Mapping: 字典，包含以下键之一：
              - "audio": 文件路径或文件对象
              - "waveform": numpy数组或torch.Tensor（形状：(channel, time)）
                + "sample_rate": 采样率（必需）
        
        返回
        -------
        Mapping
            规范化后的文件字典，包含：
            - "audio": 文件路径或文件对象
            - "uri": 文件标识符（文件名或"stream"）
            或
            - "waveform": 波形数据
            - "sample_rate": 采样率
            - "uri": 标识符
        
        异常
        ------
        ValueError
            如果文件格式无效、文件不存在或缺少必需字段
        
        注意
        -----
        如果输入是IOBase实例，直接返回，不进行文件存在性检查
        """

        if isinstance(file, Mapping):
            pass

        elif isinstance(file, (str, Path)):
            file = {"audio": str(file), "uri": Path(file).stem}

        elif isinstance(file, IOBase):
            return {"audio": file, "uri": "stream"}

        else:
            raise ValueError(AudioFileDocString)

        if "waveform" in file:
            waveform: Union[np.ndarray, Tensor] = file["waveform"]
            if len(waveform.shape) != 2 or waveform.shape[0] > waveform.shape[1]:
                raise ValueError(
                    "'waveform' must be provided as a (channel, time) torch Tensor."
                )

            sample_rate: int = file.get("sample_rate", None)
            if sample_rate is None:
                raise ValueError(
                    "'waveform' must be provided with their 'sample_rate'."
                )

            file.setdefault("uri", "waveform")

        elif "audio" in file:
            if isinstance(file["audio"], IOBase):
                return file

            path = Path(file["audio"])
            if not path.is_file():
                raise ValueError(f"File {path} does not exist")

            file.setdefault("uri", path.stem)

        else:
            raise ValueError(
                "Neither 'waveform' nor 'audio' is available for this file."
            )

        return file

    def __init__(self, sample_rate=None, mono=None):
        """初始化Audio对象
        
        参数
        ----------
        sample_rate : int, 可选
            目标采样率（Hz），None表示使用原始采样率
        mono : str, 可选
            多声道转单声道策略："random"或"downmix"，None表示保持原始声道
        """
        super().__init__()
        self.sample_rate = sample_rate  # 目标采样率
        self.mono = mono  # 单声道转换策略

    def downmix_and_resample(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """下混和重采样音频
        
        将多声道音频转换为单声道（如果指定），并重采样到目标采样率。
        这是音频预处理的关键步骤，确保输入格式统一。
        
        参数
        ----------
        waveform : (channel, time) Tensor
            输入波形，形状为(通道数, 时间样本数)
        sample_rate : int
            当前采样率（Hz）
        
        返回
        -------
        waveform : (channel, time) Tensor
            处理后的波形
            - 如果mono="downmix"或"random"：形状为(1, time)
            - 如果mono=None：保持原始形状
        sample_rate : int
            新的采样率（如果进行了重采样）
        
        处理流程
        --------
        1. 下混（如果多声道且mono不为None）：
           - "random": 随机选择一个声道
           - "downmix": 对所有声道求平均（推荐，保留更多信息）
        2. 重采样（如果目标采样率与当前不同）：
           - 使用torchaudio.functional.resample进行高质量重采样
        """
        # ========== 下混到单声道 ==========
        num_channels = waveform.shape[0]
        if num_channels > 1:
            if self.mono == "random":
                # 随机选择一个声道
                channel = random.randint(0, num_channels - 1)
                waveform = waveform[channel : channel + 1]
            elif self.mono == "downmix":
                # 对所有声道求平均（推荐方法，保留更多信息）
                waveform = waveform.mean(dim=0, keepdim=True)

        # ========== 重采样 ==========
        # 如果指定了目标采样率且与当前不同，进行重采样
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
            sample_rate = self.sample_rate

        return waveform, sample_rate

    def get_duration(self, file: AudioFile) -> float:
        """获取音频文件时长（秒）
        
        计算音频文件的总时长，支持文件路径和内存中的波形数据。
        
        参数
        ----------
        file : AudioFile
            音频文件输入（路径、文件对象或包含waveform的字典）
        
        返回
        -------
        float
            音频时长（秒）
        
        注意
        -----
        对于内存中的波形数据，直接计算长度
        对于文件，使用torchaudio.info获取信息（可能使用缓存）
        """
        file = self.validate_file(file)

        if "waveform" in file:
            # 内存中的波形数据：直接计算长度
            frames = len(file["waveform"].T)  # 转置后取时间维度长度
            sample_rate = file["sample_rate"]
        else:
            # 文件：使用torchaudio.info获取信息
            if "torchaudio.info" in file:
                # 使用缓存的信息（避免重复读取）
                info = file["torchaudio.info"]
            else:
                # 读取文件信息
                info = get_torchaudio_info(file)

            frames = info.num_frames  # 总帧数
            sample_rate = info.sample_rate  # 采样率

        # 时长 = 帧数 / 采样率
        return frames / sample_rate

    def get_num_samples(
        self, duration: float, sample_rate: Optional[int] = None
    ) -> int:
        """根据时长和采样率计算样本数
        
        这是一个确定性函数，给定时长和采样率，总是返回相同的样本数。
        用于确保音频块大小的确定性。
        
        参数
        ----------
        duration : float
            时长（秒）
        sample_rate : int, 可选
            采样率（Hz）
            如果为None，使用self.sample_rate
        
        返回
        -------
        int
            样本数（向下取整）
        
        异常
        ------
        ValueError
            如果sample_rate为None且self.sample_rate也为None
        
        注意
        -----
        使用math.floor向下取整，确保不会超出实际长度
        """
        sample_rate = sample_rate or self.sample_rate

        if sample_rate is None:
            raise ValueError(
                "`sample_rate` must be provided to compute number of samples."
            )

        # 样本数 = 时长 × 采样率（向下取整）
        return math.floor(duration * sample_rate)

    def __call__(self, file: AudioFile) -> Tuple[Tensor, int]:
        """读取音频文件并返回波形数据
        
        这是Audio类的主要接口，用于读取音频文件。
        支持多种输入格式，自动处理重采样和下混。
        
        参数
        ----------
        file : AudioFile
            音频文件输入，可以是：
            - 文件路径（str或Path）
            - 文件对象（IOBase）
            - 字典（包含"audio"键）
            - 字典（包含"waveform"和"sample_rate"键）
        
        返回
        -------
        waveform : (channel, time) torch.Tensor
            音频波形数据
            - 形状：(通道数, 时间样本数)
            - 如果mono="downmix"或"random"：形状为(1, time)
            - 数据类型：torch.float32
        sample_rate : int
            采样率（Hz）
            - 如果指定了self.sample_rate，返回目标采样率
            - 否则返回原始采样率
        
        处理流程
        --------
        1. 验证文件格式
        2. 读取音频（从文件或内存）
        3. 选择指定声道（如果指定了channel参数）
        4. 下混和重采样（如果需要）
        
        参考
        --------
        AudioFile: 支持的输入格式
        """

        file = self.validate_file(file)

        if "waveform" in file:
            waveform = file["waveform"]
            sample_rate = file["sample_rate"]

        elif "audio" in file:
            waveform, sample_rate = torchaudio.load(file["audio"])

            # rewind if needed
            if isinstance(file["audio"], IOBase):
                file["audio"].seek(0)

        channel = file.get("channel", None)

        if channel is not None:
            waveform = waveform[channel : channel + 1]

        return self.downmix_and_resample(waveform, sample_rate)

    def crop(
        self,
        file: AudioFile,
        segment: Segment,
        duration: Optional[float] = None,
        mode="raise",
    ) -> Tuple[Tensor, int]:
        """快速提取音频片段
        
        这是self(file).crop(segment, **kwargs)的优化版本，
        直接使用torchaudio的seek-and-read功能，避免加载整个文件。
        对于大文件和频繁裁剪操作，性能显著提升。
        
        参数
        ----------
        file : AudioFile
            音频文件输入
        segment : pyannote.core.Segment
            要提取的时间段
            包含start（起始时间）和end（结束时间），单位为秒
        duration : float, 可选
            覆盖Segment的持续时间，确保返回固定长度的帧数
            用于避免舍入误差导致的长度不一致
        mode : {'raise', 'pad'}, 默认'raise'
            超出边界时的处理方式：
            - 'raise': 抛出错误（默认，严格模式）
            - 'pad': 用零填充（宽松模式，适用于边界情况）
        
        返回
        -------
        waveform : (channel, time) torch.Tensor
            提取的音频片段波形
        sample_rate : int
            采样率
        
        异常
        ------
        ValueError
            当mode='raise'且请求的片段超出文件范围时
        
        性能优化
        --------
        - 使用torchaudio的frame_offset和num_frames参数
        - 避免加载整个文件到内存
        - 对于文件对象，如果seek失败，回退到加载整个文件
        
        注意
        -----
        如果指定了duration，会确保返回的帧数固定，
        即使segment.end - segment.start与duration略有不同
        """
        file = self.validate_file(file)

        if "waveform" in file:
            waveform = file["waveform"]
            frames = waveform.shape[1]
            sample_rate = file["sample_rate"]

        elif "torchaudio.info" in file:
            info = file["torchaudio.info"]
            frames = info.num_frames
            sample_rate = info.sample_rate

        else:
            info = get_torchaudio_info(file)
            frames = info.num_frames
            sample_rate = info.sample_rate

        channel = file.get("channel", None)

        # infer which samples to load from sample rate and requested chunk
        start_frame = math.floor(segment.start * sample_rate)

        if duration:
            num_frames = math.floor(duration * sample_rate)
            end_frame = start_frame + num_frames

        else:
            end_frame = math.floor(segment.end * sample_rate)
            num_frames = end_frame - start_frame

        if mode == "raise":
            if num_frames > frames:
                raise ValueError(
                    f"requested fixed duration ({duration:6f}s, or {num_frames:d} frames) is longer "
                    f"than file duration ({frames / sample_rate:.6f}s, or {frames:d} frames)."
                )

            if end_frame > frames + math.ceil(self.PRECISION * sample_rate):
                raise ValueError(
                    f"requested chunk [{segment.start:.6f}s, {segment.end:.6f}s] (frames #{start_frame:d} to #{end_frame:d}) "
                    f"lies outside of {file.get('uri', 'in-memory')} file bounds [0., {frames / sample_rate:.6f}s] ({frames:d} frames)."
                )
            else:
                end_frame = min(end_frame, frames)
                start_frame = end_frame - num_frames

            if start_frame < 0:
                raise ValueError(
                    f"requested chunk [{segment.start:.6f}s, {segment.end:.6f}s] (frames #{start_frame:d} to #{end_frame:d}) "
                    f"lies outside of {file.get('uri', 'in-memory')} file bounds [0, {frames / sample_rate:.6f}s] ({frames:d} frames)."
                )

        elif mode == "pad":
            pad_start = -min(0, start_frame)
            pad_end = max(end_frame, frames) - frames
            start_frame = max(0, start_frame)
            end_frame = min(end_frame, frames)
            num_frames = end_frame - start_frame

        if "waveform" in file:
            data = file["waveform"][:, start_frame:end_frame]

        else:
            try:
                data, _ = torchaudio.load(
                    file["audio"], frame_offset=start_frame, num_frames=num_frames
                )
                # rewind if needed
                if isinstance(file["audio"], IOBase):
                    file["audio"].seek(0)
            except RuntimeError:
                if isinstance(file["audio"], IOBase):
                    msg = "torchaudio failed to seek-and-read in file-like object."
                    raise RuntimeError(msg)

                msg = (
                    f"torchaudio failed to seek-and-read in {file['audio']}: "
                    f"loading the whole file instead."
                )

                warnings.warn(msg)
                waveform, sample_rate = self.__call__(file)
                data = waveform[:, start_frame:end_frame]

                # storing waveform and sample_rate for next time
                # as it is very likely that seek-and-read will
                # fail again for this particular file
                file["waveform"] = waveform
                file["sample_rate"] = sample_rate

        if channel is not None:
            data = data[channel : channel + 1, :]

        # pad with zeros
        if mode == "pad":
            data = F.pad(data, (pad_start, pad_end))

        return self.downmix_and_resample(data, sample_rate)

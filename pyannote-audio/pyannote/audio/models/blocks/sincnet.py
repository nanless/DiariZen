# The MIT License (MIT)
#
# Copyright (c) 2019- CNRS
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
#
# AUTHOR
# Hervé Bredin - http://herve.niderb.fr

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import Encoder, ParamSincFB

from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames,
    multi_conv_receptive_field_center,
    multi_conv_receptive_field_size,
)


class SincNet(nn.Module):
    """SincNet卷积层：可学习的Sinc滤波器组
    
    SincNet是一种可学习的频域卷积层，直接从原始波形提取特征。
    与传统卷积不同，SincNet的滤波器参数有物理意义（中心频率和带宽），
    这使得它比标准卷积更参数高效且更具可解释性。
    
    架构：
    1. ParamSincFB（参数化Sinc滤波器组）：80个滤波器，251采样点
    2. MaxPool1d：池化降维
    3. InstanceNorm1d：实例归一化
    4. 标准Conv1d层：进一步特征提取
    
    参数
    ----------
    sample_rate : int, 默认16000
        音频采样率（Hz）
        目前只支持16kHz，其他采样率需要修改kernel_size
    stride : int, 默认1
        第一层卷积的步长
        默认1表示不进行下采样，10表示10倍下采样
    
    特点
    -----
    - 可解释性：滤波器参数有物理意义
    - 参数效率：比标准卷积参数更少
    - 频域特性：直接学习频域特征
    - 端到端：从原始波形学习，无需手工特征
    
    参考
    -----
    Ravanelli, M., & Bengio, Y. (2018).
    "Speaker recognition from raw waveform with SincNet."
    IEEE Spoken Language Technology Workshop (SLT).
    """
    def __init__(self, sample_rate: int = 16000, stride: int = 1):
        super().__init__()

        # 目前只支持16kHz采样率
        if sample_rate != 16000:
            raise NotImplementedError("SincNet only supports 16kHz audio for now.")
            # TODO: 添加其他采样率支持
            # 理论上只需要将kernel_size乘以(sample_rate / 16000)
            # 但这需要仔细验证

        self.sample_rate = sample_rate  # 采样率
        self.stride = stride  # 第一层步长

        # ========== 第一层：参数化Sinc滤波器组 ==========
        # 实例归一化：对每个样本独立归一化
        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)

        # 存储各层的模块列表
        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.norm1d = nn.ModuleList()

        # 第一层：参数化Sinc滤波器组
        # 80个滤波器，每个251个采样点（约15.7ms @ 16kHz）
        self.conv1d.append(
            Encoder(
                ParamSincFB(
                    80,  # 滤波器数量（输出通道数）
                    251,  # 滤波器长度（采样点数）
                    stride=self.stride,  # 步长
                    sample_rate=sample_rate,  # 采样率
                    min_low_hz=50,  # 最低中心频率（Hz）
                    min_band_hz=50,  # 最小带宽（Hz）
                )
            )
        )
        # 最大池化：3倍下采样
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        # 实例归一化：80通道
        self.norm1d.append(nn.InstanceNorm1d(80, affine=True))

        # ========== 第二层：标准卷积 ==========
        # 80 → 60通道，5点卷积核
        self.conv1d.append(nn.Conv1d(80, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

        # ========== 第三层：标准卷积 ==========
        # 60 → 60通道，5点卷积核（保持维度）
        self.conv1d.append(nn.Conv1d(60, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """计算输出帧数
        
        根据输入样本数计算经过所有卷积和池化层后的输出帧数。
        这是一个缓存方法，相同输入只计算一次。
        
        参数
        ----------
        num_samples : int
            输入样本数（原始音频采样点数）
        
        返回
        -------
        int
            输出帧数
        
        注意
        -----
        层结构：
        1. SincNet卷积（251点，stride）
        2. MaxPool（3点，stride=3）
        3. Conv1d（5点，stride=1）
        4. MaxPool（3点，stride=3）
        5. Conv1d（5点，stride=1）
        6. MaxPool（3点，stride=3）
        """
        # 定义各层的kernel_size和stride
        kernel_size = [251, 3, 5, 3, 5, 3]  # 卷积核大小
        stride = [self.stride, 3, 1, 3, 1, 3]  # 步长
        padding = [0, 0, 0, 0, 0, 0]  # 填充
        dilation = [1, 1, 1, 1, 1, 1]  # 膨胀率

        # 使用工具函数计算多卷积后的帧数
        return multi_conv_num_frames(
            num_samples,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_center(
            frame,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)
        """

        outputs = self.wav_norm1d(waveforms)

        for c, (conv1d, pool1d, norm1d) in enumerate(
            zip(self.conv1d, self.pool1d, self.norm1d)
        ):
            outputs = conv1d(outputs)

            # https://github.com/mravanelli/SincNet/issues/4
            if c == 0:
                outputs = torch.abs(outputs)

            outputs = F.leaky_relu(norm1d(pool1d(outputs)))

        return outputs

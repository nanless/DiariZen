# MIT License
#
# Copyright (c) 2021 CNRS
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

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from torchaudio.transforms import MFCC

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
    multi_conv_num_frames,
    multi_conv_receptive_field_center,
    multi_conv_receptive_field_size,
)


class XVectorMFCC(Model):
    """X-Vector嵌入模型（基于MFCC特征）
    
    经典的说话人嵌入模型，架构为：
    MFCC（手工特征） → TDNN（时延神经网络） → Stats Pooling（统计池化） → Embedding（嵌入层）
    
    特点：
    - 传统方法：使用MFCC手工特征
    - 轻量级：模型小，速度快
    - 稳定：经过充分验证的架构
    
    参数
    ----------
    sample_rate : int, 默认16000
        音频采样率（Hz）
    num_channels : int, 默认1
        音频通道数
    mfcc : dict, 可选
        MFCC特征提取参数
        默认：{"n_mfcc": 40, "dct_type": 2, "norm": "ortho", "log_mels": False}
        - n_mfcc: MFCC系数数量（40）
        - dct_type: DCT类型（2）
        - norm: 归一化方式（"ortho"）
        - log_mels: 是否对mel谱取对数（False）
    dimension : int, 默认512
        输出嵌入向量维度
    
    架构说明
    --------
    1. MFCC: 提取40维MFCC特征（手工特征）
    2. TDNN: 5层时延神经网络
       - 层1: 40 → 512（kernel=5, dilation=1）
       - 层2: 512 → 512（kernel=3, dilation=2）
       - 层3: 512 → 512（kernel=3, dilation=3）
       - 层4: 512 → 512（kernel=1, dilation=1）
       - 层5: 512 → 1500（kernel=1, dilation=1）
    3. Stats Pooling: 统计池化（均值+标准差，1500*2=3000维）
    4. Embedding: 线性层（3000 → dimension）
    
    参考
    -----
    Snyder, D., et al. (2018).
    "X-vectors: Robust DNN embeddings for speaker recognition."
    IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
    """
    MFCC_DEFAULTS = {"n_mfcc": 40, "dct_type": 2, "norm": "ortho", "log_mels": False}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        mfcc: Optional[dict] = None,
        dimension: int = 512,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        # ========== MFCC特征提取 ==========
        mfcc = merge_dict(self.MFCC_DEFAULTS, mfcc)
        mfcc["sample_rate"] = sample_rate
        self.save_hyperparameters("mfcc", "dimension")
        self.mfcc = MFCC(**self.hparams.mfcc)

        # ========== TDNN层构建 ==========
        self.tdnns = nn.ModuleList()
        in_channel = self.hparams.mfcc["n_mfcc"]  # 输入：40维MFCC
        out_channels = [512, 512, 512, 512, 1500]  # 各层输出通道数
        self.kernel_size = [5, 3, 3, 1, 1]  # 卷积核大小
        self.dilation = [1, 2, 3, 1, 1]  # 膨胀率
        self.padding = [0, 0, 0, 0, 0]  # 填充
        self.stride = [1, 1, 1, 1, 1]  # 步长

        # 构建5层TDNN
        for out_channel, kernel_size, dilation in zip(
            out_channels, self.kernel_size, self.dilation
        ):
            self.tdnns.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        dilation=dilation,  # 使用膨胀卷积扩大感受野
                    ),
                    nn.LeakyReLU(),  # LeakyReLU激活函数
                    nn.BatchNorm1d(out_channel),  # 批归一化
                ]
            )
            in_channel = out_channel

        # ========== 统计池化和嵌入层 ==========
        self.stats_pool = StatsPool()  # 统计池化（均值+标准差）
        # 嵌入层：1500*2（统计池化输出） → dimension（最终嵌入维度）
        self.embedding = nn.Linear(in_channel * 2, self.hparams.dimension)

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return self.hparams.dimension

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """

        hop_length = self.mfcc.MelSpectrogram.spectrogram.hop_length
        n_fft = self.mfcc.MelSpectrogram.spectrogram.n_fft
        center = self.mfcc.MelSpectrogram.spectrogram.center

        num_frames = conv1d_num_frames(
            num_samples,
            kernel_size=n_fft,
            stride=hop_length,
            dilation=1,
            padding=n_fft // 2 if center else 0,
        )

        return multi_conv_num_frames(
            num_frames,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
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

        receptive_field_size = multi_conv_receptive_field_size(
            num_frames,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        hop_length = self.mfcc.MelSpectrogram.spectrogram.hop_length
        n_fft = self.mfcc.MelSpectrogram.spectrogram.n_fft

        return conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=n_fft,
            stride=hop_length,
            dilation=1,
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

        receptive_field_center = multi_conv_receptive_field_center(
            frame,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        hop_length = self.mfcc.MelSpectrogram.spectrogram.hop_length
        n_fft = self.mfcc.MelSpectrogram.spectrogram.n_fft
        center = self.mfcc.MelSpectrogram.spectrogram.center

        return conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=n_fft,
            stride=hop_length,
            padding=n_fft // 2 if center else 0,
            dilation=1,
        )

    def forward(
        self, waveforms: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        weights : torch.Tensor, optional
            Batch of weights with shape (batch, frame).
        """

        outputs = self.mfcc(waveforms).squeeze(dim=1)
        for block in self.tdnns:
            outputs = block(outputs)
        outputs = self.stats_pool(outputs, weights=weights)
        return self.embedding(outputs)


class XVectorSincNet(Model):
    SINCNET_DEFAULTS = {"stride": 10}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        sincnet: Optional[dict] = None,
        dimension: int = 512,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate

        self.save_hyperparameters("sincnet", "dimension")

        self.sincnet = SincNet(**self.hparams.sincnet)
        in_channel = 60

        self.tdnns = nn.ModuleList()
        out_channels = [512, 512, 512, 512, 1500]
        self.kernel_size = [5, 3, 3, 1, 1]
        self.dilation = [1, 2, 3, 1, 1]
        self.padding = [0, 0, 0, 0, 0]
        self.stride = [1, 1, 1, 1, 1]

        for out_channel, kernel_size, dilation in zip(
            out_channels, self.kernel_size, self.dilation
        ):
            self.tdnns.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(out_channel),
                ]
            )
            in_channel = out_channel

        self.stats_pool = StatsPool()

        self.embedding = nn.Linear(in_channel * 2, self.hparams.dimension)

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return self.hparams.dimension

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """

        num_frames = self.sincnet.num_frames(num_samples)

        return multi_conv_num_frames(
            num_frames,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
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

        receptive_field_size = multi_conv_receptive_field_size(
            num_frames,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        return self.sincnet.receptive_field_size(num_frames=receptive_field_size)

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

        receptive_field_center = multi_conv_receptive_field_center(
            frame,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        return self.sincnet.receptive_field_center(frame=receptive_field_center)

    def forward(
        self, waveforms: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        weights : torch.Tensor, optional
            Batch of weights with shape (batch, frame).
        """

        outputs = self.sincnet(waveforms).squeeze(dim=1)
        for tdnn in self.tdnns:
            outputs = tdnn(outputs)
        outputs = self.stats_pool(outputs, weights=weights)
        return self.embedding(outputs)

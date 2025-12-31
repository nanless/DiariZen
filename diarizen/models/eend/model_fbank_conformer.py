#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

"""
Fbank-Conformer说话人分离模型

该模型使用Fbank（滤波器组）特征和Conformer编码器，用于说话人分离任务。
模型架构：
1. Fbank特征提取：从原始音频波形提取Mel滤波器组特征
2. 投影和归一化：将Fbank特征投影到目标维度并归一化
3. Conformer编码：使用Conformer架构进行序列建模
4. 分类器：输出说话人分离的预测结果

与WavLM-Conformer模型的区别：
- 使用Fbank特征而不是WavLM预训练特征
- 特征提取更简单，计算更快
- 不需要预训练模型，可以直接训练
- 可选择性支持Fbank滤波器参数在训练中同步更新（通过fbank_requires_grad参数控制）

Fbank参数训练特性：
- fbank_requires_grad: 控制是否让滤波器参数参与训练（默认False，保持向后兼容）
- fbank_param_change_factor: 控制参数更新速度（值越小更新越平滑）
- fbank_param_rand_factor: 添加参数随机化正则化（值越大随机化越强）
"""

import torch
import torch.nn as nn

from functools import lru_cache  # 用于缓存计算结果

from pyannote.audio.core.model import Model as BaseModel  # 基础模型类
from diarizen.models.module.conformer import ConformerEncoder  # Conformer编码器
from diarizen.models.module.speechbrain_feats import Fbank  # Fbank特征提取器


class Model(BaseModel):
    """Fbank-Conformer说话人分离模型
    
    该模型使用Fbank（Mel滤波器组）作为特征提取器，Conformer作为序列编码器，
    用于端到端的说话人分离任务。
    
    模型流程：
    1. 输入：原始音频波形 (batch, channel, samples)
    2. Fbank特征提取：提取Mel滤波器组特征
    3. 投影和归一化：将特征维度投影到attention_in
    4. Conformer编码：使用Conformer进行序列建模
    5. 分类器：输出说话人分离预测
    
    参数
    ----------
    n_fft : int, 默认400
        FFT窗口大小（样本数），用于短时傅里叶变换
        通常设置为400（25ms @ 16kHz）
    n_mels : int, 默认80
        Mel滤波器组的数量，即Fbank特征的维度
    win_length : int, 默认25
        窗口长度（毫秒），用于STFT
    hop_length : int, 默认10
        帧移（毫秒），相邻帧之间的时间间隔
    sample_rate : int, 默认16000
        音频采样率（Hz）
    attention_in : int, 默认256
        Conformer的输入特征维度
    ffn_hidden : int, 默认1024
        Conformer中前馈网络的隐藏层维度
    num_head : int, 默认4
        Conformer中多头注意力的头数
    num_layer : int, 默认4
        Conformer编码器的层数
    kernel_size : int, 默认31
        Conformer中卷积模块的卷积核大小
    dropout : float, 默认0.1
        Dropout比率
    use_posi : bool, 默认False
        是否使用位置编码
    output_activate_function : str, 默认False
        输出激活函数类型（如"relu", "gelu"等），False表示不使用
    max_speakers_per_chunk : int, 默认4
        每个chunk中最大说话人数量
    max_speakers_per_frame : int, 默认2
        每帧中最大同时说话的说话人数量
    use_powerset : bool, 默认True
        是否使用幂集编码（用于处理重叠语音）
    chunk_size : int, 默认5
        音频chunk的大小（秒）
    num_channels : int, 默认8
        输入音频的通道数（多通道音频）
    selected_channel : int, 默认0
        选择的通道索引（从多通道中选择一个通道进行处理）
    fbank_requires_grad : bool, 默认False
        是否让Fbank滤波器参数在训练中可更新
    fbank_param_change_factor : float, 默认1.0
        Fbank参数更新速度控制因子，requires_grad=True时生效
    fbank_param_rand_factor : float, 默认0.0
        Fbank参数随机化因子，requires_grad=True时生效
    """
    def __init__(
        self,
        n_fft: int = 400,
        n_mels: int = 80,
        win_length: int = 25,  # ms
        hop_length: int = 10,  # ms
        sample_rate: int = 16000,
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
        output_activate_function: str = False,
        max_speakers_per_chunk: int = 4,
        max_speakers_per_frame: int = 2,
        use_powerset: bool = True,
        chunk_size: int = 5,
        num_channels: int = 8,
        selected_channel: int = 0,
        fbank_requires_grad: bool = False,
        fbank_param_change_factor: float = 1.0,
        fbank_param_rand_factor: float = 0.0,
    ):
        # 初始化基类，设置说话人分离任务的基本参数
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame,
            use_powerset=use_powerset
        )
        
        # 保存音频处理相关参数
        self.chunk_size = chunk_size  # chunk大小（秒）
        self.selected_channel = selected_channel  # 选择的通道索引

        # Fbank特征提取参数
        self.n_fft = n_fft  # FFT窗口大小（样本数）
        # 将窗口长度从毫秒转换为样本数
        # 例如：25ms @ 16kHz = 25 * 16000 / 1000 = 400 samples
        self.win_length_samples = win_length * sample_rate // 1000
        # 将帧移从毫秒转换为样本数
        # 例如：10ms @ 16kHz = 10 * 16000 / 1000 = 160 samples
        self.hop_length_samples = hop_length * sample_rate // 1000

        # Fbank特征提取器
        # 将原始音频波形转换为Mel滤波器组特征
        # 输出形状：(batch, frames, n_mels)
        # 可配置是否让滤波器参数在训练中可更新
        self.make_feats = Fbank(
            n_fft=n_fft,  # FFT窗口大小
            n_mels=n_mels,  # Mel滤波器数量
            win_length=win_length,  # 窗口长度（ms）
            hop_length=hop_length,  # 帧移（ms）
            requires_grad=fbank_requires_grad,  # 是否让滤波器参数在训练中可更新
            param_change_factor=fbank_param_change_factor,  # 控制参数更新速度
            param_rand_factor=fbank_param_rand_factor,  # 添加随机化正则化
        )

        # 投影层：将Fbank特征维度投影到Conformer的输入维度
        # n_mels (80) -> attention_in (256)
        self.proj = nn.Linear(n_mels, attention_in)
        # 层归一化：对投影后的特征进行归一化，稳定训练
        self.lnorm = nn.LayerNorm(attention_in)

        # Conformer编码器：用于序列建模和特征提取
        # Conformer结合了Transformer和CNN的优点，适合处理音频序列
        self.conformer = ConformerEncoder(
            attention_in=attention_in,  # 输入特征维度
            ffn_hidden=ffn_hidden,  # 前馈网络隐藏层维度
            num_head=num_head,  # 多头注意力头数
            num_layer=num_layer,  # 编码器层数
            kernel_size=kernel_size,  # 卷积核大小
            dropout=dropout,  # Dropout比率
            use_posi=use_posi,  # 是否使用位置编码
            output_activate_function=output_activate_function  # 输出激活函数
        )

        # 分类器：将Conformer输出映射到输出类别
        # attention_in -> dimension（输出类别数，可能是幂集类别数或常规类别数）
        self.classifier = nn.Linear(attention_in, self.dimension)
        # 激活函数：根据任务类型选择（通常是log_softmax用于幂集分类）
        self.activation = self.default_activation()

    @property
    def dimension(self) -> int:
        """计算模型输出的维度
        
        根据任务规范（specifications）确定输出维度：
        - 如果使用幂集编码（powerset）：返回幂集类别数
        - 否则：返回常规类别数（说话人数量）
        
        返回
        -------
        int
            输出维度（类别数）
            
        注意
        ----
        该模型不支持多任务学习，specifications必须是单一任务规范
        """
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        # 如果使用幂集编码，返回幂集类别数
        # 幂集类别数 = C(n,0) + C(n,1) + ... + C(n,k)
        # 其中n是说话人数量，k是max_speakers_per_frame
        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            # 否则返回常规类别数（说话人数量）
            return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int, center: bool = True) -> int:
        """计算输出帧数

        根据STFT（短时傅里叶变换）的参数计算输出帧数。
        Fbank特征提取使用STFT，因此输出帧数取决于输入样本数和STFT参数。

        参数
        ----------
        num_samples : int
            输入音频的样本数（采样点数）
        center : bool, 默认True
            是否对输入信号进行中心填充
            - True: 在信号两端填充，使得第一个和最后一个帧以信号边界为中心
            - False: 不进行填充，从信号开始处计算帧

        返回
        -------
        num_frames : int
            经过Fbank特征提取后的输出帧数
            
        公式
        -----
        center=True:  num_frames = 1 + num_samples // hop_length_samples
        center=False: num_frames = 1 + (num_samples - n_fft) // hop_length_samples
            
        注意
        ----
        使用@lru_cache装饰器缓存计算结果，避免重复计算
        
        参考
        -----
        https://pytorch.org/docs/stable/generated/torch.stft.html#torch.stft
        """

        if center:
            # 中心填充模式：在信号两端填充，使得可以计算更多帧
            # 公式：1 + num_samples // hop_length_samples
            return 1 + num_samples // self.hop_length_samples
        else:
            # 非中心填充模式：从信号开始处计算，需要考虑n_fft窗口大小
            # 公式：1 + (num_samples - n_fft) // hop_length_samples
            return 1 + (num_samples - self.n_fft) // self.hop_length_samples

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """计算感受野大小

        感受野定义了输出中每个点（或每帧）对应的输入范围。
        对于Fbank特征提取（基于STFT），感受野大小计算如下。

        参数
        ----------
        num_frames : int, 可选，默认1
            输出信号的帧数
            如果为1，计算单个输出帧对应的输入范围
            如果大于1，计算多个输出帧对应的总输入范围

        返回
        -------
        receptive_field_size : int
            感受野大小（以样本数/采样点为单位）
            
        公式
        -----
        receptive_field_size = n_fft + (num_frames - 1) * hop_length_samples
        
        说明
        -----
        - n_fft: 单个帧的感受野大小（FFT窗口大小）
        - (num_frames - 1) * hop_length_samples: 多个帧之间的间隔
        - 总感受野 = 第一个帧的窗口大小 + 后续帧的间隔
            
        用途
        -----
        用于理解模型在处理音频时的上下文范围
        """
        # 感受野 = FFT窗口大小 + (输出帧数-1) * 帧移
        # 第一部分是单个帧的感受野，第二部分是多个帧之间的间隔
        return self.n_fft + (num_frames - 1) * self.hop_length_samples

    def receptive_field_center(self, frame: int = 0, center: bool = True) -> int:
        """计算感受野的中心位置

        计算指定输出帧对应的感受野在输入音频中的中心位置。
        用于对齐输入和输出的时间位置。

        参数
        ----------
        frame : int, 可选，默认0
            输出帧索引（从0开始）
        center : bool, 可选，默认True
            是否使用中心填充模式
            - True: 感受野中心 = frame * hop_length_samples
            - False: 感受野中心 = frame * hop_length_samples + n_fft // 2

        返回
        -------
        receptive_field_center : int
            感受野中心在输入音频中的索引位置（以样本数/采样点为单位）
            
        公式
        -----
        center=True:  center = frame * hop_length_samples
        center=False: center = frame * hop_length_samples + n_fft // 2
            
        说明
        -----
        - center=True: 在中心填充模式下，每个帧的中心位置就是帧索引乘以帧移
        - center=False: 在非中心填充模式下，需要考虑FFT窗口的中心偏移
            
        用途
        -----
        用于时间对齐：确定输出帧对应输入音频的哪个时间点
        """
        if center:
            # 中心填充模式：感受野中心 = 帧索引 * 帧移
            # 这是因为在中心填充模式下，每个帧以hop_length_samples的倍数为中心
            return frame * self.hop_length_samples
        else:
            # 非中心填充模式：需要考虑FFT窗口的中心偏移
            # 感受野中心 = 帧位置 + FFT窗口中心偏移
            return frame * self.hop_length_samples + self.n_fft // 2
    
    @property
    def get_rf_info(self, sample_rate=16000):    
        """返回感受野信息，供数据集使用

        计算并返回模型的感受野相关信息，包括：
        - 输出帧数
        - 感受野持续时间（秒）
        - 感受野步长（秒，即相邻输出帧之间的时间间隔）

        参数
        ----------
        sample_rate : int, 默认16000
            音频采样率（Hz），用于将样本数转换为时间
            
        返回
        -------
        num_frames : int
            输出帧数（对于给定的chunk大小）
        duration : float
            感受野持续时间（秒）
        step : float
            感受野步长（秒），即相邻输出帧之间的时间间隔
            
        用途
        -----
        用于数据集生成和特征对齐，确保输入输出时间对应关系正确
        """
        # 计算单个输出帧对应的感受野大小（样本数）
        receptive_field_size = self.receptive_field_size(num_frames=1)
        # 计算感受野步长：两个输出帧之间的感受野大小差
        # 这表示相邻输出帧之间的时间间隔（以样本数表示）
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        # 计算给定chunk大小对应的输出帧数
        # chunk_size * sample_rate 将秒转换为样本数
        num_frames = self.num_frames(self.chunk_size * sample_rate)
        # 将样本数转换为时间（秒）
        duration = receptive_field_size / sample_rate  # 感受野持续时间
        step = receptive_field_step / sample_rate  # 感受野步长
        return num_frames, duration, step
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """前向传播

        模型的前向传播流程：
        1. 选择音频通道
        2. Fbank特征提取
        3. 投影和归一化
        4. Conformer编码
        5. 分类和激活

        参数
        ----------
        waveforms : torch.Tensor
            输入音频波形，形状为 (batch, channel, samples)
            - batch: 批次大小
            - channel: 音频通道数（多通道音频）
            - samples: 采样点数

        返回
        -------
        scores : torch.Tensor
            模型输出分数，形状为 (batch, frames, classes)
            - batch: 批次大小
            - frames: 输出帧数（经过Fbank特征提取后的时间步数）
            - classes: 类别数（可能是幂集类别数或常规类别数）
            输出通常是log_softmax的结果，用于后续的幂集解码
        """
        # 确保输入是3维张量 (batch, channel, samples)
        assert waveforms.dim() == 3
        # 从多通道中选择指定通道进行处理
        # 形状: (batch, channel, samples) -> (batch, samples)
        waveforms = waveforms[:, self.selected_channel, :]

        # 步骤1: Fbank特征提取
        # 将原始音频波形转换为Mel滤波器组特征
        # 输入: (batch, samples)
        # 输出: (batch, frames, n_mels)
        wav_feat = self.make_feats(waveforms)

        # 步骤2: 投影和归一化
        # 投影: (batch, frames, n_mels) -> (batch, frames, attention_in)
        outputs = self.proj(wav_feat)
        # 层归一化: 对特征进行归一化，稳定训练
        outputs = self.lnorm(outputs)
        
        # 步骤3: Conformer编码
        # 使用Conformer进行序列建模，提取高级特征
        # 形状保持不变: (batch, frames, attention_in)
        outputs = self.conformer(outputs)

        # 步骤4: 分类和激活
        # 分类器: (batch, frames, attention_in) -> (batch, frames, classes)
        outputs = self.classifier(outputs)
        # 激活函数: 通常是log_softmax，用于幂集分类
        outputs = self.activation(outputs)

        return outputs
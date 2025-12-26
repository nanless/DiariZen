#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

"""
WavLM-Conformer说话人分离模型

该模型结合了WavLM预训练特征提取器和Conformer编码器，用于说话人分离任务。
模型架构：
1. WavLM特征提取：从原始音频波形提取多层特征表示
2. 特征融合：对WavLM的多层特征进行加权求和
3. 投影和归一化：将特征投影到目标维度并归一化
4. Conformer编码：使用Conformer架构进行序列建模
5. 分类器：输出说话人分离的预测结果
"""

import os
import torch
import torch.nn as nn

from functools import lru_cache  # 用于缓存计算结果

from pyannote.audio.core.model import Model as BaseModel  # 基础模型类
from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames,  # 计算多层卷积后的输出帧数
    multi_conv_receptive_field_size,  # 计算多层卷积的感受野大小
    multi_conv_receptive_field_center  # 计算多层卷积的感受野中心位置
)

from diarizen.models.module.conformer import ConformerEncoder  # Conformer编码器
from diarizen.models.module.wav2vec2.model import wav2vec2_model as wavlm_model  # WavLM模型
from diarizen.models.module.wavlm_config import get_config  # WavLM配置获取函数

class Model(BaseModel):
    """WavLM-Conformer说话人分离模型
    
    该模型使用WavLM作为特征提取器，Conformer作为序列编码器，
    用于端到端的说话人分离任务。
    
    模型流程：
    1. 输入：原始音频波形 (batch, channel, samples)
    2. WavLM特征提取：提取多层特征表示
    3. 特征融合：对多层特征进行加权求和
    4. 投影和归一化：将特征维度投影到attention_in
    5. Conformer编码：使用Conformer进行序列建模
    6. 分类器：输出说话人分离预测
    
    参数
    ----------
    wavlm_src : str, 默认"wavlm_base"
        WavLM模型来源，可以是配置名称（如"wavlm_base"）或checkpoint文件路径
    wavlm_layer_num : int, 默认13
        WavLM模型的层数，用于特征融合的权重矩阵维度
    wavlm_feat_dim : int, 默认768
        WavLM特征维度（base模型为768，large模型为1024）
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
    chunk_size : int, 默认5
        音频chunk的大小（秒）
    num_channels : int, 默认8
        输入音频的通道数（多通道音频）
    selected_channel : int, 默认0
        选择的通道索引（从多通道中选择一个通道进行处理）
    sample_rate : int, 默认16000
        音频采样率（Hz）
    """
    def __init__(
        self,
        wavlm_src: str = "wavlm_base",
        wavlm_layer_num: int = 13,
        wavlm_feat_dim: int = 768,
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
        chunk_size: int = 5,
        num_channels: int = 8,
        selected_channel: int = 0,
        sample_rate: int = 16000,
    ):
        # 初始化基类，设置说话人分离任务的基本参数
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame
        )
        
        # 保存音频处理相关参数
        self.chunk_size = chunk_size  # chunk大小（秒）
        self.sample_rate = sample_rate  # 采样率（Hz）
        self.selected_channel = selected_channel  # 选择的通道索引

        # WavLM特征提取器
        # 加载预训练的WavLM模型，用于从原始音频提取特征
        self.wavlm_model = self.load_wavlm(wavlm_src)
        # 特征融合层：对WavLM的多层特征进行加权求和
        # 输入：wavlm_layer_num个层的特征，输出：单个融合后的特征
        # bias=False：不使用偏置，只学习各层的权重
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        # 投影层：将WavLM特征维度投影到Conformer的输入维度
        # wavlm_feat_dim (768) -> attention_in (256)
        self.proj = nn.Linear(wavlm_feat_dim, attention_in)
        # 层归一化：对投影后的特征进行归一化
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

    def non_wavlm_parameters(self):
        """返回非WavLM模型的参数列表
        
        用于训练时只更新部分参数（冻结WavLM，只训练其他部分）。
        这在微调场景中很有用，可以保持WavLM的预训练权重不变。
        
        返回
        -------
        list
            包含所有非WavLM模块参数的列表
            包括：特征融合层、投影层、归一化层、Conformer、分类器
        """
        return [
            *self.weight_sum.parameters(),  # 特征融合层参数
            *self.proj.parameters(),  # 投影层参数
            *self.lnorm.parameters(),  # 归一化层参数
            *self.conformer.parameters(),  # Conformer参数
            *self.classifier.parameters(),  # 分类器参数
        ]

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
    def num_frames(self, num_samples: int) -> int:
        """计算输出帧数

        根据WavLM的特征提取过程（多层卷积）计算输出帧数。
        WavLM使用7层卷积进行特征提取，每层的参数如下。

        参数
        ----------
        num_samples : int
            输入音频的样本数（采样点数）

        返回
        -------
        num_frames : int
            经过WavLM特征提取后的输出帧数
            
        注意
        ----
        使用@lru_cache装饰器缓存计算结果，避免重复计算
        """

        # WavLM特征提取的7层卷积参数
        # 这些参数对应WavLM模型的特征提取器配置
        kernel_size = [10, 3, 3, 3, 3, 2, 2]  # 各层卷积核大小
        stride = [5, 2, 2, 2, 2, 2, 2]  # 各层步长
        padding = [0, 0, 0, 0, 0, 0, 0]  # 各层填充
        dilation = [1, 1, 1, 1, 1, 1, 1]  # 各层膨胀率

        # 计算经过所有卷积层后的输出帧数
        return multi_conv_num_frames(
            num_samples,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """计算感受野大小

        感受野定义了输出中每个点（或每帧）对应的输入范围。
        对于WavLM特征提取器，计算多层卷积的总感受野大小。

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
            
        用途
        -----
        用于理解模型在处理音频时的上下文范围
        """

        # WavLM特征提取的7层卷积参数
        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        # 计算多层卷积的总感受野大小
        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        """计算感受野的中心位置

        计算指定输出帧对应的感受野在输入音频中的中心位置。
        用于对齐输入和输出的时间位置。

        参数
        ----------
        frame : int, 可选，默认0
            输出帧索引（从0开始）

        返回
        -------
        receptive_field_center : int
            感受野中心在输入音频中的索引位置（以样本数/采样点为单位）
            
        用途
        -----
        用于时间对齐：确定输出帧对应输入音频的哪个时间点
        """

        # WavLM特征提取的7层卷积参数
        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        # 计算多层卷积的感受野中心位置
        return multi_conv_receptive_field_center(
            frame,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    
    @property
    def get_rf_info(self):     
        """返回感受野信息，供数据集使用

        计算并返回模型的感受野相关信息，包括：
        - 输出帧数
        - 感受野持续时间（秒）
        - 感受野步长（秒，即相邻输出帧之间的时间间隔）

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
        num_frames = self.num_frames(self.chunk_size * self.sample_rate)
        # 将样本数转换为时间（秒）
        duration = receptive_field_size / self.sample_rate  # 感受野持续时间
        step = receptive_field_step / self.sample_rate  # 感受野步长
        return num_frames, duration, step

    def load_wavlm(self, source: str):
        """加载WavLM模型

        支持两种加载方式：
        1. 从预定义配置名称加载（如"wavlm_base", "wavlm_large"等）
        2. 从checkpoint文件加载（包含config和state_dict的.pt文件）

        参数
        ----------
        source : str
            - 如果是配置名称（如"wavlm_base_md_s80"）：
              使用预定义配置通过`get_config()`初始化模型
            - 如果是文件路径（如"pytorch_model.bin", "model.ckpt"等）：
              从checkpoint文件加载，使用保存的'config'和'state_dict'

        返回
        -------
        model : nn.Module
            初始化好的WavLM模型
            
        异常
        -----
        ValueError
            - checkpoint文件必须包含'config'和'state_dict'
            - 如果checkpoint中启用了pruning（剪枝），会抛出异常
        """
        if os.path.isfile(source):
            # 从checkpoint文件加载
            ckpt = torch.load(source, map_location="cpu")

            # 检查checkpoint格式
            if "config" not in ckpt or "state_dict" not in ckpt:
                raise ValueError("Checkpoint must contain 'config' and 'state_dict'.")

            # 检查是否启用了pruning（剪枝），如果启用则报错
            # 因为剪枝后的模型结构可能不完整
            for k, v in ckpt["config"].items():
                if 'prune' in k and v is not False:
                    raise ValueError(f"Pruning must be disabled. Found: {k}={v}")

            # 使用checkpoint中的config创建模型
            model = wavlm_model(**ckpt["config"])
            # 加载权重，strict=False允许部分权重不匹配
            model.load_state_dict(ckpt["state_dict"], strict=False)

        else:
            # 从预定义配置加载
            config = get_config(source)  # 获取配置字典
            model = wavlm_model(**config)  # 使用配置创建模型

        return model


    def wav2wavlm(self, in_wav, model):
        """将音频波形转换为WavLM特征

        使用WavLM模型提取多层特征表示。
        WavLM的每一层都会产生一个特征表示，这些特征会被堆叠在一起。

        参数
        ----------
        in_wav : torch.Tensor
            输入音频波形，形状为 (batch, samples)
        model : nn.Module
            WavLM模型实例

        返回
        -------
        torch.Tensor
            WavLM多层特征，形状为 (batch, frames, wavlm_layer_num, wavlm_feat_dim)
            其中：
            - batch: 批次大小
            - frames: 特征帧数（经过WavLM特征提取后的帧数）
            - wavlm_layer_num: WavLM层数（最后一维）
            - wavlm_feat_dim: 每层特征的维度
            
        处理流程
        --------
        1. 使用WavLM的extract_features方法提取所有层的特征
        2. layer_reps是一个列表，包含每一层的特征表示
        3. 使用torch.stack将所有层的特征堆叠在一起
        """
        # 提取WavLM所有层的特征表示
        # layer_reps: 列表，每个元素是一层的特征，形状为 (batch, frames, feat_dim)
        # 第二个返回值通常是其他信息（如注意力权重），这里忽略
        layer_reps, _ = model.extract_features(in_wav)
        # 将所有层的特征堆叠，在最后一个维度上堆叠
        # 结果形状: (batch, frames, wavlm_layer_num, wavlm_feat_dim)
        return torch.stack(layer_reps, dim=-1)
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """前向传播

        模型的前向传播流程：
        1. 选择音频通道
        2. WavLM特征提取
        3. 特征融合（多层特征加权求和）
        4. 投影和归一化
        5. Conformer编码
        6. 分类和激活

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
            - frames: 输出帧数（经过特征提取后的时间步数）
            - classes: 类别数（可能是幂集类别数或常规类别数）
            输出通常是log_softmax的结果，用于后续的幂集解码
        """
        # 确保输入是3维张量 (batch, channel, samples)
        assert waveforms.dim() == 3
        # 从多通道中选择指定通道进行处理
        # 形状: (batch, channel, samples) -> (batch, samples)
        waveforms = waveforms[:, self.selected_channel, :]

        # 步骤1: WavLM特征提取
        # 提取所有层的特征，形状: (batch, frames, wavlm_layer_num, wavlm_feat_dim)
        wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model)
        # 步骤2: 特征融合 - 对多层特征进行加权求和
        # weight_sum: (batch, frames, wavlm_layer_num, wavlm_feat_dim) -> (batch, frames, 1, wavlm_feat_dim)
        wavlm_feat = self.weight_sum(wavlm_feat)
        # 移除多余的维度，形状: (batch, frames, wavlm_feat_dim)
        wavlm_feat = torch.squeeze(wavlm_feat, -1)

        # 步骤3: 投影和归一化
        # 投影: (batch, frames, wavlm_feat_dim) -> (batch, frames, attention_in)
        outputs = self.proj(wavlm_feat)
        # 层归一化: 对特征进行归一化，稳定训练
        outputs = self.lnorm(outputs)
        
        # 步骤4: Conformer编码
        # 使用Conformer进行序列建模，提取高级特征
        # 形状保持不变: (batch, frames, attention_in)
        outputs = self.conformer(outputs)

        # 步骤5: 分类和激活
        # 分类器: (batch, frames, attention_in) -> (batch, frames, classes)
        outputs = self.classifier(outputs)
        # 激活函数: 通常是log_softmax，用于幂集分类
        outputs = self.activation(outputs)

        return outputs


if __name__ == '__main__':
    """测试代码：创建模型并进行前向传播测试"""
    # 使用WavLM base配置
    wavlm_conf_name = 'wavlm_base_md_s80'
    # 创建模型实例
    model = Model(wavlm_conf_name=wavlm_conf_name)
    # 打印模型结构
    print(model)
    # 创建测试输入：批次大小2，通道数1，采样点数32000（2秒@16kHz）
    x = torch.randn(2, 1, 32000)
    # 前向传播
    y = model(x)
    # 打印输出形状
    print(f'y: {y.shape}')
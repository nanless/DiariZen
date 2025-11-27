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

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class StatsPool(nn.Module):
    """统计池化层
    
    计算时序特征的加权均值和（无偏）标准差，并返回它们的拼接。
    这是说话人嵌入模型中常用的池化方法，将变长序列转换为固定长度向量。
    
    输出 = [均值, 标准差]
    维度：输入features → 输出2*features
    
    参考
    ---------
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    
    使用场景
    --------
    主要用于说话人嵌入模型（如X-Vector），将变长语音特征序列
    池化为固定长度的说话人表征向量。
    
    特点
    -----
    - 支持加权统计（可以只考虑活跃区域）
    - 无偏标准差估计
    - 可以处理多说话人情况（为每个说话人单独计算）
    """

    def _pool(self, sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """辅助函数：计算统计池化
        
        计算加权均值和标准差的核心实现。
        假设权重已经插值到与sequences相同的帧数，且只编码单个说话人的激活。
        
        参数
        ----------
        sequences : torch.Tensor
            特征序列，形状为(batch, features, frames)
        weights : torch.Tensor
            权重，形状为(batch, frames)
            应该已经插值到与sequences相同的帧数
            编码单个说话人的激活（0-1之间）
        
        返回
        -------
        torch.Tensor
            拼接后的统计特征，形状为(batch, 2 * features)
            [均值, 标准差]
        
        算法
        -----
        1. 计算加权均值：mean = sum(sequences * weights) / sum(weights)
        2. 计算加权方差：var = sum((sequences - mean)^2 * weights) / (v1 - v2/v1)
        3. 计算标准差：std = sqrt(var)
        4. 拼接：output = [mean, std]
        
        注意
        -----
        使用无偏方差估计，考虑权重的归一化
        """

        weights = weights.unsqueeze(dim=1)
        # (batch, 1, frames)

        v1 = weights.sum(dim=2) + 1e-8
        mean = torch.sum(sequences * weights, dim=2) / v1

        dx2 = torch.square(sequences - mean.unsqueeze(2))
        v2 = torch.square(weights).sum(dim=2)

        var = torch.sum(dx2 * weights, dim=2) / (v1 - v2 / v1 + 1e-8)
        std = torch.sqrt(var)

        return torch.cat([mean, std], dim=1)

    def forward(
        self, sequences: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        计算时序特征的统计池化（均值和标准差）。
        
        参数
        ----------
        sequences : torch.Tensor
            特征序列，形状为(batch, features, frames)
        weights : torch.Tensor, 可选
            权重，形状为：
            - (batch, frames)：单说话人情况
            - (batch, speakers, frames)：多说话人情况
            如果提供，使用加权统计；否则使用均匀权重
        
        注意
        ----
        `sequences`和`weights`可能使用不同的帧数，
        在这种情况下，`weights`会线性插值以匹配`sequences`的帧数。
        
        返回
        -------
        torch.Tensor
            统计特征，形状为：
            - (batch, 2 * features)：单说话人情况
            - (batch, speakers, 2 * features)：多说话人情况
            当`weights`包含`speakers`维度时，为每个说话人单独计算统计特征
        
        处理流程
        --------
        1. 如果提供了weights且帧数不匹配，线性插值weights
        2. 如果是多说话人情况，为每个说话人单独调用_pool
        3. 如果是单说话人情况，直接调用_pool
        4. 返回拼接的[均值, 标准差]
        """

        if weights is None:
            mean = sequences.mean(dim=-1)
            std = sequences.std(dim=-1, correction=1)
            return torch.cat([mean, std], dim=-1)

        if weights.dim() == 2:
            has_speaker_dimension = False
            weights = weights.unsqueeze(dim=1)
            # (batch, frames) -> (batch, 1, frames)
        else:
            has_speaker_dimension = True

        # interpolate weights if needed
        _, _, num_frames = sequences.shape
        _, _, num_weights = weights.shape
        if num_frames != num_weights:
            warnings.warn(
                f"Mismatch between frames ({num_frames}) and weights ({num_weights}) numbers."
            )
            weights = F.interpolate(weights, size=num_frames, mode="nearest")

        output = rearrange(
            torch.vmap(self._pool, in_dims=(None, 1))(sequences, weights),
            "speakers batch features -> batch speakers features",
        )

        if not has_speaker_dimension:
            return output.squeeze(dim=1)

        return output

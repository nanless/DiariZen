# MIT License
#
# Copyright (c) 2020-2021 CNRS
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


"""帧加权版本的常用损失函数

本模块提供了支持帧级加权的损失函数。
帧加权允许对不同时间帧分配不同的权重，常用于：
- 忽略预热区域（warm-up regions）
- 处理不平衡数据
- 关注重要时间段
"""

from typing import Optional

import torch
import torch.nn.functional as F


def interpolate(target: torch.Tensor, weight: Optional[torch.Tensor] = None):
    """插值权重以匹配目标帧分辨率
    
    当权重和目标的时间分辨率不同时，使用线性插值将权重调整到目标分辨率。
    这在处理不同采样率的特征时很有用。
    
    参数
    ----------
    target : torch.Tensor
        目标张量，形状为：
        - (batch_size, num_frames)：二分类或多分类
        - (batch_size, num_frames, num_classes)：多标签分类
    weight : torch.Tensor, 可选
        帧权重，形状为(batch_size, num_frames_weight, 1)
        如果为None，返回None
    
    返回
    -------
    weight : torch.Tensor 或 None
        插值后的帧权重，形状为(batch_size, num_frames, 1)
        如果输入weight为None，返回None
    
    注意
    -----
    使用线性插值（linear interpolation）进行时间维度上的插值
    """

    num_frames = target.shape[1]
    if weight is not None and weight.shape[1] != num_frames:
        weight = F.interpolate(
            weight.transpose(1, 2),
            size=num_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return weight


def binary_cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """帧加权的二元交叉熵损失
    
    支持帧级加权的二元交叉熵损失函数。
    用于二分类和多标签分类任务。
    
    参数
    ----------
    prediction : torch.Tensor
        模型预测，形状为(batch_size, num_frames, num_classes)
        通常是经过sigmoid激活的概率值（0-1之间）
    target : torch.Tensor
        目标标签，形状为：
        - (batch_size, num_frames)：二分类或多分类（会自动扩展维度）
        - (batch_size, num_frames, num_classes)：多标签分类
    weight : torch.Tensor, 可选
        帧权重，形状为(batch_size, num_frames, 1)
        用于对不同时间帧分配不同权重
        如果为None，所有帧权重相等
    
    返回
    -------
    loss : torch.Tensor
        标量损失值
    
    使用场景
    --------
    - 二分类：语音活动检测（VAD）
    - 多标签分类：说话人分离（多个说话人可能同时活跃）
    
    注意
    -----
    如果提供了weight，会自动插值以匹配target的时间分辨率
    """

    # reshape target to (batch_size, num_frames, num_classes) even if num_classes is 1
    if len(target.shape) == 2:
        target = target.unsqueeze(dim=2)

    if weight is None:
        return F.binary_cross_entropy(prediction, target.float())

    else:
        # interpolate weight
        weight = interpolate(target, weight=weight)

        return F.binary_cross_entropy(
            prediction, target.float(), weight=weight.expand(target.shape)
        )


def mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Frame-weighted mean-squared error loss

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction with shape (batch_size, num_frames, num_classes).
    target : torch.Tensor
        Target with shape (batch_size, num_frames) for binary or multi-class classification,
        or (batch_size, num_frames, num_classes) for multi-label classification.
    weight : (batch_size, num_frames, 1) torch.Tensor, optional
        Frame weight with shape (batch_size, num_frames, 1).

    Returns
    -------
    loss : torch.Tensor
    """

    # reshape target to (batch_size, num_frames, num_classes) even if num_classes is 1
    if len(target.shape) == 2:
        target = target.unsqueeze(dim=2)

    losses = F.mse_loss(prediction, target.float(), reduction="none")
    # (batch_size, num_frames, num_classes)

    if weight is None:
        return torch.mean(losses)

    else:
        # interpolate weight
        weight = interpolate(target, weight=weight).expand(losses.shape)
        # (batch_size, num_frames, num_classes)

        return torch.sum(losses * weight) / torch.sum(weight)


def nll_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Frame-weighted negative log-likelihood loss

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction with shape (batch_size, num_frames, num_classes).
    target : torch.Tensor
        Target with shape (batch_size, num_frames)
    class_weight : (num_classes, ) torch.Tensor, optional
        Class weight with shape (num_classes,  )
    weight : (batch_size, num_frames, 1) torch.Tensor, optional
        Frame weight with shape (batch_size, num_frames, 1).

    Returns
    -------
    loss : torch.Tensor
    """

    num_classes = prediction.shape[2]

    losses = F.nll_loss(
        prediction.view(-1, num_classes),
        # (batch_size x num_frames, num_classes)
        target.view(-1),
        # (batch_size x num_frames, )
        weight=class_weight,
        # (num_classes, )
        reduction="none",
    ).view(target.shape)
    # (batch_size, num_frames)

    if weight is None:
        return torch.mean(losses)

    else:
        # interpolate weight
        weight = interpolate(target, weight=weight).squeeze(dim=2)
        # (batch_size, num_frames)

        return torch.sum(losses * weight) / torch.sum(weight)

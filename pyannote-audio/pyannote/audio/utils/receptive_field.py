# MIT License
#
# Copyright (c) 2023 CNRS
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
感受野计算工具

该模块提供了一系列函数用于计算卷积神经网络的感受野（receptive field）。
感受野是深度学习中重要的概念，它定义了输出中每个点对应的输入范围。

主要功能：
1. 计算单层/多层卷积后的输出帧数
2. 计算单层/多层卷积的感受野大小
3. 计算感受野的中心位置

这些函数对于理解模型的时间分辨率和进行时间对齐非常重要。
"""

from typing import List


def conv1d_num_frames(
    num_samples, kernel_size=5, stride=1, padding=0, dilation=1
) -> int:
    """计算1D卷积后的输出帧数
    
    根据PyTorch Conv1d的公式计算输出帧数。
    
    参数
    ----------
    num_samples : int
        输入信号的样本数
    kernel_size : int, 默认5
        卷积核大小
    stride : int, 默认1
        步长
    padding : int, 默认0
        填充
    dilation : int, 默认1
        膨胀率
    
    返回
    -------
    int
        输出信号的帧数
    
    公式
    -----
    output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
    
    参考
    ------
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
    """
    return 1 + (num_samples + 2 * padding - dilation * (kernel_size - 1) - 1) // stride


def multi_conv_num_frames(
    num_samples: int,
    kernel_size: List[int] = None,
    stride: List[int] = None,
    padding: List[int] = None,
    dilation: List[int] = None,
) -> int:
    """计算多层卷积后的输出帧数
    
    依次应用多个卷积层，计算最终的输出帧数。
    
    参数
    ----------
    num_samples : int
        输入信号的样本数
    kernel_size : List[int]
        各层卷积核大小列表
    stride : List[int]
        各层步长列表
    padding : List[int]
        各层填充列表
    dilation : List[int]
        各层膨胀率列表
    
    返回
    -------
    int
        经过所有卷积层后的输出帧数
    
    处理流程
    --------
    逐层计算：每层的输出作为下一层的输入
    从第一层开始，依次应用每层的卷积操作，计算最终的输出帧数
    
    示例
    -----
    假设有2层卷积：
    - 第1层：kernel_size=10, stride=5, padding=0, dilation=1
      输入1000个样本 -> 输出199帧
    - 第2层：kernel_size=3, stride=2, padding=0, dilation=1
      输入199帧 -> 输出99帧
    最终输出：99帧
    """
    # 初始化为输入样本数
    num_frames = num_samples
    # 逐层应用卷积，每层的输出作为下一层的输入
    for k, s, p, d in zip(kernel_size, stride, padding, dilation):
        num_frames = conv1d_num_frames(
            num_frames, kernel_size=k, stride=s, padding=p, dilation=d
        )

    return num_frames


def conv1d_receptive_field_size(num_frames=1, kernel_size=5, stride=1, dilation=1):
    """计算1D卷积的感受野大小
    
    感受野定义了输出中每个点对应的输入范围。
    
    参数
    ----------
    num_frames : int, 默认1
        输出信号的帧数
    kernel_size : int, 默认5
        卷积核大小
    stride : int, 默认1
        步长
    dilation : int, 默认1
        膨胀率
    
    返回
    -------
    int
        感受野大小（样本数）
    
    公式
    -----
    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    receptive_field_size = effective_kernel_size + (num_frames - 1) * stride
    
    说明
    -----
    - 有效卷积核大小考虑了膨胀率（dilation）
      dilation=1时，有效卷积核大小 = kernel_size
      dilation=2时，有效卷积核大小 = 1 + (kernel_size-1)*2
    - 感受野大小包括所有输出帧对应的输入范围
      单个输出帧：感受野 = 有效卷积核大小
      多个输出帧：感受野 = 有效卷积核大小 + (num_frames-1) * stride
      
    示例
    -----
    kernel_size=5, stride=2, dilation=1, num_frames=1:
    有效卷积核大小 = 1 + (5-1)*1 = 5
    感受野 = 5 + (1-1)*2 = 5
    
    kernel_size=5, stride=2, dilation=1, num_frames=3:
    有效卷积核大小 = 5
    感受野 = 5 + (3-1)*2 = 9
    """
    # 计算有效卷积核大小（考虑膨胀率）
    # dilation=1: 有效大小 = kernel_size
    # dilation=2: 有效大小 = 1 + (kernel_size-1)*2
    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    # 感受野 = 有效卷积核大小 + (输出帧数-1) * 步长
    # 第一部分是单个帧的感受野，第二部分是多个帧之间的间隔
    return effective_kernel_size + (num_frames - 1) * stride


def multi_conv_receptive_field_size(
    num_frames: int,
    kernel_size: List[int] = None,
    stride: List[int] = None,
    padding: List[int] = None,
    dilation: List[int] = None,
) -> int:
    """计算多层卷积的感受野大小
    
    从输出层反向计算，逐层累加感受野大小。
    
    参数
    ----------
    num_frames : int
        输出信号的帧数
    kernel_size : List[int]
        各层卷积核大小列表
    stride : List[int]
        各层步长列表
    padding : List[int]
        各层填充列表（此函数中未使用）
    dilation : List[int]
        各层膨胀率列表
    
    返回
    -------
    int
        多层卷积的总感受野大小（样本数）
    
    算法
    -----
    从最后一层开始，反向计算每层的感受野
    每层的感受野会累加到前一层
    这是因为感受野是从输出向输入反向传播的
    
    示例
    -----
    假设有2层卷积：
    - 第2层（最后一层）：kernel_size=3, stride=2, dilation=1
      输出1帧 -> 感受野 = 3
    - 第1层：kernel_size=10, stride=5, dilation=1
      输入3帧（来自第2层的感受野）-> 感受野 = 10 + (3-1)*5 = 20
    总感受野：20个样本
    """
    # 初始化为输出帧数（最后一层的"输入"）
    receptive_field_size = num_frames

    # 从后往前遍历各层（反向计算）
    # reversed确保从最后一层开始计算
    for k, s, d in reversed(list(zip(kernel_size, stride, dilation))):
        # 计算当前层的感受野
        # 注意：这里的num_frames参数实际上是上一层（更靠近输出）的感受野大小
        receptive_field_size = conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=k,
            stride=s,
            dilation=d,
        )
    return receptive_field_size


def conv1d_receptive_field_center(
    frame=0, kernel_size=5, stride=1, padding=0, dilation=1
) -> int:
    """计算感受野的中心位置
    
    计算指定输出帧对应的感受野在输入中的中心位置。
    
    参数
    ----------
    frame : int, 默认0
        输出帧索引
    kernel_size : int, 默认5
        卷积核大小
    stride : int, 默认1
        步长
    padding : int, 默认0
        填充
    dilation : int, 默认1
        膨胀率
    
    返回
    -------
    int
        感受野中心在输入中的索引位置
    
    公式
    -----
    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    center = frame * stride + (effective_kernel_size - 1) // 2 - padding
    
    说明
    -----
    - frame * stride: 输出帧在输入中的起始位置
    - (effective_kernel_size - 1) // 2: 卷积核的中心偏移（从起始位置到中心）
    - padding: 填充会向左移动感受野，所以需要减去
    
    用途
    -----
    用于对齐输入和输出的时间位置
    例如：确定输出帧i对应的输入音频的时间点
    
    示例
    -----
    frame=0, kernel_size=5, stride=2, padding=0, dilation=1:
    有效卷积核大小 = 5
    中心 = 0*2 + (5-1)//2 - 0 = 2
    即输出帧0对应输入样本2（从0开始计数）
    """
    # 计算有效卷积核大小（考虑膨胀率）
    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    # 计算中心位置：
    # 1. frame * stride: 输出帧在输入中的起始位置
    # 2. (effective_kernel_size - 1) // 2: 从起始位置到卷积核中心的偏移
    # 3. - padding: 填充会向左移动感受野，需要减去
    return frame * stride + (effective_kernel_size - 1) // 2 - padding


def multi_conv_receptive_field_center(
    frame: int,
    kernel_size: List[int] = None,
    stride: List[int] = None,
    padding: List[int] = None,
    dilation: List[int] = None,
) -> int:
    """计算多层卷积的感受野中心位置
    
    从输出层反向计算，逐层计算感受野中心位置。
    
    参数
    ----------
    frame : int
        输出帧索引
    kernel_size : List[int]
        各层卷积核大小列表
    stride : List[int]
        各层步长列表
    padding : List[int]
        各层填充列表
    dilation : List[int]
        各层膨胀率列表
    
    返回
    -------
    int
        多层卷积的总感受野中心在输入中的索引位置
    
    算法
    -----
    从最后一层开始，反向计算每层的感受野中心
    每层的中心位置会作为前一层计算的输入（即前一层对应的"输出帧"）
    
    说明
    -----
    多层卷积的感受野中心计算需要逐层反向传播：
    - 最后一层的输出帧 -> 最后一层的感受野中心
    - 最后一层的感受野中心 -> 倒数第二层的"输出帧"
    - 倒数第二层的"输出帧" -> 倒数第二层的感受野中心
    - ... 依此类推
    
    示例
    -----
    假设有2层卷积，输出帧0：
    - 第2层：frame=0 -> 中心=2（假设）
    - 第1层：frame=2（来自第2层的中心）-> 中心=12（假设）
    最终感受野中心：12
    """
    # 初始化为输出帧索引（最后一层的"输出帧"）
    receptive_field_center = frame
    # 从后往前遍历各层（反向计算）
    for k, s, p, d in reversed(list(zip(kernel_size, stride, padding, dilation))):
        # 计算当前层的感受野中心
        # 注意：这里的frame参数实际上是上一层（更靠近输出）的感受野中心位置
        receptive_field_center = conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )

    return receptive_field_center

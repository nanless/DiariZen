#!/usr/bin/env python
# encoding: utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""
信号处理模块

本模块提供音频信号处理相关的工具函数，主要包括：
- 二值化（binarize）：将连续分数转换为二值标签
- 滞后阈值（hysteresis thresholding）：减少抖动
"""

from functools import singledispatch
from itertools import zip_longest
from typing import Optional, Union

import einops
import numpy as np
import scipy.signal
from pyannote.core import Annotation, Segment, SlidingWindowFeature, Timeline
from pyannote.core.utils.generators import pairwise


@singledispatch
def binarize(
    scores,
    onset: float = 0.5,
    offset: Optional[float] = None,
    initial_state: Optional[Union[bool, np.ndarray]] = None,
):
    """批量滞后阈值二值化（Hysteresis Thresholding）
    
    将连续分数转换为二值标签，使用滞后阈值减少抖动。
    滞后阈值使用两个阈值：
    - onset（开启阈值）：分数超过此值时标记为活跃
    - offset（关闭阈值）：分数低于此值时标记为非活跃
    只有当分数超过onset时才开启，低于offset时才关闭，避免频繁切换。
    
    参数
    ----------
    scores : numpy.ndarray 或 SlidingWindowFeature
        输入分数，形状为：
        - (num_chunks, num_frames, num_classes)：批次数据
        - (num_frames, num_classes)：单样本数据
    onset : float, 默认0.5
        开启阈值，分数超过此值时标记为活跃
    offset : float, 可选
        关闭阈值，分数低于此值时标记为非活跃
        如果为None，默认等于onset
    initial_state : np.ndarray 或 bool, 可选
        初始状态
        如果为None，根据第一个帧的分数自动推断
    
    返回
    -------
    binarized : 与scores相同的类型和形状
        二值化后的分数（布尔数组）
    
    参考
    ---------
    https://stackoverflow.com/questions/23289976/how-to-find-zero-crossings-with-hysteresis
    
    注意
    -----
    这是一个单分派泛型函数，根据输入类型自动选择对应的实现
    """
    raise NotImplementedError(
        "scores must be of type numpy.ndarray or SlidingWindowFeatures"
    )


@binarize.register
def binarize_ndarray(
    scores: np.ndarray,
    onset: float = 0.5,
    offset: Optional[float] = None,
    initial_state: Optional[Union[bool, np.ndarray]] = None,
):
    """numpy数组的滞后阈值二值化实现
    
    这是binarize函数针对numpy.ndarray类型的实现。
    使用滞后阈值算法将连续分数转换为二值标签。
    
    参数
    ----------
    scores : numpy.ndarray
        输入分数，形状为(num_frames, num_classes)
        注意：实际实现中支持批次维度(batch_size, num_frames, num_classes)
    onset : float, 默认0.5
        开启阈值
    offset : float, 可选
        关闭阈值，默认等于onset
    initial_state : np.ndarray 或 bool, 可选
        初始状态
    
    返回
    -------
    binarized : numpy.ndarray
        二值化后的分数，形状与输入相同，数据类型为bool
    
    算法说明
    --------
    1. 识别"定义良好"的帧（分数明确高于onset或低于offset）
    2. 对于每个定义良好的帧，根据阈值决定状态
    3. 对于未定义良好的帧，保持前一个定义良好帧的状态
    4. 这样可以避免在阈值附近频繁切换
    """

    offset = offset or onset

    batch_size, num_frames = scores.shape

    scores = np.nan_to_num(scores)

    if initial_state is None:
        initial_state = scores[:, 0] >= 0.5 * (onset + offset)

    elif isinstance(initial_state, bool):
        initial_state = initial_state * np.ones((batch_size,), dtype=bool)

    elif isinstance(initial_state, np.ndarray):
        assert initial_state.shape == (batch_size,)
        assert initial_state.dtype == bool

    initial_state = np.tile(initial_state, (num_frames, 1)).T

    on = scores > onset
    off_or_on = (scores < offset) | on

    # indices of frames for which the on/off state is well-defined
    well_defined_idx = np.array(
        list(zip_longest(*[np.nonzero(oon)[0] for oon in off_or_on], fillvalue=-1))
    ).T

    # corner case where well_defined_idx is empty
    if not well_defined_idx.size:
        return np.zeros_like(scores, dtype=bool) | initial_state

    # points to the index of the previous well-defined frame
    same_as = np.cumsum(off_or_on, axis=1)

    samples = np.tile(np.arange(batch_size), (num_frames, 1)).T

    return np.where(
        same_as, on[samples, well_defined_idx[samples, same_as - 1]], initial_state
    )


@binarize.register
def binarize_swf(
    scores: SlidingWindowFeature,
    onset: float = 0.5,
    offset: Optional[float] = None,
    initial_state: Optional[bool] = None,
):
    """SlidingWindowFeature的滞后阈值二值化实现
    
    这是binarize函数针对SlidingWindowFeature类型的实现。
    支持单样本和批次数据的二值化。
    
    参数
    ----------
    scores : SlidingWindowFeature
        输入分数，形状为：
        - (num_frames, num_classes)：单样本数据
        - (num_chunks, num_frames, num_classes)：批次数据
    onset : float, 默认0.5
        开启阈值，分数超过此值时标记为活跃
    offset : float, 可选
        关闭阈值，分数低于此值时标记为非活跃
        如果为None，默认等于onset
    initial_state : bool, 可选
        初始状态
        如果为None，根据第一个帧的分数自动推断
    
    返回
    -------
    binarized : SlidingWindowFeature
        二值化后的分数，形状和类型与输入相同
        数据类型为float（0.0或1.0），保持SlidingWindowFeature格式
    
    处理流程
    --------
    1. 根据输入维度选择处理方式（2D或3D）
    2. 重新排列维度以适配binarize_ndarray的输入格式
    3. 调用binarize_ndarray进行实际二值化
    4. 重新排列回原始维度
    5. 保持SlidingWindowFeature的sliding_window属性
    """

    offset = offset or onset

    if scores.data.ndim == 2:
        num_frames, num_classes = scores.data.shape
        data = einops.rearrange(scores.data, "f k -> k f", f=num_frames, k=num_classes)
        binarized = binarize(
            data, onset=onset, offset=offset, initial_state=initial_state
        )
        return SlidingWindowFeature(
            1.0
            * einops.rearrange(binarized, "k f -> f k", f=num_frames, k=num_classes),
            scores.sliding_window,
        )

    elif scores.data.ndim == 3:
        num_chunks, num_frames, num_classes = scores.data.shape
        data = einops.rearrange(
            scores.data, "c f k -> (c k) f", c=num_chunks, f=num_frames, k=num_classes
        )
        binarized = binarize(
            data, onset=onset, offset=offset, initial_state=initial_state
        )
        return SlidingWindowFeature(
            1.0
            * einops.rearrange(
                binarized, "(c k) f -> c f k", c=num_chunks, f=num_frames, k=num_classes
            ),
            scores.sliding_window,
        )

    else:
        raise ValueError(
            "Shape of scores must be (num_chunks, num_frames, num_classes) or (num_frames, num_classes)."
        )


class Binarize:
    """使用滞后阈值二值化检测分数
    
    将连续的检测分数转换为离散的激活区域（Annotation格式）。
    使用滞后阈值（hysteresis thresholding）减少抖动，并进行后处理。
    
    参数
    ----------
    onset : float, 默认0.5
        开启阈值，分数超过此值时标记为激活开始
    offset : float, 可选
        关闭阈值，分数低于此值时标记为激活结束
        如果为None，默认等于onset
        通常offset < onset（滞后阈值，避免频繁切换）
    min_duration_on : float, 默认0.0
        最小激活持续时间（秒）
        短于此时间的激活区域将被移除
    min_duration_off : float, 默认0.0
        最小非激活持续时间（秒）
        短于此时间的非激活间隙将被填充为激活
    pad_onset : float, 默认0.0
        激活区域起始时间扩展量（秒）
        将激活区域的开始时间向前移动此量
    pad_offset : float, 默认0.0
        激活区域结束时间扩展量（秒）
        将激活区域的结束时间向后移动此量
    
    工作流程
    --------
    1. 使用滞后阈值将连续分数转换为激活/非激活状态
    2. 应用时间扩展（pad_onset/pad_offset）
    3. 填充短的非激活间隙（min_duration_off）
    4. 移除短的激活区域（min_duration_on）
    5. 返回Annotation格式的结果
    
    参考
    ---------
    Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
    RNN-based Voice Activity Detection", InterSpeech 2015.
    
    使用示例
    -----
    >>> binarize = Binarize(onset=0.5, offset=0.3, min_duration_on=0.1)
    >>> annotation = binarize(scores)  # scores是SlidingWindowFeature
    """

    def __init__(
        self,
        onset: float = 0.5,
        offset: Optional[float] = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
    ):

        super().__init__()

        self.onset = onset
        self.offset = offset or onset

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        """二值化检测分数
        
        将连续的检测分数转换为离散的激活区域。
        使用滞后阈值算法，并进行后处理（填充短间隙、移除短区域）。
        
        参数
        ----------
        scores : SlidingWindowFeature
            检测分数，形状为(num_frames, num_classes)
            每个类别独立处理
        
        返回
        -------
        active : Annotation
            二值化后的激活区域
            每个区域是一个Segment，标签为对应的类别
        
        处理流程
        --------
        1. 对每个类别独立处理：
           a. 初始化状态（根据第一个帧的分数）
           b. 遍历所有时间帧，使用滞后阈值判断状态切换
           c. 记录激活区域的开始和结束时间
           d. 应用时间扩展（pad_onset/pad_offset）
        2. 合并重叠区域（由于padding可能导致重叠）
        3. 填充短的非激活间隙（min_duration_off）
        4. 移除短的激活区域（min_duration_on）
        """

        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        # 获取每个帧的时间戳（中间点）
        timestamps = [frames[i].middle for i in range(num_frames)]

        # 用于存储激活区域的Annotation对象
        active = Annotation()

        # 对每个类别独立处理
        for k, k_scores in enumerate(scores.data.T):
            # 确定标签（如果scores有labels属性则使用，否则使用索引）
            label = k if scores.labels is None else scores.labels[k]

            # 初始化状态
            start = timestamps[0]  # 当前激活区域的开始时间
            is_active = k_scores[0] > self.onset  # 初始是否为激活状态

            # 遍历所有时间帧
            for t, y in zip(timestamps[1:], k_scores[1:]):
                # 当前处于激活状态
                if is_active:
                    # 从激活切换到非激活（分数低于offset阈值）
                    if y < self.offset:
                        # 记录激活区域（应用时间扩展）
                        region = Segment(start - self.pad_onset, t + self.pad_offset)
                        active[region, k] = label
                        start = t  # 更新开始时间
                        is_active = False  # 切换到非激活状态

                # 当前处于非激活状态
                else:
                    # 从非激活切换到激活（分数高于onset阈值）
                    if y > self.onset:
                        start = t  # 更新开始时间
                        is_active = True  # 切换到激活状态

            # 如果最后仍处于激活状态，添加最后一个区域
            if is_active:
                region = Segment(start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label

        # 由于padding，某些激活区域可能重叠：合并它们
        # 同时填充同一说话人的短间隙（min_duration_off）
        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            active = active.support(collar=self.min_duration_off)

        # 移除短于min_duration_on的轨迹
        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active


class Peak:
    """峰值检测
    
    从检测分数中识别峰值，并将音频分割为以峰值作为边界的段。
    用于基于峰值的事件检测和分割。
    
    参数
    ----------
    alpha : float, 默认0.5
        峰值阈值，只有超过此值的峰值才会被识别
    min_duration : float, 默认1.0
        两个连续峰值之间的最小时间间隔（秒）
        用于过滤过于密集的峰值
    
    工作原理
    --------
    1. 使用scipy.signal.argrelmax找到局部最大值（峰值）
    2. 过滤低于阈值的峰值
    3. 确保峰值之间的间隔至少为min_duration
    4. 以峰值作为边界，将音频分割为段
    
    使用示例
    -----
    >>> peak_detector = Peak(alpha=0.6, min_duration=0.5)
    >>> timeline = peak_detector(scores)  # scores是SlidingWindowFeature
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_duration: float = 1.0,
    ):
        super(Peak, self).__init__()
        self.alpha = alpha  # 峰值阈值
        self.min_duration = min_duration  # 最小峰值间隔

    def __call__(self, scores: SlidingWindowFeature):
        """峰值检测
        
        从检测分数中识别峰值，并返回以峰值作为边界的分割结果。
        
        参数
        ----------
        scores : SlidingWindowFeature
            检测分数，必须是一维的（dimension=1）
            形状为(num_frames, 1)
        
        返回
        -------
        segmentation : Timeline
            分割结果，以峰值作为边界
            每个段是从一个峰值到下一个峰值（或开始/结束）
        
        处理流程
        --------
        1. 检查输入维度（必须是一维）
        2. 计算峰值检测的order参数（基于min_duration）
        3. 使用argrelmax找到局部最大值
        4. 过滤低于阈值的峰值
        5. 构建边界列表（开始、峰值、结束）
        6. 创建Timeline对象
        """

        if scores.dimension != 1:
            raise ValueError("Peak expects one-dimensional scores.")

        num_frames = len(scores)
        frames = scores.sliding_window

        precision = frames.step
        order = max(1, int(np.rint(self.min_duration / precision)))
        indices = scipy.signal.argrelmax(scores[:], order=order)[0]

        peak_time = np.array(
            [frames[i].middle for i in indices if scores[i] > self.alpha]
        )
        boundaries = np.hstack([[frames[0].start], peak_time, [frames[num_frames].end]])

        segmentation = Timeline()
        for i, (start, end) in enumerate(pairwise(boundaries)):
            segment = Segment(start, end)
            segmentation.add(segment)

        return segmentation

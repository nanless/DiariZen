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
    """(Batch) hysteresis thresholding

    Parameters
    ----------
    scores : SlidingWindowFeature
        (num_chunks, num_frames, num_classes)- or (num_frames, num_classes)-shaped scores.
    onset : float, optional
        Onset threshold. Defaults to 0.5.
    offset : float, optional
        Offset threshold. Defaults to `onset`.
    initial_state : np.ndarray or bool, optional
        Initial state.

    Returns
    -------
    binarized : same as scores
        Binarized scores with same shape and type as scores.

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
    """Binarize detection scores using hysteresis thresholding

    Parameters
    ----------
    onset : float, optional
        Onset threshold. Defaults to 0.5.
    offset : float, optional
        Offset threshold. Defaults to `onset`.
    min_duration_on : float, optional
        Remove active regions shorter than that many seconds. Defaults to 0s.
    min_duration_off : float, optional
        Fill inactive regions shorter than that many seconds. Defaults to 0s.
    pad_onset : float, optional
        Extend active regions by moving their start time by that many seconds.
        Defaults to 0s.
    pad_offset : float, optional
        Extend active regions by moving their end time by that many seconds.
        Defaults to 0s.

    Reference
    ---------
    Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
    RNN-based Voice Activity Detection", InterSpeech 2015.
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
        """Binarize detection scores

        Parameters
        ----------
        scores : SlidingWindowFeature
            Detection scores.

        Returns
        -------
        active : Annotation
            Binarized scores.
        """

        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]

        # annotation meant to store 'active' regions
        active = Annotation()

        for k, k_scores in enumerate(scores.data.T):

            label = k if scores.labels is None else scores.labels[k]

            # initial state
            start = timestamps[0]
            is_active = k_scores[0] > self.onset

            for t, y in zip(timestamps[1:], k_scores[1:]):

                # currently active
                if is_active:
                    # switching from active to inactive
                    if y < self.offset:
                        region = Segment(start - self.pad_onset, t + self.pad_offset)
                        active[region, k] = label
                        start = t
                        is_active = False

                # currently inactive
                else:
                    # switching from inactive to active
                    if y > self.onset:
                        start = t
                        is_active = True

            # if active at the end, add final region
            if is_active:
                region = Segment(start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label

        # because of padding, some active regions might be overlapping: merge them.
        # also: fill same speaker gaps shorter than min_duration_off
        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            active = active.support(collar=self.min_duration_off)

        # remove tracks shorter than min_duration_on
        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active


class Peak:
    """Peak detection

    Parameters
    ----------
    alpha : float, optional
        Peak threshold. Defaults to 0.5
    min_duration : float, optional
        Minimum elapsed time between two consecutive peaks. Defaults to 1 second.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_duration: float = 1.0,
    ):
        super(Peak, self).__init__()
        self.alpha = alpha
        self.min_duration = min_duration

    def __call__(self, scores: SlidingWindowFeature):
        """Peak detection

        Parameter
        ---------
        scores : SlidingWindowFeature
            Detection scores.

        Returns
        -------
        segmentation : Timeline
            Partition.
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

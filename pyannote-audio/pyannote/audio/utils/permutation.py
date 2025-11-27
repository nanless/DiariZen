# MIT License
#
# Copyright (c) 2020-2022 CNRS
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


import math
from typing import Callable, List, Optional, Tuple
from functools import singledispatch, partial

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from pyannote.core import SlidingWindowFeature


@singledispatch
def permutate(y1, y2, cost_func: Optional[Callable] = None, return_cost: bool = False):
    """寻找成本最小化的排列（排列不变性处理）
    
    在说话人分离任务中，说话人的顺序是任意的（排列不变性）。
    此函数找到y2到y1的最优排列，使得总成本最小。
    使用匈牙利算法（Hungarian algorithm）求解最优分配问题。
    
    参数
    ----------
    y1 : np.ndarray 或 torch.Tensor
        参考序列，形状为(batch_size, num_samples, num_classes_1)
    y2 : np.ndarray 或 torch.Tensor
        待排列序列，形状为：
        - (num_samples, num_classes_2)：单样本
        - (batch_size, num_samples, num_classes_2)：批次数据
    cost_func : callable, 可选
        成本函数，接收两个(num_samples, num_classes)序列，
        返回(num_classes,)的成对成本
        默认使用均方误差（MSE）
    return_cost : bool, 默认False
        是否返回成本矩阵
    
    返回
    -------
    permutated_y2 : np.ndarray 或 torch.Tensor
        排列后的y2，形状为(batch_size, num_samples, num_classes_1)
    permutations : list of tuple
        排列列表
        permutation[i] == j表示y2的第j个说话人应该映射到y1的第i个说话人
        permutation[i] == None表示y2中没有说话人映射到y1的第i个说话人
    cost : np.ndarray 或 torch.Tensor, 可选
        成本矩阵，形状为(batch_size, num_classes_1, num_classes_2)
        仅在return_cost=True时返回
    
    使用场景
    --------
    - 训练时：对齐预测和标签的说话人顺序
    - 评估时：找到最优说话人映射以计算准确指标
    - 排列不变性：处理说话人顺序不固定的问题
    
    算法
    -----
    使用匈牙利算法（linear_sum_assignment）求解最优分配问题
    """
    raise TypeError()


def mse_cost_func(Y, y, **kwargs):
    """计算类别级别的均方误差
    
    用于排列不变性处理的默认成本函数。
    计算每个说话人（类别）的均方误差。
    
    参数
    ----------
    Y, y : torch.Tensor
        两个序列，形状为(num_frames, num_classes)
        Y：参考序列
        y：待比较序列
    
    返回
    -------
    torch.Tensor
        每个类别的MSE，形状为(num_classes,)
    
    用途
    -----
    用于permutate函数，计算说话人之间的匹配成本
    """
    return torch.mean(F.mse_loss(Y, y, reduction="none"), axis=0)


def mae_cost_func(Y, y, **kwargs):
    """计算类别级别的平均绝对误差
    
    另一种成本函数，使用L1距离（平均绝对误差）而非L2距离。
    
    参数
    ----------
    Y, y : torch.Tensor
        两个序列，形状为(num_frames, num_classes)
        Y：参考序列
        y：待比较序列
    
    返回
    -------
    torch.Tensor
        每个类别的MAE，形状为(num_classes,)
    
    用途
    -----
    用于permutate函数，提供L1距离的匹配成本
    """
    return torch.mean(torch.abs(Y - y), axis=0)


@permutate.register
def permutate_torch(
    y1: torch.Tensor,
    y2: torch.Tensor,
    cost_func: Optional[Callable] = None,
    return_cost: bool = False,
) -> Tuple[torch.Tensor, List[Tuple[int]]]:

    batch_size, num_samples, num_classes_1 = y1.shape

    if len(y2.shape) == 2:
        y2 = y2.expand(batch_size, -1, -1)

    if len(y2.shape) != 3:
        msg = "Incorrect shape: should be (batch_size, num_frames, num_classes)."
        raise ValueError(msg)

    batch_size_, num_samples_, num_classes_2 = y2.shape
    if batch_size != batch_size_ or num_samples != num_samples_:
        msg = f"Shape mismatch: {tuple(y1.shape)} vs. {tuple(y2.shape)}."
        raise ValueError(msg)

    if cost_func is None:
        cost_func = mse_cost_func

    permutations = []
    permutated_y2 = []

    if return_cost:
        costs = []

    permutated_y2 = torch.zeros(y1.shape, device=y2.device, dtype=y2.dtype)

    for b, (y1_, y2_) in enumerate(zip(y1, y2)):
        # y1_ is (num_samples, num_classes_1)-shaped
        # y2_ is (num_samples, num_classes_2)-shaped
        with torch.no_grad():
            cost = torch.stack(
                [
                    cost_func(y2_, y1_[:, i : i + 1].expand(-1, num_classes_2))
                    for i in range(num_classes_1)
                ],
            )

        if num_classes_2 > num_classes_1:
            padded_cost = F.pad(
                cost,
                (0, 0, 0, num_classes_2 - num_classes_1),
                "constant",
                torch.max(cost) + 1,
            )
        else:
            padded_cost = cost

        permutation = [None] * num_classes_1
        for k1, k2 in zip(*linear_sum_assignment(padded_cost.cpu())):
            if k1 < num_classes_1:
                permutation[k1] = k2
                permutated_y2[b, :, k1] = y2_[:, k2]
        permutations.append(tuple(permutation))

        if return_cost:
            costs.append(cost)

    if return_cost:
        return permutated_y2, permutations, torch.stack(costs)

    return permutated_y2, permutations


@permutate.register
def permutate_numpy(
    y1: np.ndarray,
    y2: np.ndarray,
    cost_func: Optional[Callable] = None,
    return_cost: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int]]]:

    output = permutate(
        torch.from_numpy(y1),
        torch.from_numpy(y2),
        cost_func=cost_func,
        return_cost=return_cost,
    )

    if return_cost:
        permutated_y2, permutations, costs = output
        return permutated_y2.numpy(), permutations, costs.numpy()

    permutated_y2, permutations = output
    return permutated_y2.numpy(), permutations


def build_permutation_graph(
    segmentations: SlidingWindowFeature,
    onset: float = 0.5,
    cost_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mae_cost_func,
) -> nx.Graph:
    """Build permutation graph

    Parameters
    ----------
    segmentations : (num_chunks, num_frames, local_num_speakers)-shaped SlidingWindowFeature
        Raw output of segmentation model.
    onset : float, optionan
        Threshold above which a speaker is considered active. Defaults to 0.5
    cost_func : callable
        Cost function used to find the optimal bijective mapping between speaker activations
        of two overlapping chunks. Expects two (num_frames, num_classes) torch.tensor as input
        and returns cost as a (num_classes, ) torch.tensor. Defaults to mae_cost_func.

    Returns
    -------
    permutation_graph : nx.Graph
        Nodes are (chunk_idx, speaker_idx) tuples.
        An edge between two nodes indicate that those are likely to be the same speaker
        (the lower the value of "cost" attribute, the more likely).
    """

    cost_func = partial(cost_func, onset=onset)

    chunks = segmentations.sliding_window
    num_chunks, num_frames, _ = segmentations.data.shape
    max_lookahead = math.floor(chunks.duration / chunks.step - 1)
    lookahead = 2 * (max_lookahead,)

    permutation_graph = nx.Graph()

    for C, (chunk, segmentation) in enumerate(segmentations):
        for c in range(max(0, C - lookahead[0]), min(num_chunks, C + lookahead[1] + 1)):

            if c == C:
                continue

            # extract common temporal support
            shift = round((C - c) * num_frames * chunks.step / chunks.duration)

            if shift < 0:
                shift = -shift
                this_segmentations = segmentation[shift:]
                that_segmentations = segmentations[c, : num_frames - shift]
            else:
                this_segmentations = segmentation[: num_frames - shift]
                that_segmentations = segmentations[c, shift:]

            # find the optimal one-to-one mapping
            _, (permutation,), (cost,) = permutate(
                this_segmentations[np.newaxis],
                that_segmentations,
                cost_func=cost_func,
                return_cost=True,
            )

            for this, that in enumerate(permutation):

                this_is_active = np.any(this_segmentations[:, this] > onset)
                that_is_active = np.any(that_segmentations[:, that] > onset)

                if this_is_active:
                    permutation_graph.add_node((C, this))

                if that_is_active:
                    permutation_graph.add_node((c, that))

                if this_is_active and that_is_active:
                    permutation_graph.add_edge(
                        (C, this), (c, that), cost=cost[this, that]
                    )

    return permutation_graph

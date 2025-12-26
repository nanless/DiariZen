# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""聚类管道模块

本模块实现了说话人分离中的聚类算法。
聚类是将说话人嵌入向量分组为不同说话人的过程。

主要类：
- BaseClustering: 聚类基类
- AgglomerativeClustering: 凝聚聚类（默认，速度快）
- VBxClustering: 变分贝叶斯聚类（精度高）
- OracleClustering: Oracle聚类（仅用于评估）
"""


import random
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from einops import rearrange
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Categorical, Integer, Uniform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import oracle_segmentation
from pyannote.audio.utils.permutation import permutate

# VBx 
from diarizen.clustering.VBx import vbx_setup, cluster_vbx


def filter_embeddings_by_frames(
    binary_segmentations: np.ndarray, 
    min_frames: int = 0
) -> np.ndarray:
    """根据干净帧数过滤嵌入向量
    
    只保留有足够"干净"帧（只有一个说话人活跃）的说话人片段。
    这有助于提高聚类质量，因为重叠区域的嵌入可能不够准确。
    
    参数
    ----------
    binary_segmentations : np.ndarray
        二值化分割结果，形状为(chunks, frames, spks)
        1表示说话人活跃，0表示静音
    min_frames : int, 默认0
        每个说话人所需的最小干净帧数
        干净帧：只有该说话人活跃的帧（非重叠区域）
    
    返回
    -------
    np.ndarray
        布尔掩码，形状为(chunks, spks)
        True表示该说话人在该块中有足够的干净帧
    
    处理流程
    --------
    1. 识别只有单个说话人活跃的帧（非重叠区域）
    2. 统计每个说话人在每个块中的干净帧数
    3. 检查是否达到最小帧数要求
    """
    # 识别只有单个说话人活跃的帧（非重叠区域）
    single_active_mask = (np.sum(binary_segmentations, axis=2, keepdims=True) == 1)
    # 只保留该说话人是唯一活跃说话人的帧
    clean_frames = binary_segmentations * single_active_mask  # 形状: (chunks, frames, spks)
    # 统计每个块和说话人的干净帧数
    clean_frame_counts = np.sum(clean_frames, axis=1)  # 形状: (chunks, spks)
    # 检查是否达到最小帧数要求
    clean_segments = clean_frame_counts >= min_frames  # 形状: (chunks, spks)

    return clean_segments


class BaseClustering(Pipeline):
    """聚类基类
    
    所有聚类算法的基类，定义了聚类的基本接口和通用功能。
    
    参数
    ----------
    metric : str, 默认"cosine"
        距离度量方式，用于计算嵌入向量之间的距离
        可选："cosine"（余弦距离）、"euclidean"（欧氏距离）等
    max_num_embeddings : int, 默认1000
        最大嵌入向量数量
        如果嵌入向量太多，会随机采样到此数量（用于加速）
    constrained_assignment : bool, 默认False
        是否使用约束分配
        True：确保每个块中的说话人不会被分配到同一聚类
        False：无约束（默认）
    """
    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = 1000,
        constrained_assignment: bool = False,
    ):
        super().__init__()
        self.metric = metric  # 距离度量方式
        self.max_num_embeddings = max_num_embeddings  # 最大嵌入向量数
        self.constrained_assignment = constrained_assignment  # 是否约束分配

    def set_num_clusters(
        self,
        num_embeddings: int,
        num_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        max_clusters: Optional[int] = None,
    ):
        """设置聚类数量
        
        根据提供的参数确定最终的聚类数量范围。
        处理各种边界情况，确保聚类数量在合理范围内。
        
        参数
        ----------
        num_embeddings : int
            可用的嵌入向量数量
        num_clusters : int, 可选
            指定的聚类数量（如果提供，将覆盖min/max）
        min_clusters : int, 可选
            最小聚类数量
        max_clusters : int, 可选
            最大聚类数量
        
        返回
        -------
        num_clusters : int 或 None
            确定的聚类数量（如果min==max），否则为None
        min_clusters : int
            调整后的最小聚类数量
        max_clusters : int
            调整后的最大聚类数量
        
        处理逻辑
        --------
        1. 如果指定了num_clusters，则min和max都设为该值
        2. 否则，min默认为1，max默认为num_embeddings
        3. 确保min和max都在[1, num_embeddings]范围内
        4. 如果min==max，返回确定的聚类数量
        """
        min_clusters = num_clusters or min_clusters or 1
        min_clusters = max(1, min(num_embeddings, min_clusters))
        max_clusters = num_clusters or max_clusters or num_embeddings
        max_clusters = max(1, min(num_embeddings, max_clusters))

        if min_clusters > max_clusters:
            raise ValueError(
                f"min_clusters must be smaller than (or equal to) max_clusters "
                f"(here: min_clusters={min_clusters:g} and max_clusters={max_clusters:g})."
            )

        if min_clusters == max_clusters:
            num_clusters = min_clusters

        return num_clusters, min_clusters, max_clusters

    def filter_embeddings(
        self,
        embeddings: np.ndarray,
        segmentations: Optional[SlidingWindowFeature] = None,
        min_frames_ratio: int = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """过滤和降采样嵌入向量
        
        移除无效（NaN）的嵌入向量，并根据干净帧数过滤。
        如果嵌入向量太多，会随机采样到max_num_embeddings。
        
        参数
        ----------
        embeddings : np.ndarray
            嵌入向量序列，形状为(num_chunks, num_speakers, dimension)
        segmentations : SlidingWindowFeature, 可选
            二值化分割结果，形状为(num_chunks, num_frames, num_speakers)
            用于计算干净帧数
        min_frames_ratio : int, 默认0.1
            最小干净帧比例（相对于总帧数）
            例如：0.1表示至少需要10%的帧是干净的
        
        返回
        -------
        filtered_embeddings : np.ndarray
            过滤后的嵌入向量，形状为(num_embeddings, dimension)
        chunk_idx : np.ndarray
            块索引数组，形状为(num_embeddings,)
            指示每个嵌入向量来自哪个块
        speaker_idx : np.ndarray
            说话人索引数组，形状为(num_embeddings,)
            指示每个嵌入向量来自哪个说话人
        
        处理流程
        --------
        1. 识别活跃的说话人（分割结果非零）
        2. 识别有效的嵌入（非NaN）
        3. 根据干净帧数过滤（如果提供了segmentations）
        4. 如果嵌入太多，随机采样到max_num_embeddings
        """

        # whether speaker is active
        active = np.sum(segmentations.data, axis=1) > 0
        # whether speaker embedding extraction went fine
        valid = ~np.any(np.isnan(embeddings), axis=2)

        # whether speaker embedding extraction satisfies the minimum frames
        min_frames = round(min_frames_ratio * segmentations.data.shape[1])
        frame_mask = filter_embeddings_by_frames(segmentations.data, min_frames)
        chunk_idx, speaker_idx = np.where(active * valid * frame_mask)

        if len(chunk_idx) < 2:    
            # warning: no effective frames; input might be too short or fully overlapped
            frame_mask = filter_embeddings_by_frames(segmentations.data, 0)
            chunk_idx, speaker_idx = np.where(active * valid * frame_mask)

        # sample max_num_embeddings embeddings
        num_embeddings = len(chunk_idx)
        if num_embeddings > self.max_num_embeddings:
            indices = list(range(num_embeddings))
            random.shuffle(indices)
            indices = sorted(indices[: self.max_num_embeddings])
            chunk_idx = chunk_idx[indices]
            speaker_idx = speaker_idx[indices]

        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx  

    def constrained_argmax(self, soft_clusters: np.ndarray, const_location: np.ndarray = None) -> np.ndarray:
        """约束分配：使用匈牙利算法进行最优分配
        
        对每个块中的说话人进行聚类分配，确保同一块中的不同说话人不会被分配到同一聚类。
        使用匈牙利算法（linear_sum_assignment）找到最优的一对一分配。
        
        参数
        ----------
        soft_clusters : np.ndarray
            软聚类分配，形状为(num_chunks, num_speakers, num_clusters)
            值越大表示该说话人属于该聚类的概率越高
        const_location : np.ndarray, 可选
            约束位置掩码，形状为(num_chunks, num_speakers, num_clusters)
            如果提供，这些位置的soft_clusters值会被设为极小值（禁止分配）
        
        返回
        -------
        hard_clusters : np.ndarray
            硬聚类分配，形状为(num_chunks, num_speakers)
            值-2表示未分配，其他值表示分配的聚类索引
        
        算法说明
        --------
        1. 对NaN值进行处理（替换为最小值）
        2. 如果提供了约束位置，将对应位置的分数设为极小值
        3. 对每个块，使用匈牙利算法找到说话人到聚类的最优一对一分配
        4. 匈牙利算法确保每个说话人只分配到一个聚类，每个聚类最多分配给一个说话人
        """
        soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))
        num_chunks, num_speakers, num_clusters = soft_clusters.shape
        # num_chunks, num_speakers, num_clusters
        if const_location is not None:
            # 将约束位置设为极小值，禁止这些分配
            soft_clusters[const_location] = -10000      # TODO: try less ad-hoc options

        # 初始化硬聚类分配，-2表示未分配
        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)

        # 对每个块使用匈牙利算法进行最优分配
        for c, cost in enumerate(soft_clusters):
            # linear_sum_assignment找到最优的一对一分配（最大化总分数）
            speakers, clusters = linear_sum_assignment(cost, maximize=True)
            for s, k in zip(speakers, clusters):
                hard_clusters[c, s] = k

        return hard_clusters

    def assign_embeddings(
        self,
        embeddings: np.ndarray,
        train_chunk_idx: np.ndarray,
        train_speaker_idx: np.ndarray,
        train_clusters: np.ndarray,
        constrained: bool = False,
    ):
        """将嵌入向量分配到最近的聚类中心
        
        基于训练嵌入向量的聚类结果，计算每个聚类的中心（质心），
        然后将所有嵌入向量（包括训练集和测试集）分配到最近的聚类中心。
        
        参数
        ----------
        embeddings : np.ndarray
            完整的嵌入向量集合，形状为(num_chunks, num_speakers, dimension)
        train_chunk_idx : np.ndarray
            训练嵌入向量的块索引，形状为(num_embeddings,)
        train_speaker_idx : np.ndarray
            训练嵌入向量的说话人索引，形状为(num_embeddings,)
        train_clusters : np.ndarray
            训练嵌入向量的聚类分配，形状为(num_embeddings,)
            值k表示该嵌入向量属于第k个聚类
        constrained : bool, 默认False
            是否使用约束分配
            True：使用constrained_argmax（确保同一块中的说话人不分配到同一聚类）
            False：使用简单的argmax（默认）
        
        返回
        -------
        hard_clusters : np.ndarray
            硬聚类分配，形状为(num_chunks, num_speakers)
            值k表示该说话人被分配到第k个聚类
        soft_clusters : np.ndarray
            软聚类分配，形状为(num_chunks, num_speakers, num_clusters)
            值越大表示该说话人属于该聚类的概率越高
            计算公式：soft_clusters = 2 - distance（距离越小，分数越高）
        centroids : np.ndarray
            聚类中心向量，形状为(num_clusters, dimension)
            每个聚类中心是该聚类中所有训练嵌入向量的平均值
        
        处理流程
        --------
        1. 从训练嵌入向量计算每个聚类的中心（质心）
        2. 计算所有嵌入向量到所有聚类中心的距离
        3. 将距离转换为相似度分数（soft_clusters = 2 - distance）
        4. 根据相似度分数进行分配：
           - 约束分配：使用匈牙利算法确保同一块中的说话人不冲突
           - 非约束分配：简单选择最相似的聚类
        
        注意
        -----
        - 训练嵌入向量可能会被重新分配到不同的聚类（基于所有嵌入向量的全局最优分配）
        - 实验表明，这种重新分配通常能获得更好的结果
        """

        # TODO: option to add a new (dummy) cluster in case num_clusters < max(frame_speaker_count)

        num_clusters = np.max(train_clusters) + 1
        num_chunks, num_speakers, dimension = embeddings.shape

        # 提取训练嵌入向量
        train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]

        # 计算每个聚类的中心（质心）：该聚类中所有训练嵌入向量的平均值
        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        # 计算所有嵌入向量到所有聚类中心的距离
        # 1. 将嵌入向量reshape为(num_chunks * num_speakers, dimension)
        # 2. 计算到所有聚类中心的距离矩阵：(num_chunks * num_speakers, num_clusters)
        # 3. 重新reshape为(num_chunks, num_speakers, num_clusters)
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
            ),
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
        )
        # 将距离转换为相似度分数（距离越小，分数越高）
        # 对于余弦距离，范围是[0, 2]，所以使用2 - distance
        soft_clusters = 2 - e2k_distance

        # 将每个嵌入向量分配到最相似的聚类中心
        if constrained:
            # 约束分配：使用匈牙利算法，确保同一块中的说话人不分配到同一聚类
            hard_clusters = self.constrained_argmax(soft_clusters)
        else:
            # 非约束分配：简单选择最相似的聚类
            hard_clusters = np.argmax(soft_clusters, axis=2)

        # NOTE: train_embeddings might be reassigned to a different cluster
        # in the process. based on experiments, this seems to lead to better
        # results than sticking to the original assignment.

        return hard_clusters, soft_clusters, centroids

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: Optional[SlidingWindowFeature] = None,
        num_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        max_clusters: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """应用聚类算法
        
        这是聚类管道的主入口，执行完整的聚类流程：
        1. 过滤和采样嵌入向量
        2. 确定聚类数量
        3. 对训练嵌入向量进行聚类
        4. 将所有嵌入向量分配到聚类中心
        
        参数
        ----------
        embeddings : np.ndarray
            嵌入向量序列，形状为(num_chunks, num_speakers, dimension)
            每个块可能有多个说话人的嵌入向量
        segmentations : SlidingWindowFeature, 可选
            二值化分割结果，形状为(num_chunks, num_frames, num_speakers)
            用于过滤嵌入向量（只保留有足够干净帧的说话人）
        num_clusters : int, 可选
            指定的聚类数量。如果提供，将覆盖min/max参数
        min_clusters : int, 可选
            最小聚类数量。当num_clusters未提供时使用
        max_clusters : int, 可选
            最大聚类数量。当num_clusters未提供时使用
        
        返回
        -------
        hard_clusters : np.ndarray
            硬聚类分配，形状为(num_chunks, num_speakers)
            hard_clusters[c, s] = k 表示第c个块的第s个说话人被分配到第k个聚类
        soft_clusters : np.ndarray
            软聚类分配，形状为(num_chunks, num_speakers, num_clusters)
            soft_clusters[c, s, k] 值越大，表示第c个块的第s个说话人属于第k个聚类的概率越高
        centroids : np.ndarray
            聚类中心向量，形状为(num_clusters, dimension)
            每个聚类中心是该聚类中所有嵌入向量的平均值
        
        处理流程
        --------
        1. 过滤嵌入向量：移除无效（NaN）和低质量嵌入，如果太多则随机采样
        2. 确定聚类数量：根据参数和嵌入向量数量确定最终的聚类数量范围
        3. 特殊情况处理：如果只需要1个聚类，直接返回所有嵌入向量的平均值作为中心
        4. 训练聚类：对训练嵌入向量应用聚类算法（由子类实现）
        5. 分配嵌入：将所有嵌入向量分配到聚类中心
        """

        # 步骤1：过滤和采样嵌入向量
        train_embeddings, train_chunk_idx, train_speaker_idx = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
        )

        num_embeddings, _ = train_embeddings.shape

        # 步骤2：确定聚类数量
        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        # 步骤3：特殊情况处理（只需要1个聚类）
        if max_clusters < 2:
            # 当min_clusters = max_clusters = 1时，不需要聚类
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            # 所有嵌入向量的平均值作为唯一的聚类中心
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            return hard_clusters, soft_clusters, centroids

        # 步骤4：对训练嵌入向量进行聚类（由子类实现）
        train_clusters = self.cluster(
            train_embeddings,
            min_clusters,
            max_clusters,
            num_clusters=num_clusters,
        )

        # 步骤5：将所有嵌入向量分配到聚类中心
        hard_clusters, soft_clusters, centroids = self.assign_embeddings(
            embeddings,
            train_chunk_idx,
            train_speaker_idx,
            train_clusters,
            constrained=self.constrained_assignment,
        )

        return hard_clusters, soft_clusters, centroids


class AgglomerativeClustering(BaseClustering):
    """凝聚聚类（Agglomerative Clustering）
    
    使用层次聚类算法对说话人嵌入向量进行聚类。
    这是默认的聚类算法，速度快且效果良好。
    
    工作原理
    --------
    1. 从每个嵌入向量作为一个独立的聚类开始
    2. 迭代合并最相似的两个聚类
    3. 根据阈值或聚类数量停止合并
    4. 处理小聚类：将过小的聚类重新分配到最近的 large cluster
    
    参数
    ----------
    metric : str, 默认"cosine"
        距离度量方式，用于计算嵌入向量之间的距离
        可选："cosine"（余弦距离）、"euclidean"（欧氏距离）等
    max_num_embeddings : int, 默认np.inf
        最大嵌入向量数量（用于加速）
    constrained_assignment : bool, 默认True
        是否使用约束分配（确保同一块中的说话人不分配到同一聚类）
    
    超参数
    ----------------
    method : str
        链接方法（linkage method），可选：
        - "average": 平均链接（推荐）
        - "centroid": 质心链接
        - "complete": 完全链接
        - "median": 中位数链接
        - "single": 单链接
        - "ward": Ward链接
        - "weighted": 加权链接
    threshold : float
        聚类阈值，范围[0.0, 2.0]
        当聚类间距离超过此阈值时停止合并
        假设嵌入向量已归一化（单位长度）
    min_cluster_size : int
        最小聚类大小，范围[1, 20]
        小于此大小的聚类会被重新分配到最近的 large cluster
    """

    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = np.inf,
        constrained_assignment: bool = True,
    ):
        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
            constrained_assignment=constrained_assignment,
        )

        self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings
        self.method = Categorical(
            ["average", "centroid", "complete", "median", "single", "ward", "weighted"]
        )

        # minimum cluster size
        self.min_cluster_size = Integer(1, 20)

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: Optional[int] = None,
    ):
        """执行凝聚聚类
        
        使用层次聚类算法对嵌入向量进行聚类。
        首先构建层次树（dendrogram），然后根据阈值或目标聚类数量进行切割。
        
        参数
        ----------
        embeddings : np.ndarray
            嵌入向量，形状为(num_embeddings, dimension)
        min_clusters : int
            最小聚类数量
        max_clusters : int
            最大聚类数量
        num_clusters : int, 可选
            目标聚类数量。如果提供，会尝试找到最接近此数量的聚类结果
            如果为None，则根据threshold参数自动确定
        
        返回
        -------
        clusters : np.ndarray
            聚类分配结果，形状为(num_embeddings,)
            值k表示该嵌入向量属于第k个聚类（从0开始索引）
        
        处理流程
        --------
        1. 调整min_cluster_size以适应小数据集
        2. 特殊情况：只有一个嵌入向量时直接返回
        3. 根据metric和method选择合适的距离计算方式
        4. 构建层次树（dendrogram）
        5. 根据threshold切割层次树
        6. 处理小聚类：重新分配到最近的large cluster
        7. 如果指定了num_clusters，遍历层次树找到最接近的聚类结果
        """

        num_embeddings, _ = embeddings.shape

        # heuristic to reduce self.min_cluster_size when num_embeddings is very small
        # (0.1 value is kind of arbitrary, though)
        min_cluster_size = min(
            self.min_cluster_size, max(1, round(0.1 * num_embeddings))
        )

        # linkage function will complain when there is just one embedding to cluster
        if num_embeddings == 1:
            return np.zeros((1,), dtype=np.uint8)

        # centroid, median, and Ward method only support "euclidean" metric
        # therefore we unit-normalize embeddings to somehow make them "euclidean"
        if self.metric == "cosine" and self.method in ["centroid", "median", "ward"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric="euclidean"
            )

        # other methods work just fine with any metric
        else:
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric=self.metric
            )

        # apply the predefined threshold
        clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1

        # split clusters into two categories based on their number of items:
        # large clusters vs. small clusters
        cluster_unique, cluster_counts = np.unique(
            clusters,
            return_counts=True,
        )
        large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        num_large_clusters = len(large_clusters)

        # force num_clusters to min_clusters in case the actual number is too small
        if num_large_clusters < min_clusters:
            num_clusters = min_clusters

        # force num_clusters to max_clusters in case the actual number is too large
        elif num_large_clusters > max_clusters:
            num_clusters = max_clusters

        # look for perfect candidate if necessary
        if num_clusters is not None and num_large_clusters != num_clusters:
            # switch stopping criterion from "inter-cluster distance" stopping to "iteration index"
            _dendrogram = np.copy(dendrogram)
            _dendrogram[:, 2] = np.arange(num_embeddings - 1)

            best_iteration = num_embeddings - 1
            best_num_large_clusters = 1

            # traverse the dendrogram by going further and further away
            # from the "optimal" threshold

            for iteration in np.argsort(np.abs(dendrogram[:, 2] - self.threshold)):
                # only consider iterations that might have resulted
                # in changing the number of (large) clusters
                new_cluster_size = _dendrogram[iteration, 3]
                if new_cluster_size < min_cluster_size:
                    continue

                # estimate number of large clusters at considered iteration
                clusters = fcluster(_dendrogram, iteration, criterion="distance") - 1
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)

                # keep track of iteration that leads to the number of large clusters
                # as close as possible to the target number of clusters.
                if abs(num_large_clusters - num_clusters) < abs(
                    best_num_large_clusters - num_clusters
                ):
                    best_iteration = iteration
                    best_num_large_clusters = num_large_clusters

                # stop traversing the dendrogram as soon as we found a good candidate
                if num_large_clusters == num_clusters:
                    break

            # re-apply best iteration in case we did not find a perfect candidate
            if best_num_large_clusters != num_clusters:
                clusters = (
                    fcluster(_dendrogram, best_iteration, criterion="distance") - 1
                )
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)
                print(
                    f"Found only {num_large_clusters} clusters. Using a smaller value than {min_cluster_size} for `min_cluster_size` might help."
                )

        if num_large_clusters == 0:
            clusters[:] = 0
            return clusters

        small_clusters = cluster_unique[cluster_counts < min_cluster_size]
        if len(small_clusters) == 0:
            return clusters

        # re-assign each small cluster to the most similar large cluster based on their respective centroids
        large_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == large_k], axis=0)
                for large_k in large_clusters
            ]
        )
        small_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == small_k], axis=0)
                for small_k in small_clusters
            ]
        )
        centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)
        for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
            clusters[clusters == small_clusters[small_k]] = large_clusters[large_k]

        # re-number clusters from 0 to num_large_clusters
        _, clusters = np.unique(clusters, return_inverse=True)
        return clusters


class OracleClustering(BaseClustering):
    """Oracle聚类（完美聚类）
    
    这是一个理想化的聚类算法，使用真实标注（ground truth）来确定聚类分配。
    主要用于：
    - 性能上限评估（upper bound）
    - 对比实验
    - 调试和测试
    
    工作原理
    --------
    1. 从真实标注中提取oracle分割结果
    2. 使用排列算法（permutation）将预测分割与oracle分割对齐
    3. 根据对齐结果分配聚类标签
    
    注意
    -----
    这不是一个实际的聚类系统，而是使用真实标注作为"预测"结果。
    需要音频文件包含"annotation"键（真实说话人标注）。
    """

    def __call__(
        self,
        embeddings: Optional[np.ndarray] = None,
        segmentations: Optional[SlidingWindowFeature] = None,
        file: Optional[AudioFile] = None,
        frames: Optional[SlidingWindow] = None,
        **kwargs,
    ) -> np.ndarray:
        """应用Oracle聚类
        
        使用真实标注来确定聚类分配，这是性能的上限。
        
        参数
        ----------
        embeddings : np.ndarray, 可选
            嵌入向量序列，形状为(num_chunks, num_speakers, dimension)
            如果提供，会基于这些嵌入向量计算说话人中心
        segmentations : SlidingWindowFeature
            预测的分割结果，形状为(num_chunks, num_frames, num_speakers)
        file : AudioFile
            音频文件，必须包含"annotation"键（真实说话人标注）
        frames : SlidingWindow
            滑动窗口配置，用于对齐时间帧
        
        返回
        -------
        hard_clusters : np.ndarray
            硬聚类分配，形状为(num_chunks, num_speakers)
            hard_clusters[c, s] = k 表示第c个块的第s个说话人被分配到第k个聚类
        soft_clusters : np.ndarray
            软聚类分配，形状为(num_chunks, num_speakers, num_clusters)
            soft_clusters[c, s, k] = 1.0 表示该说话人属于该聚类，否则为0.0
        centroids : np.ndarray, 可选
            聚类中心向量，形状为(num_clusters, dimension)
            如果提供了embeddings，则计算中心；否则为None
        
        处理流程
        --------
        1. 从真实标注生成oracle分割结果
        2. 对每个块，使用排列算法将预测分割与oracle分割对齐
        3. 根据对齐结果分配聚类标签
        4. 如果提供了embeddings，计算聚类中心
        """

        num_chunks, num_frames, num_speakers = segmentations.data.shape
        window = segmentations.sliding_window

        oracle_segmentations = oracle_segmentation(file, window, frames=frames)
        #   shape: (num_chunks, num_frames, true_num_speakers)

        file["oracle_segmentations"] = oracle_segmentations

        _, oracle_num_frames, num_clusters = oracle_segmentations.data.shape

        segmentations = segmentations.data[:, : min(num_frames, oracle_num_frames)]
        oracle_segmentations = oracle_segmentations.data[
            :, : min(num_frames, oracle_num_frames)
        ]

        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)
        soft_clusters = np.zeros((num_chunks, num_speakers, num_clusters))
        for c, (segmentation, oracle) in enumerate(
            zip(segmentations, oracle_segmentations)
        ):
            _, (permutation, *_) = permutate(oracle[np.newaxis], segmentation)
            for j, i in enumerate(permutation):
                if i is None:
                    continue
                hard_clusters[c, i] = j
                soft_clusters[c, i, j] = 1.0

        if embeddings is None:
            return hard_clusters, soft_clusters, None

        (
            train_embeddings,
            train_chunk_idx,
            train_speaker_idx,
        ) = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
        )

        train_clusters = hard_clusters[train_chunk_idx, train_speaker_idx]
        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        return hard_clusters, soft_clusters, centroids


class VBxClustering(BaseClustering):
    """VBx聚类（变分贝叶斯聚类）
    
    使用变分贝叶斯（Variational Bayes）方法进行说话人聚类。
    这是精度最高的聚类算法，但计算成本也较高。
    
    工作原理
    --------
    1. AHC（Agglomerative Hierarchical Clustering）：使用层次聚类获得初始聚类
    2. LDA降维：使用线性判别分析将嵌入向量降维
    3. PLDA变换：使用概率线性判别分析（PLDA）进一步变换特征
    4. VBx迭代优化：使用变分贝叶斯方法迭代优化聚类结果
    
    参数
    ----------
    metric : str, 默认"cosine"
        距离度量方式（用于最终分配）
    max_num_embeddings : int, 默认np.inf
        最大嵌入向量数量
    constrained_assignment : bool, 默认True
        是否使用约束分配
    plda_dir : str, 默认""
        PLDA模型目录路径
        必须包含PLDA模型文件（用于特征变换）
    lda_dim : int, 默认128
        LDA降维后的维度
    maxIters : int, 默认20
        VBx迭代优化的最大迭代次数
    
    超参数
    ----------------
    ahc_criterion : str
        AHC停止准则，可选：
        - "maxclust": 基于最大聚类数量
        - "distance": 基于聚类间距离
    ahc_threshold : int 或 float
        AHC阈值
        - 如果criterion="maxclust": 整数，范围[0, 30]，表示最大聚类数量
        - 如果criterion="distance": 浮点数，范围[0.5, 0.8]，表示距离阈值
    Fa : float
        VBx超参数，范围[0.01, 0.5]
        控制聚类先验分布的参数
    Fb : float
        VBx超参数，范围[0.01, 15.0]
        控制聚类先验分布的参数
    
    参考
    -----
    VBx: A fully Bayesian method for speaker clustering
    https://github.com/BUTSpeechFIT/VBx
    """
    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = np.inf,
        constrained_assignment: bool = True,
        plda_dir: str = "",
        lda_dim: int = 128,
        maxIters: int = 20
    ):
        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
            constrained_assignment=constrained_assignment,
        )

        # AHC（层次聚类）参数
        self.ahc_criterion = Categorical(["maxclust", "distance"])
        if self.ahc_criterion == "maxclust":
            self.ahc_threshold = Integer(0, 30)    # 最大聚类数量
        else:
            self.ahc_threshold = Uniform(0.5, 0.8)  # 距离阈值（假设单位归一化嵌入）

        # VBx参数
        self.plda_dir = plda_dir  # PLDA模型目录
        self.lda_dim = lda_dim  # LDA降维维度
        self.maxIters = maxIters  # 最大迭代次数

        # VBx超参数（用于优化）
        self.Fa = Uniform(0.01, 0.5)  # VBx先验参数a
        self.Fb = Uniform(0.01, 15.0)  # VBx先验参数b

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: Optional[SlidingWindowFeature] = None,
        num_clusters: Optional[int] = None,     # 未使用，保留以保持兼容性
        min_clusters: Optional[int] = None,     # 未使用，保留以保持兼容性
        max_clusters: Optional[int] = None,     # 未使用，保留以保持兼容性
    ) -> np.ndarray:
        """应用VBx聚类
        
        使用变分贝叶斯方法进行说话人聚类。
        结合AHC初始化和VBx迭代优化。
        
        参数
        ----------
        embeddings : np.ndarray
            嵌入向量序列，形状为(num_chunks, num_speakers, dimension)
        segmentations : SlidingWindowFeature, 可选
            二值化分割结果，用于过滤嵌入向量
        num_clusters : int, 可选
            未使用，保留以保持兼容性
        min_clusters : int, 可选
            未使用，保留以保持兼容性
        max_clusters : int, 可选
            未使用，保留以保持兼容性
        
        返回
        -------
        hard_clusters : np.ndarray
            硬聚类分配，形状为(num_chunks, num_speakers)
        soft_clusters : np.ndarray
            软聚类分配，形状为(num_chunks, num_speakers, num_clusters)
        centroids : np.ndarray
            聚类中心向量，形状为(num_clusters, dimension)
        
        处理流程
        --------
        1. 过滤嵌入向量（只保留有足够干净帧的说话人）
        2. AHC初始化：使用层次聚类获得初始聚类
        3. PLDA变换：加载PLDA模型并变换特征
        4. VBx优化：使用变分贝叶斯方法迭代优化聚类
        5. 计算聚类中心：基于VBx结果计算中心
        6. 分配嵌入：将所有嵌入向量分配到聚类中心
        """
        # 步骤1：过滤嵌入向量
        train_embeddings, _, _ = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
            min_frames_ratio=0.1  # 至少需要10%的干净帧
        )
        
        # 特殊情况：嵌入向量太少，无法聚类
        if train_embeddings.shape[0] < 2:
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            return hard_clusters, soft_clusters, centroids
            
        # 步骤2：AHC（层次聚类）初始化
        # 归一化嵌入向量（单位长度）
        train_embeddings_normed = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        # 构建层次树（使用质心链接和欧氏距离）
        dendrogram = linkage(
            train_embeddings_normed, method="centroid", metric="euclidean"
        )
        # 根据阈值切割层次树，获得初始聚类
        ahc_clusters = fcluster(dendrogram, self.ahc_threshold, criterion=self.ahc_criterion) - 1
        # 重新编号聚类（从0开始）
        _, ahc_clusters = np.unique(ahc_clusters, return_inverse=True)
     
        # 步骤3：PLDA变换和VBx优化
        # 加载PLDA模型和变换函数
        x_tf, plda_tf, plda_psi = vbx_setup(self.plda_dir) 
        # 应用LDA降维和PLDA变换
        fea = plda_tf(x_tf(train_embeddings), lda_dim=self.lda_dim)
        # 提取PLDA的Phi矩阵（前lda_dim维）
        Phi = plda_psi[:self.lda_dim]
        # 应用VBx聚类优化
        q, sp = cluster_vbx(
            ahc_clusters, fea, Phi,
            Fa=self.Fa, Fb=self.Fb, maxIters=self.maxIters 
        )

        # 步骤4：计算聚类中心
        num_chunks, num_speakers, dimension = embeddings.shape
        # 只保留活跃的聚类（sp > 1e-7）
        # q是VBx优化的聚类参数，用于计算中心
        centroids = (q[:, sp > 1e-7].T @ train_embeddings.reshape(-1, dimension))
        # 注意：不需要除以范数，因为后续使用余弦相似度

        # 步骤5：计算所有嵌入向量到聚类中心的距离
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
            ),
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
        )
        # 将距离转换为相似度分数
        soft_clusters = 2 - e2k_distance 

        # 步骤6：分配嵌入向量到聚类
        if self.constrained_assignment:
            # 约束分配：使用匈牙利算法
            hard_clusters = self.constrained_argmax(
                soft_clusters, 
                const_location=None
            )
        else:
            # 非约束分配：简单选择最相似的聚类
            hard_clusters = np.argmax(soft_clusters, axis=2)

        # 重新编号聚类（从0开始连续编号）
        _, hard_clusters = np.unique(hard_clusters, return_inverse=True)
        hard_clusters = hard_clusters.reshape(num_chunks, num_speakers)

        return hard_clusters, soft_clusters, centroids

        
class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    OracleClustering = OracleClustering
    VBxClustering = VBxClustering

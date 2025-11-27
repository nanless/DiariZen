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
        soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))
        num_chunks, num_speakers, num_clusters = soft_clusters.shape
        # num_chunks, num_speakers, num_clusters
        if const_location is not None:
            soft_clusters[const_location] = -10000      # TODO: try less ad-hoc options

        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)

        for c, cost in enumerate(soft_clusters):
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
        """Assign embeddings to the closest centroid

        Cluster centroids are computed as the average of the train embeddings
        previously assigned to them.

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension)-shaped array
            Complete set of embeddings.
        train_chunk_idx : (num_embeddings,)-shaped array
        train_speaker_idx : (num_embeddings,)-shaped array
            Indices of subset of embeddings used for "training".
        train_clusters : (num_embedding,)-shaped array
            Clusters of the above subset
        constrained : bool, optional
            Use constrained_argmax, instead of (default) argmax.

        Returns
        -------
        soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
        hard_clusters : (num_chunks, num_speakers)-shaped array
        centroids : (num_clusters, dimension)-shaped array
            Clusters centroids
        """

        # TODO: option to add a new (dummy) cluster in case num_clusters < max(frame_speaker_count)

        num_clusters = np.max(train_clusters) + 1
        num_chunks, num_speakers, dimension = embeddings.shape

        train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]

        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        # compute distance between embeddings and clusters
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
        soft_clusters = 2 - e2k_distance

        # assign each embedding to the cluster with the most similar centroid
        if constrained:
            hard_clusters = self.constrained_argmax(soft_clusters)
        else:
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
        """Apply clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Has no effect when `num_clusters` is provided.

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        centroids : (num_clusters, dimension) array
            Centroid vectors of each cluster
        """

        train_embeddings, train_chunk_idx, train_speaker_idx = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
        )

        num_embeddings, _ = train_embeddings.shape

        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        if max_clusters < 2:
            # do NOT apply clustering when min_clusters = max_clusters = 1
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            return hard_clusters, soft_clusters, centroids

        train_clusters = self.cluster(
            train_embeddings,
            min_clusters,
            max_clusters,
            num_clusters=num_clusters,
        )

        hard_clusters, soft_clusters, centroids = self.assign_embeddings(
            embeddings,
            train_chunk_idx,
            train_speaker_idx,
            train_clusters,
            constrained=self.constrained_assignment,
        )

        return hard_clusters, soft_clusters, centroids


class AgglomerativeClustering(BaseClustering):
    """Agglomerative clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".

    Hyper-parameters
    ----------------
    method : {"average", "centroid", "complete", "median", "single", "ward"}
        Linkage method.
    threshold : float in range [0.0, 2.0]
        Clustering threshold.
    min_cluster_size : int in range [1, 20]
        Minimum cluster size
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
        """

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) array
            Embeddings
        min_clusters : int
            Minimum number of clusters
        max_clusters : int
            Maximum number of clusters
        num_clusters : int, optional
            Actual number of clusters. Default behavior is to estimate it based
            on values provided for `min_clusters`,  `max_clusters`, and `threshold`.

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
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
    """Oracle clustering"""

    def __call__(
        self,
        embeddings: Optional[np.ndarray] = None,
        segmentations: Optional[SlidingWindowFeature] = None,
        file: Optional[AudioFile] = None,
        frames: Optional[SlidingWindow] = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply oracle clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array, optional
            Sequence of embeddings. When provided, compute speaker centroids
            based on these embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        file : AudioFile
        frames : SlidingWindow

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        centroids : (num_clusters, dimension), optional
            Clusters centroids if `embeddings` is provided, None otherwise.
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

        self.ahc_criterion = Categorical(["maxclust", "distance"])
        if self.ahc_criterion == "maxclust":
            self.ahc_threshold = Integer(0, 30)    # set the max to 30
        else:
            self.ahc_threshold = Uniform(0.5, 0.8)  # assume unit-normalized embeddings

        # VBx
        self.plda_dir = plda_dir
        self.lda_dim = lda_dim
        self.maxIters = maxIters

        # tuned VBx hyper params
        self.Fa = Uniform(0.01, 0.5)
        self.Fb = Uniform(0.01, 15.0)

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: Optional[SlidingWindowFeature] = None,
        num_clusters: Optional[int] = None,     # not used but kept for compatibility
        min_clusters: Optional[int] = None,     # not used but kept for compatibility
        max_clusters: Optional[int] = None,     # not used but kept for compatibility
    ) -> np.ndarray:
        train_embeddings, _, _ = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
            min_frames_ratio=0.1
        )
        
        if train_embeddings.shape[0] < 2:
            # do NOT apply clustering when the number of training embeddings is less than 2
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            return hard_clusters, soft_clusters, centroids
            
        # AHC
        train_embeddings_normed = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        dendrogram = linkage(
            train_embeddings_normed, method="centroid", metric="euclidean"
        )
        ahc_clusters = fcluster(dendrogram, self.ahc_threshold, criterion=self.ahc_criterion) - 1
        _, ahc_clusters = np.unique(ahc_clusters, return_inverse=True)
     
        # VBx
        x_tf, plda_tf, plda_psi = vbx_setup(self.plda_dir) 
        fea = plda_tf(x_tf(train_embeddings), lda_dim=self.lda_dim)
        Phi = plda_psi[:self.lda_dim]
        q, sp = cluster_vbx(
            ahc_clusters, fea, Phi,
            Fa=self.Fa, Fb=self.Fb, maxIters=self.maxIters 
        )

        # calculate distance
        num_chunks, num_speakers, dimension = embeddings.shape
        centroids = (q[:, sp > 1e-7].T @ train_embeddings.reshape(-1, dimension))  # not division needed, cos-sim follows

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
        soft_clusters = 2 - e2k_distance 

        # assign each embedding to the cluster with the most similar centroid
        if self.constrained_assignment:
            hard_clusters = self.constrained_argmax(
                soft_clusters, 
                const_location=None
            )
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)

        # re-number clusters from 0 to num_large_clusters
        _, hard_clusters = np.unique(hard_clusters, return_inverse=True)
        hard_clusters = hard_clusters.reshape(num_chunks, num_speakers)

        return hard_clusters, soft_clusters, centroids

        
class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    OracleClustering = OracleClustering
    VBxClustering = VBxClustering

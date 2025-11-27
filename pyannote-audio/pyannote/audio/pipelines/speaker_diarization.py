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

"""说话人分离管道模块

本模块实现了完整的说话人分离（Speaker Diarization）管道。
说话人分离是指识别"谁在什么时候说话"的任务。

主要类：
- SpeakerDiarization: 说话人分离管道，整合了分割、嵌入和聚类三个步骤
"""

import functools
import itertools
import math
import textwrap
import warnings
from typing import Dict, Any, Callable, Optional, Text, Union

import numpy as np
import torch
from einops import rearrange
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import ParamDict, Uniform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_model,
)
from pyannote.audio.utils.signal import binarize


def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """将可迭代对象分批处理
    
    将可迭代对象分成固定大小的批次，用于批处理优化。
    最后一个批次可能不足batch_size，会用fillvalue填充。
    
    参数
    ----------
    iterable : iterable
        要分批的可迭代对象
    batch_size : int, 默认32
        批次大小
    fillvalue : 可选
        最后一个批次不足时的填充值
    
    返回
    -------
    iterator
        批次迭代器，每个批次是一个元组
    
    示例
    -----
    >>> list(batchify('ABCDEFG', 3))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', None, None)]
    """
    # 使用zip_longest将可迭代对象分成批次
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class SpeakerDiarization(SpeakerDiarizationMixin, Pipeline):
    """说话人分离管道 ⭐ 核心类
    
    这是pyannote-audio的核心管道，实现了完整的说话人分离流程。
    说话人分离是指识别"谁在什么时候说话"的任务。
    
    管道流程：
    1. 语音活动检测（VAD）：使用分割模型检测语音活动
    2. 说话人分割：将音频分割为说话人片段
    3. 说话人嵌入：为每个片段提取说话人嵌入向量
    4. 聚类：使用聚类算法将片段分组为说话人
    5. 后处理：优化分割结果
    
    参数
    ----------
    config : dict, 可选
        配置字典（用于DiariZen自定义配置）
    seg_duration : float, 可选
        分割模型的块持续时间（秒）
        如果为None，使用模型训练时的duration
    segmentation : Model, str 或 dict, 默认"pyannote/segmentation@2022.07"
        预训练分割模型
        可以是：
        - Model实例
        - HuggingFace模型ID（字符串）
        - 字典（包含模型配置）
    segmentation_step : float, 默认0.1
        分割窗口的步长，作为窗口持续时间的比例
        0.1表示90%重叠（推荐值，平衡精度和速度）
    embedding : Model, str 或 dict, 默认"speechbrain/spkrec-ecapa-voxceleb@..."
        预训练嵌入模型，用于提取说话人表征
    embedding_exclude_overlap : bool, 默认False
        提取嵌入时是否排除重叠语音区域
        True：只使用非重叠语音（更准确但可能数据不足）
        False：使用所有语音（默认，更稳定）
    clustering : str, 默认"AgglomerativeClustering"
        聚类算法，可选：
        - "AgglomerativeClustering": 凝聚聚类（默认，速度快）
        - "VBxClustering": 变分贝叶斯聚类（精度高）
        - "OracleClustering": Oracle聚类（仅用于评估）
    segmentation_batch_size : int, 默认1
        分割模型的批处理大小
        增大可提高速度，但需要更多内存
    embedding_batch_size : int, 默认1
        嵌入模型的批处理大小
        增大可提高速度，但需要更多内存
    der_variant : dict, 可选
        DER（说话人分离错误率）变体配置
        默认：{"collar": 0.0, "skip_overlap": False}
        用于评估指标计算
    use_auth_token : str, 可选
        加载私有HuggingFace模型时的认证token
        可通过`huggingface-cli login`获取
    device : torch.device, 默认cuda
        计算设备（CPU或GPU）
    
    使用示例
    -----
    >>> # 创建管道
    >>> pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1")
    >>> 
    >>> # 无约束分离（自动确定说话人数）
    >>> diarization = pipeline("/path/to/audio.wav")
    >>> 
    >>> # 指定说话人数（精确4个说话人）
    >>> diarization = pipeline("/path/to/audio.wav", num_speakers=4)
    >>> 
    >>> # 指定说话人范围（最少2个，最多10个）
    >>> diarization = pipeline("/path/to/audio.wav", min_speakers=2, max_speakers=10)
    >>> 
    >>> # 获取说话人嵌入（每个说话人的代表性嵌入向量）
    >>> diarization, embeddings = pipeline("/path/to/audio.wav", return_embeddings=True)
    >>> for s, speaker in enumerate(diarization.labels()):
    ...     print(f"{speaker}: {embeddings[s]}")  # embeddings[s]是该说话人的嵌入
    
    超参数
    ----------------
    可通过instantiate()方法优化的超参数：
    - segmentation.threshold: 分割阈值（0.1-0.9）
    - segmentation.min_duration_off: 最小静音持续时间（秒）
    - clustering.*: 聚类算法相关参数
    """

    def __init__(
        self,
        config: Union[Dict, Any] = None,
        seg_duration: float = None,
        segmentation: PipelineModel = "pyannote/segmentation@2022.07",
        segmentation_step: float = 0.1,
        embedding: PipelineModel = "speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e",
        embedding_exclude_overlap: bool = False,
        clustering: str = "AgglomerativeClustering",
        embedding_batch_size: int = 1,
        segmentation_batch_size: int = 1,
        der_variant: Optional[dict] = None,
        use_auth_token: Union[Text, None] = None,
        device: Optional[torch.device] = torch.device("cuda")
    ):
        """初始化说话人分离管道
        
        设置分割模型、嵌入模型和聚类算法，创建推理引擎。
        """
        super().__init__()

        # ========== 分割模型设置 ==========
        self.segmentation_model = segmentation
        # 加载分割模型（支持多种格式）
        model: Model = get_model(segmentation, config=config, use_auth_token=use_auth_token)
        self.segmentation_step = segmentation_step  # 分割窗口步长（比例）

        # ========== 嵌入模型设置 ==========
        self.embedding = embedding  # 嵌入模型配置
        self.embedding_batch_size = embedding_batch_size  # 嵌入批处理大小
        self.embedding_exclude_overlap = embedding_exclude_overlap  # 是否排除重叠区域

        # ========== 聚类算法设置 ==========
        self.klustering = clustering  # 聚类算法名称

        # ========== 评估指标设置 ==========
        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}

        # ========== 设备设置 ==========
        self.device = device

        # ========== 创建分割推理引擎 ==========
        # 确定分割块的持续时间
        segmentation_duration = seg_duration if seg_duration is not None else model.specifications.duration
        # 创建推理引擎（skip_aggregation=True因为我们需要逐块输出）
        self._segmentation = Inference(
            model,  # 分割模型
            duration=segmentation_duration,  # 块持续时间
            step=self.segmentation_step * segmentation_duration,  # 步长（秒）
            skip_aggregation=True,  # 不聚合，保留逐块输出
            batch_size=segmentation_batch_size,  # 批处理大小
            device=device  # 计算设备
        )

        # ========== 分割参数设置 ==========
        # 根据模型是否使用幂集编码，设置不同的可优化参数
        if self._segmentation.model.specifications.powerset:
            # 使用幂集编码：只需要最小静音持续时间参数
            self.segmentation = ParamDict(
                min_duration_off=Uniform(0.0, 1.0),  # 最小静音持续时间（秒）
            )
        else:
            # 不使用幂集编码：需要阈值和最小静音持续时间
            self.segmentation = ParamDict(
                threshold=Uniform(0.1, 0.9),  # 分割阈值（0.1-0.9）
                min_duration_off=Uniform(0.0, 1.0),  # 最小静音持续时间（秒）
            )

        # ========== 嵌入模型设置 ==========
        if self.klustering == "OracleClustering":
            # Oracle聚类不需要嵌入模型
            metric = "not_applicable"
        else:
            # 创建嵌入模型推理引擎
            self._embedding = PretrainedSpeakerEmbedding(
                self.embedding, device=device, use_auth_token=use_auth_token
            )
            # 创建音频处理器（匹配嵌入模型的采样率）
            self._audio = Audio(sample_rate=self._embedding.sample_rate, mono="downmix")
            # 获取嵌入模型的度量方式（用于聚类）
            metric = self._embedding.metric

        # ========== 聚类算法设置 ==========
        try:
            # 从Clustering枚举中获取聚类类
            Klustering = Clustering[clustering]
        except KeyError:
            # 如果聚类算法不存在，抛出错误
            raise ValueError(
                f'clustering must be one of [{", ".join(list(Clustering.__members__))}]'
            )
        # 实例化聚类算法（传入度量方式）
        self.clustering = Klustering.value(metric=metric)

    @property
    def segmentation_batch_size(self) -> int:
        return self._segmentation.batch_size

    @segmentation_batch_size.setter
    def segmentation_batch_size(self, batch_size: int):
        self._segmentation.batch_size = batch_size

    def default_parameters(self):
        raise NotImplementedError()

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1

    @property
    def CACHED_SEGMENTATION(self):
        return "training_cache/segmentation"

    def get_segmentations(self, file, hook=None, soft=False) -> SlidingWindowFeature:
        """应用分割模型获取分割结果
        
        这是管道的第一步：使用分割模型检测语音活动和说话人分割。
        
        参数
        ----------
        file : AudioFile
            音频文件输入
        hook : Optional[Callable]
            进度回调函数，在处理每个批次时调用
            可用于显示进度条
        soft : bool, 默认False
            是否返回软分割（概率值）而不是硬分割（二值化）
            True：返回概率值（0-1之间）
            False：返回二值化结果（0或1）
        
        返回
        -------
        SlidingWindowFeature
            分割结果，形状为(num_chunks, num_frames, num_speakers)
            - num_chunks: 音频块数量
            - num_frames: 每个块的帧数
            - num_speakers: 说话人数量（或幂集类别数）
        
        注意
        -----
        由于skip_aggregation=True，返回的是逐块的分割结果，
        每个块对应一个滑动窗口的输出。
        """

        if hook is not None:
            hook = functools.partial(hook, "segmentation", None)

        segmentations: SlidingWindowFeature = self._segmentation(file, hook=hook, soft=soft)
        return segmentations

    def get_embeddings(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
        hook: Optional[Callable] = None,
    ):
        """为每个(块, 说话人)对提取嵌入向量
        
        这是管道的第二步：为每个说话人片段提取说话人嵌入向量。
        嵌入向量用于后续的聚类步骤。
        
        参数
        ----------
        file : AudioFile
            音频文件输入
        binary_segmentations : SlidingWindowFeature
            二值化分割结果，形状为(num_chunks, num_frames, num_speakers)
            每个元素表示该帧是否属于该说话人（0或1）
        exclude_overlap : bool, 默认False
            提取嵌入时是否排除重叠语音区域
            True：只使用非重叠语音（更准确但可能数据不足）
            False：使用所有语音（默认，更稳定）
            注意：如果非重叠语音太短，会回退到使用整个语音
        hook : Optional[Callable]
            进度回调函数，在处理每个批次后调用
            可用于显示进度条
        
        返回
        -------
        np.ndarray
            嵌入向量数组，形状为(num_chunks, num_speakers, dimension)
            - num_chunks: 音频块数量
            - num_speakers: 说话人数量
            - dimension: 嵌入向量维度（通常512或256）
        
        处理流程
        --------
        1. 根据分割结果提取每个说话人的音频片段
        2. 如果exclude_overlap=True，排除重叠区域
        3. 对每个片段提取嵌入向量
        4. 如果片段太短，使用整个语音片段
        """

        # when optimizing the hyper-parameters of this pipeline with frozen
        # "segmentation.threshold", one can reuse the embeddings from the first trial,
        # bringing a massive speed up to the optimization process (and hence allowing to use
        # a larger search space).
        if self.training:
            # we only re-use embeddings if they were extracted based on the same value of the
            # "segmentation.threshold" hyperparameter or if the segmentation model relies on
            # `powerset` mode
            cache = file.get("training_cache/embeddings", dict())
            if ("embeddings" in cache) and (
                self._segmentation.model.specifications.powerset
                or (cache["segmentation.threshold"] == self.segmentation.threshold)
            ):
                return cache["embeddings"]

        duration = binary_segmentations.sliding_window.duration
        num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

        if exclude_overlap:
            # minimum number of samples needed to extract an embedding
            # (a lower number of samples would result in an error)
            min_num_samples = self._embedding.min_num_samples

            # corresponding minimum number of frames
            num_samples = duration * self._embedding.sample_rate
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

            # zero-out frames with overlapping speech
            clean_frames = 1.0 * (
                np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
            )
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data * clean_frames,
                binary_segmentations.sliding_window,
            )

        else:
            min_num_frames = -1
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data, binary_segmentations.sliding_window
            )

        def iter_waveform_and_mask():
            for (chunk, masks), (_, clean_masks) in zip(
                binary_segmentations, clean_segmentations
            ):
                # chunk: Segment(t, t + duration)
                # masks: (num_frames, local_num_speakers) np.ndarray

                waveform, _ = self._audio.crop(
                    file,
                    chunk,
                    duration=duration,
                    mode="pad",
                )
                # waveform: (1, num_samples) torch.Tensor

                # mask may contain NaN (in case of partial stitching)
                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

                for mask, clean_mask in zip(masks.T, clean_masks.T):
                    # mask: (num_frames, ) np.ndarray

                    if np.sum(clean_mask) > min_num_frames:
                        used_mask = clean_mask
                    else:
                        used_mask = mask

                    yield waveform[None], torch.from_numpy(used_mask)[None]
                    # w: (1, 1, num_samples) torch.Tensor
                    # m: (1, num_frames) torch.Tensor

        batches = batchify(
            iter_waveform_and_mask(),
            batch_size=self.embedding_batch_size,
            fillvalue=(None, None),
        )

        batch_count = math.ceil(num_chunks * num_speakers / self.embedding_batch_size)

        embedding_batches = []

        if hook is not None:
            hook("embeddings", None, total=batch_count, completed=0)

        for i, batch in enumerate(batches, 1):
            waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))

            waveform_batch = torch.vstack(waveforms)
            # (batch_size, 1, num_samples) torch.Tensor

            mask_batch = torch.vstack(masks)
            # (batch_size, num_frames) torch.Tensor

            embedding_batch: np.ndarray = self._embedding(
                waveform_batch, masks=mask_batch
            )
            # (batch_size, dimension) np.ndarray

            embedding_batches.append(embedding_batch)

            if hook is not None:
                hook("embeddings", embedding_batch, total=batch_count, completed=i)

        embedding_batches = np.vstack(embedding_batches)

        embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)

        # caching embeddings for subsequent trials
        # (see comments at the top of this method for more details)
        if self.training:
            if self._segmentation.model.specifications.powerset:
                file["training_cache/embeddings"] = {
                    "embeddings": embeddings,
                }
            else:
                file["training_cache/embeddings"] = {
                    "segmentation.threshold": self.segmentation.threshold,
                    "embeddings": embeddings,
                }

        return embeddings

    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        hard_clusters: np.ndarray,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        """Build final discrete diarization out of clustered segmentation

        Parameters
        ----------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Raw speaker segmentation.
        hard_clusters : (num_chunks, num_speakers) array
            Output of clustering step.
        count : (total_num_frames, 1) SlidingWindowFeature
            Instantaneous number of active speakers.

        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """

        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        num_clusters = np.max(hard_clusters) + 1
        clustered_segmentations = np.NAN * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )

        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(hard_clusters, segmentations)
        ):
            # cluster is (local_num_speakers, )-shaped
            # segmentation is (num_frames, local_num_speakers)-shaped
            for k in np.unique(cluster):
                if k == -2:
                    continue

                # TODO: can we do better than this max here?
                clustered_segmentations[c, :, k] = np.max(
                    segmentation[:, cluster == k], axis=1
                )

        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )

        return self.to_diarization(clustered_segmentations, count)

    def apply(
        self,
        file: AudioFile,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        num_speakers : int, optional
            Number of speakers, when known.
        min_speakers : int, optional
            Minimum number of speakers. Has no effect when `num_speakers` is provided.
        max_speakers : int, optional
            Maximum number of speakers. Has no effect when `num_speakers` is provided.
        return_embeddings : bool, optional
            Return representative speaker embeddings.
        hook : callable, optional
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        embeddings : np.array, optional
            Representative speaker embeddings such that `embeddings[i]` is the
            speaker embedding for i-th speaker in diarization.labels().
            Only returned when `return_embeddings` is True.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        num_speakers, min_speakers, max_speakers = self.set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        segmentations = self.get_segmentations(file, hook=hook)
        hook("segmentation", segmentations)
        #   shape: (num_chunks, num_frames, local_num_speakers)

        # binarize segmentation
        if self._segmentation.model.specifications.powerset:
            binarized_segmentations = segmentations
        else:
            binarized_segmentations: SlidingWindowFeature = binarize(
                segmentations,
                onset=self.segmentation.threshold,
                initial_state=False,
            )

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation.model._receptive_field,
            warm_up=(0.0, 0.0),
        )
        hook("speaker_counting", count)
        #   shape: (num_frames, 1)
        #   dtype: int

        # exit early when no speaker is ever active
        if np.nanmax(count.data) == 0.0:
            diarization = Annotation(uri=file["uri"])
            if return_embeddings:
                return diarization, np.zeros((0, self._embedding.dimension))

            return diarization

        if self.klustering == "OracleClustering" and not return_embeddings:
            embeddings = None
        else:
            embeddings = self.get_embeddings(
                file,
                binarized_segmentations,
                exclude_overlap=self.embedding_exclude_overlap,
                hook=hook,
            )
            hook("embeddings", embeddings)
            #   shape: (num_chunks, local_num_speakers, dimension)

        hard_clusters, _, centroids = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
            file=file,  # <== for oracle clustering
            frames=self._segmentation.model._receptive_field,  # <== for oracle clustering
        )
        # hard_clusters: (num_chunks, num_speakers)
        # centroids: (num_speakers, dimension)

        # number of detected clusters is the number of different speakers
        num_different_speakers = np.max(hard_clusters) + 1

        # detected number of speakers can still be out of bounds
        # (specifically, lower than `min_speakers`), since there could be too few embeddings
        # to make enough clusters with a given minimum cluster size.
        if (
            num_different_speakers < min_speakers
            or num_different_speakers > max_speakers
        ):
            warnings.warn(
                textwrap.dedent(
                    f"""
                The detected number of speakers ({num_different_speakers}) is outside
                the given bounds [{min_speakers}, {max_speakers}]. This can happen if the
                given audio file is too short to contain {min_speakers} or more speakers.
                Try to lower the desired minimal number of speakers.
                """
                )
            )

        # during counting, we could possibly overcount the number of instantaneous
        # speakers due to segmentation errors, so we cap the maximum instantaneous number
        # of speakers by the `max_speakers` value
        count.data = np.minimum(count.data, max_speakers).astype(np.int8)

        # reconstruct discrete diarization from raw hard clusters

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        hard_clusters[inactive_speakers] = -2
        discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        hook("discrete_diarization", discrete_diarization)

        # convert to continuous diarization
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=self.segmentation.min_duration_off,
        )
        diarization.uri = file["uri"]

        # at this point, `diarization` speaker labels are integers
        # from 0 to `num_speakers - 1`, aligned with `centroids` rows.

        if "annotation" in file and file["annotation"]:
            # when reference is available, use it to map hypothesized speakers
            # to reference speakers (this makes later error analysis easier
            # but does not modify the actual output of the diarization pipeline)
            _, mapping = self.optimal_mapping(
                file["annotation"], diarization, return_mapping=True
            )

            # in case there are more speakers in the hypothesis than in
            # the reference, those extra speakers are missing from `mapping`.
            # we add them back here
            mapping = {key: mapping.get(key, key) for key in diarization.labels()}

        else:
            # when reference is not available, rename hypothesized speakers
            # to human-readable SPEAKER_00, SPEAKER_01, ...
            mapping = {
                label: expected_label
                for label, expected_label in zip(diarization.labels(), self.classes())
            }

        diarization = diarization.rename_labels(mapping=mapping)

        # at this point, `diarization` speaker labels are strings (or mix of
        # strings and integers when reference is available and some hypothesis
        # speakers are not present in the reference)

        if not return_embeddings:
            return diarization

        # this can happen when we use OracleClustering
        if centroids is None:
            return diarization, None

        # The number of centroids may be smaller than the number of speakers
        # in the annotation. This can happen if the number of active speakers
        # obtained from `speaker_count` for some frames is larger than the number
        # of clusters obtained from `clustering`. In this case, we append zero embeddings
        # for extra speakers
        if len(diarization.labels()) > centroids.shape[0]:
            centroids = np.pad(
                centroids, ((0, len(diarization.labels()) - centroids.shape[0]), (0, 0))
            )

        # re-order centroids so that they match
        # the order given by diarization.labels()
        inverse_mapping = {label: index for index, label in mapping.items()}
        centroids = centroids[
            [inverse_mapping[label] for label in diarization.labels()]
        ]

        return diarization, centroids

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(**self.der_variant)

# MIT License
#
# Copyright (c) 2022- CNRS
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

from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.types import Label
from pyannote.metrics.diarization import DiarizationErrorRate

from pyannote.audio.core.inference import Inference
from pyannote.audio.utils.signal import Binarize


# TODO: move to dedicated module
class SpeakerDiarizationMixin:
    """说话人分离管道混入类
    
    定义了说话人分离管道共用的方法集合。
    这些方法被SpeakerDiarization等管道类继承使用。
    
    主要功能：
    - 说话人数验证和设置
    - 最优说话人映射（对齐参考和假设）
    - 说话人计数估计
    """

    @staticmethod
    def set_num_speakers(
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """验证和设置说话人数
        
        处理说话人数的约束条件：
        - 如果指定了num_speakers，它会覆盖min_speakers和max_speakers
        - 验证min_speakers ≤ max_speakers
        - 如果min_speakers == max_speakers，设置num_speakers
        
        参数
        ----------
        num_speakers : int, 可选
            精确的说话人数（如果指定，会覆盖min和max）
        min_speakers : int, 可选
            最小说话人数
        max_speakers : int, 可选
            最大说话人数
        
        返回
        -------
        num_speakers : int 或 None
            精确说话人数（如果min==max）或None
        min_speakers : int
            最小说话人数（至少为1）
        max_speakers : int 或 np.inf
            最大说话人数
        
        异常
        ------
        ValueError
            如果min_speakers > max_speakers
        """

        # override {min|max}_num_speakers by num_speakers when available
        min_speakers = num_speakers or min_speakers or 1
        max_speakers = num_speakers or max_speakers or np.inf

        if min_speakers > max_speakers:
            raise ValueError(
                f"min_speakers must be smaller than (or equal to) max_speakers "
                f"(here: min_speakers={min_speakers:g} and max_speakers={max_speakers:g})."
            )
        if min_speakers == max_speakers:
            num_speakers = min_speakers

        return num_speakers, min_speakers, max_speakers

    @staticmethod
    def optimal_mapping(
        reference: Union[Mapping, Annotation],
        hypothesis: Annotation,
        return_mapping: bool = False,
    ) -> Union[Annotation, Tuple[Annotation, Dict[Label, Label]]]:
        """寻找参考和假设之间的最优双射映射
        
        使用DER（说话人分离错误率）指标找到最优的说话人标签映射。
        这对于评估很重要，因为说话人标签的顺序是任意的。
        
        参数
        ----------
        reference : Annotation 或 Mapping
            参考标注
            可以是Annotation实例，或包含"annotation"键的字典
        hypothesis : Annotation
            假设标注（模型预测结果）
        return_mapping : bool, 默认False
            是否返回映射字典本身
        
        返回
        -------
        mapped : Annotation
            映射到参考说话人的假设标注
        mapping : dict, 可选
            说话人标签映射字典
            key：假设中的说话人标签
            value：参考中对应的说话人标签
            仅在return_mapping=True时返回
        
        用途
        -----
        - 评估时对齐预测和参考的说话人标签
        - 计算准确的DER指标
        - 可视化时显示对齐后的结果
        """

        if isinstance(reference, Mapping):
            reference = reference["annotation"]
            annotated = reference["annotated"] if "annotated" in reference else None
        else:
            annotated = None

        mapping = DiarizationErrorRate().optimal_mapping(
            reference, hypothesis, uem=annotated
        )
        mapped_hypothesis = hypothesis.rename_labels(mapping=mapping)

        if return_mapping:
            return mapped_hypothesis, mapping

        else:
            return mapped_hypothesis

    # TODO: 移除warm-up参数（应该在调用speaker_count之前应用trimming）
    @staticmethod
    def speaker_count(
        binarized_segmentations: SlidingWindowFeature,
        frames: SlidingWindow,
        warm_up: Tuple[float, float] = (0.1, 0.1),
    ) -> SlidingWindowFeature:
        """估计帧级别的瞬时说话人数
        
        从二值化分割结果估计每个时间帧的说话人数量。
        通过统计每个帧中活跃的说话人数来实现。
        
        参数
        ----------
        binarized_segmentations : SlidingWindowFeature
            二值化分割结果，形状为(num_chunks, num_frames, num_classes)
            每个元素表示该帧是否属于该说话人（0或1）
        frames : SlidingWindow
            帧分辨率（时间窗口信息）
            如果已知，提供精确的帧分辨率可以获得更好的时间精度
        warm_up : (float, float), 默认(0.1, 0.1)
            左右预热比例（相对于块持续时间）
            默认：左右各10%
            预热区域会被trim掉，不参与计数
        
        返回
        -------
        SlidingWindowFeature
            瞬时说话人数，形状为(num_frames, 1)
            每个元素表示该帧的说话人数量
        
        处理流程
        --------
        1. Trim掉预热区域
        2. 对每个帧，统计活跃说话人数（sum along classes axis）
        3. 聚合多个块的结果（overlap-add）
        4. 返回帧级说话人数
        
        用途
        -----
        - 估计重叠语音区域
        - 说话人数估计
        - 后处理优化
        """

        trimmed = Inference.trim(binarized_segmentations, warm_up=warm_up)
        count = Inference.aggregate(
            np.sum(trimmed, axis=-1, keepdims=True),
            frames,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)

        return count

    @staticmethod
    def to_annotation(
        discrete_diarization: SlidingWindowFeature,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
    ) -> Annotation:
        """

        Parameters
        ----------
        discrete_diarization : SlidingWindowFeature
            (num_frames, num_speakers)-shaped discrete diarization
        min_duration_on : float, optional
            Defaults to 0.
        min_duration_off : float, optional
            Defaults to 0.

        Returns
        -------
        continuous_diarization : Annotation
            Continuous diarization, with speaker labels as integers,
            corresponding to the speaker indices in the discrete diarization.
        """

        binarize = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

        return binarize(discrete_diarization).rename_tracks(generator="string")

    @staticmethod
    def to_diarization(
        segmentations: SlidingWindowFeature,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        """Build diarization out of preprocessed segmentation and precomputed speaker count

        Parameters
        ----------
        segmentations : SlidingWindowFeature
            (num_chunks, num_frames, num_speakers)-shaped segmentations
        count : SlidingWindow_feature
            (num_frames, 1)-shaped speaker count

        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """

        # TODO: investigate alternative aggregation
        activations = Inference.aggregate(
            segmentations,
            count.sliding_window,
            hamming=False,
            missing=0.0,
            skip_average=True,
        )
        # shape is (num_frames, num_speakers)

        _, num_speakers = activations.data.shape
        max_speakers_per_frame = np.max(count.data)
        if num_speakers < max_speakers_per_frame:
            activations.data = np.pad(
                activations.data, ((0, 0), (0, max_speakers_per_frame - num_speakers))
            )

        extent = activations.extent & count.extent
        activations = activations.crop(extent, return_data=False)
        count = count.crop(extent, return_data=False)

        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations.data)

        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            for i in range(c.item()):
                binary[t, speakers[i]] = 1.0
        return SlidingWindowFeature(binary, activations.sliding_window), activations

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1

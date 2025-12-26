# MIT License
#
# Copyright (c) 2018-2022 CNRS
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

"""Resegmentation pipeline"""

from functools import partial
from typing import Callable, Optional, Text, Union

import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform

from pyannote.audio import Inference, Model
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_model,
)
from pyannote.audio.utils.permutation import mae_cost_func, permutate
from pyannote.audio.utils.signal import binarize


class Resegmentation(SpeakerDiarizationMixin, Pipeline):
    """重新分割管道
    
    使用预训练分割模型改进现有的说话人分离假设。
    通过滑动窗口在整个文件上应用分割模型，对每个窗口位置找到输入分离结果
    与分割模型输出的最优映射，并相应地排列后者的标签。
    排列后的局部分割分数在时间上聚合，并使用滞后阈值进行后处理。
    
    工作原理
    --------
    1. 使用滑动窗口在整个文件上应用分割模型
    2. 对每个窗口位置：
       a. 获取该窗口的输入分离结果（原始假设）
       b. 获取分割模型的输出
       c. 使用排列算法找到最优映射（使两者对齐）
       d. 根据映射排列分割模型的输出
    3. 聚合所有窗口的排列后分割结果
    4. 使用滞后阈值和后处理生成最终结果
    
    应用场景
    --------
    - 改进现有分离结果的质量
    - 优化超参数：将`diarization`设为"annotation"可以找到
      分割模型的最优`onset`、`offset`、`min_duration_on`和`min_duration_off`参数
    
    参数
    ----------
    segmentation : Model, str, 或 dict, 默认"pyannote/segmentation"
        预训练分割模型
        支持格式见 pyannote.audio.pipelines.utils.get_model
    diarization : str, 默认"diarization"
        用作输入分离结果的文件键
        如果设为"annotation"，则使用真实标注（用于超参数优化）
    der_variant : dict, 可选
        DER（说话人分离错误率）变体配置
        默认：{"collar": 0.0, "skip_overlap": False}
        用于评估指标计算
    use_auth_token : str, 可选
        当加载私有HuggingFace模型时，设置认证token
        可以通过运行`huggingface-cli login`获取
    
    超参数
    ----------------
    warm_up : float
        分割模型的预热时间（秒）
        移除窗口开始和结束的预热区域（这些区域不够稳健）
    onset : float
        激活开始检测阈值（0.0-1.0）
    offset : float
        激活结束检测阈值（0.0-1.0）
    min_duration_on : float
        移除短于此时间的说话人段（秒）
    min_duration_off : float
        填充短于此时间的同一说话人间隙（秒）
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        diarization: Text = "diarization",
        der_variant: Optional[dict] = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.diarization = diarization

        model: Model = get_model(segmentation, use_auth_token=use_auth_token)
        self._segmentation = Inference(model)

        self._audio = model.audio

        # number of speakers in output of segmentation model
        self._num_speakers = len(model.specifications.classes)

        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}

        # segmentation warm-up
        self.warm_up = Uniform(0.0, 0.1)

        # hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

        # post-processing i.e. removing short speech turns
        # or filling short gaps between speech turns of one speaker
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    def default_parameters(self):
        # parameters optimized on DIHARD 3 development set
        if self.segmentation == "pyannote/segmentation":
            return {
                "warm_up": 0.05,
                "onset": 0.810,
                "offset": 0.481,
                "min_duration_on": 0.055,
                "min_duration_off": 0.098,
            }
        raise NotImplementedError()

    def classes(self):
        raise NotImplementedError()

    CACHED_SEGMENTATION = "cache/segmentation/inference"

    def apply(
        self,
        file: AudioFile,
        diarization: Optional[Annotation] = None,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """应用重新分割管道
        
        使用分割模型改进输入的说话人分离结果。
        
        参数
        ----------
        file : AudioFile
            待处理的音频文件
        diarization : Annotation, 可选
            输入的分离结果。如果为None，使用file[self.diarization]
        hook : callable, 可选
            回调函数，在每个主要步骤后调用
            调用格式：
                hook(step_name,      # 当前步骤的可读名称
                     step_artefact,  # 当前步骤生成的产物
                     file=file)      # 正在处理的文件
            耗时步骤会多次调用hook，使用相同的step_name
            并提供额外的completed和total参数用于跟踪进度
        
        返回
        -------
        resegmentation : Annotation
            改进后的说话人分离结果
        
        处理流程
        --------
        1. 应用分割模型获取分割结果
        2. 二值化分割结果
        3. 估计瞬时说话人数量
        4. 离散化输入分离结果（转换为帧级表示）
        5. 移除分割结果的预热区域
        6. 对齐输入分离结果和分割结果的说话人数量（通过padding）
        7. 对每个窗口，使用排列算法找到最优映射
        8. 应用排列到分割结果
        9. 构建离散分离结果
        10. 转换为连续标注格式
        11. 如果有参考标注，使用最优映射对齐标签
        """

        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, local_num_speakers)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(
                    file, hook=partial(hook, "segmentation", None)
                )
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(
                file, hook=partial(hook, "segmentation", None)
            )

        hook("segmentation", segmentations)

        # binarize segmentations before speaker counting
        binarized_segmentations: SlidingWindowFeature = binarize(
            segmentations,
            onset=self.onset,
            offset=self.offset,
            initial_state=False,
        )

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation.model.receptive_field,
            warm_up=(self.warm_up, self.warm_up),
        )
        hook("speaker_counting", count)

        # discretize original diarization
        # output shape is (num_frames, num_speakers)
        diarization = diarization or file[self.diarization]
        diarization = diarization.discretize(
            support=Segment(
                0.0, self._audio.get_duration(file) + self._segmentation.step
            ),
            resolution=self._segmentation.model.receptive_field,
        )
        hook("@resegmentation/original", diarization)

        # remove warm-up regions from segmentation as they are less robust
        segmentations = Inference.trim(
            segmentations, warm_up=(self.warm_up, self.warm_up)
        )
        hook("@resegmentation/trim", segmentations)

        # zero-pad diarization or segmentation so they have the same number of speakers
        _, num_speakers = diarization.data.shape
        if num_speakers > self._num_speakers:
            segmentations.data = np.pad(
                segmentations.data,
                ((0, 0), (0, 0), (0, num_speakers - self._num_speakers)),
            )
        elif num_speakers < self._num_speakers:
            diarization.data = np.pad(
                diarization.data, ((0, 0), (0, self._num_speakers - num_speakers))
            )
            num_speakers = self._num_speakers

        # find optimal permutation with respect to the original diarization
        permutated_segmentations = np.full_like(segmentations.data, np.NAN)
        _, num_frames, _ = permutated_segmentations.shape
        for c, (chunk, segmentation) in enumerate(segmentations):
            local_diarization = diarization.crop(chunk)[np.newaxis, :num_frames]
            (permutated_segmentations[c],), _ = permutate(
                local_diarization,
                segmentation,
                cost_func=mae_cost_func,
            )
        permutated_segmentations = SlidingWindowFeature(
            permutated_segmentations, segmentations.sliding_window
        )
        hook("@resegmentation/permutated", permutated_segmentations)

        # build discrete diarization
        discrete_diarization = self.to_diarization(permutated_segmentations, count)

        # convert to continuous diarization
        resegmentation = self.to_annotation(
            discrete_diarization,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

        resegmentation.uri = file["uri"]

        # when reference is available, use it to map hypothesized speakers
        # to reference speakers (this makes later error analysis easier
        # but does not modify the actual output of the resegmentation pipeline)
        if "annotation" in file and file["annotation"]:
            resegmentation = self.optimal_mapping(file["annotation"], resegmentation)

        return resegmentation

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(**self.der_variant)

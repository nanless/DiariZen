# The MIT License (MIT)
#
# Copyright (c) 2018- CNRS
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

"""Overlapped speech detection pipelines"""

from functools import partial
from typing import Callable, Optional, Text, Union

import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature, Timeline
from pyannote.database import get_annotated
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from pyannote.pipeline.parameter import Uniform

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import PipelineModel, get_model
from pyannote.audio.utils.signal import Binarize


def to_overlap(annotation: Annotation) -> Annotation:
    """从说话人标注中提取重叠语音区域
    
    通过查找同时有多个不同说话人活跃的时间段来识别重叠语音。
    
    参数
    ----------
    annotation : Annotation
        说话人标注（包含多个说话人的时间线）
    
    返回
    -------
    Annotation
        重叠语音标注，标签为"overlap"
    
    算法
    -----
    1. 遍历所有说话人段对
    2. 如果两个段来自不同说话人且时间重叠，记录重叠部分
    3. 合并所有重叠区域并转换为Annotation格式
    
    应用场景
    --------
    - 从真实标注生成重叠语音标注（用于训练/评估）
    - 分析音频中的重叠语音比例
    """
    overlap = Timeline(uri=annotation.uri)
    # 遍历所有说话人段对
    for (s1, t1), (s2, t2) in annotation.co_iter(annotation):
        l1 = annotation[s1, t1]  # 段1的说话人标签
        l2 = annotation[s2, t2]  # 段2的说话人标签
        if l1 == l2:
            continue  # 跳过同一说话人的段
        # 添加重叠部分（两个段的交集）
        overlap.add(s1 & s2)
    # 合并重叠区域并转换为Annotation格式
    return overlap.support().to_annotation(generator="string", modality="overlap")


class OracleOverlappedSpeechDetection(Pipeline):
    """Oracle（完美）重叠语音检测管道
    
    这是一个理想化的重叠语音检测管道，直接从真实标注中提取重叠语音区域。
    主要用于：
    - 性能上限评估（upper bound）
    - 对比实验
    - 调试和测试
    
    注意
    -----
    这不是一个实际的检测系统，而是使用真实标注作为"预测"结果。
    """

    def apply(self, file: AudioFile) -> Annotation:
        """返回真实标注的重叠语音检测结果
        
        参数
        ---------
        file : AudioFile
            音频文件，必须包含"annotation"键（真实说话人标注）
        
        返回
        -------
        Annotation
            重叠语音区域标注（从真实标注中提取）
        
        处理流程
        --------
        1. 从文件获取真实说话人标注
        2. 使用to_overlap函数提取重叠语音区域
        3. 返回重叠语音标注
        """
        return to_overlap(file["annotation"])


class OverlappedSpeechDetection(Pipeline):
    """重叠语音检测管道
    
    从音频中检测重叠语音区域（多个说话人同时说话的时间段）。
    使用分割模型预测每个时间帧的重叠概率，然后通过滞后阈值二值化得到最终的重叠/非重叠标签。
    
    参数
    ----------
    segmentation : Model, str, 或 dict, 默认"pyannote/segmentation"
        预训练分割模型（或重叠语音检测模型）
        支持格式见 pyannote.audio.pipelines.utils.get_model
    precision : float, 可选
        目标精确率。如果设置，将在目标精确率下优化召回率
        默认：优化精确率/召回率F-score
    recall : float, 可选
        目标召回率。如果设置，将在目标召回率下优化精确率
        默认：优化精确率/召回率F-score
        注意：precision和recall不能同时设置
    use_auth_token : str, 可选
        当加载私有HuggingFace模型时，设置认证token
        可以通过运行`huggingface-cli login`获取
    inference_kwargs : dict, 可选
        传递给Inference的关键字参数
    
    超参数
    ----------------
    onset : float
        重叠开始检测阈值（0.0-1.0）
        当模型输出超过此阈值时，标记为重叠开始
    offset : float
        重叠结束检测阈值（0.0-1.0）
        当模型输出低于此阈值时，标记为重叠结束
        通常offset < onset（滞后阈值）
    min_duration_on : float
        最小重叠持续时间（秒）
        短于此时间的重叠区域将被移除
    min_duration_off : float
        最小非重叠持续时间（秒）
        短于此时间的非重叠间隙将被填充为重叠
    
    工作流程
    --------
    1. 使用分割模型获取每个时间帧的重叠概率
    2. 如果模型输出维度>1，使用第二高的分数（表示重叠）
    3. 使用滞后阈值（onset/offset）进行二值化
    4. 后处理：移除过短的重叠区域，填充过短的非重叠间隙
    5. 返回最终的重叠语音区域标注
    
    应用场景
    --------
    - 说话人分离预处理（识别重叠区域）
    - 音频质量评估（计算重叠比例）
    - 会议分析（识别多人同时说话）
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs,
    ):
        super().__init__()

        self.segmentation = segmentation

        # load model
        model = get_model(segmentation, use_auth_token=use_auth_token)

        if model.dimension > 1:
            inference_kwargs["pre_aggregation_hook"] = lambda scores: np.partition(
                scores, -2, axis=-1
            )[:, :, -2, np.newaxis]
        self._segmentation = Inference(model, **inference_kwargs)

        if model.specifications.powerset:
            self.onset = self.offset = 0.5

        else:
            #  hyper-parameters used for hysteresis thresholding
            self.onset = Uniform(0.0, 1.0)
            self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short overlapped regions
        # or filling short gaps between overlapped regions
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

        if (precision is not None) and (recall is not None):
            raise ValueError(
                "One must choose between optimizing for target precision or target recall."
            )

        self.precision = precision
        self.recall = recall

    def default_parameters(self):
        if self.segmentation == "pyannote/segmentation":
            # parameters optimized on DIHARD 3 development set
            return {
                "onset": 0.430,
                "offset": 0.320,
                "min_duration_on": 0.091,
                "min_duration_off": 0.144,
            }

        elif self.segmentation == "pyannote/segmentation-3.0.0":
            return {
                "min_duration_on": 0.0,
                "min_duration_off": 0.0,
            }

        raise NotImplementedError()

    def classes(self):
        return ["OVERLAP"]

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    CACHED_SEGMENTATION = "cache/segmentation/inference"

    def apply(self, file: AudioFile, hook: Optional[Callable] = None) -> Annotation:
        """Apply overlapped speech detection

        Parameters
        ----------
        file : AudioFile
            Processed file.
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
        overlapped_speech : Annotation
            Overlapped speech regions.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, 1)
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

        overlapped_speech = self._binarize(segmentations)
        overlapped_speech.uri = file["uri"]
        return overlapped_speech.rename_labels(
            {label: "OVERLAP" for label in overlapped_speech.labels()}
        )

    def get_metric(self, **kwargs) -> DetectionPrecisionRecallFMeasure:
        """Get overlapped speech detection metric

        Returns
        -------
        metric : DetectionPrecisionRecallFMeasure
            Detection metric.
        """

        if (self.precision is not None) or (self.recall is not None):
            raise NotImplementedError(
                "pyannote.pipeline should use `loss` method fallback."
            )

        class _Metric(DetectionPrecisionRecallFMeasure):
            def compute_components(
                _self,
                reference: Annotation,
                hypothesis: Annotation,
                uem: Optional[Timeline] = None,
                **kwargs,
            ) -> dict:
                return super().compute_components(
                    to_overlap(reference), hypothesis, uem=uem, **kwargs
                )

        return _Metric()

    def loss(self, file: AudioFile, hypothesis: Annotation) -> float:
        """Compute recall at target precision (or vice versa)

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hypothesis : Annotation
            Hypothesized overlapped speech regions.

        Returns
        -------
        recall (or purity) : float
            When optimizing for target precision:
                If precision < target_precision, returns (precision - target_precision).
                If precision > target_precision, returns recall.
            When optimizing for target recall:
                If recall < target_recall, returns (recall - target_recall).
                If recall > target_recall, returns precision.
        """

        fmeasure = DetectionPrecisionRecallFMeasure()

        if "overlap_reference" in file:
            overlap_reference = file["overlap_reference"]

        else:
            reference = file["annotation"]
            overlap_reference = to_overlap(reference)
            file["overlap_reference"] = overlap_reference

        _ = fmeasure(overlap_reference, hypothesis, uem=get_annotated(file))
        precision, recall, _ = fmeasure.compute_metrics()

        if self.precision is not None:
            if precision < self.precision:
                return precision - self.precision
            else:
                return recall

        elif self.recall is not None:
            if recall < self.recall:
                return recall - self.recall
            else:
                return precision

    def get_direction(self):
        return "maximize"

# The MIT License (MIT)
#
# Copyright (c) 2022- CNRS
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
# Hadrien TITEUX - https://github.com/hadware
# Hervé BREDIN - http://herve.niderb.fr

from functools import partial
from typing import Callable, Optional, Text, Union

from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.pipeline.parameter import ParamDict, Uniform

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.utils.metric import MacroAverageFMeasure

from ..utils.signal import Binarize
from .utils import PipelineModel, get_model


class MultiLabelSegmentation(Pipeline):
    """通用多标签分割管道
    
    对音频进行多标签分割，可以同时检测多个类别（标签）。
    每个类别独立进行滞后阈值二值化和后处理。
    
    参数
    ----------
    segmentation : Model, str, 或 dict
        预训练多标签分割模型
        支持格式见 pyannote.audio.pipelines.utils.get_model
    fscore : bool, 默认False
        是否优化所有类别的平均F-score（精确率/召回率）
        False：优化识别错误率（Identification Error Rate）
        True：优化宏平均F-score
    share_min_duration : bool, 默认False
        如果为True，`min_duration_on`和`min_duration_off`在所有标签间共享
        如果为False，每个标签有独立的`min_duration_on`和`min_duration_off`参数
    use_auth_token : str, 可选
        当加载私有HuggingFace模型时，设置认证token
        可以通过运行`huggingface-cli login`获取
    inference_kwargs : dict, 可选
        传递给Inference的关键字参数
    
    超参数
    ----------------
    每个标签{label}分配四个超参数：
    {label}.onset, {label}.offset : float
        标签的起始/结束检测阈值（0.0-1.0）
    {label}.min_duration_on : float
        移除短于此时间的{label}区域（秒）
        如果`share_min_duration`为True，所有标签共享此参数
    {label}.min_duration_off : float
        填充短于此时间的非{label}间隙（秒）
        如果`share_min_duration`为True，所有标签共享此参数
    
    工作流程
    --------
    1. 使用分割模型获取每个时间帧的多标签概率分数
    2. 对每个标签独立进行：
       a. 提取该标签的概率分数
       b. 使用滞后阈值（onset/offset）进行二值化
       c. 后处理（移除短区域，填充短间隙）
    3. 合并所有标签的检测结果
    4. 返回最终的多标签标注
    
    应用场景
    --------
    - 多类别事件检测（如笑声、咳嗽、音乐等）
    - 多说话人活动检测
    - 音频场景分析
    - 情感识别（多标签情感）
    """

    def __init__(
        self,
        segmentation: Optional[PipelineModel] = None,
        fscore: bool = False,
        share_min_duration: bool = False,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs,
    ):

        super().__init__()

        if segmentation is None:
            raise ValueError(
                "MultiLabelSegmentation pipeline must be provided with a `segmentation` model."
            )

        self.segmentation = segmentation
        self.fscore = fscore
        self.share_min_duration = share_min_duration

        # load model
        model = get_model(segmentation, use_auth_token=use_auth_token)

        self._classes = model.specifications.classes
        self._segmentation = Inference(model, **inference_kwargs)

        # hyper-parameters used for hysteresis thresholding and postprocessing
        if self.share_min_duration:
            self.min_duration_on = Uniform(0.0, 2.0)
            self.min_duration_off = Uniform(0.0, 2.0)

            self.thresholds = ParamDict(
                **{
                    label: ParamDict(
                        onset=Uniform(0.0, 1.0),
                        offset=Uniform(0.0, 1.0),
                    )
                    for label in self._classes
                }
            )
        else:
            self.thresholds = ParamDict(
                **{
                    label: ParamDict(
                        onset=Uniform(0.0, 1.0),
                        offset=Uniform(0.0, 1.0),
                        min_duration_on=Uniform(0.0, 2.0),
                        min_duration_off=Uniform(0.0, 2.0),
                    )
                    for label in self._classes
                }
            )

    # needed by pyannote.audio Prodigy recipes
    def classes(self):
        return self._classes

    def initialize(self):
        """Initialize pipeline with current set of parameters"""
        self._binarize = {
            label: Binarize(
                onset=self.thresholds[label]["onset"],
                offset=self.thresholds[label]["offset"],
                min_duration_on=(
                    self.thresholds[label]["min_duration_on"]
                    if not self.share_min_duration
                    else self.min_duration_on
                ),  # noqa
                min_duration_off=(
                    self.thresholds[label]["min_duration_off"]
                    if not self.share_min_duration
                    else self.min_duration_off
                ),  # noqa
            )
            for label in self._classes
        }

    CACHED_SEGMENTATION = "cache/segmentation"

    def apply(self, file: AudioFile, hook: Optional[Callable] = None) -> Annotation:
        """Apply multi-label detection

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
        detection : Annotation
            Detected regions.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, num_classes)
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

        # apply hysteresis thresholding on each class separately
        detection = Annotation(uri=file["uri"])

        for i, label in enumerate(self._classes):
            # extract raw segmentation of current label
            label_segmentation = SlidingWindowFeature(
                segmentations.data[:, i : i + 1], segmentations.sliding_window
            )
            # obtain hard segments
            label_annotation: Annotation = self._binarize[label](label_segmentation)

            # add them to the pool of labels
            detection.update(
                label_annotation.rename_labels(
                    dict.fromkeys(label_annotation.labels(), label), copy=False
                )
            )

        return detection

    def get_metric(self) -> Union[MacroAverageFMeasure, IdentificationErrorRate]:
        """Return new instance of identification metric"""

        if self.fscore:
            return MacroAverageFMeasure(classes=self._classes)

        return IdentificationErrorRate()

    def get_direction(self):
        if self.fscore:
            return "maximize"
        return "minimize"

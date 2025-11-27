# MIT License
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

"""Voice activity detection pipelines"""

import tempfile
from copy import deepcopy
from functools import partial
from types import MethodType
from typing import Callable, Optional, Text, Union

import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.metrics.detection import (
    DetectionErrorRate,
    DetectionPrecisionRecallFMeasure,
)
from pyannote.pipeline.parameter import Categorical, Integer, LogUniform, Uniform
from pytorch_lightning import Trainer
from torch.optim import SGD
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio import Inference
from pyannote.audio.core.callback import GraduallyUnfreeze
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import (
    PipelineAugmentation,
    PipelineInference,
    PipelineModel,
    get_augmentation,
    get_inference,
    get_model,
)
from pyannote.audio.tasks import VoiceActivityDetection as VoiceActivityDetectionTask
from pyannote.audio.utils.signal import Binarize


class OracleVoiceActivityDetection(Pipeline):
    """Oracle（完美）语音活动检测管道
    
    这是一个理想化的VAD管道，直接返回标注数据中的语音区域。
    主要用于：
    - 性能上限评估（upper bound）
    - 对比实验
    - 调试和测试
    
    注意
    -----
    这不是一个实际的VAD系统，而是使用真实标注作为"预测"结果。
    """

    @staticmethod
    def apply(file: AudioFile) -> Annotation:
        """返回真实标注的语音活动检测结果
        
        参数
        ---------
        file : AudioFile
            音频文件，必须包含"annotation"键（真实标注）
        
        返回
        -------
        Annotation
            语音区域标注（从真实标注中提取）
        
        处理流程
        --------
        1. 从文件获取真实标注
        2. 提取所有时间线
        3. 获取支持区域（speech区域）
        4. 转换为Annotation格式
        """
        # 从真实标注中提取语音时间线
        speech = file["annotation"].get_timeline().support()
        # 转换为Annotation格式，标签为"speech"
        return speech.to_annotation(generator="string", modality="speech")


class VoiceActivityDetection(Pipeline):
    """语音活动检测（VAD）管道
    
    从音频中检测语音区域（speech regions）。
    使用分割模型预测每个时间帧的语音概率，然后通过滞后阈值二值化得到最终的语音/非语音标签。
    
    参数
    ----------
    segmentation : Model, str, 或 dict, 默认"pyannote/segmentation"
        预训练分割模型（或VAD模型）
        支持格式见 pyannote.audio.pipelines.utils.get_model
    fscore : bool, 默认False
        是否优化F-score（精确率/召回率）
        False：优化检测错误率（Detection Error Rate）
        True：优化F-score（精确率和召回率的调和平均）
    use_auth_token : str, 可选
        当加载私有HuggingFace模型时，设置认证token
        可以通过运行`huggingface-cli login`获取
    inference_kwargs : dict, 可选
        传递给Inference的关键字参数
    
    超参数
    ----------------
    onset : float
        语音开始检测阈值（0.0-1.0）
        当模型输出超过此阈值时，标记为语音开始
    offset : float
        语音结束检测阈值（0.0-1.0）
        当模型输出低于此阈值时，标记为语音结束
        通常offset < onset（滞后阈值）
    min_duration_on : float
        最小语音持续时间（秒）
        短于此时间的语音区域将被移除
    min_duration_off : float
        最小非语音持续时间（秒）
        短于此时间的非语音间隙将被填充为语音
    
    工作流程
    --------
    1. 使用分割模型获取每个时间帧的语音概率
    2. 使用滞后阈值（onset/offset）进行二值化
    3. 后处理：移除过短的语音区域，填充过短的非语音间隙
    4. 返回最终的语音区域标注
    
    应用场景
    --------
    - 语音识别预处理（只处理语音区域）
    - 说话人分离预处理（定位语音区域）
    - 音频质量评估（计算语音比例）
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.fscore = fscore

        # load model and send it to GPU (when available and not already on GPU)
        model = get_model(segmentation, use_auth_token=use_auth_token)

        inference_kwargs["pre_aggregation_hook"] = lambda scores: np.max(
            scores, axis=-1, keepdims=True
        )
        self._segmentation = Inference(model, **inference_kwargs)

        if model.specifications.powerset:
            self.onset = self.offset = 0.5
        else:
            #  hyper-parameters used for hysteresis thresholding
            self.onset = Uniform(0.0, 1.0)
            self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short speech regions
        # or filling short gaps between speech regions
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    def default_parameters(self):
        """返回默认超参数
        
        这些参数是在特定数据集上优化得到的。
        
        返回
        -------
        dict
            默认超参数字典
        
        注意
        -----
        - "pyannote/segmentation": 在DIHARD 3开发集上优化的参数
        - "pyannote/segmentation-3.0.0": 新版本模型，不需要后处理
        """
        if self.segmentation == "pyannote/segmentation":
            # 在DIHARD 3开发集上优化的参数
            return {
                "onset": 0.767,  # 语音开始阈值
                "offset": 0.377,  # 语音结束阈值
                "min_duration_on": 0.136,  # 最小语音持续时间（秒）
                "min_duration_off": 0.067,  # 最小非语音持续时间（秒）
            }

        elif self.segmentation == "pyannote/segmentation-3.0.0":
            # 新版本模型，不需要后处理
            return {
                "min_duration_on": 0.0,
                "min_duration_off": 0.0,
            }

        raise NotImplementedError()

    def classes(self):
        return ["SPEECH"]

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
        """应用语音活动检测
        
        对音频文件进行VAD处理，返回语音区域标注。
        
        参数
        ----------
        file : AudioFile
            待处理的音频文件
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
        Annotation
            语音区域标注，标签为"SPEECH"
        
        处理步骤
        --------
        1. 应用分割模型获取语音概率分数
        2. 使用滞后阈值二值化
        3. 后处理（移除短语音区域，填充短间隙）
        4. 返回最终标注
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

        speech: Annotation = self._binarize(segmentations)
        speech.uri = file["uri"]
        return speech.rename_labels({label: "SPEECH" for label in speech.labels()})

    def get_metric(self) -> Union[DetectionErrorRate, DetectionPrecisionRecallFMeasure]:
        """Return new instance of detection metric"""

        if self.fscore:
            return DetectionPrecisionRecallFMeasure(collar=0.0, skip_overlap=False)

        return DetectionErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        if self.fscore:
            return "maximize"
        return "minimize"


class AdaptiveVoiceActivityDetection(Pipeline):
    """自适应语音活动检测管道
    
    这是一个自适应的VAD管道，会对每个文件进行模型微调以提高性能。
    
    工作原理
    --------
    设M为预训练的VAD模型。
    
    对于每个文件f：
    1. 首先应用预训练模型M，获得初始的语音/非语音标签
    2. 使用这些（自动生成的，可能包含错误的）标签，以自监督方式在文件f上微调M，得到M_f
    3. 最后，应用微调后的模型M_f到文件f，获得最终（希望更好的）语音/非语音标签
    
    微调策略
    --------
    - 置信度加权：预训练模型M高置信度的帧被赋予更高权重
      直觉：模型将使用这些高置信度区域适应录音条件（如背景噪声），
      从而在最初不太自信的部分表现更好
    - 防止过拟合：
      - 使用数据增强
      - 冻结除最后几层外的所有层（渐进式解冻）
    
    参数
    ----------
    segmentation : Model, str, 或 dict, 默认"hbredin/VoiceActivityDetection-PyanNet-DIHARD"
        预训练分割模型
    augmentation : BaseWaveformTransform, 或 dict, 可选
        torch_audiomentations波形变换，用于微调时的数据增强
        默认为None（不使用增强）
    fscore : bool, 默认False
        是否优化F-score
        False：优化检测错误率
        True：优化F-score
    
    超参数
    ----------------
    num_epochs : int
        微调轮数（一轮 = 遍历文件一次）
    batch_size : int
        批次大小
    learning_rate : float
        学习率
    
    适用场景
    --------
    - 单个文件的精确VAD（可以花时间微调）
    - 录音条件特殊的文件（需要适应）
    - 对准确率要求极高的场景
    
    注意
    -----
    - 每个文件都需要微调，计算成本较高
    - 不适合批量处理大量文件
    - 需要GPU支持（微调过程）
    
    参考
    --------
    pyannote.audio.pipelines.utils.get_inference
    """

    def __init__(
        self,
        segmentation: PipelineInference = "hbredin/VoiceActivityDetection-PyanNet-DIHARD",
        augmentation: Optional[PipelineAugmentation] = None,
        fscore: bool = False,
    ):
        super().__init__()

        # pretrained segmentation model
        self.inference: Inference = get_inference(segmentation)
        self.augmentation: BaseWaveformTransform = get_augmentation(augmentation)

        self.fscore = fscore

        self.num_epochs = Integer(0, 10)
        self.batch_size = Categorical([1, 2, 4, 8, 16, 32])
        self.learning_rate = LogUniform(1e-6, 1)

    def apply(self, file: AudioFile) -> Annotation:
        # create a copy of file
        file = dict(file)

        # get segmentation scores from pretrained segmentation model
        file["seg"] = self.inference(file)

        # infer voice activity detection scores
        file["vad"] = np.max(file["seg"], axis=1, keepdims=True)

        # apply voice activity detection pipeline with default parameters
        vad_pipeline = VoiceActivityDetection("vad").instantiate(
            {
                "onset": 0.5,
                "offset": 0.5,
                "min_duration_on": 0.0,
                "min_duration_off": 0.0,
            }
        )
        file["annotation"] = vad_pipeline(file)

        # do not fine tune the model if num_epochs is zero
        if self.num_epochs == 0:
            return file["annotation"]

        # infer model confidence from segmentation scores
        # TODO: scale confidence differently (e.g. via an additional binarisation threshold hyper-parameter)
        file["confidence"] = np.min(
            np.abs((file["seg"] - 0.5) / 0.5), axis=1, keepdims=True
        )

        # create a dummy train-only protocol where `file` is the only training file
        class DummyProtocol(SpeakerDiarizationProtocol):
            name = "DummyProtocol"

            def train_iter(self):
                yield file

        vad_task = VoiceActivityDetectionTask(
            DummyProtocol(),
            duration=self.inference.duration,
            weight="confidence",
            batch_size=self.batch_size,
            augmentation=self.augmentation,
        )

        vad_model = deepcopy(self.inference.model)
        vad_model.task = vad_task

        def configure_optimizers(model):
            return SGD(model.parameters(), lr=self.learning_rate)

        vad_model.configure_optimizers = MethodType(configure_optimizers, vad_model)

        with tempfile.TemporaryDirectory() as default_root_dir:
            trainer = Trainer(
                max_epochs=self.num_epochs,
                accelerator="gpu",
                devices=1,
                callbacks=[GraduallyUnfreeze(epochs_per_stage=self.num_epochs + 1)],
                enable_checkpointing=False,
                default_root_dir=default_root_dir,
            )
            trainer.fit(vad_model)

        inference = Inference(
            vad_model,
            device=self.inference.device,
            batch_size=self.inference.batch_size,
        )
        file["vad"] = inference(file)

        return vad_pipeline(file)

    def get_metric(self) -> Union[DetectionErrorRate, DetectionPrecisionRecallFMeasure]:
        """Return new instance of detection metric"""

        if self.fscore:
            return DetectionPrecisionRecallFMeasure(collar=0.0, skip_overlap=False)

        return DetectionErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        if self.fscore:
            return "maximize"
        return "minimize"

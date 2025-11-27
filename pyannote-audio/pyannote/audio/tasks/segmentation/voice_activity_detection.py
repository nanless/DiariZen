# MIT License
#
# Copyright (c) 2020- CNRS
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

from typing import Dict, Optional, Sequence, Text, Tuple, Union

import numpy as np
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.tasks.segmentation.mixins import SegmentationTask


class VoiceActivityDetection(SegmentationTask):
    """语音活动检测（VAD）任务
    
    语音活动检测是从音频录音中检测语音区域的任务。
    
    任务定义
    --------
    这是一个二分类（0或1）序列标注任务。
    当至少有一个说话人活跃时，帧被标记为"speech"（1）。
    
    特点
    -----
    - 二分类问题：speech（1）或non-speech（0）
    - 帧级分辨率：每个时间帧都有标签
    - 多说话人场景：只要有一个说话人活跃就标记为speech
    
    参数
    ----------
    protocol : Protocol
        pyannote.database协议
    cache : str, 可选
        缓存路径。由于大型数据集的数据准备可能耗时很长，
        可以缓存到磁盘以便后续（更快地）重用。
        当`cache`不存在时，`Task.prepare_data()`从`protocol`生成训练和验证元数据并保存到磁盘。
        当`cache`存在时，跳过`Task.prepare_data()`，从磁盘加载（元）数据。
        默认为临时路径。
    duration : float, 默认2.0
        训练块（chunk）的持续时间（秒）
    warm_up : float 或 (float, float), 默认0.0
        在每个块的左右两端使用这么多秒来预热模型。
        虽然模型会处理这些左右部分，但只有剩余的中心部分用于：
        - 训练时计算损失
        - 推理时聚合分数
        默认0.0（即不预热）
    balance : Sequence[Text], 可选
        当提供时，训练样本相对于这些键均匀采样。
        例如，设置`balance`为["database","subset"]将确保每个数据库和子集组合
        在训练样本中均匀表示。
    weight : str, 可选
        当提供时，使用此键作为损失函数中的帧级权重。
        例如，可以使用模型置信度作为权重。
    batch_size : int, 默认32
        每个批次的训练样本数
    num_workers : int, 可选
        用于生成训练样本的工作进程数
        默认为 multiprocessing.cpu_count() // 2
    pin_memory : bool, 默认False
        如果为True，数据加载器将在返回张量之前将它们复制到CUDA固定内存。
        详见PyTorch文档
    augmentation : BaseWaveformTransform, 可选
        torch_audiomentations波形变换，训练时由数据加载器使用
    metric : Metric, Sequence[Metric], 或 Dict[str, Metric], 可选
        验证指标。可以是torchmetrics.MetricCollection支持的任何指标。
        默认为AUROC（ROC曲线下面积）
    
    任务规范
    --------
    - problem: BINARY_CLASSIFICATION（二分类）
    - resolution: FRAME（帧级）
    - classes: ["speech"]（单一类别：语音）
    """

    def __init__(
        self,
        protocol: Protocol,
        cache: Optional[Union[str, None]] = None,
        duration: float = 2.0,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        balance: Optional[Sequence[Text]] = None,
        weight: Optional[Text] = None,
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        augmentation: Optional[BaseWaveformTransform] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
    ):
        super().__init__(
            protocol,
            duration=duration,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric,
            cache=cache,
        )

        self.balance = balance
        self.weight = weight

        self.specifications = Specifications(
            problem=Problem.BINARY_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            min_duration=self.min_duration,
            warm_up=self.warm_up,
            classes=[
                "speech",
            ],
        )

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        """准备语音活动检测的训练块
        
        从文件中提取指定时间段的音频块，并生成对应的VAD标签。
        
        参数
        ----------
        file_id : int
            文件索引
        start_time : float
            块的开始时间（秒）
        duration : float
            块的持续时间（秒）
        
        返回
        -------
        sample : dict
            包含块数据的字典，包含以下键：
            - `X`: 波形数据（torch.Tensor）
            - `y`: 目标标签（SlidingWindowFeature实例）
              形状为(num_frames, 1)，值为0（非语音）或1（语音）
            - `meta`: 元数据字典
                - `database`: 数据库索引
                - `file`: 文件索引
                - 其他元数据字段
        
        处理流程
        --------
        1. 从文件中提取指定时间段的音频波形
        2. 获取该时间段内的所有标注
        3. 将标注离散化为帧级标签
        4. 构建SlidingWindowFeature目标
        5. 返回样本字典
        """

        file = self.get_file(file_id)

        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        sample["X"], _ = self.model.audio.crop(file, chunk, duration=duration)

        # gather all annotations of current file
        annotations = self.prepared_data["annotations-segments"][
            self.prepared_data["annotations-segments"]["file_id"] == file_id
        ]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output resolution
        step = self.model.receptive_field.step
        half = 0.5 * self.model.receptive_field.duration

        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start - half
        end_idx = np.round(end / step).astype(int)

        # frame-level targets
        num_frames = self.model.num_frames(
            round(duration * self.model.hparams.sample_rate)
        )
        y = np.zeros((num_frames, 1), dtype=np.uint8)
        for start, end in zip(start_idx, end_idx):
            y[start : end + 1, 0] = 1

        sample["y"] = SlidingWindowFeature(
            y, self.model.receptive_field, labels=["speech"]
        )

        metadata = self.prepared_data["audio-metadata"][file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample

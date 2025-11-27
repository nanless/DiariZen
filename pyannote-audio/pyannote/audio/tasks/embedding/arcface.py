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


from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import pytorch_metric_learning.losses
from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Task

from .mixins import SupervisedRepresentationLearningTaskMixin


class SupervisedRepresentationLearningWithArcFace(
    SupervisedRepresentationLearningTaskMixin,
    Task,
):
    """使用ArcFace损失的有监督表示学习任务
    
    使用ArcFace损失函数训练说话人嵌入模型。
    ArcFace是一种角度边距损失函数，通过在角度空间中添加边距来增强类间分离。
    
    任务定义
    --------
    表示学习是从音频中学习有意义的嵌入向量的任务。
    这些嵌入向量可以用于说话人识别、验证和聚类等任务。
    
    参数
    ----------
    protocol : Protocol
        pyannote.database协议
    duration : float, 默认2.0
        训练块（chunk）的持续时间（秒）
    min_duration : float, 可选
        训练块的最小持续时间（秒）
        如果设置，训练块持续时间将在`min_duration`和`duration`之间均匀采样
        默认为`duration`（即固定长度块）
    num_classes_per_batch : int, 默认32
        每个批次中的类别数
        用于确保每个批次包含多个说话人的样本
    num_chunks_per_class : int, 默认1
        每个类别的块数
        用于控制每个说话人在批次中的样本数
    margin : float, 默认28.6
        ArcFace损失的角度边距（度）
        较大的边距会增加类间分离，但可能导致训练不稳定
    scale : float, 默认64.0
        ArcFace损失的缩放因子
        用于控制损失函数的梯度大小
    num_workers : int, 可选
        用于生成训练样本的工作进程数
        默认为 multiprocessing.cpu_count() // 2
    pin_memory : bool, 默认False
        如果为True，数据加载器将在返回张量之前将它们复制到CUDA固定内存
        详见PyTorch文档
    augmentation : BaseWaveformTransform, 可选
        torch_audiomentations波形变换，训练时由数据加载器使用
    metric : Metric, Sequence[Metric], 或 Dict[str, Metric], 可选
        验证指标。可以是torchmetrics.MetricCollection支持的任何指标
        默认为AUROC（ROC曲线下面积）
    
    ArcFace损失
    -----------
    ArcFace通过在角度空间中添加边距来增强类间分离：
    - 将嵌入向量归一化到单位球面
    - 计算嵌入向量与类别权重之间的角度
    - 在角度上添加边距（margin）
    - 使用缩放因子（scale）控制梯度
    
    优势
    -----
    - 更强的类间分离能力
    - 更好的泛化性能
    - 适用于大规模说话人识别任务
    
    参考
    -----
    Deng, J., et al. (2019).
    "ArcFace: Additive Angular Margin Loss for Deep Face Recognition."
    CVPR 2019.
    """

    #  TODO: add a ".metric" property that tells how speaker embedding trained with this approach
    #  should be compared. could be a string like "cosine" or "euclidean" or a pdist/cdist-like
    #  callable. this ".metric" property should be propagated all the way to Inference (via the model).

    def __init__(
        self,
        protocol: Protocol,
        min_duration: Optional[float] = None,
        duration: float = 2.0,
        num_classes_per_batch: int = 32,
        num_chunks_per_class: int = 1,
        margin: float = 28.6,
        scale: float = 64.0,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        augmentation: Optional[BaseWaveformTransform] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
    ):

        self.num_chunks_per_class = num_chunks_per_class
        self.num_classes_per_batch = num_classes_per_batch

        self.margin = margin
        self.scale = scale

        super().__init__(
            protocol,
            duration=duration,
            min_duration=min_duration,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric,
        )

    def setup_loss_func(self):

        _, embedding_size = self.model(self.model.example_input_array).shape

        self.model.loss_func = pytorch_metric_learning.losses.ArcFaceLoss(
            len(self.specifications.classes),
            embedding_size,
            margin=self.margin,
            scale=self.scale,
        )

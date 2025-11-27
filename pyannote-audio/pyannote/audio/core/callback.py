# MIT License
#
# Copyright (c) 2020-2021 CNRS
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

from typing import List, Mapping, Optional, Text, Union

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

from pyannote.audio import Model


class GraduallyUnfreeze(Callback):
    """渐进式解冻回调：逐步解冻模型层
    
    这是一种迁移学习策略，用于微调预训练模型：
    1. 开始时冻结所有层，只训练任务相关层（在model.build()和task.setup_loss_func()中实例化的层）
    2. 训练几个epoch后，解冻更多层
    3. 重复此过程，逐步解冻所有层
    
    这种方法可以：
    - 稳定训练过程
    - 避免破坏预训练特征
    - 逐步适应新任务
    
    参数
    ----------
    schedule : dict 或 list, 可选
        解冻计划，支持多种格式（见示例）
        如果为None，自动从模型架构推断
    epochs_per_stage : int, 可选
        每个阶段之间的epoch数，默认1
        如果schedule是字典格式，此参数无效
    
    使用示例
    -----
    >>> # 创建回调
    >>> callback = GraduallyUnfreeze()
    >>> # 在训练器中使用
    >>> Trainer(callbacks=[callback]).fit(model)
    
    示例
    --------
    # 对于PyanNet架构（sincnet > lstm > linear > task_specific），
    # 以下三种方式等价，会在epoch 1解冻'linear'，epoch 2解冻'lstm'，epoch 3解冻'sincnet'
    >>> GraduallyUnfreeze()  # 自动推断顺序
    >>> GraduallyUnfreeze(schedule=['linear', 'lstm', 'sincnet'])  # 列表格式
    >>> GraduallyUnfreeze(schedule={'linear': 1, 'lstm': 2, 'sincnet': 3})  # 字典格式
    
    # 也可以指定多个层在同一epoch解冻：
    >>> GraduallyUnfreeze(schedule=['linear', ['lstm', 'sincnet']], epochs_per_stage=10)
    >>> # 等价于：
    >>> GraduallyUnfreeze(schedule={'linear': 10, 'lstm': 20, 'sincnet': 20})
    >>> # 会在epoch 10解冻'linear'，epoch 20同时解冻'lstm'和'sincnet'
    """

    def __init__(
        self,
        schedule: Union[Mapping[Text, int], List[Union[List[Text], Text]]] = None,
        epochs_per_stage: Optional[int] = None,
    ):
        """初始化渐进式解冻回调
        
        参数
        ----------
        schedule : dict 或 list, 可选
            解冻计划
            - dict: {layer_name: epoch}，指定每个层在哪个epoch解冻
            - list: [layer1, layer2, ...] 或 [layer1, [layer2, layer3], ...]，按顺序解冻
        epochs_per_stage : int, 可选
            每个阶段之间的epoch数
            如果schedule是list且未指定，默认为1
        """
        super().__init__()

        # 如果schedule是None或List且未指定epochs_per_stage，默认为1
        if (
            (schedule is None) or (isinstance(schedule, List))
        ) and epochs_per_stage is None:
            epochs_per_stage = 1

        self.epochs_per_stage = epochs_per_stage  # 每个阶段的epoch数
        self.schedule = schedule  # 解冻计划

    def on_fit_start(self, trainer: Trainer, model: Model):
        """训练开始时调用：设置初始冻结状态
        
        在训练开始前：
        1. 确定任务相关层和骨干层
        2. 解析解冻计划
        3. 冻结所有骨干层（保留任务相关层可训练）
        
        参数
        ----------
        trainer : Trainer
            PyTorch Lightning训练器
        model : Model
            要训练的模型
        """
        schedule = self.schedule

        # 获取任务相关层（这些层始终可训练）
        task_specific_layers = model.task_dependent
        # 获取所有骨干层（从后往前，从输出层到输入层）
        backbone_layers = [
            layer
            for layer, _ in reversed(ModelSummary(model, max_depth=1).named_modules)
            if layer not in task_specific_layers
        ]

        # 如果未指定schedule，使用所有骨干层（按顺序）
        if schedule is None:
            schedule = backbone_layers

        # 如果schedule是列表，转换为字典格式
        if isinstance(schedule, List):
            _schedule = dict()
            for depth, layers in enumerate(schedule):
                # 支持单个层或层列表
                layers = layers if isinstance(layers, List) else [layers]
                for layer in layers:
                    # 计算解冻epoch：(深度+1) × 每阶段epoch数
                    _schedule[layer] = (depth + 1) * self.epochs_per_stage
            schedule = _schedule

        self.schedule = schedule

        # 冻结所有骨干层（任务相关层保持可训练）
        for layer in backbone_layers:
            model.freeze_by_name(layer)

    def on_train_epoch_start(self, trainer: Trainer, model: Model):
        """每个训练epoch开始时调用：检查是否需要解冻层
        
        根据解冻计划，在当前epoch解冻相应的层。
        
        参数
        ----------
        trainer : Trainer
            PyTorch Lightning训练器
        model : Model
            要训练的模型
        """
        # 遍历解冻计划，检查是否有层需要在当前epoch解冻
        for layer, epoch in self.schedule.items():
            if epoch == trainer.current_epoch:
                # 解冻该层
                model.unfreeze_by_name(layer)

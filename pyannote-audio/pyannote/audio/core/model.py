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


import os 
from os.path import join

import copy

import warnings

import torch
import torch.nn as nn

from pathlib import Path

from enum import Enum
from dataclasses import dataclass

from importlib import import_module

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from lightning_fabric.utilities.cloud_io import _load as pl_load

from functools import cached_property
from typing import Dict, List, Optional, Text, Tuple, Union, Sequence
from urllib.parse import urlparse

from diarizen.utils import instantiate

from pyannote.core import SlidingWindow

from pyannote.audio import __version__
from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import (
    Problem, 
    Specifications, 
    Task
)

from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.utils.multi_task import map_with_specifications

from torchmetrics import Metric, MetricCollection
from pyannote.audio.torchmetrics import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    OptimalDiarizationErrorRate,
    OptimalDiarizationErrorRateThreshold,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
    SpeakerConfusionRate,
)


# 缓存目录：用于存储下载的预训练模型
# 可以通过环境变量PYANNOTE_CACHE自定义，默认为~/.cache/torch/pyannote
CACHE_DIR = os.getenv(
    "PYANNOTE_CACHE",
    os.path.expanduser("~/.cache/torch/pyannote"),
)
# HuggingFace模型权重文件名
HF_PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
# HuggingFace Lightning配置文件文件名
HF_LIGHTNING_CONFIG_NAME = "config.yaml"


# 注意：为了向后兼容，用于加载pyannote.audio 3.x之前训练的模型
class Introspection:
    """向后兼容类，用于加载旧版本模型"""
    pass


@dataclass
class Output:
    """模型输出数据结构
    
    用于描述模型的输出格式，包含：
    - num_frames: 输出帧数
    - dimension: 输出维度
    - frames: 时间窗口信息
    """
    num_frames: int  # 输出帧数
    dimension: int  # 输出维度（特征维度）
    frames: SlidingWindow  # 时间窗口信息（起始时间、持续时间、步长）

class Resolution(Enum):
    """模型输出分辨率类型
    
    定义模型输出的时间分辨率：
    - FRAME: 逐帧输出（每个时间帧都有一个输出）
    - CHUNK: 整块输出（整个音频块只有一个输出向量）
    """
    FRAME = 1  # 模型输出一系列帧（frame-level输出）
    CHUNK = 2  # 模型输出整个块的单个向量（chunk-level输出）


def average_checkpoints(
    model: nn.Module,
    checkpoint_list: str,
) -> nn.Module:
    """平均多个检查点的模型权重
    
    这个方法用于模型集成（ensemble），通过平均多个检查点的权重来获得更稳定的模型。
    常用于模型训练的最后阶段，平均多个epoch的模型。
    
    参数
    ----------
    model : nn.Module
        模型架构（用于加载权重）
    checkpoint_list : list
        检查点列表，每个元素包含'bin_path'键，指向权重文件路径
    
    返回
    -------
    nn.Module
        权重平均后的模型
    """
    states_dict_list = []
    # 遍历所有检查点，加载权重
    for ckpt_data in checkpoint_list:
        ckpt_path = ckpt_data['bin_path']
        copy_model = copy.deepcopy(model)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        copy_model.load_state_dict(checkpoint)
        states_dict_list.append(copy_model.state_dict())
    # 计算平均权重
    avg_state_dict = average_states(states_dict_list, torch.device('cpu'))
    # 创建新模型并加载平均权重
    avg_model = copy.deepcopy(model)
    avg_model.load_state_dict(avg_state_dict)
    return avg_model

def average_states(
    states_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    """平均多个状态字典的权重
    
    将多个模型的状态字典按元素平均，用于模型集成。
    
    参数
    ----------
    states_list : List[Dict[str, torch.Tensor]]
        状态字典列表，每个字典包含模型的所有参数
    device : torch.device
        计算设备（CPU或GPU）
    
    返回
    -------
    Dict[str, torch.Tensor]
        平均后的状态字典
    """
    qty = len(states_list)
    avg_state = states_list[0]
    # 累加所有权重
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] += states_list[i][key].to(device)
    # 除以数量得到平均值
    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state


class Model(nn.Module):
    """模型基类：所有音频模型的抽象基类
    
    这是pyannote.audio中所有模型的基类，提供了统一的接口和功能：
    - 统一的模型接口
    - 任务规格定义（Specifications）
    - 感受野计算
    - 参数冻结/解冻
    - 预训练模型加载
    
    子类需要实现forward()方法来完成具体的前向传播逻辑。
    
    参考: https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/core/model.py
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        max_speakers_per_chunk: int = 4,
        max_speakers_per_frame: int = 2,
        duration: int = 5,
        min_duration: int = 5,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        mono: str = "downmix"
    ):
        """
        初始化模型
        
        参数
        ----------
        sample_rate : int, 默认16000
            音频采样率（Hz）
        num_channels : int, 默认1
            音频通道数（1=单声道，2=立体声等）
        task : Task, 可选
            关联的任务对象（用于训练时）
        max_speakers_per_chunk : int, 默认4
            每个音频块的最大说话人数（用于定义类别数）
        max_speakers_per_frame : int, 默认2
            每帧的最大同时说话人数（用于幂集编码）
        duration : int, 默认5
            音频块持续时间（秒）
        min_duration : int, 默认5
            最小音频块持续时间（秒）
        warm_up : float或(float, float), 默认0.0
            预热时间（秒），用于模型稳定
            - 单个值：左右对称预热
            - 元组：(左预热时间, 右预热时间)
        mono : str, 默认"downmix"
            多声道转单声道策略："downmix"（下混）或"random"（随机选择）
        """
        super().__init__()
        
        # 如果多声道，不使用mono转换
        if num_channels > 1:
            mono = None
            
        self.num_channels = num_channels  # 音频通道数
        self.sample_rate = sample_rate  # 采样率
        # 初始化音频I/O处理器
        self.audio = Audio(sample_rate=sample_rate, mono=mono)

        # 定义任务规格（Specifications）
        # 根据max_speakers_per_frame决定是多标签还是单标签分类
        self.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION
            if max_speakers_per_frame is None
            else Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,  # 帧级输出
            duration=duration,  # 块持续时间
            min_duration=min_duration,  # 最小持续时间
            warm_up=warm_up,  # 预热时间
            classes=[f"speaker#{i+1}" for i in range(max_speakers_per_chunk)],  # 说话人类别
            powerset_max_classes=max_speakers_per_frame,  # 幂集最大类别数
            permutation_invariant=True,  # 排列不变性（说话人顺序无关）
        )

        # 如果使用幂集编码，初始化幂集转换器
        if self.specifications.powerset:
            self.powerset = Powerset(
                len(self.specifications.classes),  # 类别总数
                self.specifications.powerset_max_classes,  # 最大同时类别数
            )
        
        # 初始化验证指标集合
        self.validation_metric = MetricCollection(self.default_metric())
    
    @cached_property
    def _receptive_field(self) -> SlidingWindow:     
        """计算模型的感受野（Receptive Field）
        
        感受野定义了模型输出中每个时间点对应的输入时间范围。
        这是一个缓存属性，首次访问时计算，之后直接返回缓存结果。
        
        返回
        -------
        SlidingWindow
            感受野的时间窗口信息，包含：
            - start: 起始时间（秒）
            - duration: 持续时间（秒）
            - step: 步长（秒）
        """

        # 计算单帧输出的感受野大小（样本数）
        receptive_field_size = self.receptive_field_size(num_frames=1)
        # 计算感受野的步长（通过计算两帧输出的差值）
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        # 计算感受野的起始位置（相对于第0帧的中心）
        receptive_field_start = (
            self.receptive_field_center(frame=0) - (receptive_field_size - 1) / 2
        )
        # 转换为时间单位（秒）并返回SlidingWindow对象
        return SlidingWindow(
            start=receptive_field_start / self.sample_rate,  # 起始时间（秒）
            duration=receptive_field_size / self.sample_rate,  # 持续时间（秒）
            step=receptive_field_step / self.sample_rate,  # 步长（秒）
        )

    def forward(
        self, waveforms: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """前向传播方法（子类必须实现）
        
        这是模型的核心方法，定义了从输入到输出的计算流程。
        所有继承Model的子类都必须实现这个方法。
        
        参数
        ----------
        waveforms : torch.Tensor
            输入音频波形
            形状: (batch_size, num_channels, num_samples)
        **kwargs
            其他可选参数
        
        返回
        -------
        torch.Tensor 或 Tuple[torch.Tensor]
            模型输出
            - 单任务：返回单个张量
            - 多任务：返回张量元组
        
        异常
        ------
        NotImplementedError
            如果子类没有实现此方法
        """
        msg = "Class {self.__class__.__name__} should define a `forward` method."
        raise NotImplementedError(msg)

    # 便利函数：根据任务规格自动选择激活函数
    def default_activation(self) -> Union[nn.Module, Tuple[nn.Module]]:
        """根据任务规格自动选择默认激活函数
        
        根据任务类型（二分类、多分类、多标签分类）自动选择合适的激活函数：
        - 二分类：Sigmoid（输出0-1之间的概率）
        - 多分类：LogSoftmax（输出对数概率，用于NLL损失）
        - 多标签分类：Sigmoid（每个类别独立输出概率）
        
        返回
        -------
        nn.Module 或 Tuple[nn.Module]
            激活函数模块
            - 单任务：返回单个激活函数
            - 多任务：返回激活函数元组
        """

        def __default_activation(
            specifications: Optional[Specifications] = None,
        ) -> nn.Module:
            """内部函数：为单个任务规格选择激活函数"""
            if specifications.problem == Problem.BINARY_CLASSIFICATION:
                # 二分类：使用Sigmoid输出0-1之间的概率
                return nn.Sigmoid()

            elif specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
                # 多分类：使用LogSoftmax输出对数概率（配合NLL损失）
                return nn.LogSoftmax(dim=-1)

            elif specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
                # 多标签分类：使用Sigmoid，每个类别独立输出概率
                return nn.Sigmoid()

            else:
                msg = "TODO: implement default activation for other types of problems"
                raise NotImplementedError(msg)

        # 使用map_with_specifications处理单任务和多任务情况
        return map_with_specifications(self.specifications, __default_activation)

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """返回默认的评估指标集合
        
        根据模型是否使用幂集编码，返回不同的指标集合：
        - 使用幂集：固定阈值0.5的DER及其组件
        - 不使用幂集：最优阈值DER及其组件（自动寻找最优阈值）
        
        返回
        -------
        Dict[str, Metric]
            指标字典，包含：
            - DiarizationErrorRate: 说话人分离错误率（主要指标）
            - DiarizationErrorRate/Confusion: 说话人混淆率
            - DiarizationErrorRate/Miss: 漏检率
            - DiarizationErrorRate/FalseAlarm: 误检率
        """

        if self.specifications.powerset:
            # 使用幂集编码：使用固定阈值0.5
            return {
                "DiarizationErrorRate": DiarizationErrorRate(0.5),  # DER（阈值0.5）
                "DiarizationErrorRate/Confusion": SpeakerConfusionRate(0.5),  # 混淆率
                "DiarizationErrorRate/Miss": MissedDetectionRate(0.5),  # 漏检率
                "DiarizationErrorRate/FalseAlarm": FalseAlarmRate(0.5),  # 误检率
            }

        # 不使用幂集编码：使用最优阈值（自动寻找）
        return {
            "DiarizationErrorRate": OptimalDiarizationErrorRate(),  # 最优DER
            "DiarizationErrorRate/Threshold": OptimalDiarizationErrorRateThreshold(),  # 最优阈值
            "DiarizationErrorRate/Confusion": OptimalSpeakerConfusionRate(),  # 最优混淆率
            "DiarizationErrorRate/Miss": OptimalMissedDetectionRate(),  # 最优漏检率
            "DiarizationErrorRate/FalseAlarm": OptimalFalseAlarmRate(),  # 最优误检率
        }

    def __by_name(
        self,
        modules: Union[List[Text], Text],
        recurse: bool = True,
        requires_grad: bool = False,
    ) -> List[Text]:
        """辅助函数：按名称冻结或解冻模块
        
        这是freeze_by_name和unfreeze_by_name的内部实现函数。
        通过设置requires_grad标志来控制参数的梯度计算。
        
        参数
        ----------
        modules : str 或 List[str]
            要操作的模块名称（单个或列表）
        recurse : bool, 默认True
            是否递归处理子模块
        requires_grad : bool, 默认False
            True=解冻（允许梯度），False=冻结（禁止梯度）
        
        返回
        -------
        List[str]
            成功处理的模块名称列表
        
        异常
        ------
        ValueError
            如果指定的模块不存在
        """

        updated_modules = list()

        # 如果输入是单个字符串，转换为列表
        if isinstance(modules, str):
            modules = [modules]

        # 遍历每个模块名称
        for name in modules:
            # 获取模块对象
            module = getattr(self, name)

            # 设置所有参数的requires_grad标志
            for parameter in module.parameters(recurse=True):
                parameter.requires_grad = requires_grad
            # 设置模块的训练模式（requires_grad=True时训练模式，False时评估模式）
            module.train(requires_grad)

            # 记录已更新的模块
            updated_modules.append(name)

        # 检查是否有未找到的模块
        missing = list(set(modules) - set(updated_modules))
        if missing:
            raise ValueError(f"Could not find the following modules: {missing}.")

        return updated_modules

    def freeze_by_name(
        self,
        modules: Union[Text, List[Text]],
        recurse: bool = True,
    ) -> List[Text]:
        """冻结指定模块的参数
        
        冻结模块意味着这些参数在训练过程中不会被更新（梯度不计算）。
        常用于迁移学习：冻结预训练模型的特征提取层，只训练分类头。
        
        参数
        ----------
        modules : str 或 List[str]
            要冻结的模块名称（单个或列表）
            例如：["sincnet", "lstm"] 或 "linear"
        recurse : bool, 默认True
            如果为True，递归冻结所有子模块的参数
            如果为False，只冻结直接成员参数
        
        返回
        -------
        List[str]
            已冻结的模块名称列表
        
        异常
        ------
        ValueError
            如果指定的模块不存在
        
        示例
        ------
        >>> model.freeze_by_name("sincnet")  # 冻结sincnet模块
        >>> model.freeze_by_name(["sincnet", "lstm"])  # 冻结多个模块
        """

        return self.__by_name(
            modules,
            recurse=recurse,
            requires_grad=False,  # 冻结：禁止梯度计算
        )

    def unfreeze_by_name(
        self,
        modules: Union[List[Text], Text],
        recurse: bool = True,
    ) -> List[Text]:
        """解冻指定模块的参数
        
        解冻模块意味着这些参数在训练过程中会被更新（梯度计算）。
        常用于渐进式训练：先冻结所有层，然后逐步解冻。
        
        参数
        ----------
        modules : str 或 List[str]
            要解冻的模块名称（单个或列表）
            例如：["sincnet", "lstm"] 或 "linear"
        recurse : bool, 默认True
            如果为True，递归解冻所有子模块的参数
            如果为False，只解冻直接成员参数
        
        返回
        -------
        List[str]
            已解冻的模块名称列表
        
        异常
        ------
        ValueError
            如果指定的模块不存在
        
        示例
        ------
        >>> model.unfreeze_by_name("linear")  # 解冻linear模块
        >>> model.unfreeze_by_name(["lstm", "linear"])  # 解冻多个模块
        """

        return self.__by_name(modules, recurse=recurse, requires_grad=True)  # 解冻：允许梯度计算

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Union[Path, Text, List],
        config: Union[Path, Text] = None,
        map_location=None,
        hparams_file: Union[Path, Text] = None,
        strict: bool = True,
        use_auth_token: Union[Text, None] = None,
        cache_dir: Union[Path, Text] = None,
        **kwargs,
    ) -> "Model":
        """从预训练模型加载模型实例
        
        支持多种加载方式：
        1. 从本地文件加载
        2. 从URL加载
        3. 从HuggingFace Hub加载（最常用）
        4. 从配置文件和检查点加载（自定义模型）
        5. 平均多个检查点（模型集成）
        
        参数
        ----------
        checkpoint : Path, str 或 List
            检查点路径、HuggingFace模型ID或检查点列表
            例如：
            - "pyannote/segmentation-3.0"（HuggingFace ID）
            - "/path/to/model.bin"（本地文件）
            - ["ckpt1.bin", "ckpt2.bin"]（多个检查点，会平均）
        config : Path 或 str, 可选
            配置文件路径（用于自定义模型架构）
        map_location : 可选
            设备映射（用于CPU/GPU转换）
        hparams_file : Path 或 str, 可选
            超参数文件路径
        strict : bool, 默认True
            是否严格匹配权重（False时允许部分权重不匹配）
        use_auth_token : str, 可选
            HuggingFace认证token（用于私有模型）
        cache_dir : Path 或 str, 可选
            缓存目录（用于存储下载的模型）
        **kwargs
            其他传递给模型构造函数的参数
        
        返回
        -------
        Model
            加载的模型实例
        
        参考
        ------
        https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/core/model.py#L529-L704
        """
        # 如果提供了配置文件，使用自定义模型架构
        if config is not None:
            # 从配置文件实例化模型架构
            model = instantiate(config["model"]["path"], args=config["model"]["args"])
            # 如果checkpoint是列表，平均多个检查点
            if type(checkpoint) == list:
                print(f'Average model over {len(checkpoint)} checkpoints...')
                print(checkpoint)
                model = average_checkpoints(model, checkpoint)
            else:
                # 加载单个检查点
                ckpt_loaded = torch.load(checkpoint, map_location=torch.device('cpu'))
                model.load_state_dict(ckpt_loaded)
            return model
        
        else:
            # PyTorch Lightning需要字符串路径，不是Path对象
            checkpoint = str(checkpoint)
            if hparams_file is not None:
                hparams_file = str(hparams_file)

            # 解析检查点路径，确定加载方式
            if os.path.isfile(checkpoint):
                # 情况1：本地文件路径
                path_for_pl = checkpoint
            elif urlparse(checkpoint).scheme in ("http", "https"):
                # 情况2：HTTP/HTTPS URL
                path_for_pl = checkpoint
            else:
                # 情况3：HuggingFace Hub模型ID
                # 例如：julien-c/voice-activity-detection 或 julien-c/voice-activity-detection@main
                if "@" in checkpoint:
                    # 支持指定版本/分支/标签：model_id@revision
                    model_id = checkpoint.split("@")[0]
                    revision = checkpoint.split("@")[1]
                else:
                    model_id = checkpoint
                    revision = None

                try:
                    # 从HuggingFace Hub下载模型权重
                    path_for_pl = hf_hub_download(
                        model_id,
                        HF_PYTORCH_WEIGHTS_NAME,  # "pytorch_model.bin"
                        repo_type="model",
                        revision=revision,
                        library_name="pyannote",
                        library_version=__version__,
                        cache_dir=cache_dir,
                        use_auth_token=use_auth_token,
                    )
                except RepositoryNotFoundError:
                    # 模型未找到：可能是私有模型或需要认证
                    print(
                        f"""
    Could not download '{model_id}' model.
    It might be because the model is private or gated so make
    sure to authenticate. Visit https://hf.co/settings/tokens to
    create your access token and retry with:

    >>> Model.from_pretrained('{model_id}',
    ...                       use_auth_token=YOUR_AUTH_TOKEN)

    If this still does not work, it might be because the model is gated:
    visit https://hf.co/{model_id} to accept the user conditions."""
                    )
                    return None

                # 注意：HuggingFace下载计数器依赖于config.yaml
                # 因此即使不使用config.yaml，我们也下载它以更新下载计数
                # 如果模型没有config.yaml文件，静默失败
                try:
                    _ = hf_hub_download(
                        model_id,
                        HF_LIGHTNING_CONFIG_NAME,  # "config.yaml"
                        repo_type="model",
                        revision=revision,
                        library_name="pyannote",
                        library_version=__version__,
                        cache_dir=cache_dir,
                        use_auth_token=use_auth_token,
                    )
                except Exception:
                    pass

            # 设置设备映射（默认不转换）
            if map_location is None:
                def default_map_location(storage, loc):
                    return storage
                map_location = default_map_location

            # 从检查点获取模型类信息
            loaded_checkpoint = pl_load(path_for_pl, map_location=map_location)
            # 获取模型类的模块名和类名
            module_name: str = loaded_checkpoint["pyannote.audio"]["architecture"]["module"]
            module = import_module(module_name)  # 动态导入模块
            class_name: str = loaded_checkpoint["pyannote.audio"]["architecture"]["class"]
            Klass = getattr(module, class_name)  # 获取模型类

            try:
                # 使用PyTorch Lightning的load_from_checkpoint加载模型
                model = Klass.load_from_checkpoint(
                    path_for_pl,
                    map_location=map_location,
                    hparams_file=hparams_file,
                    strict=strict,
                    **kwargs,
                )
            except RuntimeError as e:
                # 如果模型包含任务相关的损失函数，可能需要设置strict=False
                if "loss_func" in str(e):
                    msg = (
                        "Model has been trained with a task-dependent loss function. "
                        "Set 'strict' to False to load the model without its loss function "
                        "and prevent this warning from appearing. "
                    )
                    warnings.warn(msg)
                    # 使用strict=False重新加载
                    model = Klass.load_from_checkpoint(
                        path_for_pl,
                        map_location=map_location,
                        hparams_file=hparams_file,
                        strict=False,
                        **kwargs,
                    )
                    return model

                raise e

            return model

if __name__ == '__main__':
    nnet = Model()
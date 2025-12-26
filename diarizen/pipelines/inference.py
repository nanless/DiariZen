# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

"""
DiariZen推理管道模块

该模块实现了DiariZen说话人分离的推理管道，继承自pyannote.audio的SpeakerDiarizationPipeline。
主要功能：
1. 加载DiariZen模型和配置
2. 执行说话人分离推理（segmentation）
3. 可选地执行嵌入提取和聚类（embedding + clustering）
4. 生成RTTM格式的输出

支持两种模式：
- segmentation_only: 只使用segmentation结果，直接转换为diarization（快速模式）
- full_pipeline: 完整的embedding + clustering流程（更准确但更慢）
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import toml
import numpy as np
import torch
import torchaudio

from scipy.ndimage import median_filter  # 用于中值滤波平滑segmentation结果

from huggingface_hub import snapshot_download, hf_hub_download  # 用于下载模型
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline  # 基类
from pyannote.audio.utils.signal import Binarize  # 用于将segmentation转换为最终diarization
from pyannote.database.protocol.protocol import ProtocolFile  # 协议文件类型

from diarizen.pipelines.utils import scp2path


class DiariZenPipeline(SpeakerDiarizationPipeline):
    """DiariZen说话人分离推理管道

    继承自pyannote.audio的SpeakerDiarizationPipeline，实现了DiariZen模型的推理流程。
    
    主要特点：
    - 支持powerset和multilabel两种输出模式
    - 支持segmentation-only和full-pipeline两种推理模式
    - 支持整句推理（不使用滑动窗口）
    - 自动处理设备分配和模型加载
    
    参数
    ----------
    diarizen_hub : Path
        DiariZen模型hub目录路径，应包含config.toml和pytorch_model.bin
    embedding_model : str
        嵌入模型路径（用于full-pipeline模式）
    config_parse : Optional[Dict[str, Any]], 可选
        用于覆盖配置文件的参数字典
    rttm_out_dir : Optional[str], 可选
        RTTM输出目录路径
    device : Optional[torch.device], 可选
        计算设备（CPU或CUDA），如果为None则自动检测
    segmentation_only : bool, 默认False
        是否只使用segmentation，跳过embedding和clustering
    binarize_onset : float, 默认0.5
        说话开始阈值（0-1），用于将segmentation转换为diarization
    binarize_offset : Optional[float], 可选
        说话结束阈值（0-1），如果为None则使用binarize_onset
    binarize_min_duration_on : float, 默认0.0
        说话段最小持续时间（秒）
    binarize_min_duration_off : float, 默认0.0
        静音段最小持续时间（秒）
    """
    def __init__(
        self, 
        diarizen_hub,
        embedding_model,
        config_parse: Optional[Dict[str, Any]] = None,
        rttm_out_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        segmentation_only: bool = False,
        binarize_onset: float = 0.5,
        binarize_offset: Optional[float] = None,
        binarize_min_duration_on: float = 0.0,
        binarize_min_duration_off: float = 0.0,
    ):
        # 加载配置文件
        config_path = Path(diarizen_hub / "config.toml")
        config = toml.load(config_path.as_posix())

        # 如果提供了config_parse，用它覆盖配置文件中的相应部分
        if config_parse is not None:
            print('Overriding with parsed config.')
            if "inference" in config_parse and "args" in config_parse["inference"]:
                config["inference"]["args"].update(config_parse["inference"]["args"])
            if "clustering" in config_parse and "args" in config_parse["clustering"]:
                config["clustering"]["args"].update(config_parse["clustering"]["args"])
       
        # 提取推理和聚类配置
        inference_config = config["inference"]["args"]
        clustering_config = config["clustering"]["args"]
        
        print(f'Loaded configuration: {config}')

        # 设备选择：如果没有指定设备，自动检测
        if device is None:
            # 检查CUDA是否可用（考虑CUDA_VISIBLE_DEVICES环境变量）
            cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
            device = torch.device("cuda:0") if cuda_available else torch.device("cpu")

        # DiariZen模型需要使用config来加载，因为检查点格式与标准pyannote.audio格式不同
        # 创建一个包含模型路径和配置的字典，传递给get_model
        segmentation_model_path = str(Path(diarizen_hub / "pytorch_model.bin"))
        segmentation_model_dict = {
            "checkpoint": segmentation_model_path,
            "config": config
        }
        
        # 处理整句推理逻辑：如果seg_duration特别大（>10000秒），认为是整句推理模式
        self.full_utterance_mode = False
        seg_duration = inference_config["seg_duration"]
        if seg_duration > 10000:
            print(f"检测到整句推理请求")
            self.full_utterance_mode = True

        # 调用父类初始化，设置segmentation、embedding和clustering组件
        super().__init__(
            segmentation=segmentation_model_dict,  # DiariZen分割模型
            segmentation_step=inference_config["segmentation_step"],  # 分割步长
            embedding=embedding_model,  # 嵌入模型路径
            embedding_exclude_overlap=True,  # 排除重叠区域的嵌入
            clustering=clustering_config["method"],  # 聚类方法（VBx或AHC）
            embedding_batch_size=inference_config["batch_size"],  # 嵌入批次大小
            segmentation_batch_size=inference_config["batch_size"],  # 分割批次大小
        )
        
        # 在初始化后设置设备
        self.device = device
        
        # 递归移动所有子模块到指定设备的辅助函数
        def move_to_device_recursive(module):
            """递归地将模块及其所有子模块移动到指定设备"""
            module.to(device)
            for child in module.children():
                move_to_device_recursive(child)
        
        # 强制将所有模型组件移动到指定设备
        # 处理segmentation模型
        if hasattr(self, '_segmentation'):
            if hasattr(self._segmentation, 'model'):
                # 递归移动模型及其所有子模块（包括wavlm_model等）
                move_to_device_recursive(self._segmentation.model)
                # 确保模型在eval模式（禁用dropout等训练特性）
                self._segmentation.model.eval()
            # 移动conversion对象（Powerset转换器）到正确设备
            if hasattr(self._segmentation, 'conversion'):
                self._segmentation.conversion = self._segmentation.conversion.to(device)
            # 确保Inference对象也使用正确的设备
            if hasattr(self._segmentation, 'device'):
                self._segmentation.device = device
            # 如果Inference对象有_device属性，也设置它
            if hasattr(self._segmentation, '_device'):
                self._segmentation._device = device
                
        # 处理embedding模型
        if hasattr(self, '_embedding'):
            # PretrainedSpeakerEmbedding使用model_属性存储模型
            if hasattr(self._embedding, 'model_'):
                move_to_device_recursive(self._embedding.model_)
                self._embedding.model_.eval()
            # 也检查是否有model属性（向后兼容）
            if hasattr(self._embedding, 'model'):
                move_to_device_recursive(self._embedding.model)
                self._embedding.model.eval()
            if hasattr(self._embedding, 'device'):
                self._embedding.device = device
            if hasattr(self._embedding, '_device'):
                self._embedding._device = device

        # 保存推理和聚类配置参数
        self.apply_median_filtering = inference_config["apply_median_filtering"]  # 是否应用中值滤波
        self.min_speakers = clustering_config["min_speakers"]  # 最小说话人数量
        self.max_speakers = clustering_config["max_speakers"]  # 最大说话人数量

        # 根据聚类方法设置管道参数
        if clustering_config["method"] == "AgglomerativeClustering":
            # 凝聚聚类（AHC）参数
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "method": "centroid",  # 使用质心方法
                    "min_cluster_size": clustering_config["min_cluster_size"],  # 最小簇大小
                    "threshold": clustering_config["ahc_threshold"],  # AHC阈值
                }
            }
        elif clustering_config["method"] == "VBxClustering":
            # VBx聚类参数
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "ahc_criterion": clustering_config["ahc_criterion"],  # AHC准则
                    "ahc_threshold": clustering_config["ahc_threshold"],  # AHC阈值
                    "Fa": clustering_config["Fa"],  # VBx Fa参数
                    "Fb": clustering_config["Fb"],  # VBx Fb参数
                }
            }
            # VBx需要PLDA目录
            self.clustering.plda_dir = str(Path(diarizen_hub / "plda"))
            self.clustering.lda_dim = clustering_config["lda_dim"]  # LDA维度
            self.clustering.maxIters = clustering_config["max_iters"]  # 最大迭代次数
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_config['method']}")

        # 实例化管道组件
        self.instantiate(self.PIPELINE_PARAMS)

        # 设置输出目录
        if rttm_out_dir is not None:
            os.makedirs(rttm_out_dir, exist_ok=True)
        self.rttm_out_dir = rttm_out_dir
        self.segmentation_only = segmentation_only  # 是否只使用segmentation，跳过embedding和clustering
        
        # Binarize参数（用于将segmentation转换为最终diarization）
        self.binarize_onset = binarize_onset  # 说话开始阈值
        self.binarize_offset = binarize_offset if binarize_offset is not None else binarize_onset  # 说话结束阈值
        self.binarize_min_duration_on = binarize_min_duration_on  # 说话段最小持续时间
        self.binarize_min_duration_off = binarize_min_duration_off  # 静音段最小持续时间

        # 根据模型规格自动检测是否为powerset模式
        self.is_powerset = self._segmentation.model.specifications.powerset

        # 修复Multilabel模式下Identity转换层不支持soft参数的问题
        # 如果模型不是powerset模式，创建一个支持soft参数的Identity层
        if not self.is_powerset:
            from torch import nn
            class IdentityWithSoft(nn.Module):
                """支持soft参数的Identity层"""
                def forward(self, x, soft=False):
                    return x
            self._segmentation.conversion = IdentityWithSoft().to(device)

    @classmethod
    def from_pretrained(
        cls, 
        repo_id: str, 
        cache_dir: str = None,
        rttm_out_dir: str = None,
        device: Optional[torch.device] = None,
        segmentation_only: bool = False,
        binarize_onset: float = 0.5,
        binarize_offset: Optional[float] = None,
        binarize_min_duration_on: float = 0.0,
        binarize_min_duration_off: float = 0.0,
        config_parse: Optional[Dict[str, Any]] = None,
    ) -> "DiariZenPipeline":
        """从预训练模型创建DiariZenPipeline实例

        支持从本地路径或HuggingFace Hub加载模型。
        自动下载嵌入模型（wespeaker-voxceleb-resnet34-LM）。

        参数
        ----------
        repo_id : str
            模型仓库ID（HuggingFace Hub）或本地路径
        cache_dir : str, 可选
            模型缓存目录
        rttm_out_dir : str, 可选
            RTTM输出目录
        device : Optional[torch.device], 可选
            计算设备
        segmentation_only : bool, 默认False
            是否只使用segmentation
        binarize_onset : float, 默认0.5
            说话开始阈值
        binarize_offset : Optional[float], 可选
            说话结束阈值
        binarize_min_duration_on : float, 默认0.0
            说话段最小持续时间
        binarize_min_duration_off : float, 默认0.0
            静音段最小持续时间
        config_parse : Optional[Dict[str, Any]], 可选
            配置覆盖参数字典

        返回
        -------
        DiariZenPipeline
            初始化好的管道实例
        """
        # 检查是否是本地路径
        local_path = Path(repo_id)
        if local_path.exists() and local_path.is_dir():
            # 使用本地路径
            diarizen_hub = local_path
        else:
            # 从HuggingFace Hub下载
            diarizen_hub = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir
            )

        # 下载嵌入模型（wespeaker-voxceleb-resnet34-LM）
        embedding_model = hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin",
            cache_dir=cache_dir
        )

        # 创建并返回管道实例
        return cls(
            diarizen_hub=Path(diarizen_hub).expanduser().absolute(),
            embedding_model=embedding_model,
            config_parse=config_parse,
            rttm_out_dir=rttm_out_dir,
            device=device,
            segmentation_only=segmentation_only,
            binarize_onset=binarize_onset,
            binarize_offset=binarize_offset,
            binarize_min_duration_on=binarize_min_duration_on,
            binarize_min_duration_off=binarize_min_duration_off,
        )

    def __call__(self, in_wav, sess_name=None):
        """执行说话人分离推理

        这是管道的主要调用方法，执行完整的说话人分离流程。

        参数
        ----------
        in_wav : str 或 ProtocolFile
            输入音频文件路径或协议文件对象
        sess_name : str, 可选
            会话名称，用于输出文件命名

        返回
        -------
        Annotation
            pyannote.core.Annotation对象，包含说话人分离结果
        """
        assert isinstance(in_wav, (str, ProtocolFile)), "input must be either a str or a ProtocolFile"
        # 如果是ProtocolFile，提取音频路径
        in_wav = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']
        
        print('Extracting segmentations.')
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(in_wav)
        # 强制使用第一个通道（SDM数据）
        waveform = torch.unsqueeze(waveform[0], 0)
        # 确保输入数据在正确的设备上
        waveform = waveform.to(self.device)

        # 如果是整句模式，动态设置duration为当前音频长度，避免产生巨大的Padding
        if hasattr(self, "full_utterance_mode") and self.full_utterance_mode:
            duration = waveform.shape[-1] / sample_rate
            self._segmentation.duration = duration
            self._segmentation.step = duration
        
        # CPU优化：使用torch.inference_mode()来加速推理（禁用梯度计算和自动微分）
        # inference_mode比no_grad更快，因为它完全禁用了自动微分图构建
        with torch.inference_mode():
            # 获取segmentation结果（帧级别的说话人活跃度预测）
            segmentations = self.get_segmentations({"waveform": waveform, "sample_rate": sample_rate}, soft=False)

        # 应用中值滤波平滑segmentation结果（可选）
        # 
        # ========== 中值滤波的作用和算法原理 ==========
        # 
        # 【算法原理】
        # 中值滤波（Median Filter）是一种非线性滤波方法，其核心思想是：
        # 1. 对每个像素/数据点，取一个固定大小的邻域窗口
        # 2. 将窗口内的所有值按大小排序
        # 3. 取排序后的中位数（中间值）作为该点的输出值
        # 
        # 示例（窗口大小为5）：
        #   输入序列: [0.1, 0.2, 0.9, 0.3, 0.2]  (0.9是异常值)
        #   排序后:   [0.1, 0.2, 0.2, 0.3, 0.9]
        #   中位数:   0.2 (第3个值)
        #   输出:     0.2 (异常值0.9被去除)
        # 
        # 【与均值滤波的区别】
        # 1. 均值滤波：计算窗口内所有值的平均值
        #    - 优点：平滑效果好
        #    - 缺点：会模糊边界，异常值会影响所有输出
        #    - 示例：[0.1, 0.2, 0.9, 0.3, 0.2] → 均值=0.34 (被异常值拉高)
        # 
        # 2. 中值滤波：取窗口内的中位数
        #    - 优点：能去除异常值，保持边界清晰
        #    - 缺点：对高斯噪声的平滑效果不如均值滤波
        #    - 示例：[0.1, 0.2, 0.9, 0.3, 0.2] → 中位数=0.2 (异常值被去除)
        # 
        # 【在说话人分离中的应用】
        # 1. 去除帧级抖动（Flickering）：
        #    - 模型输出的segmentation结果在帧级别可能存在噪声和抖动
        #    - 某些帧可能因为音频质量、背景噪声等原因产生错误的预测
        #    - 中值滤波可以有效去除这些孤立的异常值
        # 
        # 2. 保持边界清晰：
        #    - 与均值滤波不同，中值滤波不会模糊边界
        #    - 它只去除异常值，保留真实的说话人切换边界
        #    - 这对于说话人分离的准确性很重要
        # 
        # 3. 提高时间一致性：
        #    - 说话人的活跃状态在短时间内应该是连续的
        #    - 中值滤波确保相邻帧的预测更加一致
        #    - 减少因模型不确定性导致的频繁切换
        # 
        # 【参数说明】
        # size=(1, 11, 1):
        #    - 第一个1: 不处理批次维度（num_chunks）
        #    - 11: 在时间维度（num_frames）上使用11帧的窗口
        #    - 最后一个1: 不处理特征维度（num_classes）
        #    - 11帧窗口：假设帧率为10ms/帧，11帧约110ms，这是一个合理的平滑窗口
        # 
        # mode='reflect':
        #    - 边界处理模式，使用反射填充避免边界效应
        #    - 例如：[a, b, c] → [c, b, a, b, c] (在两端反射填充)
        #    - 这样可以避免边界处的数据丢失
        # 
        # 【时间复杂度】
        # O(n * k * log(k))，其中n是数据点数，k是窗口大小
        # 对于11帧窗口，log(11)≈3.46，实际运行很快
        # 
        # 注意：中值滤波是可选的，在某些情况下（如需要保留细微变化）可以关闭
        if self.apply_median_filtering:
            segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

        # binarize segmentation（对于powerset模式，这里还是powerset格式）
        binarized_segmentations = segmentations

        # 估计帧级别的瞬时说话人数量
        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation.model._receptive_field,
            warm_up=(0.0, 0.0),  # 不使用warm-up
        )

        # 根据模式选择处理流程
        if self.segmentation_only:
            # 模式1：只使用segmentation，跳过embedding和clustering（快速模式）
            
            if self.is_powerset:
                # 如果模型输出是powerset格式，需要转换为multi-label
                print("Converting powerset to multilabel (segmentation-only mode).")
                
                # 获取conversion对象（Powerset转换器）
                conversion = self._segmentation.conversion
                
                # segmentations.data形状: (num_chunks, num_frames, num_powerset_classes)
                num_chunks, num_frames, num_powerset_classes = segmentations.data.shape
                
                # 转换为torch tensor并移动到正确设备
                powerset_tensor = torch.from_numpy(segmentations.data).to(self.device)
                
                # 转换为multi-label: (num_chunks, num_frames, num_speakers)
                multilabel_tensor = conversion.to_multilabel(powerset_tensor, soft=False)
                multilabel_data = multilabel_tensor.cpu().numpy()
            else:
                # 如果模型已经是multi-label输出，直接使用
                print("Using native multilabel output (segmentation-only mode).")
                multilabel_data = segmentations.data
            
            # 创建multi-label segmentation对象
            from pyannote.core import SlidingWindowFeature
            multilabel_segmentations = SlidingWindowFeature(
                multilabel_data, 
                segmentations.sliding_window
            )
            
            # 直接从multi-label segmentation得到diarization
            discrete_diarization, _ = self.to_diarization(multilabel_segmentations, count)
        else:
            # 模式2：使用完整的流程：embedding + clustering（更准确但更慢）
            print("Extracting Embeddings.")
            # CPU优化：在inference_mode下提取嵌入
            with torch.inference_mode():
                # 提取说话人嵌入向量
                embeddings = self.get_embeddings(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    binarized_segmentations,
                    exclude_overlap=self.embedding_exclude_overlap,  # 排除重叠区域
                )

            # embeddings形状: (num_chunks, local_num_speakers, dimension)
            print("Clustering.")
            # 执行聚类，将嵌入向量分组为说话人
            hard_clusters, _, _ = self.clustering(
                embeddings=embeddings,
                segmentations=binarized_segmentations,
                min_clusters=self.min_speakers,  # 最小说话人数量
                max_clusters=self.max_speakers  # 最大说话人数量
            )

            # 在计数过程中，由于segmentation错误可能会高估瞬时说话人数量
            # 因此将最大瞬时说话人数量限制为max_speakers
            count.data = np.minimum(count.data, self.max_speakers).astype(np.int8)

            # 跟踪不活跃的说话人（在整个chunk中都没有说话）
            inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
            # 形状: (num_chunks, num_speakers)

            # 从原始硬聚类重建离散diarization
            hard_clusters[inactive_speakers] = -2  # 将不活跃说话人标记为-2
            discrete_diarization, _ = self.reconstruct(
                segmentations,
                hard_clusters,
                count,
            )

        # 将离散diarization转换为最终annotation
        # Binarize用于应用阈值和最小持续时间过滤
        to_annotation = Binarize(
            onset=self.binarize_onset,  # 说话开始阈值
            offset=self.binarize_offset,  # 说话结束阈值
            min_duration_on=self.binarize_min_duration_on,  # 说话段最小持续时间
            min_duration_off=self.binarize_min_duration_off  # 静音段最小持续时间
        )
        result = to_annotation(discrete_diarization)
        result.uri = sess_name  # 设置会话名称
        
        # 如果指定了输出目录，写入RTTM文件
        if self.rttm_out_dir is not None:
            assert sess_name is not None
            rttm_out = os.path.join(self.rttm_out_dir, sess_name + ".rttm")
            with open(rttm_out, "w") as f:
                f.write(result.to_rttm())
        return result
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This script performs diarization using DiariZen pipeline ",
        add_help=True,
        usage="%(prog)s [options]",
    )

    # Required paths
    parser.add_argument(
        "--in_wav_scp",
        type=str,
        required=True,
        help="Path to wav.scp."
    )
    parser.add_argument(
        "--diarizen_hub",
        type=str,
        required=True,
        help="Path to DiariZen model hub directory."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to pretrained embedding model."
    )

    # inference parameters
    parser.add_argument(
        "--seg_duration",
        type=int,
        default=16,
        help="Segment duration in seconds.",
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Shifting ratio during segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Input batch size for inference.",
    )
    parser.add_argument(
        "--apply_median_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply median filtering to segmentation output.",
    )

    # clustering parameters
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="VBxClustering",
        choices=["VBxClustering", "AgglomerativeClustering"],
        help="Clustering method to use.",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=1,
        help="Minimum number of speakers.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=20,
        help="Maximum number of speakers.",
    )
    parser.add_argument(
        "--ahc_criterion",
        type=str,
        default="distance",
        help="AHC criterion (for VBx).",
    )
    parser.add_argument(
        "--ahc_threshold",
        type=float,
        default=0.6,
        help="AHC threshold.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=13,
        help="Minimum cluster size (for AHC).",
    )
    parser.add_argument(
        "--Fa",
        type=float,
        default=0.07,
        help="VBx Fa parameter.",
    )
    parser.add_argument(
        "--Fb",
        type=float,
        default=0.8,
        help="VBx Fb parameter.",
    )
    parser.add_argument(
        "--lda_dim",
        type=int,
        default=128,
        help="VBx LDA dimension.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="VBx maximum iterations.",
    )

    # Output
    parser.add_argument(
        "--rttm_out_dir",
        type=str,
        default=None,
        required=False,
        help="Path to output folder.",
    )

    args = parser.parse_args()
    print(args)

    inference_config = {
        "seg_duration": args.seg_duration,
        "segmentation_step": args.segmentation_step,
        "batch_size": args.batch_size,
        "apply_median_filtering": args.apply_median_filtering
    }

    clustering_config = {
        "method": args.clustering_method,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers
    }
    if args.clustering_method == "AgglomerativeClustering":
        clustering_config.update({
            "ahc_threshold": args.ahc_threshold,
            "min_cluster_size": args.min_cluster_size
        })
    elif args.clustering_method == "VBxClustering":
        clustering_config.update({
            "ahc_criterion": args.ahc_criterion,
            "ahc_threshold": args.ahc_threshold,
            "Fa": args.Fa,
            "Fb": args.Fb,
            "lda_dim": args.lda_dim,
            "max_iters": args.max_iters
        })
    else:
        raise ValueError(f"Unsupported clustering method: {args.clustering_method}")

    config_parse = {
        "inference": {"args": inference_config},
        "clustering": {"args": clustering_config}
    }

    diarizen_pipeline = DiariZenPipeline(
        diarizen_hub=Path(args.diarizen_hub),
        embedding_model=args.embedding_model,
        config_parse=config_parse,
        rttm_out_dir=args.rttm_out_dir
    )

    audio_f = scp2path(args.in_wav_scp)
    for audio_file in audio_f:
        sess_name = Path(audio_file).stem.split('.')[0]
        print(f'Prosessing: {sess_name}')
        diarizen_pipeline(audio_file, sess_name=sess_name)

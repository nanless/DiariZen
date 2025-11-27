# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import toml
import numpy as np
import torch
import torchaudio

from scipy.ndimage import median_filter

from huggingface_hub import snapshot_download, hf_hub_download
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.utils.signal import Binarize
from pyannote.database.protocol.protocol import ProtocolFile

from diarizen.pipelines.utils import scp2path


class DiariZenPipeline(SpeakerDiarizationPipeline):
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
        config_path = Path(diarizen_hub / "config.toml")
        config = toml.load(config_path.as_posix())

        if config_parse is not None:
            print('Overriding with parsed config.')
            config["inference"]["args"] = config_parse["inference"]["args"]
            config["clustering"]["args"] = config_parse["clustering"]["args"]
       
        inference_config = config["inference"]["args"]
        clustering_config = config["clustering"]["args"]
        
        print(f'Loaded configuration: {config}')

        # 如果没有指定设备，使用默认逻辑
        if device is None:
            # 检查 CUDA 是否可用（考虑 CUDA_VISIBLE_DEVICES 环境变量）
            cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
            device = torch.device("cuda:0") if cuda_available else torch.device("cpu")

        # DiariZen 模型需要使用 config 来加载，因为检查点格式与标准 pyannote.audio 格式不同
        # 创建一个包含模型路径和配置的字典，传递给 get_model
        segmentation_model_path = str(Path(diarizen_hub / "pytorch_model.bin"))
        segmentation_model_dict = {
            "checkpoint": segmentation_model_path,
            "config": config
        }
        
        super().__init__(
            segmentation=segmentation_model_dict,
            segmentation_step=inference_config["segmentation_step"],
            embedding=embedding_model,
            embedding_exclude_overlap=True,
            clustering=clustering_config["method"],     
            embedding_batch_size=inference_config["batch_size"],
            segmentation_batch_size=inference_config["batch_size"],
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
        if hasattr(self, '_segmentation'):
            if hasattr(self._segmentation, 'model'):
                # 递归移动模型及其所有子模块（包括 wavlm_model 等）
                move_to_device_recursive(self._segmentation.model)
                # 确保模型在 eval 模式
                self._segmentation.model.eval()
            # 移动 conversion 对象（Powerset 转换器）到正确设备
            if hasattr(self._segmentation, 'conversion'):
                self._segmentation.conversion = self._segmentation.conversion.to(device)
            # 确保 Inference 对象也使用正确的设备
            if hasattr(self._segmentation, 'device'):
                self._segmentation.device = device
            # 如果 Inference 对象有 _device 属性，也设置它
            if hasattr(self._segmentation, '_device'):
                self._segmentation._device = device
                
        if hasattr(self, '_embedding'):
            # PretrainedSpeakerEmbedding 使用 model_ 属性存储模型
            if hasattr(self._embedding, 'model_'):
                move_to_device_recursive(self._embedding.model_)
                self._embedding.model_.eval()
            # 也检查是否有 model 属性（向后兼容）
            if hasattr(self._embedding, 'model'):
                move_to_device_recursive(self._embedding.model)
                self._embedding.model.eval()
            if hasattr(self._embedding, 'device'):
                self._embedding.device = device
            if hasattr(self._embedding, '_device'):
                self._embedding._device = device

        self.apply_median_filtering = inference_config["apply_median_filtering"]
        self.min_speakers = clustering_config["min_speakers"]
        self.max_speakers = clustering_config["max_speakers"]

        if clustering_config["method"] == "AgglomerativeClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": clustering_config["min_cluster_size"],
                    "threshold": clustering_config["ahc_threshold"],
                }
            }
        elif clustering_config["method"] == "VBxClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "ahc_criterion": clustering_config["ahc_criterion"],
                    "ahc_threshold": clustering_config["ahc_threshold"],
                    "Fa": clustering_config["Fa"],
                    "Fb": clustering_config["Fb"],
                }
            }
            self.clustering.plda_dir = str(Path(diarizen_hub / "plda"))
            self.clustering.lda_dim = clustering_config["lda_dim"]
            self.clustering.maxIters = clustering_config["max_iters"]
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_config['method']}")

        self.instantiate(self.PIPELINE_PARAMS)

        if rttm_out_dir is not None:
            os.makedirs(rttm_out_dir, exist_ok=True)
        self.rttm_out_dir = rttm_out_dir
        self.segmentation_only = segmentation_only  # 是否只使用 segmentation，跳过 embedding 和 clustering
        
        # Binarize参数（用于将segmentation转换为最终diarization）
        self.binarize_onset = binarize_onset
        self.binarize_offset = binarize_offset if binarize_offset is not None else binarize_onset
        self.binarize_min_duration_on = binarize_min_duration_on
        self.binarize_min_duration_off = binarize_min_duration_off

        assert self._segmentation.model.specifications.powerset is True

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
    ) -> "DiariZenPipeline":
        # 检查是否是本地路径
        local_path = Path(repo_id)
        if local_path.exists() and local_path.is_dir():
            diarizen_hub = local_path
        else:
            diarizen_hub = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir
            )

        embedding_model = hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin",
            cache_dir=cache_dir
        )

        return cls(
            diarizen_hub=Path(diarizen_hub).expanduser().absolute(),
            embedding_model=embedding_model,
            rttm_out_dir=rttm_out_dir,
            device=device,
            segmentation_only=segmentation_only,
            binarize_onset=binarize_onset,
            binarize_offset=binarize_offset,
            binarize_min_duration_on=binarize_min_duration_on,
            binarize_min_duration_off=binarize_min_duration_off,
        )

    def __call__(self, in_wav, sess_name=None):
        assert isinstance(in_wav, (str, ProtocolFile)), "input must be either a str or a ProtocolFile"
        in_wav = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']
        
        print('Extracting segmentations.')
        waveform, sample_rate = torchaudio.load(in_wav) 
        waveform = torch.unsqueeze(waveform[0], 0)      # force to use the SDM data
        # 确保输入数据在正确的设备上
        waveform = waveform.to(self.device)
        
        # CPU 优化：使用 torch.inference_mode() 来加速推理（禁用梯度计算和自动微分）
        # inference_mode 比 no_grad 更快，因为它完全禁用了自动微分图构建
        with torch.inference_mode():
            segmentations = self.get_segmentations({"waveform": waveform, "sample_rate": sample_rate}, soft=False)

        if self.apply_median_filtering:
            segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

        # binarize segmentation
        binarized_segmentations = segmentations     # powerset

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation.model._receptive_field,
            warm_up=(0.0, 0.0),
        )

        if self.segmentation_only:
            # 只使用 segmentation，跳过 embedding 和 clustering
            # 将 powerset segmentation 转换为 multi-label
            print("Converting powerset to multilabel (segmentation-only mode).")
            
            # 获取 conversion 对象（Powerset 转换器）
            conversion = self._segmentation.conversion
            
            # 将 powerset segmentation 转换为 multi-label
            # segmentations.data 形状: (num_chunks, num_frames, num_powerset_classes)
            num_chunks, num_frames, num_powerset_classes = segmentations.data.shape
            
            # 转换为 torch tensor 并移动到正确设备
            powerset_tensor = torch.from_numpy(segmentations.data).to(self.device)
            
            # 转换为 multi-label: (num_chunks, num_frames, num_speakers)
            multilabel_tensor = conversion.to_multilabel(powerset_tensor, soft=False)
            multilabel_data = multilabel_tensor.cpu().numpy()
            
            # 创建 multi-label segmentation
            from pyannote.core import SlidingWindowFeature
            multilabel_segmentations = SlidingWindowFeature(
                multilabel_data, 
                segmentations.sliding_window
            )
            
            # 直接从 multi-label segmentation 得到 diarization
            discrete_diarization, _ = self.to_diarization(multilabel_segmentations, count)
        else:
            # 使用完整的流程：embedding + clustering
            print("Extracting Embeddings.")
            # CPU 优化：在 inference_mode 下提取嵌入
            with torch.inference_mode():
                embeddings = self.get_embeddings(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    binarized_segmentations,
                    exclude_overlap=self.embedding_exclude_overlap,
                )

            # shape: (num_chunks, local_num_speakers, dimension)
            print("Clustering.")
            hard_clusters, _, _ = self.clustering(
                embeddings=embeddings,
                segmentations=binarized_segmentations,
                min_clusters=self.min_speakers,  
                max_clusters=self.max_speakers
            )

            # during counting, we could possibly overcount the number of instantaneous
            # speakers due to segmentation errors, so we cap the maximum instantaneous number
            # of speakers by the `max_speakers` value
            count.data = np.minimum(count.data, self.max_speakers).astype(np.int8)

            # keep track of inactive speakers
            inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
            #   shape: (num_chunks, num_speakers)

            # reconstruct discrete diarization from raw hard clusters
            hard_clusters[inactive_speakers] = -2
            discrete_diarization, _ = self.reconstruct(
                segmentations,
                hard_clusters,
                count,
            )

        # convert to annotation
        to_annotation = Binarize(
            onset=self.binarize_onset,
            offset=self.binarize_offset,
            min_duration_on=self.binarize_min_duration_on,
            min_duration_off=self.binarize_min_duration_off
        )
        result = to_annotation(discrete_diarization)
        result.uri = sess_name
        
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

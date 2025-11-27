# MIT License
#
# Copyright (c) 2021 CNRS
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

import warnings
from functools import cached_property
from pathlib import Path
from typing import Optional, Text, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from torch.nn.utils.rnn import pad_sequence

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.inference import BaseInference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import CACHE_DIR
from pyannote.audio.pipelines.utils import PipelineModel, get_model

try:
    from speechbrain.pretrained import (
        EncoderClassifier as SpeechBrain_EncoderClassifier,
    )

    SPEECHBRAIN_IS_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_IS_AVAILABLE = False

try:
    from nemo.collections.asr.models import (
        EncDecSpeakerLabelModel as NeMo_EncDecSpeakerLabelModel,
    )

    NEMO_IS_AVAILABLE = True
except ImportError:
    NEMO_IS_AVAILABLE = False

try:
    import onnxruntime as ort

    ONNX_IS_AVAILABLE = True
except ImportError:
    ONNX_IS_AVAILABLE = False


class NeMoPretrainedSpeakerEmbedding(BaseInference):
    """NeMo预训练说话人嵌入模型
    
    使用NVIDIA NeMo框架的预训练说话人嵌入模型。
    NeMo提供了高质量的说话人识别模型。
    
    参数
    ----------
    embedding : str, 默认"nvidia/speakerverification_en_titanet_large"
        NeMo模型名称或路径
    device : torch.device, 可选
        计算设备（CPU或GPU）
    
    注意
    -----
    需要安装NeMo库才能使用
    访问 https://nvidia.github.io/NeMo/ 获取安装说明
    """
    def __init__(
        self,
        embedding: Text = "nvidia/speakerverification_en_titanet_large",
        device: Optional[torch.device] = None,
    ):
        if not NEMO_IS_AVAILABLE:
            raise ImportError(
                f"'NeMo' must be installed to use '{embedding}' embeddings. "
                "Visit https://nvidia.github.io/NeMo/ for installation instructions."
            )

        super().__init__()
        self.embedding = embedding
        self.device = device or torch.device("cpu")

        self.model_ = NeMo_EncDecSpeakerLabelModel.from_pretrained(self.embedding)
        self.model_.freeze()
        self.model_.to(self.device)

    def to(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.model_.to(device)
        self.device = device
        return self

    @cached_property
    def sample_rate(self) -> int:
        return self.model_._cfg.train_ds.get("sample_rate", 16000)

    @cached_property
    def dimension(self) -> int:
        input_signal = torch.rand(1, self.sample_rate).to(self.device)
        input_signal_length = torch.tensor([self.sample_rate]).to(self.device)
        _, embeddings = self.model_(
            input_signal=input_signal, input_signal_length=input_signal_length
        )
        _, dimension = embeddings.shape
        return dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        lower, upper = 2, round(0.5 * self.sample_rate)
        middle = (lower + upper) // 2
        while lower + 1 < upper:
            try:
                input_signal = torch.rand(1, middle).to(self.device)
                input_signal_length = torch.tensor([middle]).to(self.device)

                _ = self.model_(
                    input_signal=input_signal, input_signal_length=input_signal_length
                )

                upper = middle
            except RuntimeError:
                lower = middle

            middle = (lower + upper) // 2

        return upper

    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1

        waveforms = waveforms.squeeze(dim=1)

        if masks is None:
            signals = waveforms.squeeze(dim=1)
            wav_lens = signals.shape[1] * torch.ones(batch_size)

        else:
            batch_size_masks, _ = masks.shape
            assert batch_size == batch_size_masks

            # TODO: speed up the creation of "signals"
            # preliminary profiling experiments show
            # that it accounts for 15% of __call__
            # (the remaining 85% being the actual forward pass)

            imasks = F.interpolate(
                masks.unsqueeze(dim=1), size=num_samples, mode="nearest"
            ).squeeze(dim=1)

            imasks = imasks > 0.5

            signals = pad_sequence(
                [waveform[imask] for waveform, imask in zip(waveforms, imasks)],
                batch_first=True,
            )

            wav_lens = imasks.sum(dim=1)

        max_len = wav_lens.max()

        # corner case: every signal is too short
        if max_len < self.min_num_samples:
            return np.NAN * np.zeros((batch_size, self.dimension))

        too_short = wav_lens < self.min_num_samples
        wav_lens[too_short] = max_len

        _, embeddings = self.model_(
            input_signal=waveforms.to(self.device),
            input_signal_length=wav_lens.to(self.device),
        )

        embeddings = embeddings.cpu().numpy()
        embeddings[too_short.cpu().numpy()] = np.NAN

        return embeddings


class SpeechBrainPretrainedSpeakerEmbedding(BaseInference):
    """SpeechBrain预训练说话人嵌入模型
    
    使用SpeechBrain框架的预训练说话人嵌入模型。
    SpeechBrain提供了多种高质量的说话人识别模型（如ECAPA-TDNN）。
    
    参数
    ----------
    embedding : str, 默认"speechbrain/spkrec-ecapa-voxceleb"
        SpeechBrain模型名称
        支持HuggingFace Hub上的模型ID
    device : torch.device, 可选
        计算设备（CPU或GPU）
    use_auth_token : str, 可选
        当加载私有HuggingFace模型时，设置认证token
        可以通过运行`huggingface-cli login`获取
    
    使用示例
    -----
    >>> # 创建嵌入提取器
    >>> get_embedding = SpeechBrainPretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
    >>> # 提取嵌入（无掩码）
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size
    >>> # 提取嵌入（带掩码，只使用活跃区域）
    >>> assert binary_masks.ndim == 1
    >>> assert binary_masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=binary_masks)
    
    注意
    -----
    需要安装speechbrain库才能使用
    访问 https://speechbrain.github.io 获取安装说明
    """

    def __init__(
        self,
        embedding: Text = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[torch.device] = None,
        use_auth_token: Union[Text, None] = None,
    ):
        if not SPEECHBRAIN_IS_AVAILABLE:
            raise ImportError(
                f"'speechbrain' must be installed to use '{embedding}' embeddings. "
                "Visit https://speechbrain.github.io for installation instructions."
            )

        super().__init__()
        if "@" in embedding:
            self.embedding = embedding.split("@")[0]
            self.revision = embedding.split("@")[1]
        else:
            self.embedding = embedding
            self.revision = None
        self.device = device or torch.device("cpu")
        self.use_auth_token = use_auth_token

        self.classifier_ = SpeechBrain_EncoderClassifier.from_hparams(
            source=self.embedding,
            savedir=f"{CACHE_DIR}/speechbrain",
            run_opts={"device": self.device},
            use_auth_token=self.use_auth_token,
            revision=self.revision,
        )

    def to(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.classifier_ = SpeechBrain_EncoderClassifier.from_hparams(
            source=self.embedding,
            savedir=f"{CACHE_DIR}/speechbrain",
            run_opts={"device": device},
            use_auth_token=self.use_auth_token,
            revision=self.revision,
        )
        self.device = device
        return self

    @cached_property
    def sample_rate(self) -> int:
        return self.classifier_.audio_normalizer.sample_rate

    @cached_property
    def dimension(self) -> int:
        dummy_waveforms = torch.rand(1, 16000).to(self.device)
        *_, dimension = self.classifier_.encode_batch(dummy_waveforms).shape
        return dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        with torch.inference_mode():
            lower, upper = 2, round(0.5 * self.sample_rate)
            middle = (lower + upper) // 2
            while lower + 1 < upper:
                try:
                    _ = self.classifier_.encode_batch(
                        torch.randn(1, middle).to(self.device)
                    )
                    upper = middle
                except RuntimeError:
                    lower = middle

                middle = (lower + upper) // 2

        return upper

    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1

        waveforms = waveforms.squeeze(dim=1)

        if masks is None:
            signals = waveforms.squeeze(dim=1)
            wav_lens = signals.shape[1] * torch.ones(batch_size)

        else:
            batch_size_masks, _ = masks.shape
            assert batch_size == batch_size_masks

            # TODO: speed up the creation of "signals"
            # preliminary profiling experiments show
            # that it accounts for 15% of __call__
            # (the remaining 85% being the actual forward pass)

            imasks = F.interpolate(
                masks.unsqueeze(dim=1), size=num_samples, mode="nearest"
            ).squeeze(dim=1)

            imasks = imasks > 0.5

            signals = pad_sequence(
                [
                    waveform[imask].contiguous()
                    for waveform, imask in zip(waveforms, imasks)
                ],
                batch_first=True,
            )

            wav_lens = imasks.sum(dim=1)

        max_len = wav_lens.max()

        # corner case: every signal is too short
        if max_len < self.min_num_samples:
            return np.NAN * np.zeros((batch_size, self.dimension))

        too_short = wav_lens < self.min_num_samples
        wav_lens = wav_lens / max_len
        wav_lens[too_short] = 1.0

        embeddings = (
            self.classifier_.encode_batch(signals, wav_lens=wav_lens)
            .squeeze(dim=1)
            .cpu()
            .numpy()
        )

        embeddings[too_short.cpu().numpy()] = np.NAN

        return embeddings


class ONNXWeSpeakerPretrainedSpeakerEmbedding(BaseInference):
    """ONNX格式的WeSpeaker预训练说话人嵌入模型
    
    使用ONNX Runtime运行WeSpeaker模型，提供高效的推理速度。
    WeSpeaker是WeNet团队开发的说话人识别工具包。
    
    参数
    ----------
    embedding : str
        WeSpeaker预训练模型路径或HuggingFace模型ID
        例如："hbredin/wespeaker-voxceleb-resnet34-LM"
    device : torch.device, 可选
        计算设备（CPU或GPU）
        支持CUDA加速（如果安装了onnxruntime-gpu）
    
    使用示例
    -----
    >>> # 创建嵌入提取器
    >>> get_embedding = ONNXWeSpeakerPretrainedSpeakerEmbedding("hbredin/wespeaker-voxceleb-resnet34-LM")
    >>> # 提取嵌入（无掩码）
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size
    >>> # 提取嵌入（带掩码）
    >>> assert binary_masks.ndim == 1
    >>> assert binary_masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=binary_masks)
    
    特点
    -----
    - 使用ONNX Runtime，推理速度快
    - 支持CPU和GPU加速
    - 使用FBank特征（而非原始波形）
    
    注意
    -----
    需要安装onnxruntime库才能使用
    对于GPU加速，需要安装onnxruntime-gpu
    """

    def __init__(
        self,
        embedding: Text = "hbredin/wespeaker-voxceleb-resnet34-LM",
        device: Optional[torch.device] = None,
    ):
        if not ONNX_IS_AVAILABLE:
            raise ImportError(
                f"'onnxruntime' must be installed to use '{embedding}' embeddings."
            )

        super().__init__()

        if not Path(embedding).exists():
            try:
                embedding = hf_hub_download(
                    repo_id=embedding,
                    filename="speaker-embedding.onnx",
                )
            except RepositoryNotFoundError:
                raise ValueError(
                    f"Could not find '{embedding}' on huggingface.co nor on local disk."
                )

        self.embedding = embedding

        self.to(device or torch.device("cpu"))

    def to(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        if device.type == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device.type == "cuda":
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "cudnn_conv_algo_search": "DEFAULT",  # EXHAUSTIVE / HEURISTIC / DEFAULT
                    },
                )
            ]
        else:
            warnings.warn(
                f"Unsupported device type: {device.type}, falling back to CPU"
            )
            device = torch.device("cpu")
            providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        self.session_ = ort.InferenceSession(
            self.embedding, sess_options=sess_options, providers=providers
        )

        self.device = device
        return self

    @cached_property
    def sample_rate(self) -> int:
        return 16000

    @cached_property
    def dimension(self) -> int:
        dummy_waveforms = torch.rand(1, 1, 16000)
        features = self.compute_fbank(dummy_waveforms)
        embeddings = self.session_.run(
            output_names=["embs"], input_feed={"feats": features.numpy()}
        )[0]
        _, dimension = embeddings.shape
        return dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        lower, upper = 2, round(0.5 * self.sample_rate)
        middle = (lower + upper) // 2
        while lower + 1 < upper:
            try:
                features = self.compute_fbank(torch.randn(1, 1, middle))

            except AssertionError:
                lower = middle
                middle = (lower + upper) // 2
                continue

            embeddings = self.session_.run(
                output_names=["embs"], input_feed={"feats": features.numpy()}
            )[0]

            if np.any(np.isnan(embeddings)):
                lower = middle
            else:
                upper = middle
            middle = (lower + upper) // 2

        return upper

    @cached_property
    def min_num_frames(self) -> int:
        return self.compute_fbank(torch.randn(1, 1, self.min_num_samples)).shape[1]

    def compute_fbank(
        self,
        waveforms: torch.Tensor,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
    ) -> torch.Tensor:
        """Extract fbank features

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)

        Returns
        -------
        fbank : (batch_size, num_frames, num_mel_bins)

        Source: https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker/bin/infer_onnx.py#L30C1-L50
        """

        waveforms = waveforms * (1 << 15)
        features = torch.stack(
            [
                kaldi.fbank(
                    waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    sample_frequency=self.sample_rate,
                    window_type="hamming",
                    use_energy=False,
                )
                for waveform in waveforms
            ]
        )

        return features - torch.mean(features, dim=1, keepdim=True)

    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1

        features = self.compute_fbank(waveforms.to(self.device))
        _, num_frames, _ = features.shape

        if masks is None:
            embeddings = self.session_.run(
                output_names=["embs"], input_feed={"feats": features.numpy(force=True)}
            )[0]

            return embeddings

        batch_size_masks, _ = masks.shape
        assert batch_size == batch_size_masks

        imasks = F.interpolate(
            masks.unsqueeze(dim=1), size=num_frames, mode="nearest"
        ).squeeze(dim=1)

        imasks = imasks > 0.5

        embeddings = np.NAN * np.zeros((batch_size, self.dimension))

        for f, (feature, imask) in enumerate(zip(features, imasks)):
            masked_feature = feature[imask]
            if masked_feature.shape[0] < self.min_num_frames:
                continue

            embeddings[f] = self.session_.run(
                output_names=["embs"],
                input_feed={"feats": masked_feature.numpy(force=True)[None]},
            )[0][0]

        return embeddings


class PyannoteAudioPretrainedSpeakerEmbedding(BaseInference):
    """pyannote.audio预训练说话人嵌入模型
    
    使用pyannote.audio框架训练的说话人嵌入模型。
    这是pyannote.audio原生的嵌入模型接口。
    
    参数
    ----------
    embedding : PipelineModel
        pyannote.audio模型
        可以是：
        - HuggingFace模型ID（如"pyannote/embedding"）
        - 模型路径
        - Model实例
    device : torch.device, 可选
        计算设备（CPU或GPU）
    use_auth_token : str, 可选
        当加载私有HuggingFace模型时，设置认证token
        可以通过运行`huggingface-cli login`获取
    
    使用示例
    -----
    >>> # 创建嵌入提取器
    >>> get_embedding = PyannoteAudioPretrainedSpeakerEmbedding("pyannote/embedding")
    >>> # 提取嵌入（无掩码）
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size
    >>> # 提取嵌入（带掩码，只使用活跃区域）
    >>> assert masks.ndim == 1
    >>> assert masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=masks)
    
    特点
    -----
    - 原生pyannote.audio模型接口
    - 支持多种模型格式
    - 自动处理设备管理
    """

    def __init__(
        self,
        embedding: PipelineModel = "pyannote/embedding",
        device: Optional[torch.device] = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()
        self.embedding = embedding
        self.device = device or torch.device("cpu")
        print(f'self.embedding: {self.embedding}')
        self.model_: Model = get_model(self.embedding, use_auth_token=use_auth_token)
        self.model_.eval()
        self.model_.to(self.device)

    def to(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.model_.to(device)
        self.device = device
        return self

    @cached_property
    def sample_rate(self) -> int:
        return self.model_.audio.sample_rate

    @cached_property
    def dimension(self) -> int:
        return self.model_.dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        with torch.inference_mode():
            lower, upper = 2, round(0.5 * self.sample_rate)
            middle = (lower + upper) // 2
            while lower + 1 < upper:
                try:
                    _ = self.model_(torch.randn(1, 1, middle).to(self.device))
                    upper = middle
                except Exception:
                    lower = middle

                middle = (lower + upper) // 2

        return upper

    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        with torch.inference_mode():
            if masks is None:
                embeddings = self.model_(waveforms.to(self.device))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    embeddings = self.model_(
                        waveforms.to(self.device), weights=masks.to(self.device)
                    )
        return embeddings.cpu().numpy()


def PretrainedSpeakerEmbedding(
    embedding: PipelineModel,
    device: Optional[torch.device] = None,
    use_auth_token: Union[Text, None] = None,
):
    """预训练说话人嵌入模型的统一工厂函数
    
    根据模型名称自动选择合适的嵌入模型实现类。
    支持多种预训练模型框架。
    
    参数
    ----------
    embedding : PipelineModel
        嵌入模型标识符，可以是：
        - pyannote模型（如"pyannote/embedding"）
        - SpeechBrain模型（如"speechbrain/spkrec-ecapa-voxceleb"）
        - NeMo模型（如"nvidia/speakerverification_en_titanet_large"）
        - WeSpeaker模型（如"hbredin/wespeaker-voxceleb-resnet34-LM"）
        - 本地模型路径
    device : torch.device, 可选
        计算设备（CPU或GPU）
    use_auth_token : str, 可选
        当加载私有HuggingFace模型时，设置认证token
        可以通过运行`huggingface-cli login`获取
    
    返回
    -------
    BaseInference
        对应的预训练说话人嵌入模型实例
    
    使用示例
    -----
    >>> # pyannote模型
    >>> get_embedding = PretrainedSpeakerEmbedding("pyannote/embedding")
    >>> # SpeechBrain模型
    >>> get_embedding = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
    >>> # NeMo模型
    >>> get_embedding = PretrainedSpeakerEmbedding("nvidia/speakerverification_en_titanet_large")
    >>> # 提取嵌入
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size
    >>> # 带掩码提取嵌入
    >>> assert masks.ndim == 1
    >>> assert masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=masks)
    
    自动选择逻辑
    -----------
    - 包含"pyannote" → PyannoteAudioPretrainedSpeakerEmbedding
    - 包含"speechbrain" → SpeechBrainPretrainedSpeakerEmbedding
    - 包含"nvidia" → NeMoPretrainedSpeakerEmbedding
    - 包含"wespeaker" → ONNXWeSpeakerPretrainedSpeakerEmbedding
    - 其他情况 → PyannoteAudioPretrainedSpeakerEmbedding（默认）
    """

    # 根据模型名称自动选择合适的实现类
    if isinstance(embedding, str) and "pyannote" in embedding:
        return PyannoteAudioPretrainedSpeakerEmbedding(
            embedding, device=device, use_auth_token=use_auth_token
        )

    elif isinstance(embedding, str) and "speechbrain" in embedding:
        return SpeechBrainPretrainedSpeakerEmbedding(
            embedding, device=device, use_auth_token=use_auth_token
        )

    elif isinstance(embedding, str) and "nvidia" in embedding:
        return NeMoPretrainedSpeakerEmbedding(embedding, device=device)

    elif isinstance(embedding, str) and "wespeaker" in embedding:
        return ONNXWeSpeakerPretrainedSpeakerEmbedding(embedding, device=device)

    else:
        # 默认回退到pyannote（用于加载本地模型）
        return PyannoteAudioPretrainedSpeakerEmbedding(
            embedding, device=device, use_auth_token=use_auth_token
        )


class SpeakerEmbedding(Pipeline):
    """说话人嵌入管道
    
    从音频文件中提取说话人嵌入向量。
    假设每个文件只包含一个说话人，从整个文件中提取单个嵌入向量。
    
    参数
    ----------
    embedding : Model, str, 或 dict, 默认"pyannote/embedding"
        预训练嵌入模型
        支持格式见 pyannote.audio.pipelines.utils.get_model
    segmentation : Model, str, 或 dict, 可选
        预训练分割模型（或语音活动检测模型）
        用于加权提取嵌入（只关注语音活跃区域）
        支持格式见 pyannote.audio.pipelines.utils.get_model
        默认为None（不使用语音活动检测）
    use_auth_token : str, 可选
        当加载私有HuggingFace模型时，设置认证token
        可以通过运行`huggingface-cli login`获取
    
    使用示例
    -----
    >>> from pyannote.audio.pipelines import SpeakerEmbedding
    >>> # 创建管道
    >>> pipeline = SpeakerEmbedding()
    >>> # 提取说话人嵌入
    >>> emb1 = pipeline("speaker1.wav")
    >>> emb2 = pipeline("speaker2.wav")
    >>> # 计算说话人相似度（余弦距离）
    >>> from scipy.spatial.distance import cdist
    >>> distance = cdist(emb1, emb2, metric="cosine")[0,0]
    
    工作流程
    --------
    1. 加载音频文件
    2. （可选）使用分割模型获取语音活动得分
    3. 使用嵌入模型提取说话人嵌入
    4. 如果提供了分割模型，使用语音活动得分加权聚合嵌入
    
    应用场景
    --------
    - 说话人验证（验证两个音频是否为同一说话人）
    - 说话人识别（识别音频中的说话人）
    - 说话人聚类（将多个音频按说话人分组）
    """

    def __init__(
        self,
        embedding: PipelineModel = "pyannote/embedding",
        segmentation: Optional[PipelineModel] = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()

        self.embedding = embedding
        self.segmentation = segmentation

        self.embedding_model_: Model = get_model(
            embedding, use_auth_token=use_auth_token
        )

        if self.segmentation is not None:
            segmentation_model: Model = get_model(
                self.segmentation, use_auth_token=use_auth_token
            )
            self._segmentation = Inference(
                segmentation_model,
                pre_aggregation_hook=lambda scores: np.max(
                    scores, axis=-1, keepdims=True
                ),
            )

    def apply(self, file: AudioFile) -> np.ndarray:
        """应用管道处理音频文件
        
        参数
        ----------
        file : AudioFile
            音频文件路径或对象
        
        返回
        -------
        np.ndarray
            说话人嵌入向量（1维数组）
        """
        device = self.embedding_model_.device

        # 读取音频文件并发送到GPU
        waveform = self.embedding_model_.audio(file)[0][None].to(device)

        if self.segmentation is None:
            # 不使用语音活动检测，均匀加权
            weights = None
        else:
            # 获取语音活动得分
            weights = self._segmentation(file).data
            # HACK -- 修复NaN值（应该在上游修复）
            weights[np.isnan(weights)] = 0.0
            # 使用三次方增强语音区域的权重
            weights = torch.from_numpy(weights**3)[None, :, 0].to(device)

        # 提取说话人嵌入（使用权重加权）
        with torch.no_grad():
            return self.embedding_model_(waveform, weights=weights).cpu().numpy()


def main(
    protocol: str = "VoxCeleb.SpeakerVerification.VoxCeleb1",
    subset: str = "test",
    embedding: str = "pyannote/embedding",
    segmentation: Optional[str] = None,
):
    import typer
    from pyannote.database import FileFinder, get_protocol
    from pyannote.metrics.binary_classification import det_curve
    from scipy.spatial.distance import cdist
    from tqdm import tqdm

    pipeline = SpeakerEmbedding(embedding=embedding, segmentation=segmentation)

    protocol = get_protocol(protocol, preprocessors={"audio": FileFinder()})

    y_true, y_pred = [], []

    emb = dict()

    trials = getattr(protocol, f"{subset}_trial")()

    for t, trial in enumerate(tqdm(trials)):
        audio1 = trial["file1"]["audio"]
        if audio1 not in emb:
            emb[audio1] = pipeline(audio1)

        audio2 = trial["file2"]["audio"]
        if audio2 not in emb:
            emb[audio2] = pipeline(audio2)

        y_pred.append(cdist(emb[audio1], emb[audio2], metric="cosine")[0][0])
        y_true.append(trial["reference"])

    _, _, _, eer = det_curve(y_true, np.array(y_pred), distances=True)
    typer.echo(
        f"{protocol.name} | {subset} | {embedding} | {segmentation} | EER = {100 * eer:.3f}%"
    )


if __name__ == "__main__":
    import typer

    typer.run(main)

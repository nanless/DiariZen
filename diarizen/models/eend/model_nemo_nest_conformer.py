#!/usr/bin/env python3
# pyright: reportMissingImports=false

"""NeMo NEST-Conformer diarization model.

This module keeps the same head/trainer contract as model_wavlm_conformer:
the SSL backbone is exposed as ``wavlm_model`` and the diarization head is
returned by ``non_wavlm_parameters()``.  That lets existing diar_ssl recipes
reuse the dual-optimizer fine-tuning flow while replacing WavLM with NVIDIA
NEST from NeMo/HuggingFace.
"""

import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from pyannote.audio.core.model import Model as BaseModel

from diarizen.models.module.conformer import ConformerEncoder


def _maybe_add_nemo_root(nemo_root: str) -> None:
    if not nemo_root:
        return
    nemo_root = Path(nemo_root).expanduser().absolute().as_posix()
    if os.path.isdir(nemo_root) and nemo_root not in sys.path:
        sys.path.insert(0, nemo_root)


def _parse_layer_list(layers: str, num_layers: int) -> Optional[List[int]]:
    if layers == "all":
        return None
    if layers == "last":
        return [num_layers - 1]
    return [int(layer.strip()) for layer in layers.split(",") if layer.strip()]


class NemoNestBackbone(nn.Module):
    """Thin NeMo feature extractor returning a list of [B, D, T] tensors."""

    def __init__(self, ssl_model: nn.Module, layers: str = "all"):
        super().__init__()

        from nemo.collections.asr.modules import ConformerMultiLayerFeatureExtractor

        self.preprocessor = ssl_model.preprocessor
        self.encoder = ssl_model.encoder

        layer_idx_list = _parse_layer_list(layers, len(self.encoder.layers))
        self.feature_extractor = ConformerMultiLayerFeatureExtractor(
            self.encoder,
            aggregator=None,
            layer_idx_list=layer_idx_list,
        )

        self.num_selected_layers = (
            len(self.encoder.layers) if layer_idx_list is None else len(layer_idx_list)
        )
        self.feature_dim = int(
            getattr(self.encoder, "d_model", getattr(self.encoder, "_feat_out", 0))
        )
        cfg = getattr(self.preprocessor, "_cfg", {})
        self.window_stride = float(getattr(cfg, "window_stride", cfg.get("window_stride", 0.01)))
        self.window_size = float(getattr(cfg, "window_size", cfg.get("window_size", 0.025)))
        self.subsampling_factor = int(getattr(self.encoder, "subsampling_factor", 1))

    @property
    def frame_shift(self) -> float:
        return self.window_stride * self.subsampling_factor

    def forward(self, input_signal: torch.Tensor):
        input_signal_length = torch.full(
            (input_signal.size(0),),
            input_signal.size(-1),
            dtype=torch.long,
            device=input_signal.device,
        )
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal,
            length=input_signal_length,
        )
        return self.feature_extractor(
            audio_signal=processed_signal,
            length=processed_signal_length,
        )


class Model(BaseModel):
    """Diarization model using NVIDIA NEST SSL features plus a Conformer head."""

    def __init__(
        self,
        nest_model_name: str = "nvidia/ssl_en_nest_xlarge_v1.0",
        nemo_root: str = "/root/code/github_repos/NeMo",
        nest_layers: str = "all",
        nest_layer_num: int = 0,
        nest_feat_dim: int = 0,
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
        output_activate_function: str = False,
        max_speakers_per_chunk: int = 4,
        max_speakers_per_frame: int = 2,
        chunk_size: int = 5,
        num_channels: int = 8,
        selected_channel: int = 0,
        sample_rate: int = 16000,
    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame,
        )

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel

        self.wavlm_model = self.load_nest(nest_model_name, nemo_root, nest_layers)
        if nest_layer_num <= 0:
            nest_layer_num = self.wavlm_model.num_selected_layers
        if nest_feat_dim <= 0:
            nest_feat_dim = self.wavlm_model.feature_dim

        self.weight_sum = nn.Linear(nest_layer_num, 1, bias=False)
        self.proj = nn.Linear(nest_feat_dim, attention_in)
        self.lnorm = nn.LayerNorm(attention_in)

        self.conformer = ConformerEncoder(
            attention_in=attention_in,
            ffn_hidden=ffn_hidden,
            num_head=num_head,
            num_layer=num_layer,
            kernel_size=kernel_size,
            dropout=dropout,
            use_posi=use_posi,
            output_activate_function=output_activate_function,
        )

        self.classifier = nn.Linear(attention_in, self.dimension)
        self.activation = self.default_activation()

    def non_wavlm_parameters(self):
        return [
            *self.weight_sum.parameters(),
            *self.proj.parameters(),
            *self.lnorm.parameters(),
            *self.conformer.parameters(),
            *self.classifier.parameters(),
        ]

    @property
    def dimension(self) -> int:
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")
        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        return len(self.specifications.classes)

    def load_nest(self, source: str, nemo_root: str, layers: str):
        _maybe_add_nemo_root(nemo_root)

        try:
            from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel
            from nemo.core.classes.common import typecheck
        except ImportError as exc:
            raise ImportError(
                "NeMo is required for NEST fine-tuning. Clone NVIDIA/NeMo and set "
                "NEMO_ROOT so it is importable, or install NeMo in the active env."
            ) from exc

        typecheck.set_typecheck_enabled(enabled=False)

        if source.endswith(".nemo") and os.path.isfile(source):
            ssl_model = EncDecDenoiseMaskedTokenPredModel.restore_from(
                restore_path=source,
                map_location="cpu",
            )
        else:
            ssl_model = EncDecDenoiseMaskedTokenPredModel.from_pretrained(
                model_name=source,
                map_location="cpu",
            )

        return NemoNestBackbone(ssl_model, layers=layers)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        duration = num_samples / self.sample_rate
        return max(1, int(math.ceil(duration / self.wavlm_model.frame_shift)))

    def receptive_field_size(self, num_frames: int = 1) -> int:
        window = int(round(self.wavlm_model.window_size * self.sample_rate))
        step = int(round(self.wavlm_model.frame_shift * self.sample_rate))
        return window + (num_frames - 1) * step

    @property
    def get_rf_info(self):
        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        num_frames = self.num_frames(self.chunk_size * self.sample_rate)
        duration = receptive_field_size / self.sample_rate
        step = receptive_field_step / self.sample_rate
        return num_frames, duration, step

    def wav2nest(self, in_wav: torch.Tensor, model: nn.Module) -> torch.Tensor:
        layer_reps, _ = model(in_wav)
        layer_reps = [layer.transpose(1, 2) for layer in layer_reps]
        return torch.stack(layer_reps, dim=-1)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        assert waveforms.dim() == 3
        waveforms = waveforms[:, self.selected_channel, :]

        nest_feat = self.wav2nest(waveforms, self.wavlm_model)
        nest_feat = self.weight_sum(nest_feat)
        nest_feat = torch.squeeze(nest_feat, -1)

        outputs = self.proj(nest_feat)
        outputs = self.lnorm(outputs)
        outputs = self.conformer(outputs)
        outputs = self.classifier(outputs)
        outputs = self.activation(outputs)

        return outputs

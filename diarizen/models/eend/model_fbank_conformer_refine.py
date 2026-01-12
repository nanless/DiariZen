#!/usr/bin/env python3
#
# Licensed under the MIT license.
#
"""
Fbank-Conformer 说话人分离模型（RoPE + FlashAttention 优化版）

- 结构基本沿用 `model_fbank_conformer.py`：
  waveforms -> Fbank -> proj + LayerNorm -> ConformerEncoder -> classifier + activation
- ConformerEncoder 部分改为基于 RoPE 的实现，并使用
  `torch.nn.functional.scaled_dot_product_attention` 加速多头注意力计算。

与原版的主要区别
----------------
- 去掉相对位置编码 `RelativePositionalEncoding`，改用 RoPE（Rotary Position Embedding）
- MHA 使用 `scaled_dot_product_attention`（可利用 FlashAttention）
"""

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyannote.audio.core.model import Model as BaseModel
from diarizen.models.module.speechbrain_feats import Fbank

# 优先使用 FlashAttention backend（在 GPU + 支持的 dtype 条件下）
try:  # 在 CPU 或旧版本 PyTorch 上静默忽略
    from torch.backends.cuda import sdp_kernel

    # 将 scaled_dot_product_attention 的默认 backend 设为 FLASH_ATTENTION，
    # 若当前设备 / dtype 不支持，会自动回退到其它实现。
    sdp_kernel.set_default_backend(sdp_kernel.SDPBackend.FLASH_ATTENTION)
except Exception:
    pass

try:
    # 推荐使用 rotary_embedding_torch，如果不可用可以很方便改为简单 RoPE 实现
    from rotary_embedding_torch import RotaryEmbedding
except Exception:  # pragma: no cover - 仅在环境缺库时走到
    RotaryEmbedding = None


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()


class PositionwiseFeedForward(nn.Module):
    """Positionwise FFN with pre-LN & scaled residual."""

    def __init__(self, in_size: int, ffn_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(in_size)
        self.w1 = nn.Linear(in_size, ffn_hidden)
        self.act = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.w2 = nn.Linear(ffn_hidden, in_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.w1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.w2(x)
        x = self.dropout2(x)
        # conformer-style 0.5 residual
        return residual + 0.5 * x


class ConvolutionModule(nn.Module):
    """Conformer conv module（同原项目实现，使用 LayerNorm + GLU + depthwise conv）"""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 31,
        dropout_rate: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.ln = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        x = self.ln(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, C)
        return residual + x


class MultiHeadSelfAttentionRoPE(nn.Module):
    """Multi-head self-attention with RoPE + scaled_dot_product_attention.

    - 输入: x [B, T, C]
    - 不再使用显式相对位置编码，而是对 q/k 施加 RoPE 旋转。
    """

    def __init__(
        self,
        n_units: int,
        num_heads: int,
        dropout: float,
        rotary_embed=None,
        causal: bool = False,
    ):
        super().__init__()
        assert n_units % num_heads == 0
        self.num_heads = num_heads
        self.d_k = n_units // num_heads
        self.q_proj = nn.Linear(n_units, n_units)
        self.k_proj = nn.Linear(n_units, n_units)
        self.v_proj = nn.Linear(n_units, n_units)
        self.o_proj = nn.Linear(n_units, n_units)
        self.dropout_p = dropout
        self.rotary_embed = rotary_embed
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        bsz, t, _ = x.shape

        q = self.q_proj(x).view(bsz, t, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, t, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t, self.num_heads, self.d_k).transpose(1, 2)
        # (B, H, T, D)

        if self.rotary_embed is not None:
            # rotary_embedding_torch 接受 (..., seq_len, dim)
            # 这里 T 在 dim=-2
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        dropout_p = self.dropout_p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=self.causal
        )
        # out: (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(bsz, t, self.num_heads * self.d_k)
        return self.o_proj(out)


class ConformerMHA(nn.Module):
    """Conformer MHA block with pre-LN + residual (RoPE 版本)."""

    def __init__(
        self,
        in_size: int = 256,
        num_head: int = 4,
        dropout: float = 0.1,
        rotary_embed=None,
        causal: bool = False,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(in_size)
        self.mha = MultiHeadSelfAttentionRoPE(
            n_units=in_size,
            num_heads=num_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
            causal=causal,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.mha(x)
        x = self.dropout(x)
        return residual + x


class ConformerBlock(nn.Module):
    """标准 Conformer Block（RoPE + FlashAttention 版本）"""

    def __init__(
        self,
        in_size: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        rotary_embed=None,
        causal: bool = False,
    ):
        super().__init__()
        self.ffn1 = PositionwiseFeedForward(
            in_size=in_size,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
        )
        self.mha = ConformerMHA(
            in_size=in_size,
            num_head=num_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
            causal=causal,
        )
        self.conv = ConvolutionModule(
            channels=in_size,
            kernel_size=kernel_size,
            dropout_rate=dropout,
        )
        self.ffn2 = PositionwiseFeedForward(
            in_size=in_size,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
        )
        self.ln = nn.LayerNorm(in_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.mha(x)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.ln(x)


class ConformerEncoderRoPE(nn.Module):
    """基于 RoPE 的 ConformerEncoder（无显式相对位置编码）"""

    def __init__(
        self,
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = True,
        output_activate_function="ReLU",
        causal: bool = False,
    ):
        super().__init__()

        if use_posi:
            if RotaryEmbedding is None:
                raise ImportError(
                    "rotary_embedding_torch is required for RoPE but is not installed."
                )
            dim_head = attention_in // num_head
            self.rotary_embed = RotaryEmbedding(dim=dim_head)
        else:
            self.rotary_embed = None

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    in_size=attention_in,
                    ffn_hidden=ffn_hidden,
                    num_head=num_head,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    rotary_embed=self.rotary_embed,
                    causal=causal,
                )
                for _ in range(num_layer)
            ]
        )

        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            elif output_activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {output_activate_function}"
                )
        else:
            self.activate_function = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.activate_function is not None:
            x = self.activate_function(x)
        return x


class Model(BaseModel):
    """Fbank-Conformer 说话人分离模型（RoPE + FlashAttention 版本）

    与原 `model_fbank_conformer.Model` 接口保持一致，仅内部编码器替换为
    `ConformerEncoderRoPE`。
    """

    def __init__(
        self,
        n_fft: int = 400,
        n_mels: int = 80,
        win_length: int = 25,  # ms
        hop_length: int = 10,  # ms
        sample_rate: int = 16000,
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = True,
        output_activate_function=False,
        max_speakers_per_chunk: int = 4,
        max_speakers_per_frame: int = 2,
        use_powerset: bool = True,
        chunk_size: int = 5,
        num_channels: int = 8,
        selected_channel: int = 0,
        fbank_requires_grad: bool = False,
        fbank_param_change_factor: float = 1.0,
        fbank_param_rand_factor: float = 0.0,
        # 额外：是否使用因果注意力
        causal: bool = False,
    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame,
            use_powerset=use_powerset,
        )

        self.chunk_size = chunk_size
        self.selected_channel = selected_channel

        self.n_fft = n_fft
        self.win_length_samples = win_length * sample_rate // 1000
        self.hop_length_samples = hop_length * sample_rate // 1000

        self.make_feats = Fbank(
            n_fft=n_fft,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length,
            requires_grad=fbank_requires_grad,
            param_change_factor=fbank_param_change_factor,
            param_rand_factor=fbank_param_rand_factor,
        )

        self.proj = nn.Linear(n_mels, attention_in)
        self.lnorm = nn.LayerNorm(attention_in)

        self.conformer = ConformerEncoderRoPE(
            attention_in=attention_in,
            ffn_hidden=ffn_hidden,
            num_head=num_head,
            num_layer=num_layer,
            kernel_size=kernel_size,
            dropout=dropout,
            use_posi=use_posi,
            output_activate_function=output_activate_function,
            causal=causal,
        )

        self.classifier = nn.Linear(attention_in, self.dimension)
        self.activation = self.default_activation()

    @property
    def dimension(self) -> int:
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")
        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int, center: bool = True) -> int:
        if center:
            return 1 + num_samples // self.hop_length_samples
        return 1 + (num_samples - self.n_fft) // self.hop_length_samples

    def receptive_field_size(self, num_frames: int = 1) -> int:
        return self.n_fft + (num_frames - 1) * self.hop_length_samples

    def receptive_field_center(self, frame: int = 0, center: bool = True) -> int:
        if center:
            return frame * self.hop_length_samples
        return frame * self.hop_length_samples + self.n_fft // 2

    @property
    def get_rf_info(self, sample_rate=16000):
        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        num_frames = self.num_frames(self.chunk_size * sample_rate)
        duration = receptive_field_size / sample_rate
        step = receptive_field_step / sample_rate
        return num_frames, duration, step

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        assert waveforms.dim() == 3
        waveforms = waveforms[:, self.selected_channel, :]

        feats = self.make_feats(waveforms)
        x = self.proj(feats)
        x = self.lnorm(x)
        x = self.conformer(x)
        x = self.classifier(x)
        x = self.activation(x)
        return x


__all__ = ["Model", "ConformerEncoderRoPE"]



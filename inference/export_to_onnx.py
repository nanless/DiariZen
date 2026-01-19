#!/usr/bin/env python3
"""
Export DiariZen segmentation model checkpoint to ONNX.

This exporter creates a small wrapper that:
1) runs the original model (powerset log_softmax output)
2) decodes powerset -> multilabel with hard argmax

So ONNX output is (batch, frames, num_speakers) with values in {0,1}.

Run with:
conda run --no-capture-output -n diarizen python inference/export_to_onnx.py ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import toml
import torch
import torch.nn as nn

from diarizen.utils import instantiate


class PowersetToMultilabelHard(nn.Module):
    def __init__(self, base_model: nn.Module, quantize_logprobs: float = 0.0):
        super().__init__()
        self.base_model = base_model
        assert getattr(base_model.specifications, "powerset", False), "expected powerset model"
        mapping = base_model.powerset.mapping  # (num_powerset_classes, num_speakers)
        self.register_buffer("mapping", mapping.float(), persistent=True)
        self.quantize_logprobs = float(quantize_logprobs)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # base_model output: log_softmax over powerset classes
        powerset_logprobs = self.base_model(waveforms)  # (B, T, P)
        # Optional: quantize to reduce argmax flip due to tiny numerical differences across runtimes.
        # This slightly changes behavior, but helps achieve hard 0/1 parity when desired.
        if self.quantize_logprobs and self.quantize_logprobs > 0:
            q = self.quantize_logprobs
            powerset_logprobs = torch.round(powerset_logprobs / q) * q
        idx = torch.argmax(powerset_logprobs, dim=-1)  # (B, T)
        # gather mapping rows -> (B, T, S)
        return self.mapping[idx]


def main():
    parser = argparse.ArgumentParser("Export DiariZen checkpoint to ONNX")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment dir containing checkpoints/")
    parser.add_argument("--config", type=str, required=True, help="Path to resolved config__*.toml")
    parser.add_argument("--ckpt-name", type=str, required=True, help="Checkpoint subdir name, e.g. best or epoch_0010")
    parser.add_argument("--out-onnx", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Export device")
    parser.add_argument("--dummy-seconds", type=float, default=8.0, help="Dummy waveform length (seconds) for tracing")
    parser.add_argument(
        "--quantize-logprobs",
        type=float,
        default=0.0,
        help="If >0, quantize powerset logprobs by this step before argmax to improve hard parity.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    config_path = Path(args.config)
    ckpt_dir = exp_dir / "checkpoints" / args.ckpt_name
    ckpt_path = ckpt_dir / "pytorch_model.bin"
    out_onnx = Path(args.out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    assert exp_dir.is_dir(), f"exp-dir not found: {exp_dir}"
    assert config_path.is_file(), f"config not found: {config_path}"
    assert ckpt_path.is_file(), f"checkpoint not found: {ckpt_path}"

    config = toml.load(config_path)
    model_args = config["model"]["args"].copy()
    model = instantiate(config["model"]["path"], args=model_args)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device(args.device)
    model = model.to(device)
    wrapped = PowersetToMultilabelHard(model, quantize_logprobs=args.quantize_logprobs).to(device).eval()

    sample_rate = int(model_args.get("sample_rate", 16000))
    dummy_len = int(round(args.dummy_seconds * sample_rate))
    dummy = torch.zeros(1, 1, dummy_len, dtype=torch.float32, device=device)

    torch.onnx.export(
        wrapped,
        dummy,
        out_onnx.as_posix(),
        input_names=["waveforms"],
        output_names=["multilabel"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "waveforms": {0: "batch", 2: "samples"},
            "multilabel": {0: "batch", 1: "frames"},
        },
    )

    print(f"Exported: {out_onnx}")


if __name__ == "__main__":
    main()


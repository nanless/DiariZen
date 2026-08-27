#!/usr/bin/env python3
"""Generate a 10-second synthetic waveform that activates a target powerset class."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from benchmark_segmentation_precision import load_pytorch_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-npy", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--latent-samples", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    args = parser.parse_args()

    torch.manual_seed(20_260_826)
    device = torch.device("cuda")
    wrapped = load_pytorch_model(args.config, args.ckpt, device, torch.float32)
    model = wrapped.base_model
    mapping = model.powerset.mapping
    target_candidates = torch.nonzero(mapping.sum(dim=1) == 1, as_tuple=False).flatten()
    if target_candidates.numel() == 0:
        raise RuntimeError("No single-speaker powerset class found")
    target_class = int(target_candidates[0].item())
    latent = torch.nn.Parameter(torch.randn(1, 1, args.latent_samples, device=device) * 0.03)
    optimizer = torch.optim.Adam([latent], lr=args.learning_rate)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        waveform = functional.interpolate(latent, size=160_000, mode="linear", align_corners=False)
        waveform = torch.tanh(waveform)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            scores = model(waveform)
            target_score = scores[..., target_class].mean()
            competing_score = torch.logsumexp(
                torch.cat([scores[..., :target_class], scores[..., target_class + 1 :]], dim=-1),
                dim=-1,
            ).mean()
            loss = competing_score - target_score + 0.001 * waveform.square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predicted = scores.argmax(dim=-1)
            target_pct = float((predicted == target_class).float().mean() * 100.0)
        print(
            f"step={step + 1} loss={float(loss):.6f} target_class={target_class} "
            f"target_frames_pct={target_pct:.2f}",
            flush=True,
        )
        if target_pct >= 95.0:
            break

    with torch.no_grad():
        waveform = functional.interpolate(latent, size=160_000, mode="linear", align_corners=False)
        waveform = torch.tanh(waveform).clamp(-1.0, 1.0)
    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, waveform.cpu().numpy().astype(np.float32))
    print(f"saved={args.out_npy} min={waveform.min().item():.5f} max={waveform.max().item():.5f}")


if __name__ == "__main__":
    main()

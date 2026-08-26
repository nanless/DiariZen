#!/usr/bin/env python3
"""Static INT8 (QDQ) quantization for DiariZen segmentation ONNX.

Calibration uses raw waveforms (same input as inference). Default output::
  <fp32_stem>.static-int8.onnx

Example::

  conda run --no-capture-output -n diarizen python inference/quantize_segmentation_onnx_static.py \\
    --input inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx \\
    --calibration-root example \\
    --calibration-root /path/to/test_audios/bench_largescale_20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import onnxruntime as ort

from inference.utils import list_audio_files, load_audio_mono_16k


class _WaveformCalibrationDataReader:
    def __init__(self, waveforms: List[np.ndarray], input_name: str):
        self.waveforms = waveforms
        self.input_name = input_name
        self.index = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.index >= len(self.waveforms):
            return None
        item = {self.input_name: self.waveforms[self.index]}
        self.index += 1
        return item

    def rewind(self) -> None:
        self.index = 0


def collect_calibration_waveforms(
    roots: Iterable[str | Path],
    *,
    max_files: int = 32,
    sample_rate: int = 16000,
) -> tuple[List[np.ndarray], Dict[str, int]]:
    waveforms: List[np.ndarray] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"calibration root not found: {root}")
        for audio_path in list_audio_files(root.as_posix()):
            mono, _ = load_audio_mono_16k(audio_path, sample_rate=sample_rate)
            waveforms.append(mono[None, ...].astype(np.float32, copy=False))  # (1, 1, S)
            if max_files > 0 and len(waveforms) >= max_files:
                break
        if max_files > 0 and len(waveforms) >= max_files:
            break
    if not waveforms:
        raise RuntimeError("no calibration waveforms collected")
    meta = {
        "num_calibration_samples": len(waveforms),
        "min_samples": int(min(w.shape[-1] for w in waveforms)),
        "max_samples": int(max(w.shape[-1] for w in waveforms)),
    }
    return waveforms, meta


def quantize_segmentation_onnx_static(
    input_path: str | Path,
    output_path: str | Path,
    calibration_waveforms: List[np.ndarray],
    *,
    quant_format: str = "QDQ",
    activation_type: str = "QUInt8",
    weight_type: str = "QInt8",
    calibration_method: str = "MinMax",
    per_channel: bool = True,
    reduce_range: bool = False,
    op_types_to_quantize: Optional[List[str]] = None,
) -> Dict[str, object]:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    format_map = {"qdq": QuantFormat.QDQ, "qoperator": QuantFormat.QOperator}
    activation_map = {"quint8": QuantType.QUInt8, "qint8": QuantType.QInt8}
    weight_map = {"quint8": QuantType.QUInt8, "qint8": QuantType.QInt8}
    calibration_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }

    quant_format_key = quant_format.lower()
    activation_key = activation_type.lower()
    weight_key = weight_type.lower()
    calibration_key = calibration_method.lower()
    if quant_format_key not in format_map:
        raise ValueError(f"Unsupported quant format: {quant_format}")
    if activation_key not in activation_map:
        raise ValueError(f"Unsupported activation type: {activation_type}")
    if weight_key not in weight_map:
        raise ValueError(f"Unsupported weight type: {weight_type}")
    if calibration_key not in calibration_map:
        raise ValueError(f"Unsupported calibration method: {calibration_method}")

    input_name = ort.InferenceSession(
        input_path.as_posix(), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name
    reader = _WaveformCalibrationDataReader(calibration_waveforms, input_name=input_name)
    quantize_static(
        input_path.as_posix(),
        output_path.as_posix(),
        reader,
        quant_format=format_map[quant_format_key],
        activation_type=activation_map[activation_key],
        weight_type=weight_map[weight_key],
        calibrate_method=calibration_map[calibration_key],
        per_channel=per_channel,
        reduce_range=reduce_range,
        op_types_to_quantize=op_types_to_quantize,
    )

    return {
        "input_path": input_path.as_posix(),
        "output_path": output_path.as_posix(),
        "size_mb": round(output_path.stat().st_size / 1024 / 1024, 3),
        "quant_format": quant_format_key,
        "activation_type": activation_key,
        "weight_type": weight_key,
        "calibration_method": calibration_key,
        "per_channel": per_channel,
        "reduce_range": reduce_range,
        "quantized_ops": op_types_to_quantize or "all_supported_ops",
        "num_calibration_samples": len(calibration_waveforms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="FP32 segmentation ONNX")
    parser.add_argument("--output", type=str, default="", help="Static INT8 output ONNX")
    parser.add_argument(
        "--calibration-root",
        type=str,
        action="append",
        required=True,
        help="Directory with wav files for calibration (repeatable)",
    )
    parser.add_argument("--max-calibration-files", type=int, default=32)
    parser.add_argument("--quant-format", type=str, default="QDQ", choices=["QDQ", "QOperator", "qdq", "qoperator"])
    parser.add_argument("--activation-type", type=str, default="QUInt8", choices=["QUInt8", "QInt8", "quint8", "qint8"])
    parser.add_argument("--weight-type", type=str, default="QInt8", choices=["QUInt8", "QInt8", "quint8", "qint8"])
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="MinMax",
        choices=["MinMax", "Entropy", "Percentile", "minmax", "entropy", "percentile"],
    )
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reduce-range", action="store_true")
    parser.add_argument(
        "--op-types",
        type=str,
        default="MatMul,Gemm",
        help="Comma-separated op types to quantize. Default: MatMul,Gemm (Conv stays FP32).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}.static-int8.onnx")
    )
    op_types = [s.strip() for s in args.op_types.split(",") if s.strip()] or None

    waveforms, calib_meta = collect_calibration_waveforms(
        args.calibration_root,
        max_files=args.max_calibration_files,
    )
    meta = quantize_segmentation_onnx_static(
        input_path,
        output_path,
        waveforms,
        quant_format=args.quant_format,
        activation_type=args.activation_type,
        weight_type=args.weight_type,
        calibration_method=args.calibration_method,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        op_types_to_quantize=op_types,
    )
    meta.update(calib_meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

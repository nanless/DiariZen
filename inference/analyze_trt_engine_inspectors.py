#!/usr/bin/env python3
"""Aggregate TensorRT engine-inspector precision and reformat evidence."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


FILENAME = re.compile(r"segmentation_10s_bs(?P<batch>\d+)_(?P<precision>.+)\.inspector\.json")
LOW_PRECISION_TOKEN = {
    "bf16": "bfloat16",
    "fp8": "fp8",
    "int8": "int8",
    "int4": "int4",
}


def contains_dtype(layer: dict, token: str) -> bool:
    tensors = layer.get("Inputs", []) + layer.get("Outputs", [])
    return any(token in str(tensor.get("Format/Datatype", "")).lower() for tensor in tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for path in sorted(args.engine_dir.glob("*.inspector.json")):
        match = FILENAME.fullmatch(path.name)
        if not match:
            continue
        precision = match.group("precision")
        layers = json.loads(path.read_text()).get("Layers", [])
        layer_types = Counter(str(layer.get("LayerType", "")) for layer in layers)
        datatypes = Counter()
        for layer in layers:
            for tensor in layer.get("Inputs", []) + layer.get("Outputs", []):
                datatypes[str(tensor.get("Format/Datatype", ""))] += 1

        qdq_layers = [
            layer
            for layer in layers
            if "QuantizeLinear" in str(layer.get("Metadata", ""))
            or "DequantizeLinear" in str(layer.get("Metadata", ""))
        ]
        matmul_layers = [
            layer
            for layer in layers
            if "MatMul" in str(layer.get("Metadata", ""))
            or "Gemm" in str(layer.get("Metadata", ""))
        ]
        token = LOW_PRECISION_TOKEN.get(precision)
        low_precision_matmuls = (
            sum(contains_dtype(layer, token) for layer in matmul_layers) if token else 0
        )
        total = len(layers)
        result = {
            "precision": precision,
            "batch_size": int(match.group("batch")),
            "inspector": str(path),
            "total_layers": total,
            "layer_type_counts": dict(layer_types),
            "datatype_occurrences": dict(datatypes),
            "qdq_bearing_layers": len(qdq_layers),
            "qdq_layer_pct": round(len(qdq_layers) * 100.0 / total, 3) if total else 0.0,
            "reformat_layers": layer_types.get("Reformat", 0),
            "reformat_layer_pct": round(layer_types.get("Reformat", 0) * 100.0 / total, 3)
            if total
            else 0.0,
            "matmul_gemm_layers": len(matmul_layers),
            "low_precision_matmul_gemm_layers": low_precision_matmuls,
            "low_precision_matmul_gemm_hit_pct": round(
                low_precision_matmuls * 100.0 / len(matmul_layers), 3
            )
            if matmul_layers
            else 0.0,
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    report = {"engine_dir": str(args.engine_dir), "inspector_count": len(results), "results": results}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Minimal TensorRT builder diagnostic, intentionally independent of PyTorch."""

from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path

import tensorrt as trt


faulthandler.enable()


def log(message: str) -> None:
    print(message, flush=True)


def make_config(builder: trt.Builder, fp16: bool) -> trt.IBuilderConfig:
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.builder_optimization_level = 0
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    return config


def build_trivial(logger: trt.Logger) -> None:
    log("trivial: create builder")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    input_tensor = network.add_input("input", trt.float32, (1, 16))
    output_tensor = network.add_identity(input_tensor).get_output(0)
    output_tensor.name = "output"
    network.mark_output(output_tensor)
    log("trivial: build start")
    serialized = builder.build_serialized_network(network, make_config(builder, True))
    if serialized is None:
        raise RuntimeError("trivial build returned None")
    log(f"trivial: build ok ({len(bytes(serialized))} bytes)")


def build_onnx(logger: trt.Logger, onnx_path: Path, fp16: bool) -> None:
    log("onnx: create builder/network/parser")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    log("onnx: parse start")
    parsed = parser.parse(onnx_path.read_bytes())
    log(f"onnx: parse done parsed={parsed} layers={network.num_layers} errors={parser.num_errors}")
    if not parsed:
        for i in range(parser.num_errors):
            log(str(parser.get_error(i)))
        raise RuntimeError("parse failed")
    profile = builder.create_optimization_profile()
    name = network.get_input(0).name
    profile.set_shape(name, (1, 1, 160_000), (1, 1, 160_000), (1, 1, 160_000))
    config = make_config(builder, fp16)
    config.add_optimization_profile(profile)
    log("onnx: build start")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("ONNX build returned None")
    log(f"onnx: build ok ({len(bytes(serialized))} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip-trivial", action="store_true")
    args = parser.parse_args()
    log(f"TensorRT {trt.__version__}")
    logger = trt.Logger(trt.Logger.INFO)
    if not args.skip_trivial:
        build_trivial(logger)
    if args.onnx:
        build_onnx(logger, args.onnx, args.fp16)


if __name__ == "__main__":
    main()

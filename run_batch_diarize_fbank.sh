#!/usr/bin/env bash

# 运行 Fbank-Conformer 模型推理的脚本
# 自动寻找 recipes/diar_ssl/exp/kaldi_merged_1205_1207_fbank_conformer 下最新的 config 和 checkpoint

set -eo pipefail

# 基础路径配置
REPO_DIR="$(cd -- "$(dirname "$0")" && pwd)"
EXP_NAME="kaldi_merged_1205_1207_fbank_conformer"
EXP_DIR="$REPO_DIR/recipes/diar_ssl/exp/$EXP_NAME"

# 输入输出配置（可根据需要修改或通过环境变量传入）
IN_ROOT="${IN_ROOT:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios}"
OUT_BASE_DIR="${OUT_BASE_DIR:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios_Diarizen_fbank_conformer}"
DEVICE="${DEVICE:-auto}"

# 激活环境
set +u
source /root/miniforge3/bin/activate diarizen
set -u

echo "=========================================="
echo "Fbank-Conformer 推理工作流"
echo "=========================================="

# 1. 自动寻找最新配置
CONFIG_PATH=$(ls -t "$EXP_DIR"/config__*.toml 2>/dev/null | head -n 1)
if [[ -z "$CONFIG_PATH" ]]; then
    # 如果 exp 下没有，尝试使用 conf 目录下的原始配置
    CONFIG_PATH="$REPO_DIR/recipes/diar_ssl/conf/${EXP_NAME}.toml"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "错误: 未找到配置文件" >&2
    exit 1
fi
echo "使用配置: $CONFIG_PATH"

# 2. 自动寻找最新或指定的 checkpoint
# 默认使用 best，如果没有则取最大的 epoch_XXXX
CKPT_DIR="$EXP_DIR/checkpoints/best"
if [[ ! -d "$CKPT_DIR" || ! -f "$CKPT_DIR/pytorch_model.bin" ]]; then
    LATEST_EPOCH=$(ls -1 "$EXP_DIR/checkpoints" | grep -E "^epoch_[0-9]+$" | sort -V | tail -n 1)
    if [[ -n "$LATEST_EPOCH" ]]; then
        CKPT_DIR="$EXP_DIR/checkpoints/$LATEST_EPOCH"
    else
        echo "错误: 未找到任何 checkpoint" >&2
        exit 1
    fi
fi
echo "使用模型: $CKPT_DIR"

# 3. 设置推理参数
# 如果设置了 FULL_UTTERANCE=true，则使用整句推理，否则默认 8s 窗口
FULL_UTTERANCE="${FULL_UTTERANCE:-true}"
OUT_DIR_SUFFIX="_$(basename "$CKPT_DIR")"

if [[ "$FULL_UTTERANCE" == "true" ]]; then
    EXTRA_ARGS="--full-utterance"
    OUT_DIR="${OUT_BASE_DIR}_full${OUT_DIR_SUFFIX}"
    echo "⏱️ 推理模式: 整句推理 (Full Utterance)"
else
    SEG_DURATION="${SEG_DURATION:-8.0}"
    EXTRA_ARGS="--seg-duration $SEG_DURATION"
    OUT_DIR="${OUT_BASE_DIR}_seg${SEG_DURATION}${OUT_DIR_SUFFIX}"
    echo "⏱️ 推理模式: 滑动窗口 (${SEG_DURATION}s)"
fi

echo "📂 输入目录: $IN_ROOT"
echo "📂 输出目录: $OUT_DIR"
echo "=========================================="

# 4. 执行推理
python "$REPO_DIR/batch_diarize_finetuned.py" \
    "$IN_ROOT" \
    --out-dir "$OUT_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --config "$CONFIG_PATH" \
    --device "$DEVICE" \
    $EXTRA_ARGS \
    --segmentation-only \
    --plot

echo ""
echo "推理完成！结果保存在: $OUT_DIR"
echo "总结文件: $OUT_DIR/summary.json"


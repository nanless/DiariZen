#!/usr/bin/env bash

# 使用指定实验目录下「最新」config toml 与「最新」epoch checkpoint 运行 simple_diarize.py
# （按文件名排序取最新 toml；按 epoch 编号取最大且含 pytorch_model.bin 的目录）

set -eo pipefail

# 脚本所在目录
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

# 设置 PYTHONPATH
export PYTHONPATH="$REPO_DIR/pyannote-audio:$REPO_DIR:$PYTHONPATH"

# 配置参数
EXP_NAME="${EXP_NAME:-generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large}"
EXP_DIR="$REPO_DIR/recipes/diar_ssl/exp/$EXP_NAME"
CHECKPOINTS_DIR="$EXP_DIR/checkpoints"

# 最新 toml（config__*.toml 按版本排序取最后一个）
if [[ -n "${CONFIG_PATH:-}" ]]; then
  :
else
  CONFIG_PATH=$(ls -1 "$EXP_DIR"/config__*.toml 2>/dev/null | sort -V | tail -n1) || true
fi

# 输入和输出目录（新目录名区分本次实验与「latest」单次推理）
IN_ROOT="${IN_ROOT:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios}"
OUT_BASE_DIR="${OUT_BASE_DIR:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios_Diarizen_simple_0317_longaudio_vad_label_all_ft_large_latest}"

# 并发进程数（CPU 推理时可适当提高；CUDA 下建议保持 1）
NUM_WORKERS="${NUM_WORKERS:-16}"

echo "=========================================="
echo "开始运行 simple_diarize.py（最新 checkpoint + 最新 toml）"
echo "=========================================="
echo "实验名称: $EXP_NAME"
echo "输入目录: $IN_ROOT"
echo "输出目录: $OUT_BASE_DIR"
echo "Config: $CONFIG_PATH"
echo "=========================================="

# 检查必要文件
if [[ ! -d "$IN_ROOT" ]]; then
    echo "错误: 输入目录不存在: $IN_ROOT" >&2
    exit 1
fi

if [[ ! -d "$CHECKPOINTS_DIR" ]]; then
    echo "错误: checkpoint 目录不存在: $CHECKPOINTS_DIR" >&2
    exit 1
fi

if [[ -z "$CONFIG_PATH" || ! -f "$CONFIG_PATH" ]]; then
    echo "错误: 未找到或无效的 config toml: ${CONFIG_PATH:-}" >&2
    exit 1
fi

LATEST_EPOCH=$(ls -1 "$CHECKPOINTS_DIR" | grep -E '^epoch_[0-9]+$' | sort -V | tail -n1)
if [[ -z "$LATEST_EPOCH" ]]; then
    echo "错误: 未找到 epoch_* checkpoint 目录" >&2
    exit 1
fi

ckpt_dir="$CHECKPOINTS_DIR/$LATEST_EPOCH"
if [[ ! -f "$ckpt_dir/pytorch_model.bin" ]]; then
    echo "错误: 最新 epoch 缺少 pytorch_model.bin: $ckpt_dir" >&2
    exit 1
fi

echo "使用 checkpoint: $ckpt_dir"

mkdir -p "$OUT_BASE_DIR"

conda run -n diarizen python "$REPO_DIR/simple_diarize.py" \
    "$IN_ROOT" \
    --ckpt-dir "$ckpt_dir" \
    --config "$CONFIG_PATH" \
    --out-dir "$OUT_BASE_DIR" \
    --device cpu \
    --num-workers "$NUM_WORKERS"

if [[ $? -eq 0 ]]; then
    echo "✓ 处理完成"
else
    echo "✗ 处理失败" >&2
    exit 1
fi

echo ""
echo "=========================================="
echo "任务已完成！"
echo "=========================================="

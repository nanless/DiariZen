#!/usr/bin/env bash

# 针对特定实验运行所有 epoch 的 simple_diarize.py 脚本（用于 next 批次数据）
#
# 默认实验:
#   kaldi_merged_1219_all_ft_large
# 默认输入:
#   /root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios
# 默认输出基础目录:
#   /root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios_Diarizen_simple_large_1219_all
#
# 可通过环境变量覆盖：
#   EXP_NAME, IN_ROOT, OUT_BASE_DIR, CONFIG_PATH, NUM_WORKERS, DEVICE

set -eo pipefail

# 脚本所在目录
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

# 设置 PYTHONPATH
export PYTHONPATH="$REPO_DIR/pyannote-audio:$REPO_DIR:$PYTHONPATH"

# 配置参数
# EXP_NAME="${EXP_NAME:-kaldi_merged_1219_all_ft_large}"
EXP_NAME="kaldi_merged_1219_all_ft_base"
EXP_DIR="$REPO_DIR/recipes/diar_ssl/exp/$EXP_NAME"
CHECKPOINTS_DIR="$EXP_DIR/checkpoints"
# CONFIG_PATH="${CONFIG_PATH:-$EXP_DIR/config__2026_01_20--18_03_58.toml}"
CONFIG_PATH="${CONFIG_PATH:-$EXP_DIR/config__2026_01_09--00_41_09.toml}"

# 输入和输出目录（根据需要修改或通过环境变量传入）
IN_ROOT="${IN_ROOT:-/root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios}"
# OUT_BASE_DIR="${OUT_BASE_DIR:-/root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios_Diarizen_simple_large_1219_all}"
OUT_BASE_DIR="${OUT_BASE_DIR:-/root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios_Diarizen_simple_base_1219_all}"

# 并发进程数（CPU 推理时可适当提高；CUDA 下建议保持 1）
NUM_WORKERS="${NUM_WORKERS:-16}"
DEVICE="${DEVICE:-cpu}"

echo "=========================================="
echo "开始批量运行 simple_diarize.py (for next)"
echo "=========================================="
echo "实验名称: $EXP_NAME"
echo "输入目录: $IN_ROOT"
echo "输出基础目录: $OUT_BASE_DIR"
echo "Config: $CONFIG_PATH"
echo "device: $DEVICE"
echo "num_workers: $NUM_WORKERS"
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

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "错误: config 文件不存在: $CONFIG_PATH" >&2
  exit 1
fi

# 获取所有 epoch 目录（按数字排序）以及 best 目录
EPOCHS=$(ls -1 "$CHECKPOINTS_DIR" | grep -E "^(epoch_[0-9]+|best)$" | sort -V)
if [[ -z "$EPOCHS" ]]; then
  echo "错误: 未找到任何 epoch checkpoint 目录" >&2
  exit 1
fi

# 遍历每个 epoch
for epoch in $EPOCHS; do
  ckpt_dir="$CHECKPOINTS_DIR/$epoch"
  out_dir="$OUT_BASE_DIR/$epoch"

  # 检查 checkpoint 是否完整
  if [[ ! -f "$ckpt_dir/pytorch_model.bin" ]]; then
    echo "跳过 $epoch: 缺少 pytorch_model.bin"
    continue
  fi

  echo ""
  echo ">>> 处理 $epoch ..."
  echo ">>> 输出目录: $out_dir"

  mkdir -p "$out_dir"

  conda run -n diarizen python "$REPO_DIR/simple_diarize.py" \
    "$IN_ROOT" \
    --ckpt-dir "$ckpt_dir" \
    --config "$CONFIG_PATH" \
    --out-dir "$out_dir" \
    --device "$DEVICE" \
    --num-workers "$NUM_WORKERS"

  if [[ $? -eq 0 ]]; then
    echo "✓ $epoch 处理完成"
  else
    echo "✗ $epoch 处理失败"
  fi
done

echo ""
echo "=========================================="
echo "所有任务已完成！"
echo "=========================================="


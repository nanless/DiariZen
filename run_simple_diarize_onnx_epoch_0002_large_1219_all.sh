#!/usr/bin/env bash

# 使用 inference/simple_diarize_onnx.py 跑 ONNX 推理（默认 epoch_0016, large_1219_all）
#
# 默认输入:
#   /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios
# 默认输出:
#   /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios_Diarizen_simple_large_1219_all/epoch_0016_onnx
#
# 可通过环境变量覆盖：
#   IN_ROOT, OUT_DIR, CKPT_NAME, MAX_FILES, CONFIG_PATH

set -eo pipefail

# 脚本所在目录
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

# 设置 PYTHONPATH（保持与其它脚本一致）
export PYTHONPATH="$REPO_DIR/pyannote-audio:$REPO_DIR:$PYTHONPATH"

# 模型与输出
EXP_NAME="kaldi_merged_1219_all_ft_large"
CKPT_NAME="${CKPT_NAME:-epoch_0016}"
ONNX_PATH="$REPO_DIR/inference/models/$EXP_NAME/${CKPT_NAME}_multilabel_hard.onnx"
EXP_DIR="$REPO_DIR/recipes/diar_ssl/exp/$EXP_NAME"
CONFIG_PATH="${CONFIG_PATH:-$EXP_DIR/config__2025_12_26--11_44_15.toml}"

# 输入和输出目录（按需求默认到你指定的路径）
IN_ROOT="${IN_ROOT:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios}"
OUT_DIR="${OUT_DIR:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios_Diarizen_simple_large_1219_all/${CKPT_NAME}_onnx}"

MAX_FILES="${MAX_FILES:-0}"

echo "=========================================="
echo "开始运行 simple_diarize_onnx.py (${CKPT_NAME}, large_1219_all)"
echo "=========================================="
echo "实验名称: $EXP_NAME"
echo "ckpt: $CKPT_NAME"
echo "输入目录: $IN_ROOT"
echo "输出目录: $OUT_DIR"
echo "ONNX: $ONNX_PATH"
echo "providers: cpu"
echo "max_files: $MAX_FILES"
echo "=========================================="

# 检查必要文件
if [[ ! -d "$IN_ROOT" ]]; then
  echo "错误: 输入目录不存在: $IN_ROOT" >&2
  exit 1
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ONNX 不存在，开始导出: $ONNX_PATH"
  mkdir -p "$(dirname "$ONNX_PATH")"
  conda run --no-capture-output -n diarizen python "$REPO_DIR/inference/export_to_onnx.py" \
    --exp-dir "$EXP_DIR" \
    --config "$CONFIG_PATH" \
    --ckpt-name "$CKPT_NAME" \
    --out-onnx "$ONNX_PATH"
fi

mkdir -p "$OUT_DIR"

# Fix common Pillow/libstdc++ mismatch (GLIBCXX_3.4.29 not found) by preloading conda's libstdc++.
# This makes matplotlib/PIL PNG saving work on older system images.
DIARIZEN_PREFIX="$(conda run -n diarizen python -c "import sys; print(sys.prefix)")"
if [[ -f "$DIARIZEN_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$DIARIZEN_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  export LD_LIBRARY_PATH="$DIARIZEN_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# 使用 conda run 执行 ONNX 推理
conda run --no-capture-output -n diarizen python "$REPO_DIR/inference/simple_diarize_onnx.py" \
  "$IN_ROOT" \
  --onnx "$ONNX_PATH" \
  --out-dir "$OUT_DIR" \
  --providers cpu \
  --max-files "$MAX_FILES"

echo "=========================================="
echo "完成！输出目录: $OUT_DIR"
echo "=========================================="

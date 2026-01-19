#!/usr/bin/env bash

# 使用 inference/simple_diarize_onnx.py 跑 ONNX 推理（固定 epoch_0002, large_1219_all）
#
# 默认输入:
#   /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios
# 默认输出:
#   /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios_Diarizen_simple_large_1219_all/epoch_0002_onnx
#
# 可通过环境变量覆盖：
#   IN_ROOT, OUT_DIR, PROVIDERS, MAX_FILES

set -eo pipefail

# 脚本所在目录
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

# 设置 PYTHONPATH（保持与其它脚本一致）
export PYTHONPATH="$REPO_DIR/pyannote-audio:$REPO_DIR:$PYTHONPATH"

# 模型与输出
EXP_NAME="kaldi_merged_1219_all_ft_large"
ONNX_PATH="$REPO_DIR/inference/models/$EXP_NAME/epoch_0002_multilabel_hard.onnx"

# 输入和输出目录（按需求默认到你指定的路径）
IN_ROOT="${IN_ROOT:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios}"
OUT_DIR="${OUT_DIR:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios_Diarizen_simple_large_1219_all/epoch_0002_onnx}"

# ONNXRuntime providers（默认 cpu；如果你的环境支持 CUDAExecutionProvider，可设为 cuda）
PROVIDERS="${PROVIDERS:-cpu}"
MAX_FILES="${MAX_FILES:-0}"

echo "=========================================="
echo "开始运行 simple_diarize_onnx.py (epoch_0002, large_1219_all)"
echo "=========================================="
echo "实验名称: $EXP_NAME"
echo "输入目录: $IN_ROOT"
echo "输出目录: $OUT_DIR"
echo "ONNX: $ONNX_PATH"
echo "providers: $PROVIDERS"
echo "max_files: $MAX_FILES"
echo "=========================================="

# 检查必要文件
if [[ ! -d "$IN_ROOT" ]]; then
  echo "错误: 输入目录不存在: $IN_ROOT" >&2
  exit 1
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "错误: ONNX 文件不存在: $ONNX_PATH" >&2
  echo "提示: 先运行 inference/run_export_kaldi_merged_1219_all_ft_large_epoch_0002.sh" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# 使用 conda run 执行 ONNX 推理
conda run --no-capture-output -n diarizen python "$REPO_DIR/inference/simple_diarize_onnx.py" \
  "$IN_ROOT" \
  --onnx "$ONNX_PATH" \
  --out-dir "$OUT_DIR" \
  --providers "$PROVIDERS" \
  --max-files "$MAX_FILES"

echo "=========================================="
echo "完成！输出目录: $OUT_DIR"
echo "=========================================="


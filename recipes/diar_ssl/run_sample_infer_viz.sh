#!/usr/bin/env bash

# 一键运行 sample_infer_viz.py，对比基线 vs 微调，无需额外手动参数。
# 默认使用 kaldi_merged_1205_1207_ft 的训练/标注与最新 checkpoint/config。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RECIPE_DIR="$REPO_DIR/recipes/diar_ssl"

# 可按需覆盖的参数（通常保持默认即可）
EXP_NAME="${EXP_NAME:-kaldi_merged_1205_1207_ft}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
DEVICE="${DEVICE:-auto}"  # auto | cpu | cuda/gpu
OUT_DIR="${OUT_DIR:-$RECIPE_DIR/exp/$EXP_NAME/sample_viz}"
# 若需要指定不同的基线模型或本地基线 checkpoint，可设置：
#   BASE_MODEL="BUT-FIT/diarizen-wavlm-large-s80-md"
#   BASE_CKPT_DIR="/path/to/base/checkpoint_dir"
# 默认使用仓库缓存的 base 模型 checkpoint
BASE_MODEL="${BASE_MODEL:-}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-/root/code/github_repos/DiariZen/cache/models--BUT-FIT--diarizen-wavlm-base-s80-md/snapshots/a9857fc34908197fb5336d9d0562f291834a04b2}"

WAV_SCP="$RECIPE_DIR/data/$EXP_NAME/train/wav.scp"
RTTM="$RECIPE_DIR/data/$EXP_NAME/train/rttm"

# 选取最新的 checkpoint 目录（按时间排序）
CKPT_DIR="${CKPT_DIR:-}"
if [[ -z "$CKPT_DIR" ]]; then
    CKPT_DIR="$(ls -1dt "$RECIPE_DIR/exp/$EXP_NAME"/checkpoints/epoch_* 2>/dev/null | head -n1 || true)"
fi

# 选取最新的训练 config（config__*.toml）
CONFIG_PATH="${CONFIG_PATH:-}"
if [[ -z "$CONFIG_PATH" ]]; then
    CONFIG_PATH="$(ls -1t "$RECIPE_DIR/exp/$EXP_NAME"/config__*.toml 2>/dev/null | head -n1 || true)"
fi

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "缺少文件：$1" >&2
        exit 1
    fi
}

require_dir_with_model() {
    if [[ ! -d "$1" || ! -f "$1/pytorch_model.bin" ]]; then
        echo "缺少 checkpoint 目录或 pytorch_model.bin：$1" >&2
        exit 1
    fi
}

require_file "$WAV_SCP"
require_file "$RTTM"
require_file "$CONFIG_PATH"
require_dir_with_model "$CKPT_DIR"

mkdir -p "$OUT_DIR"

CMD=(python "$RECIPE_DIR/sample_infer_viz.py"
    --wav-scp "$WAV_SCP"
    --rttm "$RTTM"
    --ckpt-dir "$CKPT_DIR"
    --config "$CONFIG_PATH"
    --out-dir "$OUT_DIR"
    --num-samples "$NUM_SAMPLES"
    --device "$DEVICE"
)

# 若提供基线模型或基线 checkpoint，按需追加
if [[ -n "$BASE_MODEL" ]]; then
    CMD+=(--base-model "$BASE_MODEL")
fi
if [[ -n "$BASE_CKPT_DIR" ]]; then
    CMD+=(--base-ckpt-dir "$BASE_CKPT_DIR")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


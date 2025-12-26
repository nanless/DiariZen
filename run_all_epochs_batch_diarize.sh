#!/usr/bin/env bash

# 遍历所有 epoch checkpoint，批量运行 batch_diarize_finetuned.py
# 使用 diarizen conda 环境

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

# 激活 diarizen conda 环境（临时禁用 -u 以避免 conda 激活脚本中的未绑定变量错误）
set +u
source /root/miniforge3/bin/activate diarizen
set -u

# 配置参数（可根据需要修改）
IN_ROOT="${IN_ROOT:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios}"
EXP_NAME="${EXP_NAME:-kaldi_merged_1219_all_temp_ft_large}"
CHECKPOINTS_DIR="$REPO_DIR/recipes/diar_ssl/exp/$EXP_NAME/checkpoints"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/recipes/diar_ssl/exp/$EXP_NAME/config__2025_12_24--02_01_24.toml}"
OUT_BASE_DIR="${OUT_BASE_DIR:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios_Diarizen_simufinetune_1219_all_temp_large}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/cache}"
DEVICE="${DEVICE:-auto}"

# 其他可选参数
SEGMENTATION_ONLY="${SEGMENTATION_ONLY:-true}"
FULL_UTTERANCE="${FULL_UTTERANCE:-true}"
BINARIZE_ONSET="${BINARIZE_ONSET:-0.5}"
BINARIZE_OFFSET="${BINARIZE_OFFSET:-}"
BINARIZE_MIN_DURATION_ON="${BINARIZE_MIN_DURATION_ON:-0.0}"
BINARIZE_MIN_DURATION_OFF="${BINARIZE_MIN_DURATION_OFF:-0.0}"
PLOT="${PLOT:-true}"

# 根据 FULL_UTTERANCE 设置额外的推理参数
EXTRA_INFER_ARGS=""
if [[ "$FULL_UTTERANCE" == "true" ]]; then
    EXTRA_INFER_ARGS="--full-utterance"
    echo "使用整句推理模式 (Full Utterance)"
else
    SEG_DURATION="${SEG_DURATION:-16.0}" # 默认 16s 窗口
    EXTRA_INFER_ARGS="--seg-duration $SEG_DURATION"
    echo "使用滑动窗口推理模式 (${SEG_DURATION}s)"
fi

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

# 获取所有 epoch 目录（按数字排序，排除 best）
EPOCHS=($(ls -1 "$CHECKPOINTS_DIR" | grep -E "^epoch_[0-9]+$" | sort -V))

if [[ ${#EPOCHS[@]} -eq 0 ]]; then
    echo "错误: 未找到任何 epoch checkpoint 目录" >&2
    exit 1
fi

echo "=========================================="
echo "批量运行 batch_diarize_finetuned.py"
echo "=========================================="
echo "输入目录: $IN_ROOT"
echo "输出基础目录: $OUT_BASE_DIR"
echo "Checkpoint 目录: $CHECKPOINTS_DIR"
echo "Config: $CONFIG_PATH"
echo "找到 ${#EPOCHS[@]} 个 epoch: ${EPOCHS[*]}"
echo "=========================================="
echo ""

# 遍历每个 epoch
for epoch in "${EPOCHS[@]}"; do
    ckpt_dir="$CHECKPOINTS_DIR/$epoch"
    out_dir="$OUT_BASE_DIR${epoch}"
    
    # 检查 checkpoint 是否存在
    if [[ ! -d "$ckpt_dir" || ! -f "$ckpt_dir/pytorch_model.bin" ]]; then
        echo "跳过 $epoch: 缺少 pytorch_model.bin"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "处理 epoch: $epoch"
    echo "Checkpoint: $ckpt_dir"
    echo "输出目录: $out_dir"
    echo "=========================================="
    
    # 构建命令
    CMD=(
        python "$REPO_DIR/batch_diarize_finetuned.py"
        "$IN_ROOT"
        --out-dir "$out_dir"
        --ckpt-dir "$ckpt_dir"
        --config "$CONFIG_PATH"
        --cache-dir "$CACHE_DIR"
        --device "$DEVICE"
        $EXTRA_INFER_ARGS
    )
    
    # 添加可选参数
    if [[ "$SEGMENTATION_ONLY" == "true" ]]; then
        CMD+=(--segmentation-only)
    else
        CMD+=(--full-pipeline)
    fi
    
    CMD+=(--binarize-onset "$BINARIZE_ONSET")
    
    if [[ -n "$BINARIZE_OFFSET" ]]; then
        CMD+=(--binarize-offset "$BINARIZE_OFFSET")
    fi
    
    CMD+=(--binarize-min-duration-on "$BINARIZE_MIN_DURATION_ON")
    CMD+=(--binarize-min-duration-off "$BINARIZE_MIN_DURATION_OFF")
    
    if [[ "$PLOT" == "true" ]]; then
        CMD+=(--plot)
    else
        CMD+=(--no-plot)
    fi
    
    echo "运行命令: ${CMD[*]}"
    echo ""
    
    # 运行命令
    if "${CMD[@]}"; then
        echo "✓ 完成 epoch: $epoch"
    else
        echo "✗ 失败 epoch: $epoch (退出码: $?)"
        echo "继续处理下一个 epoch..."
    fi
    
    echo ""
done

echo "=========================================="
echo "所有 epoch 处理完成！"
echo "=========================================="

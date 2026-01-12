#!/usr/bin/env bash

# 针对特定实验运行所有 epoch 的 simple_diarize.py 脚本
# 替换了原有的 batch_diarize_finetuned.py 逻辑

set -eo pipefail

# 脚本所在目录
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

# 设置 PYTHONPATH
export PYTHONPATH="$REPO_DIR/pyannote-audio:$REPO_DIR:$PYTHONPATH"

# 配置参数
EXP_NAME="kaldi_merged_1219_all_ft_base"
EXP_DIR="$REPO_DIR/recipes/diar_ssl/exp/$EXP_NAME"
CHECKPOINTS_DIR="$EXP_DIR/checkpoints"
CONFIG_PATH="$EXP_DIR/config__2026_01_09--00_41_09.toml"

# 输入和输出目录（根据需要修改或通过环境变量传入）
IN_ROOT="${IN_ROOT:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios}"
OUT_BASE_DIR="${OUT_BASE_DIR:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios_Diarizen_simple_base_1219_all}"

echo "=========================================="
echo "开始批量运行 simple_diarize.py"
echo "=========================================="
echo "实验名称: $EXP_NAME"
echo "输入目录: $IN_ROOT"
echo "输出基础目录: $OUT_BASE_DIR"
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
    
    # 使用 conda run 执行 simple_diarize.py
    # 默认使用 GPU (cuda)，如果需要 CPU 可以添加 --device cpu
    conda run -n diarizen python "$REPO_DIR/simple_diarize.py" \
        "$IN_ROOT" \
        --ckpt-dir "$ckpt_dir" \
        --config "$CONFIG_PATH" \
        --out-dir "$out_dir" \
        --device cuda
        
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

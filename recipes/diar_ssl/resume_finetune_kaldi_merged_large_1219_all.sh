#!/usr/bin/env bash

# 用途：从已训练的 checkpoint 恢复训练 diarizen WavLM-Large-Conformer。
# 前提：已有训练好的 checkpoint 在 exp/$EXP_NAME/checkpoints/ 目录下。
# 示例：
#   bash resume_finetune_kaldi_merged_large_1219_all.sh                    # 使用默认 EXP_NAME
#   EXP_NAME=my_exp bash resume_finetune_kaldi_merged_large_1219_all.sh   # 指定实验名
#   EXP_NAME=my_exp NUM_GPUS=2 bash resume_finetune_kaldi_merged_large_1219_all.sh
#   LR_WAVLM=1e-6 LR_HEAD=5e-5 bash resume_finetune_kaldi_merged_large_1219_all.sh  # 指定新学习率
#
# 注意：脚本会自动在 tmux session 中运行，确保断开连接后训练继续。
#       使用 tmux attach -t diarizen_large_ft_resume 查看训练状态。

# 严格模式：命令/管道/未定义变量出错时立即退出，避免隐藏错误
set -euo pipefail

# ====== tmux session 管理 ======
# 自动在 tmux session 中运行，确保断开连接后训练继续
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
TMUX_SESSION_NAME="diarizen_large_ft_resume"
LOG_FILE="$SCRIPT_DIR/resume_finetune_kaldi_merged_large_1219_all.log"

# 检查是否已经在 tmux session 中
if [[ -z "${TMUX:-}" ]]; then
    # 不在 tmux 中，检查 session 是否已存在
    if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
        echo "Found existing tmux session: $TMUX_SESSION_NAME"
        echo "Attaching to existing session. Use 'tmux attach -t $TMUX_SESSION_NAME' to view."
        # 在现有 session 中运行脚本（不重复创建）
        tmux send-keys -t "$TMUX_SESSION_NAME" "cd '$SCRIPT_DIR' && IN_TMUX=1 bash '$0' $*" C-m
        exit 0
    else
        # 创建新的 tmux session 并运行脚本
        echo "Creating new tmux session: $TMUX_SESSION_NAME"
        echo "Use 'tmux attach -t $TMUX_SESSION_NAME' to view training progress."
        tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$SCRIPT_DIR" \
            "IN_TMUX=1 bash '$0' $*"
        exit 0
    fi
fi
# 如果已经在 tmux 中，继续执行下面的代码
# =============================================================

# 解析目录：SCRIPT_DIR 为当前脚本所在目录；REPO_DIR 为仓库根；RECIPE_DIR 为本配方目录
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RECIPE_DIR="$REPO_DIR/recipes/diar_ssl"

# ====== runtime env defaults (can be overridden by env) ======
# MKL 配置避免与 PyTorch 冲突，默认使用 LP64/GNU 线程
: "${MKL_INTERFACE_LAYER:=LP64}"
: "${MKL_THREADING_LAYER:=GNU}"
export MKL_INTERFACE_LAYER MKL_THREADING_LAYER

# 将仓库根目录加入 Python 搜索路径，方便导入本地模块
: "${PYTHONPATH:=$REPO_DIR}"
export PYTHONPATH

# CUDA/多进程与 DataLoader 相关默认值；可用外部环境变量覆盖
: "${CUDA_VISIBLE_DEVICES:=0,1}"
: "${NUM_GPUS:=4}"
: "${TRAIN_NUM_WORKERS:=16}"
: "${DEV_NUM_WORKERS:=16}"
: "${PREFETCH_FACTOR:=4}"
: "${PERSISTENT_WORKERS:=true}"
# =============================================================

# ---- user adjustable knobs --------------------------------------------------
# 数据、模型及训练参数，可按需通过环境变量覆盖，便于快速切换实验。
# 关键说明：
# - EXP_NAME：实验名，必须与之前训练时使用的名称一致，用于定位 checkpoint
# - LR_WAVLM / LR_HEAD：WavLM 主干和头部的学习率，可通过环境变量指定新值
# - 其他参数（NUM_GPUS, PORT等）应与原训练保持一致，或根据当前环境调整
# - RESUME 自动设置为 1，无需手动指定
EXP_NAME="${EXP_NAME:-kaldi_merged_1219_all_ft_large}"
NUM_GPUS="${NUM_GPUS:-$NUM_GPUS}"
PORT="${PORT:-11346}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-$TRAIN_NUM_WORKERS}"
DEV_NUM_WORKERS="${DEV_NUM_WORKERS:-$DEV_NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-$PREFETCH_FACTOR}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-$PERSISTENT_WORKERS}"
CONDA_ENV="${CONDA_ENV:-diarizen}"
# 学习率参数：如果未指定，将从配置文件中读取；如果指定，将更新配置文件
LR_WAVLM="${LR_WAVLM:-5e-6}"
LR_HEAD="${LR_HEAD:-5e-5}"
# 强制设置 RESUME=1
RESUME=1
# -----------------------------------------------------------------------------

# 生成各阶段使用的路径
CONF_OUT="$RECIPE_DIR/conf/${EXP_NAME}.toml"
EXP_DIR="$RECIPE_DIR/exp/$EXP_NAME"
CHECKPOINTS_DIR="$EXP_DIR/checkpoints"

echo "[1/3] Checking checkpoint availability..."
# 检查配置文件是否存在
if [[ ! -f "$CONF_OUT" ]]; then
    echo "Error: Configuration file not found: $CONF_OUT" >&2
    echo "Please ensure EXP_NAME matches the original training experiment name." >&2
    exit 1
fi

# 检查 checkpoint 目录是否存在
if [[ ! -d "$CHECKPOINTS_DIR" ]]; then
    echo "Error: Checkpoints directory not found: $CHECKPOINTS_DIR" >&2
    echo "Please ensure the experiment has been trained before resuming." >&2
    exit 1
fi

# 查找所有 checkpoint（格式：epoch_XXXX）
checkpoints=($(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -name "epoch_[0-9][0-9][0-9][0-9]" | sort))

if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "Error: No checkpoints found in $CHECKPOINTS_DIR" >&2
    echo "Expected checkpoint format: epoch_XXXX (e.g., epoch_0001, epoch_0002)" >&2
    exit 1
fi

# 显示可用的 checkpoint
echo "Found ${#checkpoints[@]} checkpoint(s):"
for ckpt in "${checkpoints[@]}"; do
    ckpt_name=$(basename "$ckpt")
    echo "  - $ckpt_name"
done

latest_ckpt=$(basename "${checkpoints[-1]}")
echo "Will resume from latest checkpoint: $latest_ckpt"

echo "[2/4] Updating learning rates in config (if specified)..."
# 如果指定了新的学习率，更新配置文件
if [[ -n "$LR_WAVLM" ]]; then
    # 更新 optimizer_small 的学习率
    # 匹配 [optimizer_small.args] 部分，直到下一个 [ 开头的行，更新其中的 lr = 行
    sed -i '/^\[optimizer_small\.args\]/,/^\[/ { s/^lr = .*/lr = '"$LR_WAVLM"'/; }' "$CONF_OUT"
    # 验证更新是否成功
    if grep -q "^lr = $LR_WAVLM" "$CONF_OUT" 2>/dev/null; then
        echo "  - Updated LR_WAVLM to: $LR_WAVLM"
    else
        echo "  - Warning: Failed to update optimizer_small.args.lr in config file"
    fi
fi

if [[ -n "$LR_HEAD" ]]; then
    # 更新 optimizer_big 的学习率
    # 匹配 [optimizer_big.args] 部分，直到下一个 [ 开头的行，更新其中的 lr = 行
    sed -i '/^\[optimizer_big\.args\]/,/^\[/ { s/^lr = .*/lr = '"$LR_HEAD"'/; }' "$CONF_OUT"
    # 验证更新是否成功
    if grep -q "^lr = $LR_HEAD" "$CONF_OUT" 2>/dev/null; then
        echo "  - Updated LR_HEAD to: $LR_HEAD"
    else
        echo "  - Warning: Failed to update optimizer_big.args.lr in config file"
    fi
fi

if [[ -z "$LR_WAVLM" && -z "$LR_HEAD" ]]; then
    echo "  - No new learning rates specified, using values from config file"
fi

echo "[3/4] Activating conda environment..."
if command -v conda >/dev/null 2>&1; then
    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    fi
fi

echo "[4/4] Resuming training from checkpoint on $NUM_GPUS GPU(s)..."
echo "Log file: $LOG_FILE"
echo "Use 'tmux attach -t $TMUX_SESSION_NAME' to view training progress."
cd "$RECIPE_DIR"

# 使用 accelerate 多进程启动训练，从最新 checkpoint 恢复
# - --num_processes 与 NUM_GPUS 对齐
# - --main_process_port 需避免与其他作业冲突
# - -R 标志表示从最新 checkpoint 继续
accelerate launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$PORT" \
    run_dual_opt.py -C "$CONF_OUT" -M train -R 2>&1 | tee -a "$LOG_FILE"

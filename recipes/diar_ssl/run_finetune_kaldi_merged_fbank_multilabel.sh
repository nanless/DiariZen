#!/usr/bin/env bash

# 用途：从已有 checkpoint 继续训练 fbank-conformer 模型（multilabel 模式，调整学习率）
# 前提：已有训练好的 checkpoint 目录
# 示例：
#   bash run_finetune_kaldi_merged_fbank_multilabel.sh                         # 默认配置
#   CKPT_DIR=/path/to/ckpt LR=5e-5 bash run_finetune_kaldi_merged_fbank_multilabel.sh

# 严格模式：命令/管道/未定义变量出错时立即退出
set -euo pipefail

# ====== tmux session 管理 ======
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
TMUX_SESSION_NAME="diarizen_fbank_multilabel_finetune"
LOG_FILE="$SCRIPT_DIR/run_finetune_kaldi_merged_fbank_multilabel.log"

# 检查是否已经在 tmux session 中
if [[ -z "${TMUX:-}" ]]; then
    if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
        echo "Found existing tmux session: $TMUX_SESSION_NAME"
        echo "Attaching to existing session. Use 'tmux attach -t $TMUX_SESSION_NAME' to view."
        tmux send-keys -t "$TMUX_SESSION_NAME" "cd '$SCRIPT_DIR' && IN_TMUX=1 bash '$0' $*" C-m
        exit 0
    else
        echo "Creating new tmux session: $TMUX_SESSION_NAME"
        echo "Use 'tmux attach -t $TMUX_SESSION_NAME' to view training progress."
        tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$SCRIPT_DIR" \
            "IN_TMUX=1 bash '$0' $*"
        exit 0
    fi
fi
# =============================================================

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RECIPE_DIR="$REPO_DIR/recipes/diar_ssl"

# ====== runtime env defaults ======
: "${MKL_INTERFACE_LAYER:=LP64}"
: "${MKL_THREADING_LAYER:=GNU}"
export MKL_INTERFACE_LAYER MKL_THREADING_LAYER

: "${PYTHONPATH:=$REPO_DIR}"
export PYTHONPATH

: "${CUDA_VISIBLE_DEVICES:=0,1}"
: "${NUM_GPUS:=2}"
: "${TRAIN_NUM_WORKERS:=16}"
: "${DEV_NUM_WORKERS:=16}"
: "${PREFETCH_FACTOR:=4}"
: "${PERSISTENT_WORKERS:=true}"
# =============================================================

# ---- user adjustable knobs --------------------------------------------------
# 原实验配置（保持与原训练一致）
EXP_NAME="${EXP_NAME:-kaldi_merged_1205_1207_fbank_conformer_multilabel}"
DATA_OUT="$RECIPE_DIR/data/$EXP_NAME"
# 使用最好的 checkpoint: epoch_0035 (Loss: 0.054, DER: 0.119)
# best 目录是 epoch_0035 的副本（Loss: 0.0537）
# 可以通过 BASE_CKPT_DIR 环境变量覆盖
BASE_CKPT_DIR="${BASE_CKPT_DIR:-$RECIPE_DIR/exp/$EXP_NAME/checkpoints/epoch_0035}"

# 新的 finetune 实验配置
FINETUNE_EXP_NAME="${FINETUNE_EXP_NAME:-${EXP_NAME}_finetune}"
SEED="${SEED:-3407}"
NUM_GPUS="${NUM_GPUS:-$NUM_GPUS}"
PORT="${PORT:-11349}"  # 使用不同端口避免与其他训练冲突
CHUNK_SIZE="${CHUNK_SIZE:-8}"
CHUNK_SHIFT="${CHUNK_SHIFT:-6}"
DEV_CHUNK_SHIFT="${DEV_CHUNK_SHIFT:-8}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-$TRAIN_NUM_WORKERS}"
DEV_NUM_WORKERS="${DEV_NUM_WORKERS:-$DEV_NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-$PREFETCH_FACTOR}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-$PERSISTENT_WORKERS}"
SUBSET_SESSIONS="${SUBSET_SESSIONS:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
LR="${LR:-1e-4}"  # 新的学习率（从 3e-4 降低到 1e-4）
NUM_LAYER="${NUM_LAYER:-8}"
FULL_UTTERANCE="${FULL_UTTERANCE:-1}"
CONDA_ENV="${CONDA_ENV:-diarizen}"
# 不使用 avg_ckpt_num，直接加载指定的 checkpoint 目录
# -----------------------------------------------------------------------------

TRAIN_DIR="$DATA_OUT/train"
DEV_DIR="$DATA_OUT/dev"
CONF_OUT="$RECIPE_DIR/conf/${FINETUNE_EXP_NAME}.toml"

echo "[1/3] Validating checkpoint directory..."
if [[ ! -d "$BASE_CKPT_DIR" ]]; then
    echo "Error: Checkpoint directory not found: $BASE_CKPT_DIR" >&2
    echo "Please specify the checkpoint directory using BASE_CKPT_DIR environment variable." >&2
    echo "Example: BASE_CKPT_DIR=/path/to/checkpoints/epoch_00XX bash $0" >&2
    exit 1
fi

# 检查必需的文件
if [[ ! -f "$BASE_CKPT_DIR/pytorch_model.bin" ]]; then
    echo "Error: pytorch_model.bin not found in $BASE_CKPT_DIR" >&2
    exit 1
fi

echo "Found checkpoint directory: $BASE_CKPT_DIR"

echo "[2/3] Validating data directory..."
if [[ ! -f "$TRAIN_DIR/wav.scp" || ! -f "$DEV_DIR/wav.scp" ]]; then
    echo "Error: Data not found. Please run run_train_kaldi_merged_fbank_multilabel.sh first to prepare data." >&2
    exit 1
fi
echo "Data directory validated: $DATA_OUT"

echo "[3/3] Writing finetune config to $CONF_OUT ..."
cat > "$CONF_OUT" <<EOF
[meta]
save_dir = "exp"
seed = $SEED
# save_dir 下会创建 $FINETUNE_EXP_NAME 目录用于保存新的 checkpoints

[finetune]
finetune = true
ckpt_dir = "$BASE_CKPT_DIR"
# finetune = true 表示从已有 checkpoint 继续训练
# ckpt_dir 指向具体的 checkpoint 目录（包含 pytorch_model.bin）

[trainer]
path = "trainer_single_opt_multilabel.Trainer"
[trainer.args]
max_epochs = $MAX_EPOCHS
gradient_percentile = 90
gradient_history_size = 1000
save_max_score = false
save_ckpt_interval = 1
max_patience = 10
max_num_checkpoints = 100
gradient_accumulation_steps = 1
validation_interval = 1
freeze_wavlm = false
lr_decay = false
use_one_cycle_lr = false

[optimizer]
path = "torch.optim.AdamW"
[optimizer.args]
lr = $LR
# 使用较小的学习率进行 finetune

[model]
path = "diarizen.models.eend.model_fbank_conformer.Model"
[model.args]
n_fft = 400
n_mels = 128
win_length = 25
hop_length = 10
sample_rate = 16000
attention_in = 512
ffn_hidden = 1024
num_head = 4
num_layer = ${NUM_LAYER:-8}
dropout = 0.1
chunk_size = $CHUNK_SIZE
use_posi = false
output_activate_function = false
selected_channel = 0
max_speakers_per_chunk = 4
max_speakers_per_frame = 2
use_powerset = false
# 说明：
# - use_powerset = false：使用 multilabel 模式
# - 模型输出是 (B, T, num_speakers) 的 sigmoid 概率
# - 损失函数使用 binary_cross_entropy

[train_dataset]
path = "dataset.DiarizationDataset"
[train_dataset.args]
scp_file = "data/$EXP_NAME/train/wav.scp"
rttm_file = "data/$EXP_NAME/train/rttm"
uem_file = "data/$EXP_NAME/train/all.uem"
chunk_size = $CHUNK_SIZE
chunk_shift = $CHUNK_SHIFT
sample_rate = 16000
full_utterance = $([[ "$FULL_UTTERANCE" == "1" ]] && echo true || echo false)
max_sessions = $([[ -n "$SUBSET_SESSIONS" ]] && echo "$SUBSET_SESSIONS" || echo "0")
max_chunks = 0

[train_dataset.dataloader]
batch_size = $BATCH_SIZE
num_workers = $TRAIN_NUM_WORKERS
prefetch_factor = $PREFETCH_FACTOR
persistent_workers = $([[ "$PERSISTENT_WORKERS" == "true" ]] && echo true || echo false)
drop_last = true
pin_memory = true

[validate_dataset]
path = "dataset.DiarizationDataset"
[validate_dataset.args]
scp_file = "data/$EXP_NAME/dev/wav.scp"
rttm_file = "data/$EXP_NAME/dev/rttm"
uem_file = "data/$EXP_NAME/dev/all.uem"
chunk_size = $CHUNK_SIZE
chunk_shift = $DEV_CHUNK_SHIFT
sample_rate = 16000
full_utterance = $([[ "$FULL_UTTERANCE" == "1" ]] && echo true || echo false)
max_sessions = $([[ -n "$SUBSET_SESSIONS" ]] && echo "$SUBSET_SESSIONS" || echo "0")
max_chunks = 0

[validate_dataset.dataloader]
batch_size = $VAL_BATCH_SIZE
num_workers = $DEV_NUM_WORKERS
prefetch_factor = $PREFETCH_FACTOR
persistent_workers = $([[ "$PERSISTENT_WORKERS" == "true" ]] && echo true || echo false)
drop_last = true
pin_memory = true
EOF

if command -v conda >/dev/null 2>&1; then
    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    fi
fi

echo "[4/4] Launching finetune training with accelerate on $NUM_GPUS GPU(s)..."
echo "Loading checkpoint: $BASE_CKPT_DIR"
echo "New learning rate: $LR"
echo "Log file: $LOG_FILE"
echo "Use 'tmux attach -t $TMUX_SESSION_NAME' to view training progress."
cd "$RECIPE_DIR"

# 使用 accelerate 启动 finetune 训练
# 注意：不使用 -R (resume) 标志，因为我们通过 finetune=true 来加载权重
accelerate launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$PORT" \
    run_single_opt.py -C "$CONF_OUT" -M train 2>&1 | tee -a "$LOG_FILE"


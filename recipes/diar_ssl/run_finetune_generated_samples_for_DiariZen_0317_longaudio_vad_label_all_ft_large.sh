#!/usr/bin/env bash

# 命名与 kaldi_merged_1219_all_ft_large 一致：<Kaldi 数据目录名>_ft_large
# 本数据：generated_samples_for_DiariZen_0317_longaudio_vad_label_all
# → EXP / conf / data 子目录：generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large
#
# 从 kaldi_merged_1219_all_ft_large/checkpoints/epoch_0016 加载 pytorch_model.bin，新 exp 下训练（不加 -R）。
#
# 示例：
#   bash run_finetune_generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large.sh
#   BATCH_SIZE=16 NUM_GPUS=1 bash run_finetune_generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large.sh
#
# 若出现「Creating new tmux session」后立刻 tmux ls 报 no server：表示 tmux 里那条命令已退出
#（常见为数据路径/校验失败、conda、accelerate 报错）。本会话结束后无其它 session 时 tmux 服务会一起关掉。
# 请查看同目录下 .log；或前台排错：cd 本目录后执行
#   IN_TMUX=1 bash ./run_finetune_generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large.sh
#
# 小数据快速跑满 1 个 epoch 并写 checkpoint：见同目录 run_smoke_finetune_generated_samples_0317_longaudio_vad_label_all_ft_large.sh
# dataset 的 max_sessions：取 wav.scp / reco2dur 中前 N 条（整数，0=不限制）
#
# 为何可能「一个 ckpt 都没存」：若训练在 epoch 末尾的 training_epoch_end 崩溃，则尚未执行后面的
# _save_checkpoint；已修复 diarizen/trainer_dual_opt.py 中对 None 的汇总逻辑。
#
# 按 step 存盘：默认每 5000 个 steps_trained 存 checkpoints/step_XXXXXXXX/；设 SAVE_CKPT_EVERY_N_STEPS=0 可关。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
TMUX_SESSION_NAME="diarizen_gs0317_longaudio_vad_label_all_ft_large"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/run_finetune_generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large.log}"

# USE_TMUX=0：当前 shell 直接跑完全流程（便于 smoke）；默认仍自动挂 tmux
if [[ -z "${TMUX:-}" && "${USE_TMUX:-1}" != "0" ]]; then
    if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
        echo "Found existing tmux session: $TMUX_SESSION_NAME"
        tmux send-keys -t "$TMUX_SESSION_NAME" "cd '$SCRIPT_DIR' && IN_TMUX=1 bash '$0' $*" C-m
        exit 0
    else
        echo "Creating new tmux session: $TMUX_SESSION_NAME"
        tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$SCRIPT_DIR" \
            "IN_TMUX=1 bash '$0' $*"
        exit 0
    fi
fi

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RECIPE_DIR="$REPO_DIR/recipes/diar_ssl"

# tmux 内从校验阶段起就写入日志，避免只有 accelerate 段有输出、会话秒退时无从查因
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== $(date -Is) start (IN_TMUX) pid=$$ ==="

: "${MKL_INTERFACE_LAYER:=LP64}"
: "${MKL_THREADING_LAYER:=GNU}"
export MKL_INTERFACE_LAYER MKL_THREADING_LAYER

: "${PYTHONPATH:=$REPO_DIR}"
export PYTHONPATH

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${NUM_GPUS:=4}"
: "${TRAIN_NUM_WORKERS:=16}"
: "${DEV_NUM_WORKERS:=16}"
: "${PREFETCH_FACTOR:=4}"
: "${PERSISTENT_WORKERS:=true}"

DATA_SRC="${DATA_SRC:-/root/group-shared/voiceprint/data/speech/speaker_diarization/generated_samples_for_DiariZen_0317_longaudio_vad_label_all/kaldi_data}"
EXP_NAME="${EXP_NAME:-generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large_from_0002}"
FT_SOURCE_EPOCH_DIR="${FT_SOURCE_EPOCH_DIR:-$RECIPE_DIR/exp/kaldi_merged_1219_all_ft_large/checkpoints/epoch_0002}"

VAL_RATIO="${VAL_RATIO:-0.05}"
SEED="${SEED:-3407}"
NUM_GPUS="${NUM_GPUS:-$NUM_GPUS}"
PORT="${PORT:-11347}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
CHUNK_SHIFT="${CHUNK_SHIFT:-6}"
DEV_CHUNK_SHIFT="${DEV_CHUNK_SHIFT:-8}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-$TRAIN_NUM_WORKERS}"
DEV_NUM_WORKERS="${DEV_NUM_WORKERS:-$DEV_NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-$PREFETCH_FACTOR}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-$PERSISTENT_WORKERS}"
# DiarizationDataset：仅取前 N 个 session（train/dev 分别限制）；0=全量
MAX_TRAIN_SESSIONS="${MAX_TRAIN_SESSIONS:-0}"
MAX_DEV_SESSIONS="${MAX_DEV_SESSIONS:-0}"
MAX_TRAIN_CHUNKS="${MAX_TRAIN_CHUNKS:-0}"
MAX_DEV_CHUNKS="${MAX_DEV_CHUNKS:-0}"
# full utterance 最长约 30s 时 batch 20/卡易 OOM；默认改小并用梯度累积维持更新幅度
BATCH_SIZE="${BATCH_SIZE:-10}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
LR_WAVLM="${LR_WAVLM:-3e-6}"
LR_HEAD="${LR_HEAD:-1e-5}"
FULL_UTTERANCE="${FULL_UTTERANCE:-1}"
CONDA_ENV="${CONDA_ENV:-diarizen}"
RESUME="${RESUME:-0}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SAVE_CKPT_EVERY_N_STEPS="${SAVE_CKPT_EVERY_N_STEPS:-5000}"
MAX_NUM_STEP_CHECKPOINTS="${MAX_NUM_STEP_CHECKPOINTS:-500}"

DATA_OUT="$RECIPE_DIR/data/$EXP_NAME"
TRAIN_DIR="$DATA_OUT/train"
DEV_DIR="$DATA_OUT/dev"
CONF_OUT="$RECIPE_DIR/conf/${EXP_NAME}.toml"

check_path() {
    if [[ ! -s "$1" ]]; then
        echo "Missing required file: $1" >&2
        exit 1
    fi
}

echo "[1/4] Validating inputs..."
check_path "$DATA_SRC/wav.scp"
check_path "$DATA_SRC/rttm"
check_path "$DATA_SRC/reco2dur"
if [[ ! -d "$FT_SOURCE_EPOCH_DIR" ]]; then
    echo "FT_SOURCE_EPOCH_DIR not found: $FT_SOURCE_EPOCH_DIR" >&2
    exit 1
fi
if [[ ! -s "$FT_SOURCE_EPOCH_DIR/pytorch_model.bin" ]]; then
    echo "Missing model weights: $FT_SOURCE_EPOCH_DIR/pytorch_model.bin" >&2
    exit 1
fi

if [[ "$FORCE_REBUILD_DATA" == "1" ]]; then
    rm -rf "$DATA_OUT"
fi

echo "[2/4] Preparing train/dev Kaldi files under $DATA_OUT ..."
if [[ ! -f "$TRAIN_DIR/wav.scp" || ! -f "$DEV_DIR/wav.scp" ]]; then
    python "$RECIPE_DIR/split_kaldi_data.py" "$DATA_SRC" "$DATA_OUT" "$VAL_RATIO" "$SEED"
else
    echo "  - Existing data split found; set FORCE_REBUILD_DATA=1 to rebuild."
fi

echo "[3/4] Writing config to $CONF_OUT ..."
cat > "$CONF_OUT" <<EOF
[meta]
save_dir = "exp"
seed = $SEED

[finetune]
finetune = true
ckpt_dir = "$FT_SOURCE_EPOCH_DIR"

[trainer]
path = "trainer_dual_opt.Trainer"
[trainer.args]
max_epochs = $MAX_EPOCHS
gradient_percentile = 90
gradient_history_size = 1000
save_max_score = false
save_ckpt_interval = 1
save_ckpt_every_n_steps = $SAVE_CKPT_EVERY_N_STEPS
max_num_step_checkpoints = $MAX_NUM_STEP_CHECKPOINTS
max_patience = 10
max_num_checkpoints = 100
gradient_accumulation_steps = $GRADIENT_ACCUMULATION_STEPS
validation_interval = 1
freeze_wavlm = false
lr_decay = false
use_one_cycle_lr = false

[optimizer_small]
path = "torch.optim.AdamW"
[optimizer_small.args]
lr = $LR_WAVLM

[optimizer_big]
path = "torch.optim.AdamW"
[optimizer_big.args]
lr = $LR_HEAD

[model]
path = "diarizen.models.eend.model_wavlm_conformer.Model"
[model.args]
wavlm_src = "wavlm_large_s80_md"
wavlm_layer_num = 25
wavlm_feat_dim = 1024
attention_in = 256
ffn_hidden = 1024
num_head = 4
num_layer = 4
dropout = 0.1
chunk_size = $CHUNK_SIZE
use_posi = false
output_activate_function = false
selected_channel = 0
max_speakers_per_chunk = 4
max_speakers_per_frame = 4

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
max_sessions = $MAX_TRAIN_SESSIONS
max_chunks = $MAX_TRAIN_CHUNKS

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
max_sessions = $MAX_DEV_SESSIONS
max_chunks = $MAX_DEV_CHUNKS

[validate_dataset.dataloader]
batch_size = $VAL_BATCH_SIZE
num_workers = $DEV_NUM_WORKERS
prefetch_factor = $PREFETCH_FACTOR
persistent_workers = $([[ "$PERSISTENT_WORKERS" == "true" ]] && echo true || echo false)
drop_last = true
pin_memory = true
EOF

if [[ "$SKIP_TRAIN" == "1" ]]; then
    echo "[4/4] SKIP_TRAIN=1 set; stopping after config generation."
    exit 0
fi

if command -v conda >/dev/null 2>&1; then
    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    fi
fi

echo "[4/4] Launching fine-tuning (weights from $FT_SOURCE_EPOCH_DIR → exp/$EXP_NAME/) on $NUM_GPUS GPU(s)..."
echo "Log file: $LOG_FILE"
cd "$RECIPE_DIR"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$PORT" \
    run_dual_opt.py -C "$CONF_OUT" -M train

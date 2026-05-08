#!/usr/bin/env bash

# Purpose: continue fine-tuning from a fixed continued-training checkpoint on 2 GPUs
# with halved learning rates (LR_NEST=5e-7, LR_HEAD=5e-5) and NO gradient accumulation.
#
# Defaults:
# - source experiment: kaldi_merged_1219_all_ft_nest_xlarge_2gpu_continue_lr1e-6_1e-4
# - fixed checkpoint: exp/<SRC_EXP_NAME>/checkpoints/epoch_0003
# - 2 GPUs: CUDA_VISIBLE_DEVICES=0,1, NUM_GPUS=2
# - keep batch size defaults, gradient accumulation steps: 1
#
# Examples:
#   bash run_finetune_kaldi_merged_nest_xlarge_1219_all_continue_epoch0003_2gpu_lr5e-7_5e-5.sh
#   EXP_NAME=my_next_stage PORT=11362 \
#     bash run_finetune_kaldi_merged_nest_xlarge_1219_all_continue_epoch0003_2gpu_lr5e-7_5e-5.sh
#   CKPT_DIR=/path/to/checkpoints/epoch_0003 CUDA_VISIBLE_DEVICES=2,3 \
#     bash run_finetune_kaldi_merged_nest_xlarge_1219_all_continue_epoch0003_2gpu_lr5e-7_5e-5.sh

set -euo pipefail

# ====== tmux session management ======
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
TMUX_SESSION_NAME="diarizen_nest_xlarge_ft_continue_e0003_2gpu_lr5e-7_5e-5"
LOG_FILE="$SCRIPT_DIR/run_finetune_kaldi_merged_nest_xlarge_1219_all_continue_epoch0003_2gpu_lr5e-7_5e-5.log"

if [[ -z "${TMUX:-}" ]]; then
    if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
        echo "Found existing tmux session: $TMUX_SESSION_NAME"
        echo "Sending command to existing session. Use 'tmux attach -t $TMUX_SESSION_NAME' to view."
        tmux send-keys -t "$TMUX_SESSION_NAME" "cd '$SCRIPT_DIR' && { IN_TMUX=1 bash '$SCRIPT_PATH' $*; echo \"Training command exited with code \$?\"; } 2>&1 | tee -a '$LOG_FILE'; exec bash" C-m
        exit 0
    fi

    echo "Creating new tmux session: $TMUX_SESSION_NAME"
    echo "Use 'tmux attach -t $TMUX_SESSION_NAME' to view training progress."
    tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$SCRIPT_DIR" \
        "{ IN_TMUX=1 bash '$SCRIPT_PATH' $*; echo \"Training command exited with code \$?\"; } 2>&1 | tee -a '$LOG_FILE'; exec bash"
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RECIPE_DIR="$REPO_DIR/recipes/diar_ssl"

# ====== runtime env defaults ======
: "${MKL_INTERFACE_LAYER:=LP64}"
: "${MKL_THREADING_LAYER:=GNU}"
export MKL_INTERFACE_LAYER MKL_THREADING_LAYER

NEMO_ROOT="${NEMO_ROOT:-/root/code/github_repos/NeMo}"
export PYTHONPATH="$REPO_DIR:$NEMO_ROOT:${PYTHONPATH:-}"

: "${CUDA_VISIBLE_DEVICES:=0,1}"
: "${NUM_GPUS:=2}"
: "${TRAIN_NUM_WORKERS:=4}"
: "${DEV_NUM_WORKERS:=4}"
: "${PREFETCH_FACTOR:=2}"
: "${PERSISTENT_WORKERS:=true}"
export CUDA_VISIBLE_DEVICES

# ---- user adjustable knobs --------------------------------------------------
DATA_SRC="${DATA_SRC:-/root/group-shared/voiceprint/data/speech/speaker_diarization/kaldi_merged_1219_all}"
SRC_EXP_NAME="${SRC_EXP_NAME:-kaldi_merged_1219_all_ft_nest_xlarge_2gpu_continue_lr1e-6_1e-4}"
SRC_EPOCH="${SRC_EPOCH:-0003}"
EXP_NAME="${EXP_NAME:-${SRC_EXP_NAME}_continue_epoch${SRC_EPOCH}_2gpu_lr5e-7_5e-5}"
VAL_RATIO="${VAL_RATIO:-0.05}"
SEED="${SEED:-3407}"
PORT="${PORT:-11353}"

CHUNK_SIZE="${CHUNK_SIZE:-8}"
CHUNK_SHIFT="${CHUNK_SHIFT:-6}"
DEV_CHUNK_SHIFT="${DEV_CHUNK_SHIFT:-8}"
SUBSET_SESSIONS="${SUBSET_SESSIONS:-}"
FULL_UTTERANCE="${FULL_UTTERANCE:-1}"

BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
LR_NEST="${LR_NEST:-5e-7}"
LR_HEAD="${LR_HEAD:-5e-5}"
SCHEDULER_NAME="${SCHEDULER_NAME:-cosine_schedule_with_warmup}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

NEST_MODEL="${NEST_MODEL:-nvidia/ssl_en_nest_xlarge_v1.0}"
NEST_LAYERS="${NEST_LAYERS:-all}"
NEST_LAYER_NUM="${NEST_LAYER_NUM:-0}"
NEST_FEAT_DIM="${NEST_FEAT_DIM:-0}"

CONDA_ENV="${CONDA_ENV:-diarizen-nemo}"
RESUME="${RESUME:-0}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
CKPT_DIR="${CKPT_DIR:-}"
# -----------------------------------------------------------------------------

DATA_OUT="$RECIPE_DIR/data/$EXP_NAME"
TRAIN_DIR="$DATA_OUT/train"
DEV_DIR="$DATA_OUT/dev"
CONF_OUT="$RECIPE_DIR/conf/${EXP_NAME}.toml"
SRC_CHECKPOINTS_DIR="$RECIPE_DIR/exp/$SRC_EXP_NAME/checkpoints"

check_path() {
    if [[ ! -s "$1" ]]; then
        echo "Missing required file: $1" >&2
        exit 1
    fi
}

check_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Missing required directory: $1" >&2
        exit 1
    fi
}

echo "[1/4] Validating inputs..."
check_path "$DATA_SRC/wav.scp"
check_path "$DATA_SRC/rttm"
check_path "$DATA_SRC/reco2dur"
check_dir "$NEMO_ROOT/nemo"
check_path "$NEMO_ROOT/nemo/collections/asr/models/ssl_models.py"

if [[ -z "$CKPT_DIR" ]]; then
    check_dir "$SRC_CHECKPOINTS_DIR"
    CKPT_DIR="$SRC_CHECKPOINTS_DIR/epoch_$SRC_EPOCH"
fi

check_dir "$CKPT_DIR"
check_path "$CKPT_DIR/pytorch_model.bin"
echo "Using source checkpoint: $CKPT_DIR"

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
ckpt_dir = "$CKPT_DIR"

[trainer]
path = "trainer_dual_opt.Trainer"
[trainer.args]
max_epochs = $MAX_EPOCHS
gradient_percentile = 90
gradient_history_size = 1000
save_max_score = false
save_ckpt_interval = 1
max_patience = 10
max_num_checkpoints = 100
gradient_accumulation_steps = $GRADIENT_ACCUMULATION_STEPS
validation_interval = 1
scheduler_name = "$SCHEDULER_NAME"
warmup_steps = $WARMUP_STEPS
warmup_ratio = $WARMUP_RATIO
freeze_wavlm = false
lr_decay = false
use_one_cycle_lr = false

[optimizer_small]
path = "torch.optim.AdamW"
[optimizer_small.args]
lr = $LR_NEST

[optimizer_big]
path = "torch.optim.AdamW"
[optimizer_big.args]
lr = $LR_HEAD

[model]
path = "diarizen.models.eend.model_nemo_nest_conformer.Model"
[model.args]
nest_model_name = "$NEST_MODEL"
nemo_root = "$NEMO_ROOT"
nest_layers = "$NEST_LAYERS"
nest_layer_num = $NEST_LAYER_NUM
nest_feat_dim = $NEST_FEAT_DIM
attention_in = 256
ffn_hidden = 1024
num_head = 4
num_layer = 4
dropout = 0.1
chunk_size = $CHUNK_SIZE
use_posi = true
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

if [[ "$SKIP_TRAIN" == "1" ]]; then
    echo "[4/4] SKIP_TRAIN=1 set; stopping after config generation."
    exit 0
fi

if command -v conda >/dev/null 2>&1; then
    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
        # shellcheck disable=SC1091
        set +u
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        set -u
    fi
fi

echo "[4/4] Launching continued NEST fine-tuning on $NUM_GPUS GPU(s)..."
echo "Source experiment: $SRC_EXP_NAME"
echo "Source checkpoint: $CKPT_DIR"
echo "Target experiment: $EXP_NAME"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Learning rates: LR_NEST=$LR_NEST, LR_HEAD=$LR_HEAD"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NEMO_ROOT: $NEMO_ROOT"
echo "NEST_MODEL: $NEST_MODEL"
echo "Log file: $LOG_FILE"
echo "Use 'tmux attach -t $TMUX_SESSION_NAME' to view training progress."
cd "$RECIPE_DIR"

RESUME_FLAG=""
if [[ "$RESUME" == "1" ]]; then
    RESUME_FLAG="-R"
    echo "Resuming from latest checkpoint in target experiment..."
fi

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$PORT" \
    run_dual_opt.py -C "$CONF_OUT" -M train $RESUME_FLAG 2>&1 | tee -a "$LOG_FILE"

#!/usr/bin/env bash
# 小数据 smoke：少量 session、2 个 epoch、单卡，验证能跑完 epoch 并写入 checkpoint。
# 产出目录：exp/generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large_smoke/checkpoints/
#
#   bash run_smoke_finetune_generated_samples_0317_longaudio_vad_label_all_ft_large.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
MAIN="$SCRIPT_DIR/run_finetune_generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large.sh"
DATA_ROOT="$SCRIPT_DIR/data"
FULL_SPLIT_NAME="generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large"

export USE_TMUX=0
export LOG_FILE="$SCRIPT_DIR/run_smoke_finetune_generated_samples_0317_longaudio_vad_label_all_ft_large.log"
export EXP_NAME="${EXP_NAME:-generated_samples_for_DiariZen_0317_longaudio_vad_label_all_ft_large_smoke}"
export MAX_TRAIN_SESSIONS="${MAX_TRAIN_SESSIONS:-64}"
export MAX_DEV_SESSIONS="${MAX_DEV_SESSIONS:-16}"
export MAX_EPOCHS="${MAX_EPOCHS:-2}"
export NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# 长音频 full utterance 易 OOM；默认 batch=2、workers=0 便于在单卡上冒烟（仍须保证 GPU 有足够空闲显存）
export BATCH_SIZE="${BATCH_SIZE:-2}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
export DEV_NUM_WORKERS="${DEV_NUM_WORKERS:-0}"
export FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"

# 复用已有全量 split，避免再跑一遍 split_kaldi_data（若尚无 smoke 目录且全量目录存在则软链）
if [[ ! -d "$DATA_ROOT/$EXP_NAME/train" && -d "$DATA_ROOT/$FULL_SPLIT_NAME/train" ]]; then
    echo "[smoke] linking data/$FULL_SPLIT_NAME -> data/$EXP_NAME"
    ln -sfn "$FULL_SPLIT_NAME" "$DATA_ROOT/$EXP_NAME"
fi

exec bash "$MAIN"

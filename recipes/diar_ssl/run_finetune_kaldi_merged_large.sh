#!/usr/bin/env bash

# 用途：在 Kaldi 合并数据上微调 diarizen WavLM-Large-Conformer。
# 前提：已准备好 Kaldi 三件套（wav.scp / rttm / reco2dur）和 large 模型 checkpoint。
# 示例：
#   bash run_finetune_kaldi_merged_large.sh                             # 默认 4 卡，默认数据路径
#   DATA_SRC=/data/kaldi EXP_NAME=my_exp NUM_GPUS=2 bash run_finetune_kaldi_merged_large.sh
#   FORCE_REBUILD_DATA=1 VAL_RATIO=0.2 bash run_finetune_kaldi_merged_large.sh
#
# 注意：脚本会自动在 tmux session 中运行，确保断开连接后训练继续。
#       使用 tmux attach -t diarizen_large_ft 查看训练状态。

# 严格模式：命令/管道/未定义变量出错时立即退出，避免隐藏错误
set -euo pipefail

# ====== tmux session 管理 ======
# 自动在 tmux session 中运行，确保断开连接后训练继续
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
TMUX_SESSION_NAME="diarizen_large_ft"
LOG_FILE="$SCRIPT_DIR/run_finetune_kaldi_merged_large_resume.log"

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
# - CUDA_VISIBLE_DEVICES：控制可见 GPU，逗号分隔；NUM_GPUS 必须与之匹配
# - TRAIN/DEV_NUM_WORKERS：DataLoader 进程数，受 CPU/IO 影响
# - PREFETCH_FACTOR / PERSISTENT_WORKERS：调优 IO pipeline，避免 worker 反复重启
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${NUM_GPUS:=4}"
: "${TRAIN_NUM_WORKERS:=16}"
: "${DEV_NUM_WORKERS:=16}"
: "${PREFETCH_FACTOR:=4}"
: "${PERSISTENT_WORKERS:=true}"
# =============================================================

# ---- user adjustable knobs --------------------------------------------------
# 数据、模型及训练参数，可按需通过环境变量覆盖，便于快速切换实验。
# 关键说明：
# - DATA_SRC：Kaldi 源目录，必须包含 wav.scp / rttm / reco2dur
# - EXP_NAME：实验名，决定 data/$EXP_NAME 与 conf/$EXP_NAME.toml 等输出位置
# - VAL_RATIO：dev 比例；异常值自动回退到 9:1；与 SEED 搭配确保可复现
# - CHUNK_SIZE / CHUNK_SHIFT：训练窗口长度与步长（秒），窗口偏小可避免短音频被丢弃
# - DEV_CHUNK_SHIFT：验证步长，可适当加大以降低评估开销
# - SUBSET_SESSIONS：逗号分隔的 session 列表用于子集调试；留空=全量
# - FULL_UTTERANCE：1=整段，0=随机 chunk；整段显存占用更高但序列完整
# - BATCH_SIZE / VAL_BATCH_SIZE：需结合显存调整，过大易 OOM（large 模型显存占用更高）
# - LR_WAVLM / LR_HEAD：主干与头部的学习率分离
# - PORT：accelerate 主进程通信端口，冲突时可改
# - BASE_CKPT：微调起点 checkpoint，指向 WavLM-Large 模型路径
# - CONDA_ENV：需提前创建好依赖环境，默认 diarizen
# - RESUME=1：从最近 checkpoint 继续；SKIP_TRAIN=1：仅生成配置后退出
# - NUM_GPUS：应与 CUDA_VISIBLE_DEVICES 个数一致，否则 accelerate 会报错
DATA_SRC="${DATA_SRC:-/root/group-shared/voiceprint/data/speech/speaker_diarization/kaldi_merged_1205_1207}"
EXP_NAME="${EXP_NAME:-kaldi_merged_1205_1207_ft_large}"
VAL_RATIO="${VAL_RATIO:-0.1}"          # 10% for dev by default
SEED="${SEED:-3407}"
NUM_GPUS="${NUM_GPUS:-$NUM_GPUS}"      # default 4 from env defaults
PORT="${PORT:-11346}"                   # different port to avoid conflict
CHUNK_SIZE="${CHUNK_SIZE:-2}"          # keep small so short clips are not dropped
CHUNK_SHIFT="${CHUNK_SHIFT:-1}"
DEV_CHUNK_SHIFT="${DEV_CHUNK_SHIFT:-2}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-$TRAIN_NUM_WORKERS}"
DEV_NUM_WORKERS="${DEV_NUM_WORKERS:-$DEV_NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-$PREFETCH_FACTOR}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-$PERSISTENT_WORKERS}"
SUBSET_SESSIONS="${SUBSET_SESSIONS:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"         # reduced from 96 due to large model size
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}" # reduced from 96 due to large model size
MAX_EPOCHS="${MAX_EPOCHS:-30}"
LR_WAVLM="${LR_WAVLM:-5e-6}"           # slightly lower lr for large model
LR_HEAD="${LR_HEAD:-1e-4}"
FULL_UTTERANCE="${FULL_UTTERANCE:-1}"
BASE_CKPT="${BASE_CKPT:-$REPO_DIR/cache/models--BUT-FIT--diarizen-wavlm-large-s80-md/snapshots/7030f2c7fe847c49b2390511bb4c3f8b90dbc022/pytorch_model.bin}"
CONDA_ENV="${CONDA_ENV:-diarizen}"
RESUME="${RESUME:-0}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
# -----------------------------------------------------------------------------

# 生成各阶段使用的路径，DATA_OUT 下会存放 train/dev 的 Kaldi 文件
# CONF_OUT 是最终写出的训练 TOML，后续直接被 run_dual_opt.py 读取
DATA_OUT="$RECIPE_DIR/data/$EXP_NAME"
TRAIN_DIR="$DATA_OUT/train"
DEV_DIR="$DATA_OUT/dev"
CONF_OUT="$RECIPE_DIR/conf/${EXP_NAME}.toml"

check_path() {
    # 仅检查文件是否存在且非空，提前暴露路径或下载问题
    # 若缺失，请确认 DATA_SRC 路径是否正确、文件是否同步完毕
    if [[ ! -s "$1" ]]; then
        echo "Missing required file: $1" >&2
        exit 1
    fi
}

echo "[1/4] Validating inputs..."
check_path "$DATA_SRC/wav.scp"
check_path "$DATA_SRC/rttm"
check_path "$DATA_SRC/reco2dur"
check_path "$BASE_CKPT"

if [[ "$FORCE_REBUILD_DATA" == "1" ]]; then
    # 强制重建时先清空旧的拆分结果，确保重新划分
    # 注意：会覆盖 data/$EXP_NAME 下的已存在划分，若需保留请先备份
    rm -rf "$DATA_OUT"
fi

echo "[2/4] Preparing train/dev Kaldi files under $DATA_OUT ..."
# 使用独立 Python 脚本完成数据拆分，保持原有随机划分与最小 dev 样本逻辑
# - 已有拆分且未指定 FORCE_REBUILD_DATA 时直接复用，保证可重复性
# - 需要重新划分（如调整 VAL_RATIO/SEED 或追加新录音）时设置 FORCE_REBUILD_DATA=1
# - 输出 train/dev 下的 wav.scp、rttm、all.uem 三件套
if [[ ! -f "$TRAIN_DIR/wav.scp" || ! -f "$DEV_DIR/wav.scp" ]]; then
    python "$RECIPE_DIR/split_kaldi_data.py" "$DATA_SRC" "$DATA_OUT" "$VAL_RATIO" "$SEED"
else
    echo "  - Existing data split found; set FORCE_REBUILD_DATA=1 to rebuild."
fi

echo "[3/4] Writing config to $CONF_OUT ..."
# 生成训练用 TOML 配置，供 run_dual_opt.py 直接读取
cat > "$CONF_OUT" <<EOF
[meta]
save_dir = "exp"
seed = $SEED
# save_dir 相对 recipes/diar_ssl，所有日志与 checkpoint 均写入此目录

[finetune]
finetune = true
ckpt_dir = "$BASE_CKPT"
# ckpt_dir 为微调起点权重路径（WavLM-Large）

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
gradient_accumulation_steps = 1
validation_interval = 1
freeze_wavlm = false
lr_decay = false
use_one_cycle_lr = false
# 说明：
# - validation_interval=1：每轮验证一次；max_patience=10：验证早停耐心
# - max_num_checkpoints=100：最多保留 100 个 checkpoint
# - gradient_accumulation_steps=1：可按显存调高以累积小 batch
# - freeze_wavlm=false：默认微调主干；如想只训头部可设为 true

[optimizer_small]
path = "torch.optim.AdamW"
[optimizer_small.args]
lr = $LR_WAVLM
# optimizer_small 通常用于 WavLM 主干，学习率较小（large 模型使用更小的 lr）

[optimizer_big]
path = "torch.optim.AdamW"
[optimizer_big.args]
lr = $LR_HEAD
# optimizer_big 用于上层头部，学习率相对更大

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
# 说明：
# - wavlm_layer_num=25 表示使用全部层（large 模型共 25 层）
# - wavlm_feat_dim=1024 是 large 模型的特征维度（base 是 768）
# - attention_in/ffn_hidden/num_head/num_layer 为 Conformer 超参
# - chunk_size 与数据集的 chunk_size 保持一致
# - max_speakers_per_chunk/max_speakers_per_frame 控制说话人数上限（影响标签裁剪）

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
# full_utterance=1 时按整段裁切，max_sessions 可用于子集调试
# rttm_file 指向标注；uem_file 提供全时长区间，避免裁切越界
# max_chunks=0 表示不限制每条录音生成的 chunk 数量

[train_dataset.dataloader]
batch_size = $BATCH_SIZE
num_workers = $TRAIN_NUM_WORKERS
prefetch_factor = $PREFETCH_FACTOR
persistent_workers = $([[ "$PERSISTENT_WORKERS" == "true" ]] && echo true || echo false)
drop_last = true
pin_memory = true
# DataLoader 说明：
# - drop_last=true 保持 batch 大小一致，便于分布式梯度同步
# - pin_memory=true 加速从 CPU 传输到 GPU
# - 若 CPU/IO 紧张，可下调 num_workers 或 prefetch_factor
# - large 模型 batch_size 减半以适应显存限制

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
# 开发集 chunk_shift 可调大，减少评估步数；评估仍使用与训练一致的模型超参
# dev 使用专属的 wav/rttm/uem 子集，保持与 train 划分对应

[validate_dataset.dataloader]
batch_size = $VAL_BATCH_SIZE
num_workers = $DEV_NUM_WORKERS
prefetch_factor = $PREFETCH_FACTOR
persistent_workers = $([[ "$PERSISTENT_WORKERS" == "true" ]] && echo true || echo false)
drop_last = true
pin_memory = true
# 若验证阶段显存紧张，可进一步减小 VAL_BATCH_SIZE 或增加 DEV_CHUNK_SHIFT
EOF

if [[ "$SKIP_TRAIN" == "1" ]]; then
    echo "[4/4] SKIP_TRAIN=1 set; stopping after config generation."
    exit 0
fi

if command -v conda >/dev/null 2>&1; then
    # 如未处于目标环境则自动切换，确保依赖满足
    # 若使用系统 Python，可忽略此块或自行 source venv
    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    fi
fi

echo "[4/4] Launching fine-tuning with accelerate on $NUM_GPUS GPU(s)..."
echo "Log file: $LOG_FILE"
echo "Use 'tmux attach -t $TMUX_SESSION_NAME' to view training progress."
cd "$RECIPE_DIR"

RESUME_FLAG=""
if [[ "$RESUME" == "1" ]]; then
    RESUME_FLAG="-R"
    echo "Resuming from latest checkpoint..."
fi

# 使用 accelerate 多进程启动训练，可选恢复上次断点
# - --num_processes 与 NUM_GPUS 对齐；若只用单卡，可设置 NUM_GPUS=1
# - --main_process_port 需避免与其他作业冲突
# - RESUME_FLAG 会添加 -R，从最近 checkpoint 继续
# - 输出同时显示在终端和日志文件中
accelerate launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$PORT" \
    run_dual_opt.py -C "$CONF_OUT" -M train $RESUME_FLAG 2>&1 | tee -a "$LOG_FILE"


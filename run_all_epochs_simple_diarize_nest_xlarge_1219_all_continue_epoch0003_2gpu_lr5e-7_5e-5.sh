#!/usr/bin/env bash

# 批量使用 simple_diarize.py 跑
#   kaldi_merged_1219_all_ft_nest_xlarge_2gpu_continue_lr1e-6_1e-4_continue_epoch0003_2gpu_lr5e-7_5e-5
# 下所有 epoch checkpoint，并在每个 epoch 结束后自动评估 is-multispeaker 指标，最终汇总结果。
#
# 输出结构：
#   $OUT_BASE_DIR/<exp_name>/<epoch>/
#
# 可通过环境变量覆盖：
#   IN_ROOT, OUT_BASE_DIR, NUM_WORKERS, DEVICE, CUDA_VISIBLE_DEVICES, CONDA_ENV, LABELS_CSV

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

export PYTHONPATH="$REPO_DIR/pyannote-audio:$REPO_DIR:${PYTHONPATH:-}"

IN_ROOT="${IN_ROOT:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios}"
OUT_BASE_DIR="${OUT_BASE_DIR:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios_Diarizen_simple_nest_xlarge_1219_all_all_epochs}"
NUM_WORKERS="${NUM_WORKERS:-1}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_ENV="${CONDA_ENV:-diarizen-nemo}"
LABELS_CSV="${LABELS_CSV:-/root/code/own/download_gp_online_audios_for_speakerdetection_1113/audio_multispeaker_labels.csv}"
METRICS_SCRIPT="/root/code/own/download_gp_online_audios_for_speakerdetection_1113/compute_is_multispeaker_metrics_diarizen.py"

export CUDA_VISIBLE_DEVICES

EXP_DIR="/root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_nest_xlarge_2gpu_continue_lr1e-6_1e-4_continue_epoch0003_2gpu_lr5e-7_5e-5"

# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────
check_dir() {
  if [[ ! -d "$1" ]]; then
    echo "错误: 目录不存在: $1" >&2
    exit 1
  fi
}

check_file() {
  if [[ ! -f "$1" ]]; then
    echo "错误: 文件不存在: $1" >&2
    exit 1
  fi
}

find_latest_config() {
  local exp_dir="$1"
  local config_path=""
  config_path=$(find "$exp_dir" -maxdepth 1 -type f -name 'config__*.toml' | sort -V | tail -n1 || true)
  echo "$config_path"
}

# ──────────────────────────────────────────────
# 全局汇总数组（epoch → accuracy）
# ──────────────────────────────────────────────
declare -a SUMMARY_EPOCHS=()
declare -a SUMMARY_ACC=()
declare -a SUMMARY_F1=()

# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────
echo "=========================================="
echo "开始批量运行 simple_diarize.py"
echo "实验: $(basename "$EXP_DIR")"
echo "输入目录: $IN_ROOT"
echo "输出基础目录: $OUT_BASE_DIR"
echo "device: $DEVICE  |  num_workers: $NUM_WORKERS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "conda env: $CONDA_ENV"
echo "=========================================="

check_dir "$IN_ROOT"
check_dir "$EXP_DIR"
mkdir -p "$OUT_BASE_DIR"

exp_name="$(basename "$EXP_DIR")"
checkpoints_dir="$EXP_DIR/checkpoints"
config_path="$(find_latest_config "$EXP_DIR")"

check_dir "$checkpoints_dir"
if [[ -z "$config_path" ]]; then
  echo "错误: 未找到 config__*.toml: $EXP_DIR" >&2
  exit 1
fi
check_file "$config_path"

echo "config: $config_path"
echo ""

# 枚举所有 epoch（4 位数字格式）
epochs=$(find "$checkpoints_dir" -maxdepth 1 -mindepth 1 -type d -name 'epoch_[0-9][0-9][0-9][0-9]' -printf '%f\n' | sort -V)
if [[ -z "$epochs" ]]; then
  echo "错误: 未找到任何 epoch_* checkpoint: $checkpoints_dir" >&2
  exit 1
fi

echo "找到以下 epoch:"
echo "$epochs"
echo ""

# ──────────────────────────────────────────────
# 遍历每个 epoch
# ──────────────────────────────────────────────
while IFS= read -r epoch; do
  [[ -z "$epoch" ]] && continue

  ckpt_dir="$checkpoints_dir/$epoch"
  out_dir="$OUT_BASE_DIR/$exp_name/$epoch"

  if [[ ! -f "$ckpt_dir/pytorch_model.bin" ]]; then
    echo "跳过 $exp_name/$epoch: 缺少 pytorch_model.bin"
    continue
  fi

  mkdir -p "$out_dir"

  echo ""
  echo "=========================================="
  echo ">>> 开始推理: $exp_name / $epoch"
  echo ">>> checkpoint: $ckpt_dir"
  echo ">>> 输出: $out_dir"
  echo "=========================================="

  conda run --no-capture-output -n "$CONDA_ENV" python "$REPO_DIR/simple_diarize.py" \
    "$IN_ROOT" \
    --ckpt-dir "$ckpt_dir" \
    --config "$config_path" \
    --out-dir "$out_dir" \
    --device "$DEVICE" \
    --num-workers "$NUM_WORKERS"

  echo ">>> 推理完成: $exp_name / $epoch"

  # ── 评估 ──
  summary_json="$out_dir/summary.json"
  if [[ ! -f "$summary_json" ]]; then
    echo "警告: 未找到 summary.json，跳过评估: $summary_json"
    SUMMARY_EPOCHS+=("$epoch")
    SUMMARY_ACC+=("N/A")
    SUMMARY_F1+=("N/A")
    continue
  fi

  echo ""
  echo ">>> 开始评估: $epoch"
  conda run --no-capture-output -n "$CONDA_ENV" python "$METRICS_SCRIPT" \
    --summary_json "$summary_json" \
    --labels_csv "$LABELS_CSV"

  # 从生成的汇总 JSON 中提取关键指标
  eval_json="$out_dir/evaluation_results/is_multispeaker_accuracy_summary.json"
  if [[ -f "$eval_json" ]]; then
    acc=$(python3 -c "import json,sys; d=json.load(open('$eval_json')); print(d['summary']['accuracy'])" 2>/dev/null || echo "N/A")
    f1=$(python3  -c "import json,sys; d=json.load(open('$eval_json')); print(d['summary']['f1_score'])"  2>/dev/null || echo "N/A")
  else
    acc="N/A"
    f1="N/A"
  fi

  SUMMARY_EPOCHS+=("$epoch")
  SUMMARY_ACC+=("$acc")
  SUMMARY_F1+=("$f1")

  echo ">>> 评估完成: $epoch  accuracy=$acc  f1=$f1"

done <<< "$epochs"

# ──────────────────────────────────────────────
# 最终汇总
# ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "全部 epoch 测试与评估完成"
echo "实验: $exp_name"
echo "输出目录: $OUT_BASE_DIR/$exp_name"
echo "=========================================="
echo ""
printf "%-16s  %-10s  %-10s\n" "epoch" "accuracy" "f1_score"
printf "%-16s  %-10s  %-10s\n" "----------------" "----------" "----------"
for i in "${!SUMMARY_EPOCHS[@]}"; do
  printf "%-16s  %-10s  %-10s\n" "${SUMMARY_EPOCHS[$i]}" "${SUMMARY_ACC[$i]}" "${SUMMARY_F1[$i]}"
done
echo "=========================================="

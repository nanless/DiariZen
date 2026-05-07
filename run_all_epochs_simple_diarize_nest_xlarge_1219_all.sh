#!/usr/bin/env bash

# 批量使用 simple_diarize.py 跑两个 NEST xlarge 实验目录下的所有 epoch checkpoint。
# 默认输入目录固定为：
#   /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios
#
# 默认实验目录：
#   /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_nest_xlarge_2gpu
#   /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_nest_xlarge_2gpu_continue_lr1e-6_1e-4
#
# 输出结构：
#   $OUT_BASE_DIR/<exp_name>/<epoch_xxxx>/
#
# 可通过环境变量覆盖：
#   IN_ROOT, OUT_BASE_DIR, NUM_WORKERS, DEVICE, CONDA_ENV

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

export CUDA_VISIBLE_DEVICES

EXP_DIRS=(
  "/root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_nest_xlarge_2gpu"
  "/root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_nest_xlarge_2gpu_continue_lr1e-6_1e-4"
)

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
  if [[ -z "$config_path" ]]; then
    echo ""  # caller handles the error
    return 0
  fi
  echo "$config_path"
}

run_one_experiment() {
  local exp_dir="$1"
  local exp_name checkpoints_dir config_path epochs epoch ckpt_dir out_dir

  exp_name="$(basename "$exp_dir")"
  checkpoints_dir="$exp_dir/checkpoints"
  config_path="$(find_latest_config "$exp_dir")"

  echo ""
  echo "=========================================="
  echo "实验: $exp_name"
  echo "实验目录: $exp_dir"
  echo "checkpoint 目录: $checkpoints_dir"
  echo "config: $config_path"
  echo "=========================================="

  check_dir "$exp_dir"
  check_dir "$checkpoints_dir"
  if [[ -z "$config_path" ]]; then
    echo "错误: 未找到 config__*.toml: $exp_dir" >&2
    exit 1
  fi
  check_file "$config_path"

  epochs=$(find "$checkpoints_dir" -maxdepth 1 -mindepth 1 -type d -name 'epoch_[0-9][0-9][0-9][0-9]' -printf '%f\n' | sort -V)
  if [[ -z "$epochs" ]]; then
    echo "错误: 未找到任何 epoch_* checkpoint: $checkpoints_dir" >&2
    exit 1
  fi

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
    echo ">>> 开始: $exp_name / $epoch"
    echo ">>> 输入: $IN_ROOT"
    echo ">>> 输出: $out_dir"

    conda run --no-capture-output -n "$CONDA_ENV" python "$REPO_DIR/simple_diarize.py" \
      "$IN_ROOT" \
      --ckpt-dir "$ckpt_dir" \
      --config "$config_path" \
      --out-dir "$out_dir" \
      --device "$DEVICE" \
      --num-workers "$NUM_WORKERS"

    echo ">>> 完成: $exp_name / $epoch"
  done <<< "$epochs"
}

echo "=========================================="
echo "开始批量运行 simple_diarize.py"
echo "输入目录: $IN_ROOT"
echo "输出基础目录: $OUT_BASE_DIR"
echo "device: $DEVICE"
echo "num_workers: $NUM_WORKERS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "conda env: $CONDA_ENV"
echo "=========================================="

check_dir "$IN_ROOT"
mkdir -p "$OUT_BASE_DIR"

for exp_dir in "${EXP_DIRS[@]}"; do
  run_one_experiment "$exp_dir"
done

echo ""
echo "=========================================="
echo "全部 epoch 测试已完成"
echo "输出目录: $OUT_BASE_DIR"
echo "=========================================="

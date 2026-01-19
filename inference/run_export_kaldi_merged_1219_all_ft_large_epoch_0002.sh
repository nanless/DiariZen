#!/usr/bin/env bash
#
# Export ONNX for: recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large/checkpoints/epoch_0002
#
# Output:
#   inference/models/kaldi_merged_1219_all_ft_large/epoch_0002_multilabel_hard.onnx
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Keep behavior consistent with other scripts in this repo
export PYTHONPATH="${REPO_DIR}/pyannote-audio:${REPO_DIR}:${PYTHONPATH:-}"

EXP_NAME="kaldi_merged_1219_all_ft_large"
EXP_DIR="${REPO_DIR}/recipes/diar_ssl/exp/${EXP_NAME}"
CONFIG_PATH="${EXP_DIR}/config__2025_12_26--11_44_15.toml"
CKPT_NAME="epoch_0002"

OUT_ONNX="${REPO_DIR}/inference/models/${EXP_NAME}/${CKPT_NAME}_multilabel_hard.onnx"

conda run --no-capture-output -n diarizen python "${REPO_DIR}/inference/export_to_onnx.py" \
  --exp-dir "${EXP_DIR}" \
  --config "${CONFIG_PATH}" \
  --ckpt-name "${CKPT_NAME}" \
  --out-onnx "${OUT_ONNX}"

echo "Done: ${OUT_ONNX}"


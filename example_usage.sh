#!/bin/bash
# 使用示例：simple_diarize.py 推理脚本

# 设置路径
AUDIO_DIR="/root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios"
OUTPUT_DIR="/root/code/own/download_gp_online_audios_for_speakerdetection_1113/diarization_output"
CKPT_DIR="/root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_temp_ft_large/checkpoints/epoch_0004"
CONFIG="/root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_temp_ft_large/config__2025_12_24--02_01_24.toml"

# 运行推理（使用 GPU）
./run_simple_diarize.sh \
  "${AUDIO_DIR}" \
  --out-dir "${OUTPUT_DIR}" \
  --ckpt-dir "${CKPT_DIR}" \
  --config "${CONFIG}" \
  --device cuda

echo "完成！结果保存在: ${OUTPUT_DIR}"
echo "查看汇总: cat ${OUTPUT_DIR}/summary.json"


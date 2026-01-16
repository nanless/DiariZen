## inference

这个目录用于 **无训练/无 pyannote pipeline 依赖** 的推理。

### 目标
- 把当前实验 `kaldi_merged_1219_all_ft_base` 的 **best(base)** 和 `epoch_0010` checkpoint 导出为 ONNX
- 提供一个只依赖 `onnxruntime + numpy + soundfile` 的推理脚本（输出 RTTM）
- 提供 PyTorch vs ONNX 的 **输出差异** 与 **RTF(Real-time factor)** benchmark

### 目录结构
- `export_to_onnx.py`: 从 `config.toml + pytorch_model.bin` 导出 ONNX（包含 powerset->multilabel 的硬解码）
- `infer_onnx.py`: 纯 ONNX 推理脚本（递归扫描音频目录，输出 RTTM）
- `benchmark_pytorch_vs_onnx.py`: 在同一批音频上对比 PyTorch vs ONNX 输出差异，并统计详细 RTF
- `utils.py`: 轻量工具函数（扫文件、读音频、frames->segments、写 RTTM）
- `models/`: 导出的 ONNX 模型默认输出目录

### Quickstart

#### 1) 导出 ONNX（best 和 epoch_0010）

```bash
conda run --no-capture-output -n diarizen python inference/export_to_onnx.py \
  --exp-dir /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_base \
  --config /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_base/config__2026_01_09--00_41_09.toml \
  --ckpt-name best \
  --out-onnx inference/models/kaldi_merged_1219_all_ft_base/best_multilabel_hard.onnx

conda run --no-capture-output -n diarizen python inference/export_to_onnx.py \
  --exp-dir /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_base \
  --config /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_base/config__2026_01_09--00_41_09.toml \
  --ckpt-name epoch_0010 \
  --out-onnx inference/models/kaldi_merged_1219_all_ft_base/epoch_0010_multilabel_hard.onnx
```

#### 2) 纯 ONNX 推理（输出 RTTM）

```bash
conda run --no-capture-output -n diarizen python inference/infer_onnx.py \
  /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios \
  --onnx inference/models/kaldi_merged_1219_all_ft_base/best_multilabel_hard.onnx \
  --out-dir /tmp/diar_onnx_out
```

#### 3) PyTorch vs ONNX 对比 + RTF

```bash
conda run --no-capture-output -n diarizen python inference/benchmark_pytorch_vs_onnx.py \
  /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios \
  --exp-dir /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_base \
  --config /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_base/config__2026_01_09--00_41_09.toml \
  --ckpt-name best \
  --onnx inference/models/kaldi_merged_1219_all_ft_base/best_multilabel_hard.onnx \
  --device cuda \
  --max-files 50
```

#### 备注：如何做到 PyTorch vs ONNX **严格 0 差异**

由于 CUDA 上的数值路径（尤其 TF32、以及 cuBLAS 的非确定性）会导致极少数帧的 argmax 翻转，
要让 `diff_max=0`，请用以下设置跑 benchmark：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n diarizen python inference/benchmark_pytorch_vs_onnx.py \
  /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios \
  --exp-dir /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_base \
  --config /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_base/config__2026_01_09--00_41_09.toml \
  --ckpt-name best \
  --onnx inference/models/kaldi_merged_1219_all_ft_base/best_multilabel_hard.onnx \
  --device cuda \
  --providers cuda,cpu \
  --torch-tf32 0 --ort-tf32 0 --deterministic 1
```


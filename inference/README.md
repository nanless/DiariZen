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

### L4 最长 16 秒、并发 50 的生产结论

`kaldi_merged_1219_all_ft_large/epoch_0016` 在 `dev_L4_1gpus` 上的新增加速实验全部使用固定 **16 秒 synthetic** 输入，不使用真实音频。详细数字、限制和产物见：

- `inference/models/kaldi_merged_1219_all_ft_large/SEGMENTATION_BENCHMARK_REPORT.md`
- `inference/ADR_L4_ONLINE_SERVING_16S_CONCURRENCY50.md`
- `docs/plans/2026-08-27-l4-inference-acceleration-design.md`

最终主路径是 TensorRT 10.10 FP16、batch=1、两个预分配 execution contexts。每个 context 独占 pinned host/device buffer 和 CUDA stream，engine 只反序列化一次。单槽位活跃时 replay 完整 H2D→TensorRT→D2H CUDA Graph，单 context 的 50 请求 burst mean/p95 改善约 2.67%/2.92%，host enqueue 降至约 8µs；双槽六模式差异约 0.21%，因此不实现复杂的固定 hybrid 调度，最终双槽 pinned enqueue 实测约 91.9 req/s。动态合批、4/8 contexts、当前 FP8/INT8/INT4、强制 Flash、O4/O5 builder plan 均不采用。

其他后端也完成了固定 16 秒对照：ORT CUDA Graph 比普通 ORT 快 2.09%，但仍比 TensorRT 慢 3.54×；真正按“先 SDPA patch、再 compile”执行的 PyTorch `compile(mode="reduce-overhead")+SDPA+AMP` 为 18.464ms，比 eager AMP 快 2.12×，但仍比 TensorRT 慢 1.72×。ORT Graph 是可选一级回退；PyTorch 组合路径是实验性二级候选。两者都只有 synthetic 路径一致性证据，真实音频未验收前不自动启用。

相关脚本：

- `benchmark_l4_runtime_acceleration.py`：TensorRT enqueue/pinned/CUDA Graph 配对测试。
- `tensorrt_cuda_graph_runner.py`：只依赖 NumPy、TensorRT 和 libcudart 的完整链路参考 runner。
- `benchmark_trt_online_burst_acceleration.py`：50 请求、单/双 context 的六模式交错压测。
- `benchmark_trt_builder_search.py`：builder optimization/workspace/aux stream 搜索。
- `benchmark_ort_cuda_graph_fixed16s.py`：ORT IOBinding/CUDA Graph。
- `benchmark_pytorch_compile_fixed16s.py`、`benchmark_attention_sdpa_fixed16s.py`：PyTorch compile/SDPA。

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

#### 1.1) 导出 ONNX（kaldi_merged_1219_all_ft_large 的 epoch_0002）

```bash
# 一键脚本（推荐）
./inference/run_export_kaldi_merged_1219_all_ft_large_epoch_0002.sh

# 或者手动指定参数：
conda run --no-capture-output -n diarizen python inference/export_to_onnx.py \
  --exp-dir /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large \
  --config /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large/config__2025_12_26--11_44_15.toml \
  --ckpt-name epoch_0002 \
  --out-onnx inference/models/kaldi_merged_1219_all_ft_large/epoch_0002_multilabel_hard.onnx
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

#### 4) PyTorch vs ONNX 精度对比（DER/JER…，需要 reference RTTM）

这个评测会：
- 用 PyTorch checkpoint 和 ONNX 在同一批音频上生成 RTTM
- 使用仓库内置 `dscore` 对系统 RTTM 与 reference RTTM 进行打分
- 输出 `report.json`（包含 overall/per-file 的 DER/JER/B3/NMI…）

```bash
conda run --no-capture-output -n diarizen python inference/compare_pytorch_vs_onnx_accuracy.py \
  /path/to/audios \
  --ref-rttm-dir /path/to/reference_rttm_dir \
  --exp-dir /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large \
  --config /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large/config__2025_12_26--11_44_15.toml \
  --ckpt-name epoch_0002 \
  --onnx inference/models/kaldi_merged_1219_all_ft_large/epoch_0002_multilabel_hard.onnx \
  --out-dir /tmp/diar_pytorch_vs_onnx_acc_epoch_0002
```

如果你的 reference RTTM 的 `file_id` 是音频文件名（不带扩展名），保持默认 `--session-id stem` 即可；
如果 reference RTTM 使用的是 DiariZen 常见的 `relative_path__style`，则加 `--session-id relative`。

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

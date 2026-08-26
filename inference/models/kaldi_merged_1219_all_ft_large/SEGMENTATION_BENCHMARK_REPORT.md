# 分割模型离线推理 Benchmark 详细报告 — `epoch_0016`

> **实验名称**：`kaldi_merged_1219_all_ft_large` / checkpoint `epoch_0016`  
> **测试范围**：**仅 segmentation 模型**（不含 speaker embedding、VBx 聚类、滑窗重叠、RTTM 后处理）  
> **测试日期**：2026-08-26  
> **测试 GPU**：NVIDIA A800-SXM4-80GB（双卡，benchmark 使用空闲的 **GPU 0**）  
> **对应线上模型**：与 `speaker_diarize_infer` 中 `epoch_0016_multilabel_hard.onnx` 为同一文件（MD5: `0a2142c0874e553206633e65b8348dd1`）

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [执行摘要](#2-执行摘要)
3. [模型架构与剪枝说明](#3-模型架构与剪枝说明)
4. [测试环境与方法论](#4-测试环境与方法论)
5. [速度 Benchmark 详细结果](#5-速度-benchmark-详细结果)
6. [精度 Benchmark 详细结果](#6-精度-benchmark-详细结果)
7. [INT8 量化实验（dynamic / static）](#7-int8-量化实验dynamic--static)
8. [TensorRT / FP8 部署方案对照](#8-tensorrt--fp8-部署方案对照)
9. [问题与限制](#9-问题与限制)
10. [推荐生产配置](#10-推荐生产配置)
11. [复现步骤](#11-复现步骤)
12. [产物与脚本索引](#12-产物与脚本索引)
13. [结论与后续工作](#13-结论与后续工作)

---

## 1. 背景与目标

### 1.1 为什么要做这组测试

`epoch_0016` 是当前说话人分割（segmentation）主模型，已在 `speaker_diarize_infer` 中以 ONNX 形式部署。为评估 **GPU 离线 batch 推理** 能力，需要回答：

- PyTorch vs ONNX Runtime 谁更快？batch 能开到多大？
- INT8 量化能否在保持精度的前提下进一步加速？
- 按 GPT5.6 建议的 TensorRT FP16 / selective FP8 路线是否可行？

### 1.2 测试假设（严格限定）

| 假设 | 说明 |
|------|------|
| 输入 | 16 kHz 单通道，固定 **10 秒** → `[B, 1, 160000]` |
| 输出 | 帧级 multilabel `{0,1}`，shape `[B, frames, 4]`（最多 4 说话人） |
| 不包含 | embedding 提取、聚类、长音频滑窗、VAD、后处理 |
| 并发语义 | 「50 路并发」= 50 条 10s 音频同时到达，拼成 **microbatch 一次 forward** |

### 1.3 评估指标

**速度**：wall-clock 延迟（ms）、均摊每条（ms/item）、RTF、每秒可处理音频时长（audio_sec/s）、p50/p95。

**精度**：
- 帧级完全一致率（frame exact match）
- speaker-cell 一致率（每帧每个说话人维度）
- cross DER / cross JER（将 FP32 参考输出与待测输出分别转 RTTM，用 dscore 交叉评分，collar=0）

---

## 2. 执行摘要

| 维度 | 结论 |
|------|------|
| **推荐部署** | **ONNX Runtime + CUDAExecutionProvider（FP32）** |
| **50 条 10s 墙钟** | **233 ms**（均摊 4.7 ms/条），约为 PyTorch FP16 的 **1.5×** 加速 |
| **精度** | ORT vs PyTorch FP32：帧一致 **99.998%**，cross DER **0.004%**（几乎无损） |
| **PyTorch FP16** | 大 batch 有加速，但不如 ORT；cross DER +0.26%（可接受） |
| **dynamic INT8** | 精度尚可（DER +4.3%），但 GPU **慢 12×**，不可用 |
| **static INT8** | 精度崩溃（DER 53.7%），GPU 慢 1.4×，不可用 |
| **TensorRT FP8** | 本机 TRT 无法初始化；A800 无 FP8 TC → 需 **L4** 复测 |

**一句话**：当前最高性价比方案是 **ORT CUDA FP32**，不要在本模型上做 INT8 PTQ。

---

## 3. 模型架构与剪枝说明

### 3.1 推理链路

```
原始波形 [B, 1, 160000]  (10s @ 16kHz)
    │
    ▼
WavLM Large s80 结构化剪枝主干（63.1M 参数）
    │  CNN 特征提取（通道 153~512 不等）
    │  24 层 Transformer（部分层去掉 Attention，head 数 5~8 不等）
    │  25 路特征加权（CNN 输出 + 24 层 hidden states）
    ▼
4 层 Conformer（attention_dim=256, FFN=1024, 4 heads, ~6.4M 参数）
    ▼
Powerset 分类头 → argmax 硬解码 → multilabel [B, frames, 4]
```

**总参数量**：约 **69.46M**（实测 checkpoint 统计）

| 子模块 | 参数量 |
|--------|--------|
| WavLM s80 主干 | 63.10M |
| Conformer + 分类头 | 6.36M |
| **合计** | **69.46M** |

### 3.2 相对原始 WavLM Large 剪了多少

本模型 backbone 为 `wavlm_large_s80_md`（`target_sparsity = 0.8` 结构化剪枝 + 下游 finetune）：

| 指标 | 原始 WavLM Large | s80 剪枝后 | 变化 |
|------|------------------|------------|------|
| 参数量 | 315.5M | 63.1M | **减少 80%**（保留约 20%） |
| MACs（论文） | 17.8G | 3.8G | 减少约 79%，推理约 2.6× |
| 剪枝方式 | — | 结构化 | conv 通道、attention head、FFN 中间维、整层 attention 删除 |

剪枝细节（`WAVLM_LARGE_S80_MD` 配置）：
- CNN 通道：512 → 153/224/255/302/368/211 等
- 完全去掉 Attention 的层：第 9、12、16、17 层
- 每层 FFN 中间维：96~1770 不等（原始统一 4096）
- 预训练权重来源：`BUT-FIT/diarizen-wavlm-large-s80-md`

### 3.3 训练与导出

| 阶段 | 路径 / 配置 |
|------|------------|
| Finetune 配置 | `recipes/diar_ssl/conf/kaldi_merged_1219_all_ft_large.toml` |
| Checkpoint | `recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large/checkpoints/epoch_0016/pytorch_model.bin` |
| ONNX 导出脚本 | `inference/run_export_kaldi_merged_1219_all_ft_large_epoch_0002.sh` |
| ONNX 特点 | 导出时内置 powerset → hard multilabel 解码，推理端无需额外后处理 |

---

## 4. 测试环境与方法论

### 4.1 硬件与软件

| 项目 | 配置 |
|------|------|
| GPU | 2× NVIDIA A800-SXM4-80GB |
| GPU 使用策略 | benchmark 前检查 `nvidia-smi`，仅使用空闲 GPU 0 |
| Python 环境 | conda `diarizen`，PyTorch 2.1.1，CUDA 12.1 |
| ONNX Runtime | CUDA EP + CPU EP |
| TensorRT | 系统 10.13.3.9，**CUDA init error 35**，`diarizen` 内 pip 安装失败 |

### 4.2 速度测试方法

1. 输入：全零 tensor `[B, 1, 160000]`（消除 I/O 干扰，测纯算力）
2. Warmup：5 次
3. 计时：20 次取均值，报告 p50/p95
4. PyTorch：GPU 端 `torch.cuda.synchronize()` 前后计时
5. ORT：CUDA EP，`GraphOptimizationLevel.ORT_ENABLE_ALL`
6. Batch 规模：1 / 8 / 32 / 50

### 4.3 精度测试方法

- **评估集**：21 条真实音频，总时长 1074.5s
  - `example/EN2002a_30s.wav`
  - `speaker_diarize_infer/test_audios/bench_largescale_20/` 下 20 条
- **参考系**：PyTorch FP32（CPU），powerset argmax 后转 multilabel
- **待测**：ONNX Runtime（CPU EP，因 CUDA EP 长音频 bug）
- **DER 计算**：帧标签 → RTTM → dscore cross DER（collar=0，无 collar 容忍）

---

## 5. 速度 Benchmark 详细结果

### 5.1 固定 10s 输入 — 主表（A800 GPU 0）

| Backend | bs=1 | bs=8 | bs=32 | bs=50 | 均摊/条 (bs=50) | p95 (bs=50) |
|---------|------|------|-------|-------|-----------------|-------------|
| **ORT CUDA FP32** | 12.5 ms | 43.4 ms | 152.7 ms | **232.7 ms** | **4.7 ms** | 233.0 ms |
| PyTorch FP16 | 100.1 ms | 140.4 ms | 289.7 ms | 349.0 ms | 7.0 ms | 375.4 ms |
| PyTorch FP32 | 79.8 ms | 198.5 ms | 400.3 ms | 532.4 ms | 10.6 ms | 572.6 ms |

**RTF（bs=50）**：ORT 0.00047，PyTorch FP16 0.00070，PyTorch FP32 0.00106

**吞吐量（bs=50）**：ORT **2148 audio-sec/s**（即每秒可处理约 2148 秒等效 10s 音频）

### 5.2 为什么 ORT 比 PyTorch 快约 4~5×

1. **静态图优化**：算子融合、常量折叠、内存规划
2. **无 Python eager 调度开销**
3. **导出时固化解码**：ONNX 内含 powerset→multilabel，少一层 Python 逻辑
4. **CUDA EP 对 Transformer/Conv 栈有成熟 fusion**

### 5.3 不同音频长度 — 单条 forward（历史数据）

| 音频长度 | PyTorch GPU | ORT CUDA | 加速比 |
|---------|-------------|----------|--------|
| 8s | 57.5 ms | 11.9 ms | 4.8× |
| 10s | 66 ms | 13 ms | 5.1× |
| 30s | 164.5 ms | 33.7 ms | 4.9× |
| 120s | 1249 ms | 311 ms | 4.0× |

### 5.4 50 路并发场景（microbatch）

| 音频长度 | 方案 | 墙钟 (50条) | 说明 |
|---------|------|------------|------|
| **10s** | ORT bs=50 | **~233–238 ms** | 可直接一次跑完 |
| **10s** | PyTorch bs=50 | ~349–540 ms | 显存充足时 bs=50 |
| 30s | ORT 串行 50 次 | ~1.7 s | 无 batching |
| 30s | ORT bs=32 | ~2.0 s | bs=50 OOM |
| 60s | ORT 串行 50 次 | ~4.7 s | — |

### 5.5 INT8 量化速度（10s × 50，GPU 0）

| 模型 | batch=50 墙钟 | 相对 FP32 ORT | 备注 |
|------|--------------|---------------|------|
| FP32 ONNX | **233 ms** | 1.0× | 基线 |
| static INT8 QDQ | 330 ms | 1.4× 慢 | 16 Memcpy 节点 |
| dynamic INT8 | **2901 ms** | **12× 慢** | 456 Memcpy 节点，GPU 极不友好 |

### 5.6 L4 粗估（未实测，仅供参考）

L4 显存带宽约为 A800 的 1/5~1/6。10s×50 ORT FP32 粗估 **1.0–1.5 s**（带宽敏感型负载）。

---

## 6. 精度 Benchmark 详细结果

### 6.1 FP32 路径对比（21 条音频，1074.5s）

| 对比组 | 帧一致率 | cell 一致率 | cross DER | cross JER |
|--------|----------|-------------|-----------|-----------|
| **ORT CPU vs PyTorch FP32** | **99.998%** | **99.9995%** | **0.004%** | **0.002%** |
| PyTorch FP16 vs FP32（全长度） | 99.870% | 99.960% | 0.263% | 0.284% |
| 10s 切片 ORT vs FP32 | **100%** | **100%** | — | — |
| 10s 切片 FP16 vs FP32 | 99.722% | 99.894% | — | — |

**解读**：
- ONNX 导出与 PyTorch FP32 **几乎 bit-level 一致**（仅 1 帧 / 53702 帧有差异）
- FP16 autocast 在短音频上差异更小；长音频累积误差导致 DER 约 +0.26%
- 生产若用 FP16 PyTorch，需评估是否满足 DER ≤ +0.2% 的验收线

### 6.2 INT8 vs FP32 ONNX（CPU ORT，21 条音频）

| 方案 | 帧一致率 | cell 一致率 | cross DER | cross JER | 体积 |
|------|----------|-------------|-----------|-----------|------|
| dynamic INT8 | 97.88% | 99.28% | **4.28%** | 9.95% | 98 MB |
| static INT8 MinMax | 72.63% | 91.54% | **53.72%** | 22.35% | 98 MB |
| static INT8 Entropy+10s | 71.25% | 90.67% | **53.80%** | 23.27% | 98 MB |

### 6.3 异常样本（dynamic INT8）

| 音频 | 时长 | 帧不一致率 | cross DER 贡献 |
|------|------|-----------|---------------|
| `3078f036c56248b19467cae6d2b86777.wav` | 18.9s | **43.4%** | 主要 outlier |
| `c57495123d65410baa9775e5c8639102.wav` | 284.3s | 3.5% | 段数差异大（355→247 段） |

大部分音频帧一致率 > 99%，INT8 误差集中在边界帧和重叠语音区域。

---

## 7. INT8 量化实验（dynamic / static）

### 7.1 Dynamic INT8（不推荐 GPU 部署）

```
方法：onnxruntime.quantization.quantize_dynamic
量化算子：MatMul, Gemm（Conv 保持 FP32）
产物：epoch_0016_multilabel_hard.matmul-dynamic-int8.onnx（266MB → 98MB）
```

| 维度 | 结果 |
|------|------|
| GPU 速度 | 比 FP32 **慢 12×**（大量 Memcpy） |
| CPU 速度 | 仅快约 3% |
| 精度 | DER +4.3%，多数场景可接受 |
| 结论 | **体积减小但 GPU 无收益，不推荐** |

### 7.2 Static INT8 QDQ（不推荐）

```
方法：onnxruntime.quantization.quantize_static（QDQ 格式）
脚本：inference/quantize_segmentation_onnx_static.py
校准：21 条真实音频（MinMax）；另试 Entropy + 10s 固定长度 padding
量化算子：MatMul, Gemm（Conv 保持 FP32）
产物：epoch_0016_multilabel_hard.static-int8.onnx
```

| 维度 | 结果 |
|------|------|
| GPU 速度 | 比 FP32 慢 1.4× |
| 精度 | DER 53.7%，**完全不可用** |
| Entropy 校准 | 无改善 |
| 含 Conv 量化 | 尝试时 OOM |
| 结论 | **static INT8 不适合本模型** |

### 7.3 运行 static 量化所需环境变量

```bash
export LD_LIBRARY_PATH=/root/miniforge3/envs/diarizen/lib:$LD_LIBRARY_PATH
# 解决 onnx 导入时 CXXABI_1.3.15 问题
```

---

## 8. TensorRT / FP8 部署方案对照

依据 GPT5.6 建议，理想部署路线为：

```
TensorRT FP16 基线
  → selective FP8（大型 WavLM Linear）
  → SmoothQuant INT8 对照
  → Conformer 保持 FP16
  → LayerNorm / residual / CNN 保持 FP16
```

### 8.1 实测状态

| 方案 | 状态 | 原因 |
|------|------|------|
| TensorRT FP16 | ❌ 未跑 | TRT CUDA init error 35 |
| Selective FP8 | ❌ 未跑 | 需 L4/H100 FP8 TC + TRT builder |
| SmoothQuant INT8 | ❌ 未跑 | 需 TRT explicit Q/DQ |
| PyTorch FP16 基线 | ✅ | 已完成 |
| ORT CUDA FP32 | ✅ | 当前最优 |

### 8.2 待 L4 环境执行的测试清单

1. 为 bs=1/4/8/32 各建独立 TRT engine（固定 `[B,1,160000]`）
2. WavLM 大型对齐 Linear → FP8；LayerNorm 保持 FP16/FP32 统计
3. SmoothQuant alpha 扫描：0.4 / 0.5 / 0.6 / 0.7
4. Profile：Q/DQ 融合率、Reformat 占比、FP8 GEMM 命中率
5. 精度验收线：DER 增幅 ≤ 0.2%，p95 延迟改善 ≥ 10%

---

## 9. 问题与限制

| 问题 | 影响 | 临时规避 |
|------|------|----------|
| ORT CUDA EP 长音频 `rel_attn Gather` 报错 | 部分 >60s 音频无法用 CUDA EP | 精度评估改用 CPU EP；或拆分短窗 |
| PyTorch GPU 长音频 OOM | 284s 音频 attention 显存爆炸 | 精度参考用 CPU FP32 |
| TensorRT 不可用 | 无法验证 FP16/FP8 编译收益 | 在 L4 机器复测 |
| A800 无 FP8 TC | FP8 benchmark 无意义 | 换 L4 |
| INT8 PTQ 失败 | 无法通过量化压缩延迟 | 保持 FP32 ORT |

---

## 10. 推荐生产配置

```yaml
# 分割模型 serving 推荐配置（基于 2026-08-26 实测）
backend: onnxruntime
provider: CUDAExecutionProvider  # 失败时 fallback CPUExecutionProvider
model: epoch_0016_multilabel_hard.onnx
input_dtype: float32
input_shape: [B, 1, 160000]      # 10s 固定；可变长需另行处理
batch_strategy:
  audio_10s: bs=50               # 50 路并发一次 forward
  audio_30s: bs=32               # bs=50 OOM
  audio_60s: bs=16               # 按显存调整
expected_latency_10s_x50: ~233ms # A800 参考值
precision_vs_fp32: negligible    # DER +0.004%
avoid:
  - dynamic_int8_on_gpu
  - static_int8
  - pytorch_eager_production
```

---

## 11. 复现步骤

### 11.1 导出 ONNX

```bash
cd /root/code/github_repos/DiariZen
bash inference/run_export_kaldi_merged_1219_all_ft_large_epoch_0002.sh
# 默认 CKPT_NAME=epoch_0016
# 输出：inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx
```

### 11.2 速度 Benchmark

```bash
export CUDA_VISIBLE_DEVICES=0

conda run --no-capture-output -n diarizen python inference/benchmark_segmentation_precision.py \
  --mode speed \
  --config recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large/config__2025_12_26--11_44_15.toml \
  --ckpt recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large/checkpoints/epoch_0016/pytorch_model.bin \
  --onnx inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx \
  --out-json inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_precision_benchmark.json \
  --batch-sizes 1,8,32,50 \
  --skip-trt
```

### 11.3 精度 Benchmark

```bash
conda run --no-capture-output -n diarizen python inference/benchmark_segmentation_precision.py \
  --mode accuracy \
  --config recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large/config__2025_12_26--11_44_15.toml \
  --ckpt recipes/diar_ssl/exp/kaldi_merged_1219_all_ft_large/checkpoints/epoch_0016/pytorch_model.bin \
  --onnx inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx \
  --out-json inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_precision_benchmark.json \
  --accuracy-ref pytorch_fp32_cpu
```

### 11.4 Static INT8 量化

```bash
LD_LIBRARY_PATH=/root/miniforge3/envs/diarizen/lib:$LD_LIBRARY_PATH \
conda run --no-capture-output -n diarizen python inference/quantize_segmentation_onnx_static.py \
  --input inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx \
  --calibration-root example \
  --calibration-root /path/to/bench_largescale_20 \
  --max-calibration-files 21 \
  --calibration-method MinMax \
  --op-types MatMul,Gemm
```

---

## 12. 产物与脚本索引

### 12.1 脚本（纳入 git）

| 文件 | 说明 |
|------|------|
| `inference/benchmark_segmentation_precision.py` | 速度 + 精度 benchmark |
| `inference/quantize_segmentation_onnx_static.py` | Static INT8 QDQ 量化 |
| `inference/run_export_kaldi_merged_1219_all_ft_large_epoch_0002.sh` | ONNX 导出 |

### 12.2 报告与 JSON（纳入 git）

| 文件 | 说明 |
|------|------|
| `SEGMENTATION_BENCHMARK_REPORT.md` | 本文档 |
| `epoch_0016_precision_benchmark.json` | 速度 + FP32/FP16 精度 |
| `epoch_0016_int8_accuracy_report.json` | dynamic INT8 逐文件精度 |
| `epoch_0016_static_int8_accuracy_report.json` | static INT8 逐文件精度 |

### 12.3 本地模型文件（**.gitignore 排除，不入库**）

| 文件 | 大小 |
|------|------|
| `epoch_0016_multilabel_hard.onnx` | 266 MB |
| `epoch_0016_multilabel_hard.matmul-dynamic-int8.onnx` | 98 MB |
| `epoch_0016_multilabel_hard.static-int8.onnx` | 99 MB |
| `epoch_0016_multilabel_hard.static-int8-entropy-10s.onnx` | 99 MB |

---

## 13. 结论与后续工作

### 13.1 结论

1. **生产首选**：`epoch_0016_multilabel_hard.onnx` + ORT CUDA FP32
2. **50 路 10s 并发**：A800 上约 **233 ms** 完成，分割不是瓶颈
3. **INT8 PTQ 全面失败**：dynamic 慢、static 精度崩，均不采用
4. **TensorRT FP8 方案**：理论最优但需 L4 环境，本机无法验证

### 13.2 后续工作（按优先级）

| 优先级 | 任务 | 环境要求 |
|--------|------|----------|
| P0 | L4 上跑 TensorRT FP16 engine | L4 + TRT |
| P0 | L4 上跑 selective FP8 | L4 + TRT |
| P1 | 修复 ORT CUDA 长音频 Gather bug | 改 ONNX 导出或 ORT 版本 |
| P1 | SmoothQuant INT8 alpha 扫描 | L4 + TRT |
| P2 | 选择性 QAT（若 PTQ 不达标） | GPU 训练 |
| P2 | PyTorch 在线 25 层 FP32 加权（降显存） | 代码改动 |

---

*报告生成：DiariZen inference benchmark，2026-08-26*

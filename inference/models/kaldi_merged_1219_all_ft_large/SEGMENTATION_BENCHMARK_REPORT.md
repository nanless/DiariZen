# 分割模型离线推理 Benchmark 详细报告 — `epoch_0016`

> **实验名称**：`kaldi_merged_1219_all_ft_large` / checkpoint `epoch_0016`  
> **测试范围**：**仅 segmentation 模型**（不含 speaker embedding、VBx 聚类、滑窗重叠、RTTM 后处理）  
> **测试日期**：2026-08-26～2026-08-27<br>
> **测试 GPU**：NVIDIA A800-SXM4-80GB（原始测试）与 NVIDIA L4 24GB（固定 10s、最大 16s 与并发 50 复测）<br>
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
| 输入 | 原始基准为固定 10 秒；线上容量补测覆盖 16 kHz 单通道、最长 **16 秒** → `[B, 1, 256000]` |
| 输出 | 帧级 multilabel `{0,1}`，shape `[B, frames, 4]`（最多 4 说话人） |
| 不包含 | embedding 提取、聚类、长音频滑窗、VAD、后处理 |
| 并发语义 | 原始 A800 测试为 50 条 10s 拼 microbatch；L4 线上补测为 50 条 16s 同时到达并进入 batch=1 context/stream 队列 |

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
| **推荐部署** | L4 最长 16s 首选 **TensorRT 10.10 FP16**；固定 buffer + pinned memory + 独立 context/stream，单活跃槽位使用完整 CUDA Graph |
| **50 条 10s 墙钟** | **233 ms**（均摊 4.7 ms/条），约为 PyTorch FP16 的 **1.5×** 加速 |
| **精度** | ORT vs PyTorch FP32：帧一致 **99.998%**，cross DER **0.004%**（几乎无损） |
| **PyTorch FP16** | 大 batch 有加速，但不如 ORT；cross DER +0.26%（可接受） |
| **dynamic INT8** | 精度尚可（DER +4.3%），但 GPU **慢 12×**，不可用 |
| **static INT8** | 精度崩溃（DER 53.7%），GPU 慢 1.4×，不可用 |
| **A800 TensorRT / FP8** | 本机 TRT 无法初始化；A800 无 FP8 TC，因此未测 |
| **L4 ORT FP32** | 10s×1/8/32/50：**20.5 / 163.0 / 748.7 / 1184.2 ms**；bs=50 约比 A800 慢 5.1× |
| **L4 INT8** | dynamic/static 均慢于 ORT FP32；三种 INT8 在 bs=50 均 OOM |
| **L4 TensorRT FP16** | 10.10.0.31 已解决旧版崩溃；bs=1/4/8/32 比 ORT FP32 快 **2.96–3.26×** |
| **L4 TensorRT 多精度** | 已完成严格 FP32、TF32、FP16、BF16、FP8、INT8、INT4 weight-only；四档 batch 中均为 **FP16 最快** |
| **TensorRT PTQ 精度** | FP8 frame exact 约 **98.99–99.40%**；INT8 仅约 **0.20–0.60%** 且运行时非确定，不可用；INT4 weight-only 为 **96.79%** 且无加速 |
| **L4 线上 16s** | FP16 batch=1 **10.741ms**；最终双 context pinned 链路 50 条总完成 mean/p95 **543.9/547.3ms**、请求完成 p95 **522.2ms**、约 **91.9 req/s** |
| **16s CUDA Graph** | 单 context 全链路 graph 比普通 enqueue mean/p95 快约 **2.6%/2.9%**，host enqueue 从约 1.425ms 降至 0.0079ms；双 context 时 6 种组合差异仅约 0.21%，不宣称额外吞吐收益 |
| **TensorRT builder 扫描** | O4/O5、8/16GiB workspace、0/2 auxiliary streams 均未稳定优于现有 **O3/8GiB** plan；保留现有 engine |
| **其他加速路径** | ORT CUDA Graph 仅快 **2.09%** 且仍比 TRT 慢 **3.54×**；真正的 PyTorch `compile(reduce-overhead)+SDPA+AMP` 为 **18.464ms**，比 eager AMP 快 **2.12×**，但仍比 TRT 慢 **1.72×** |
| **线上 batch 策略** | **禁用动态合批**；16s 的 bs=2/4/8/16/32 每条成本均高于 bs=1，bs=32 吞吐仅为 bs=1 的 63.1% |
| **可变时长** | 动态 duration plan 覆盖 2–16s；结合固定 10s/16s plan 做长度路由，短音频不必全部补到 16s |

**一句话**：线上最长 16s、并发上限 50 时，使用 **1 张 L4 + TensorRT FP16 + batch=1 + 2 个预分配 execution contexts + pinned/设备 buffer 复用 + 按时长路由**；单槽低并发走 CUDA Graph，50 请求突发由双槽队列调度，不做动态合批，ORT CUDA FP32 仅作为独立进程回退。

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

### 5.6 L4 固定 10 秒实测（2026-08-26）

本节只使用固定 `[B, 1, 160000]` 的 10 秒 synthetic 输入；速度输入为全零 tensor。未运行真实音频、长音频或 DER/JER。每档 warmup 5 次、正式计时 20 次。

#### 5.6.1 环境

| 项目 | L4 配置 |
|------|---------|
| GPU | NVIDIA L4 24GB，compute capability 8.9 |
| Driver / CUDA | 535.129.03 / CUDA 12.4 |
| PyTorch | 2.1.1，CUDA 12.1（conda `diarizen`） |
| ONNX Runtime | 1.22.0，CUDA EP + CPU EP |
| TensorRT | 10.10.0.31（Conda env `diarizen-trt1010`，路径 `/root/miniforge3/envs/diarizen-trt1010`） |

#### 5.6.2 FP32 / FP16 基线

| Backend | bs=1 | bs=8 | bs=32 | bs=50 | p95（bs=50） | audio-sec/s（bs=50） |
|---------|------|------|-------|-------|---------------|----------------------|
| **ORT CUDA FP32** | **20.49 ms** | **162.99 ms** | **748.72 ms** | **1184.15 ms** | 1185.72 ms | 422.2 |
| PyTorch FP16 | 27.53 ms | 164.55 ms | 749.01 ms | 1184.71 ms | 1186.68 ms | 422.0 |
| PyTorch FP32 | 30.02 ms | 250.37 ms | 1063.89 ms | 1696.65 ms | 1701.84 ms | 294.7 |

结论：ORT CUDA 只在 bs=1 明显领先 PyTorch FP16（约 1.34×）；bs=8/32/50 二者几乎相同。ORT bs=50 的 1184 ms 与 A800 的 233 ms 相比慢约 **5.1×**，此前 1.0–1.5s 的粗估被实测验证。

#### 5.6.3 INT8 速度与容量边界

| 模型 | bs=1 | bs=8 | bs=32 | bs=50 | 相对 ORT FP32 | ORT 节点分配（CUDA / CPU） |
|------|------|------|-------|-------|----------------|-----------------------------|
| dynamic INT8 | 106.93 ms | 543.63 ms | 2298.06 ms | **OOM** | 慢 3.1–5.2× | 2170 / 678 |
| static INT8 MinMax | 40.09 ms | 195.46 ms | 994.64 ms | **OOM** | 慢 1.2–2.0× | 2882 / 394 |
| static INT8 Entropy+10s | 38.54 ms | 196.65 ms | 997.65 ms | **OOM** | 慢 1.2–1.9× | 2882 / 394 |

- dynamic INT8 触发 ORT 警告：插入 **456 个 Memcpy**；static INT8 为 16 个。CPU fallback 与设备搬运抵消了量化收益。
- 三种 INT8 在 bs=50 均于首层 LayerNorm 申请 `3,276,697,600` 字节 buffer 时失败；L4 上 INT8 的可用上限为 bs=32。
- 用 seeds 1001/1002/1003 的 3 条 10 秒高斯噪声（均值 0、标准差 0.05）检查，三种 INT8 相对 FP32 ONNX 均为 cell/frame **100% exact match**（1497 帧）。这只是 synthetic 功能一致性检查，不能替代真实音频 DER。

#### 5.6.4 TensorRT FP16：问题修复与实测

旧环境 TensorRT 10.0.1 能解析 ONNX、能构建 FP32，却在 FP16 `build_serialized_network` 内稳定段错误。解决方案是在不修改 `diarizen/cosyvoice` 的前提下创建独立环境，并升级到 TensorRT **10.10.0.31**。环境最初为 Python venv，已于 2026-08-27 迁移为可由 `conda env list` 管理的同名 Conda env；Python/PyTorch/CUDA/TensorRT 版本保持不变，并已重新验证 FP16 engine 反序列化及 10 秒 synthetic 推理。新版成功将图解析/优化为 6192 层并生成全部 FP16 engines。

TensorRT 检测到 LayerNorm FP16 溢出风险，自动建议/选择让相关 Reduce/Pow 使用更高精度；因此这里的“FP16”是 TensorRT mixed-precision FP16，而不是不安全的全算子强制 FP16。

| Batch | TensorRT mean | p50 | p95 | ORT FP32 mean | 加速比 | audio-sec/s | Engine | Build |
|-------|---------------|-----|-----|---------------|--------|-------------|--------|-------|
| 1 | **6.283 ms** | 6.418 ms | 6.434 ms | 20.49 ms | **3.26×** | 1591.5 | 163.6 MB | 175.5s |
| 4 | **25.662 ms** | 25.704 ms | 26.000 ms | 75.97 ms | **2.96×** | 1558.8 | 184.4 MB | 219.3s |
| 8 | **53.450 ms** | 53.200 ms | 55.965 ms | 162.99 ms | **3.05×** | 1496.7 | 210.4 MB | 241.9s |
| 32 | **240.275 ms** | 239.981 ms | 241.748 ms | 748.72 ms | **3.12×** | 1331.8 | 385.4 MB | 366.9s |

#### 5.6.5 TensorRT 各精度横向实测

使用 NVIDIA ModelOpt 0.46.0 生成显式 Q/DQ 图；FP8/INT8 量化 MatMul/Gemm，未量化层保留 FP16/FP32。INT4 使用 block=128 的 weight-only DQ；为满足 TensorRT 的整除约束，排除 56 个输入维度不能被 128 整除的权重，保留 128 个 INT4 权重节点。

| TensorRT 路径 | bs=1 | bs=4 | bs=8 | bs=32 | 引擎检查证据（bs=1/4） | 结论 |
|---------------|------|------|------|-------|--------------------------|------|
| **FP16 mixed** | **6.283 ms** | **25.662 ms** | **53.450 ms** | **240.275 ms** | FP16 builder + 高精度 LayerNorm fallback | **四档最快** |
| INT8 explicit Q/DQ | 6.502 ms | 28.726 ms | 60.439 ms | 270.433 ms | 206 / 207 个 INT8 标记 | 速度第二，但 synthetic 精度崩溃 |
| FP8 explicit Q/DQ | 6.982 ms | 31.444 ms | 66.072 ms | 302.986 ms | 330 / 331 个 FP8 标记 | 真正命中 FP8，但慢于 FP16 |
| BF16 mixed | 9.755 ms | 37.925 ms | 78.789 ms | 352.353 ms | 175 / 176 个 BF16 标记 | 可用但无性能优势 |
| FP32 + TF32 | 13.361 ms | 55.502 ms | 116.962 ms | 526.436 ms | 462 / 474 个 Float 标记 | 比严格 FP32 降低约 15–23% 延迟 |
| INT4 weight-only | 13.702 ms | 55.930 ms | 120.233 ms | 534.227 ms | 128 个 INT4 DQ 权重；0 个 INT4 计算标记 | 实际 Float 计算，不加速 |
| 严格 FP32 | 17.249 ms | 68.271 ms | 147.735 ms | 616.571 ms | TF32 已关闭 | 精度控制组 |
| FP4 / NVFP4 | — | — | — | — | L4 为 Ada SM8.9，无 FP4 Tensor Core | **硬件不支持，不做伪回退测速** |

FP8 首次构建时报非 INT8 Q/DQ 类型推导失败；将网络创建方式改为 `STRONGLY_TYPED`，并去掉与强类型网络冲突的 builder precision flag 后成功。检查器确认 FP8/INT8 引擎实际含对应低精度层，因此这些数字不是仅改文件名或自动回退得到的结果。

Engine inspector 的进一步分析解释了低精度没有超过 FP16 的原因：FP8 的 Q/DQ-bearing layers 占 **48.7–62.0%**、Reformat 占 **4.46–5.65%**，低精度 MatMul/Gemm 命中率约 **85.1–85.3%**；INT8 的对应值为 **43.3–47.8%**、**6.67–7.39%**、**87.4–87.7%**。INT4 的低精度 MatMul/Gemm 命中率为 **0%**。量化转换与仍需 FP16/FP32 执行的算子抵消了低精度 GEMM 收益。

#### 5.6.6 非静音 synthetic 输出一致性

普通 sine/noise synthetic 被模型全部判为静音，不足以验证精度。为避免使用真实音频，使用 PyTorch 梯度生成了一条固定 10 秒 adversarial synthetic waveform：ORT FP32 每条输出中有 480 个正类 cell，再复制到各 batch 对比 TensorRT hard multilabel。

| TensorRT 路径 | bs=1 frame exact | bs=4 | bs=8 | bs=32 | 判断 |
|---------------|------------------|------|------|-------|------|
| 严格 FP32 | 100.0000% | 99.7495% | 99.8246% | 99.5741% | batch/tactic 边界有少量 hard-decision 差异 |
| FP32 + TF32 | 100.0000% | 99.7495% | 99.7996% | 99.5741% | 与严格 FP32 接近 |
| FP16 mixed | 100.0000% | 99.7495% | 99.7996% | 99.5554% | 当前速度/一致性最佳折中 |
| BF16 mixed | 100.0000% | 99.7495% | 99.8246% | 99.0105% | bs=32 差异略增 |
| FP8 explicit Q/DQ | 99.3988% | 99.2986% | 99.3487% | 98.9917% | 可运行，但不快于 FP16 |
| INT8 explicit Q/DQ | 0.2004% | 0.5010% | 0.4008% | 0.3945% | **PTQ 校准失真且非确定，不可用** |
| INT4 weight-only | 96.7936% | 96.7936% | 96.7936% | 96.7936% | 精度更差且更慢 |

INT8 的正类总数与参考接近，但位置几乎全部错位（cell exact 仅约 52.6%），排除了“只是全零/全一输出”的假象。表中 INT8 是一次 warmup 后的代表性回归；独立进程复测 frame exact 仍在约 0.20–0.60% 间波动。全 28 engine 在迁移后的 Conda env 中连续执行两次：除 INT8 外均为 **100% repeat exact**；INT8 仅为 **99.80–99.90% cell repeat exact**，说明同一 context、同一输入仍存在 hard-decision 波动。以上是比全静音输入更严格的 synthetic 功能检查，但按本轮约束没有真实音频，不能换算为 DER/JER，也不能替代上线验收。

### 5.7 L4 线上最大 16 秒、并发 50（2026-08-27）

本节仍只使用 synthetic tensor，不使用真实音频。固定 16 秒测试 warmup 10 次、计时 50 次；动态时长最终复测计时 100 次；50 请求突发测试每档 30 次。

#### 5.7.1 固定 16 秒 FP16：大 batch 反而降低吞吐

| Batch | mean | p95 | 每条 mean | req/s | audio-sec/s |
|-------|------|-----|-----------|-------|-------------|
| **1** | **10.741 ms** | **10.823 ms** | **10.741 ms** | **93.10** | **1489.6** |
| 2 | 25.572 ms | 26.476 ms | 12.786 ms | 78.21 | 1251.3 |
| 4 | 54.980 ms | 56.457 ms | 13.745 ms | 72.75 | 1164.1 |
| 8 | 119.112 ms | 121.107 ms | 14.889 ms | 67.16 | 1074.6 |
| 16 | 256.401 ms | 259.514 ms | 16.025 ms | 62.40 | 998.4 |
| 32 | 544.677 ms | 547.546 ms | 17.021 ms | 58.75 | 940.0 |

L4 为 Ada 架构；该模型在 batch=1 时已能很好利用 GPU/L2 cache。batch=32 的每条 GPU 时间比 batch=1 高 **58.5%**，吞吐只剩 **63.1%**。50 条请求若拆成 32+16+2，纯 GPU 时间约 **826.7ms**；batch=1 顺序执行的理论值约 **537.1ms**，因此线上禁用动态合批。

#### 5.7.2 50 个 16 秒请求同时到达：跨 inference streams

所有请求均调用固定 16s、batch=1 engine；每个 stream 使用独立 TensorRT execution context。

| Contexts / streams | 50 条总完成 mean | 总完成 p95 | 请求完成 p50 mean | 请求完成 p95 mean | 吞吐 |
|--------------------|------------------|------------|-------------------|-------------------|------|
| 1 | 546.610 ms | 548.810 ms | 278.639 ms | 519.827 ms | 91.47 req/s |
| **2** | **534.770 ms** | **537.643 ms** | **278.626 ms** | **514.271 ms** | **93.50 req/s** |
| 4 | 555.050 ms | 557.857 ms | 310.746 ms | 535.477 ms | 90.08 req/s |
| 8 | 571.648 ms | 574.098 ms | 330.803 ms | 554.939 ms | 87.47 req/s |

2 contexts 是最佳点；4/8 contexts 因共享 SM/L2/DRAM 产生争用。短复测测得 1/2 contexts 常驻设备内存约 **601/813 MiB**。本表是早期未计完整 pinned H2D/D2H 的历史执行基线；93.5 req/s 不再作为最终 admission 口径。最终生产形态复测见 5.8.2：91.923 req/s，admission 向下取整为 73 RPS。

“并发 50”不等于“50 RPS”：前者是同时在途请求数，后者才决定队列是否持续增长。若 50 条最坏请求在同一时刻突发，单卡请求完成 p95 约 514ms；若要求该突发 p95 明显低于 500ms，需要增加 GPU 副本，而不是增大 batch。

#### 5.7.3 动态时长 batch=1

一个 FP16 plan 使用 `[1,1,32000] / [1,1,160000] / [1,1,256000]` 作为 min/opt/max profile，对应 2/10/16 秒。plan 为 153.7 MiB，构建 195.0s。

| 时长 | mean | p95 | req/s | 推荐路径 |
|------|------|-----|-------|----------|
| 2s | 3.449 ms | 3.465 ms | 289.95 | 动态时长 |
| 4s | 3.970 ms | 3.990 ms | 251.90 | 动态时长 |
| 6s | 5.004 ms | 5.066 ms | 199.83 | 动态时长 |
| 8s | 6.209 ms | 6.245 ms | 161.06 | 动态时长（与固定 10s 基本持平） |
| 10s | 6.909 ms | 6.944 ms | 144.75 | 固定 10s（6.283ms） |
| 12s | 9.711 ms | 9.981 ms | 102.97 | 动态时长 |
| 13s | 10.732 ms | 10.940 ms | 93.18 | 固定 16s（mean 接近，p95 更低） |
| 14s | 11.645 ms | 11.924 ms | 85.87 | 固定 16s |
| 16s | 13.895 ms | 14.166 ms | 71.97 | 固定 16s |

最低 GPU 时间的路由为：`<2s` 补到 2s；`2–8s` 动态 plan；`>8–10s` 固定 10s；`>10–12s` 动态 plan；`>12–16s` 固定 16s。若更重视运维简单而非短音频成本，可只保留固定 10s/16s 两档，但 2s 请求会从约 3.45ms 增至约 6.28ms。

#### 5.7.4 16 秒 synthetic parity

固定 16s 的 bs=1/2/4/8/16/32 均成功输出 `[B,799,4]`，相对 ORT CPU FP32 reference 的 cell/frame exact 均为 **100%**。普通 harmonic pseudo-speech 在该 hard multilabel 模型上全为零，因此这里只证明 16 秒图执行、输出 shape 和静音边界一致；非零类别仍由 10 秒 adversarial synthetic parity 覆盖。本轮按约束没有运行真实音频或 DER/JER。

### 5.8 L4 固定 16 秒进一步加速实验（2026-08-27）

本节响应“除 TensorRT 外，有收益的路径都尝试”的要求，统一使用固定 `[1,1,256000]` 的 **16 秒 synthetic** 输入；没有运行 10 秒输入或真实音频。结论只比较同一脚本、同一轮交错测量中的配对结果，避免把温度、频率和运行顺序造成的跨进程波动当成加速。

#### 5.8.1 TensorRT 预分配、pinned memory 与 CUDA Graph

对现有 O3/8GiB 固定 16 秒 FP16 plan 预分配 input/output device buffer、pinned host buffer、stream 和 execution context，并使用 CUDA Runtime stream capture 捕获完整执行图。30 次 warmup、200 次计时结果：

| 路径 | GPU mean | GPU p95 | host enqueue mean | 相对对应 enqueue |
|------|----------|---------|-------------------|------------------|
| 普通 `enqueue_device` | 10.973ms | 11.250ms | 1.415ms | 基线 |
| **`cudagraph_device`** | **10.725ms** | **10.995ms** | **0.011ms** | mean **+2.31%**，p95 **+2.26%** |
| 普通 pinned H2D→TRT→D2H | 11.053ms | 11.178ms | — | 基线 |
| **完整 graph pinned H2D→TRT→D2H** | **10.799ms** | **10.950ms** | — | mean **+2.30%**，p95 **+2.04%** |

独立、只依赖 `numpy+tensorrt+libcudart` 的生产 runner 再次捕获完整 pinned H2D→TRT→pinned D2H 图，得到 396 个节点（391 kernel、2 memcpy、3 memset）。其 wall mean 从预分配 pinned 的 11.431ms 降至 **11.132ms**，快 **2.62%**；host enqueue 从 1.425ms 降至 **0.0079ms**，约 **181×**。两套实现的 graph 输出与普通 enqueue 均 bit-exact，主实现捕获 394 个节点。这里的 hard output 全零，因此 exact 只证明同一静音边界输入下 graph 没有改变输出；不能证明 16 秒非零类别或阈值附近准确率。

曾尝试直接用 `torch.cuda.CUDAGraph` 包裹 TensorRT，第一次出现不可信的 0.014ms。节点检查发现实际没有捕获 TensorRT 的外部 CUDA 工作，因此该数字已作废。最终改用 `cudaStreamBeginCapture/cudaStreamEndCapture`，并把“节点数必须大于零 + graph/enqueue 输出完全一致”设为硬门槛，避免空图假加速。

#### 5.8.2 50 请求突发下的 Graph 调度

同一 engine 只反序列化一次，各 worker 使用独立 context/buffer/stream；所有 worker/graph 只创建一次，组合按正序/逆序交错，3 次 warmup、30 轮、每轮 50 个 16 秒请求。

单 context 的配对测试中，CUDA Graph 将 device burst mean/p95 从 563.085/570.318ms 降至 **548.028/553.685ms**，分别改善 **2.67%/2.92%**；完整 pinned 链路也改善 **2.60%/2.93%**。

双 context 的最终六模式同轮验证如下：

| 双槽模式 | 50 请求总完成 mean | 总完成 p95 | 请求完成 p95 mean | 吞吐 |
|----------|-------------------|------------|---------------------|------|
| enqueue device ×2 | 556.324ms | 561.755ms | 534.810ms | 89.876 req/s |
| graph device ×2 | 555.938ms | 561.577ms | 533.724ms | 89.938 req/s |
| graph+enqueue device | 556.754ms | 561.993ms | **531.885ms** | 89.806 req/s |
| enqueue pinned ×2 | **555.586ms** | 561.570ms | 533.464ms | **89.995 req/s** |
| graph pinned ×2 | 556.606ms | 562.077ms | 534.355ms | 89.830 req/s |
| graph+enqueue pinned | 555.731ms | **561.357ms** | **531.076ms** | 89.972 req/s |

六种模式的总完成 mean 最大只差约 **0.21%**，处于运行噪声范围；hybrid 的单次更优结果未在同轮验证中转化为稳定吞吐优势。因此生产上不为双槽引入复杂的固定 hybrid 策略：只有一个槽位忙时优先 replay 完整 CUDA Graph；积压时允许两个预分配槽位并行。

最终生产形态另以 `flock` 独占 GPU，对双 context 的完整 pinned 链路做 3 次 warmup、30 轮复测：普通 enqueue 的 50 请求总完成 mean/p95 为 **543.935/547.260ms**，请求完成 p95 mean 为 **522.152ms**，吞吐 **91.923 req/s**；双 graph 为 91.590 req/s，仍无收益。生产容量因此按最终普通 enqueue pinned 链路 **91.9 req/s** 计算，而 5.7.2 的 93.5 req/s 仅保留为历史无传输基线。

#### 5.8.3 TensorRT builder/tactic 搜索

所有候选都执行相同的直接 CUDA Runtime graph benchmark；选择规则要求 mean 与 p95 都有稳定收益，否则不替换当前 plan。

| Builder 配置 | 构建时间 | graph mean | graph p95 | 决策 |
|--------------|----------|------------|-----------|------|
| **O3 / 8GiB / auto** | 现有 | **10.704ms** | 10.993ms | **保留** |
| O4 / 8GiB / auto | 232.7s | 10.819ms | 10.970ms | mean 回退，拒绝 |
| O5 / 8GiB / auto | 645.1s | 10.923ms | 11.056ms | 拒绝 |
| O5 / 8GiB / aux=0 | — | 10.822ms | 10.955ms | 拒绝 |
| O5 / 8GiB / aux=2 | — | 10.858ms | 11.002ms | 实际未分配 aux stream，拒绝 |
| O5 / 16GiB / auto | — | 10.790ms | **10.934ms** | p95 略好但 mean 回退，拒绝 |

更高 builder optimization level、加大 workspace 或请求 auxiliary streams 都没有稳定降低延迟；O5 还显著增加构建成本。现有 O3/8GiB plan 是这轮搜索后的 Pareto 选择。

#### 5.8.4 ONNX Runtime CUDA Graph / IOBinding

20 次 warmup、100 次计时：普通 `session.run` 为 38.816ms，固定设备地址 IOBinding 为 38.789ms（仅 **0.07%**），`enable_cuda_graph=1` + IOBinding 为 **38.003ms**，比普通路径快 **2.09%**。输出与基线完全一致，但仍比 TensorRT 固定 16 秒 10.741ms 慢 **3.54×**。profile 记录了 **358 次 CPU EP shape-op 事件**，并非 358 个唯一节点；它说明仍存在 host shape 工作，CUDA Graph 不能消除全部开销。因此只把 ORT Graph 作为回退路径的实验性选项，不改变主后端选择。

#### 5.8.5 PyTorch AMP、SDPA 与 `torch.compile`

10 次 warmup、50 次计时：

| PyTorch 路径 | mean | p95 | 相对 eager AMP | 相对 TRT FP16 |
|--------------|------|-----|----------------|----------------|
| eager FP32 | 56.720ms | 57.920ms | — | 5.28× 慢 |
| eager AMP FP16 | 39.099ms | 43.707ms | 基线 | 3.64× 慢 |
| SDPA + AMP | 36.308ms | 36.964ms | **+7.14%** | 3.38× 慢 |
| compile default + AMP | 19.864ms | 20.396ms | **1.97×** | 1.85× 慢 |
| **compile reduce-overhead + AMP** | **19.298ms** | 19.849ms | **2.03×** | 1.80× 慢 |
| compile max-autotune + AMP | **19.081ms** | **19.536ms** | **2.05×** | **1.78× 慢** |
| **compile reduce-overhead + SDPA + AMP** | **18.464ms** | **18.916ms** | **2.12×** | **1.72× 慢** |
| compile max-autotune + SDPA + AMP | 18.501ms | 19.002ms | 2.11× | 1.72× 慢 |

SDPA 微基准中，WavLM dense relative bias attention 的 auto kernel 比 manual attention 快 **2.21×**；Conformer 无 mask attention 快 **2.25×**。强制 Flash 在 WavLM 非空 mask 下不可用，在 Conformer 上也慢于 auto，因此采用 PyTorch 自动选择，不强制 Flash。

真正的组合路径已经按“先安装 SDPA patch、再 `torch.compile`”补测，JSON 中保存 `sdpa_patch_installed_before_compile=true`。`reduce-overhead+SDPA+AMP` 比非 SDPA 的 reduce-overhead 再快 **4.32%**，且 mean/p95 均略优于 SDPA max-autotune；后者继续提示 L4 SM 数不足，首次编译时间还可能命中持久 Inductor cache。因此 PyTorch 实验性二级回退选择 `reduce-overhead + SDPA + AMP`，启动阶段预编译固定 16 秒 shape。

两条组合路径的 hard output 在这条 synthetic 输入上均为 0/3196 mismatch，但 raw output 相对 eager FP32 未通过 `rtol=atol=1e-3` allclose（reduce-overhead max/mean abs 为 0.01407/0.00279）。这仍是一条全零 hard-output 输入，不能证明真实或非零 16 秒音频精度；没有真实验收前不能把该路径自动提升为生产主回退。该模型当前 `fullgraph=False` 下有 22 个 unique graphs 和 9 个 graph breaks，仍有进一步改写模型 `forward` 的空间，但在当前结果下不会超过 TensorRT。

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
| TensorRT FP16 | ✅ L4 完成 | 升级至 TRT 10.10.0.31 后，bs=1/4/8/32 engine 全部构建并测速 |
| ModelOpt FP8 explicit Q/DQ | ✅ L4 完成 | 强类型 engine 成功；330–331 个 FP8 层标记，但四档均慢于 FP16 |
| ModelOpt INT8 explicit Q/DQ | ⚠️ L4 完成但不可用 | 206–207 个 INT8 层，速度第二；frame exact 仅约 0.20–0.60%，repeat exact 也非 100% |
| ModelOpt INT4 weight-only | ⚠️ L4 完成但不推荐 | 128 个 DQ 权重，计算检查器无 INT4 层；比 FP16 慢 2.18–2.25× |
| FP4 / NVFP4 | ⛔ L4 不支持 | FP4 Tensor Core 需要 Blackwell；L4 是 Ada SM8.9 |
| SmoothQuant INT8 alpha 扫描 | ⛔ 当前工具链不支持 | 已检查 ModelOpt 0.46 ONNX 包，无 SmoothQuant 实现；需另写图变换/引入新工具链，且当前 INT8 已不满足确定性 |
| PyTorch FP16 基线 | ✅ | 已完成 |
| ORT CUDA FP32 | ✅ | 精度基线与生产回退 |

### 8.2 L4 测试清单完成情况

1. ~~在 L4 验证 TensorRT builder 与固定 10s 图~~：已完成；TRT 10.10 已跑完 FP32/TF32/FP16/BF16/FP8/INT8/INT4 的 bs=1/4/8/32。
2. ~~运行 ORT FP32 / PyTorch FP32 / PyTorch FP16 固定 10s 基线~~：已完成，batch=1/8/32/50。
3. ~~运行已有 dynamic/static INT8 固定 10s 模型~~：已完成，batch=1/8/32；batch=50 OOM。
4. ~~生成并运行 FP8/INT8 explicit Q/DQ 图~~：已完成；engine inspector 已确认低精度层命中。
5. ~~生成并运行 INT4 weight-only 图~~：已完成；L4 上无 INT4 计算层和速度收益。
6. 按本轮约束，不运行真实音频与 DER/JER；只报告 synthetic 输出一致性。
7. ~~分析 Q/DQ、Reformat 和低精度 GEMM 命中率~~：已通过 24 个 detailed inspector JSON 完成。
8. ~~迁移并回归 Conda 环境~~：已完成；28 个 engine 全部在 `diarizen-trt1010` 中成功执行。

---

## 9. 问题与限制

| 问题 | 影响 | 临时规避 |
|------|------|----------|
| ORT CUDA EP 长音频 `rel_attn Gather` 报错 | 部分 >60s 音频无法用 CUDA EP | 精度评估改用 CPU EP；或拆分短窗 |
| PyTorch GPU 长音频 OOM | 284s 音频 attention 显存爆炸 | 精度参考用 CPU FP32 |
| L4 TensorRT 10.0.1 FP16 builder 段错误 | 旧环境无法生成 FP16 engine | **已解决**：独立 Conda env `diarizen-trt1010` 使用 10.10.0.31；原 venv 备份保留 |
| FP8 普通网络类型推导失败 | 非 INT8 Q/DQ 无法构建 | **已解决**：量化 engine 使用 `STRONGLY_TYPED`，由 Q/DQ/ONNX 类型决定精度 |
| L4 INT8 bs=50 OOM | INT8 最大实测 batch 为 32 | L4 上限制 batch≤32；不要因模型体积变小假设运行显存也更小 |
| TensorRT INT8 PTQ synthetic 精度崩溃 | 虽有速度但不可部署 | 保持 TensorRT FP16；如继续 INT8，需 SmoothQuant/QAT 并重新验收 |
| INT4 block 不整除 | 原始 INT4 图无法被 TensorRT 解析 | 排除 56 个非 128 整除权重后可构建，但 L4 无 INT4 计算收益 |
| L4 不支持 FP4 | 无法进行原生 FP4 benchmark | 仅在 Blackwell GPU 上测试 NVFP4 |

---

## 10. 推荐生产配置

```yaml
# 分割模型 serving 推荐配置（基于 2026-08-27 L4 实测）
backend: tensorrt
tensorrt_version: 10.10.0.31
conda_env: /root/miniforge3/envs/diarizen-trt1010
precision: fp16_mixed
model: epoch_0016_multilabel_hard.onnx
input_dtype: float32
sample_rate: 16000
max_audio_seconds: 16

engines:
  dynamic_2s_16s: trt_l4_trt1010_dynamic/segmentation_bs1_2s-16s_opt10s_fp16.plan
  fixed_10s: trt_l4_trt1010/segmentation_10s_bs1_fp16.plan
  fixed_16s: trt_l4_trt1010_16s/segmentation_16s_bs1_fp16.plan

duration_router:
  - "duration < 2s: pad to 2s -> dynamic_2s_16s"
  - "2s <= duration <= 8s: ceil to 1s bucket -> dynamic_2s_16s"
  - "8s < duration <= 10s: pad to 10s -> fixed_10s"
  - "10s < duration <= 12s: ceil to 1s bucket -> dynamic_2s_16s"
  - "12s < duration <= 16s: pad to 16s -> fixed_16s"

scheduler:
  batch_size: 1
  dynamic_batching: false
  execution_contexts_per_active_engine: 2
  max_inflight_requests: 50
  queue_capacity: 100
  overload: reject_429_or_route_to_second_l4

runtime:
  deserialize_engine_once: true
  buffers: preallocated_pinned_host_and_device_per_context
  streams: one_independent_stream_per_context
  single_active_context: replay_full_h2d_trt_d2h_cuda_graph
  queued_burst: dispatch_to_two_preallocated_contexts
  cuda_graph_guard: require_nonzero_node_count_and_exact_enqueue_parity

transport:
  protocol: grpc
  payload: pcm_s16le_binary
  preprocess_client_side: mono_16khz
  avoid: json_base64_audio

warmup:
  on_startup: true
  shapes_seconds: [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 16]

fallback:
  backend: onnxruntime
  provider: CUDAExecutionProvider
  deployment: separate_process_in_conda_env_diarizen
  optional_cuda_graph_iobinding_experimental: true
  secondary_pytorch_experimental_not_auto_enabled: reduce_overhead_compile_sdpa_amp

capacity_guardrail_worst_case_16s:
  measured_requests_per_second_final_pinned_two_contexts: 91.923
  target_utilization: 0.8
  admission_requests_per_second: 73
  burst_50_total_p95_ms_gpu_only: 547.3
  burst_50_request_completion_p95_ms_gpu_only: 522.2

avoid:
  - tensorrt_dynamic_batching
  - fixed_batch_greater_than_1
  - four_or_more_execution_contexts
  - dynamic_int8_on_gpu
  - static_int8
  - tensorrt_int8_ptq_current_calibration
  - int4_weight_only_on_l4
  - fp4_on_l4
  - pytorch_eager_production
  - per_request_cuda_malloc_or_engine_deserialization
  - forced_flash_attention
```

服务层应只反序列化一次 engine；每个 context 独占一套 pinned host buffer、CUDA input/output buffer 和 stream，禁止逐请求 `cudaMalloc`。固定 shape/地址初始化后捕获完整 H2D→TensorRT→D2H graph；仅一个槽位活跃时 replay graph，积压时向两个槽位轮转派发。双槽 graph/enqueue 的配对差异约 0.21%，所以不引入难以维护的固定 hybrid 状态机，容量按双槽普通 pinned enqueue 计算。动态输入向上取整到 1 秒 bucket，避免每个任意 sample 数触发 shape 切换。请求传二进制 PCM16，不传 JSON/base64；解码和重采样尽量放上游。服务启动时加载 plan、捕获 graph 并 warmup 所有路由 bucket，健康检查至少验证 2s/10s/16s shape。记录 `decode_ms`、`queue_ms`、`h2d_ms`、`gpu_ms`、`postprocess_ms`、端到端 p50/p95/p99、队列深度、拒绝数和 GPU 显存。

`diarizen-trt1010` 中的 ORT 1.18 CUDA provider 会因缺少 `libcudnn.so.8` 回退 CPU，所以 ORT CUDA 回退必须运行在现有 `diarizen` 环境的独立进程/容器，不能在主 TensorRT 进程内假设 CUDA EP 可用。TensorRT plan 与 TRT 版本/GPU 架构绑定；GPU 型号或 TensorRT 大版本变化时从 ONNX 重建。

最终完整 pinned 链路的 50 条最坏输入总完成 p95 为 547.3ms、请求完成 p95 mean 为 522.2ms，均为 GPU-only，不含网络、音频解码、排队前端和后处理。满载理论成本公式为：`每请求 GPU 成本 ≈ L4 每小时价格 / 330900`；按 73 RPS admission 的生产口径使用 `L4 每小时价格 / 262800`。如果到达率长期低于 73 RPS，一张 L4 是最低成本方案；若硬性要求 50 条同时到达的端到端 p95 显著低于约 0.55s，则必须增加 L4 副本。

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

### 11.5 L4 固定 10 秒 INT8 variants

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n diarizen \
python inference/benchmark_ort_fixed10s_variants.py \
  --reference inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx \
  --model dynamic_int8=inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.matmul-dynamic-int8.onnx \
  --model static_int8=inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.static-int8.onnx \
  --model static_int8_entropy_10s=inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.static-int8-entropy-10s.onnx \
  --batch-sizes 1,8,32,50 \
  --warmup 5 \
  --repeats 20 \
  --seeds 1001,1002,1003 \
  --out-json inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_l4_int8_fixed10s.json
```

### 11.6 L4 TensorRT 10.10 多精度

```bash
# 独立 Conda 环境，不修改 diarizen/cosyvoice；可重复执行
bash inference/setup_tensorrt_l4_conda.sh

CUDA_VISIBLE_DEVICES=0 /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen-trt1010 python \
  inference/benchmark_tensorrt_fixed10s.py \
  --onnx inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx \
  --engine-dir inference/models/kaldi_merged_1219_all_ft_large/trt_l4_trt1010 \
  --out-json inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_l4_trt1010_fp16_fixed10s.json \
  --precision fp16 \
  --batch-sizes 1,4,8,32 \
  --workspace-gb 8 \
  --optimization-level 3 \
  --warmup 5 \
  --repeats 20

# FP32+TF32 / strict FP32 / BF16 使用同一脚本切换 --precision。
# FP8/INT8/INT4 先由 inference/quantize_segmentation_modelopt.py
# 生成显式 Q/DQ ONNX，再传入同一 benchmark；量化路径会自动创建 STRONGLY_TYPED 网络。

# Conda 迁移后全 28 engine 回归
CUDA_VISIBLE_DEVICES=0 /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen-trt1010 python inference/validate_tensorrt_l4_conda.py \
  --fp16-engine-dir inference/models/kaldi_merged_1219_all_ft_large/trt_l4_trt1010 \
  --precision-engine-dir inference/models/kaldi_merged_1219_all_ft_large/trt_l4_precisions \
  --artifact-dir /tmp/diarizen_trt_synthetic_refs \
  --out-json inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_l4_trt1010_conda_all_engines_validation.json

# Detailed inspector 汇总
/root/miniforge3/bin/conda run --no-capture-output -n diarizen-trt1010 \
  python inference/analyze_trt_engine_inspectors.py \
  --engine-dir inference/models/kaldi_merged_1219_all_ft_large/trt_l4_precisions \
  --out-json inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_l4_trt1010_engine_inspector_analysis.json
```

### 11.7 L4 线上 16 秒与并发 50

```bash
# 固定 16 秒、各 batch
CUDA_VISIBLE_DEVICES=0 /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen-trt1010 python inference/benchmark_tensorrt_fixed10s.py \
  --onnx inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx \
  --engine-dir inference/models/kaldi_merged_1219_all_ft_large/trt_l4_trt1010_16s \
  --out-json inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_l4_trt1010_fp16_fixed16s.json \
  --input-seconds 16 --batch-sizes 1,2,4,8,16,32 --precision fp16 \
  --workspace-gb 8 --optimization-level 3 --warmup 10 --repeats 50

# batch=1、50 请求、跨 inference streams
CUDA_VISIBLE_DEVICES=0 /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen-trt1010 python inference/benchmark_trt_online_burst.py \
  --engine inference/models/kaldi_merged_1219_all_ft_large/trt_l4_trt1010_16s/segmentation_16s_bs1_fp16.plan \
  --input-seconds 16 --requests 50 --streams 1,2,4,8 \
  --warmup-bursts 3 --repeats 30 \
  --out-json inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_l4_trt1010_fp16_16s_burst50.json

# 一个 batch=1 动态时长 profile
CUDA_VISIBLE_DEVICES=0 /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen-trt1010 python inference/benchmark_trt_dynamic_duration.py \
  --onnx inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx \
  --engine inference/models/kaldi_merged_1219_all_ft_large/trt_l4_trt1010_dynamic/segmentation_bs1_2s-16s_opt10s_fp16.plan \
  --out-json inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_l4_trt1010_fp16_dynamic_duration_bs1.json \
  --profile-seconds 2,10,16 --benchmark-seconds 2,4,6,8,10,12,13,14,16 \
  --workspace-gb 8 --warmup 10 --repeats 100
```

### 11.8 L4 固定 16 秒进一步加速

```bash
MODEL_DIR=inference/models/kaldi_merged_1219_all_ft_large
ENGINE=$MODEL_DIR/trt_l4_trt1010_16s/segmentation_16s_bs1_fp16.plan

# TensorRT enqueue / pinned I/O / 直接 CUDA Runtime Graph
flock /tmp/diarizen_l4_benchmark.lock env CUDA_VISIBLE_DEVICES=0 \
  /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen-trt1010 python inference/benchmark_l4_runtime_acceleration.py \
  --engine "$ENGINE" --warmup 30 --repeats 200 \
  --out-json "$MODEL_DIR/epoch_0016_l4_runtime_acceleration_fixed16s.json"

# 生产参考 runner：完整 pinned H2D -> TRT -> pinned D2H graph
# artifact 可由 ORT 脚本的 --artifact-npz 生成
flock /tmp/diarizen_l4_benchmark.lock env CUDA_VISIBLE_DEVICES=0 \
  /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen-trt1010 python inference/tensorrt_cuda_graph_runner.py \
  --engine "$ENGINE" --artifact /tmp/diarizen_fixed16s_artifact.npz \
  --warmup 20 --repeats 200 \
  --out "$MODEL_DIR/epoch_0016_l4_trt_production_runner_fixed16s.json"

# builder optimization/workspace/aux-stream 搜索
flock /tmp/diarizen_l4_benchmark.lock env CUDA_VISIBLE_DEVICES=0 \
  /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen-trt1010 python inference/benchmark_trt_builder_search.py \
  --onnx "$MODEL_DIR/epoch_0016_multilabel_hard.onnx" \
  --baseline-engine "$ENGINE" \
  --engine-dir "$MODEL_DIR/trt_l4_trt1010_16s_builder_search" \
  --out-json "$MODEL_DIR/epoch_0016_l4_trt_builder_search_fixed16s.json"

# 50 请求：单/双 context、enqueue/graph/pinned/hybrid 交错对照
flock /tmp/diarizen_l4_benchmark.lock env CUDA_VISIBLE_DEVICES=0 \
  /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen-trt1010 python inference/benchmark_trt_online_burst_acceleration.py \
  --engine "$ENGINE" --requests 50 --streams 1,2 \
  --modes enqueue_device,cudagraph_device,hybrid_device,enqueue_pinned_e2e,cudagraph_pinned_e2e,hybrid_pinned_e2e \
  --warmup-bursts 3 --repeats 30 \
  --out-json "$MODEL_DIR/epoch_0016_l4_trt1010_fp16_16s_burst50_hybrid.json"

# 最终生产容量口径：双 context、完整 pinned I/O
flock /tmp/diarizen_l4_benchmark.lock env CUDA_VISIBLE_DEVICES=0 \
  /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen-trt1010 python inference/benchmark_trt_online_burst_acceleration.py \
  --engine "$ENGINE" --requests 50 --streams 2 \
  --modes enqueue_pinned_e2e,cudagraph_pinned_e2e \
  --warmup-bursts 3 --repeats 30 \
  --out-json "$MODEL_DIR/epoch_0016_l4_trt1010_fp16_16s_burst50_final_capacity.json"

# ORT CUDA Graph + IOBinding；在 diarizen 环境执行
flock /tmp/diarizen_l4_benchmark.lock env CUDA_VISIBLE_DEVICES=0 \
  /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen python inference/benchmark_ort_cuda_graph_fixed16s.py \
  --onnx "$MODEL_DIR/epoch_0016_multilabel_hard.onnx" \
  --warmup 20 --repeats 100 --artifact-npz /tmp/diarizen_fixed16s_artifact.npz \
  --out-json "$MODEL_DIR/epoch_0016_l4_ort_cuda_graph_fixed16s.json"

# PyTorch eager/AMP/SDPA/compile 固定 16 秒；各 mode 分别执行
flock /tmp/diarizen_l4_benchmark.lock env CUDA_VISIBLE_DEVICES=0 \
  /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen python inference/benchmark_pytorch_compile_fixed16s.py \
  --mode compile_reduce_overhead_sdpa_amp_fp16 --warmup 10 --repeats 50 \
  --out "$MODEL_DIR/diarizen_pytorch_compile_reduce_overhead_sdpa_amp_fp16_fixed16s.json"

flock /tmp/diarizen_l4_benchmark.lock env CUDA_VISIBLE_DEVICES=0 \
  /root/miniforge3/bin/conda run --no-capture-output \
  -n diarizen python inference/benchmark_attention_sdpa_fixed16s.py \
  --warmup 10 --repeats 50 \
  --out-json "$MODEL_DIR/diarizen_attention_fixed16s.json"
```

---

## 12. 产物与脚本索引

### 12.1 脚本（纳入 git）

| 文件 | 说明 |
|------|------|
| `inference/benchmark_segmentation_precision.py` | 速度 + 精度 benchmark |
| `inference/benchmark_ort_fixed10s_variants.py` | L4 上多个 ORT/INT8 模型的固定 10s 速度、provider 分配与 synthetic 一致性 |
| `inference/benchmark_tensorrt_fixed10s.py` | 构建固定 shape TensorRT FP32/TF32/FP16/BF16/FP8/INT8/INT4 engines、导出 inspector 并测速 |
| `inference/benchmark_trt_fixed10s_parity.py` | ORT FP32 reference 与各 TensorRT 精度 hard multilabel synthetic parity |
| `inference/benchmark_trt_online_burst.py` | 固定 batch=1 engine 的 50 请求、跨 execution context/stream 突发压测 |
| `inference/benchmark_trt_dynamic_duration.py` | 构建 batch=1、2–16s 动态时长 profile 并逐时长测速 |
| `inference/generate_adversarial_synthetic_10s.py` | 生成能激活正类的 10 秒 adversarial synthetic waveform |
| `inference/diagnose_tensorrt_build.py` | TensorRT 最小网络、ONNX parse、FP32/FP16 build 隔离诊断 |
| `inference/quantize_segmentation_modelopt.py` | ModelOpt FP8/INT8 explicit Q/DQ 与 INT4 weight-only 图生成 |
| `inference/setup_tensorrt_l4_conda.sh` | 幂等创建/修复 `diarizen-trt1010` Conda env，并验证版本、依赖与 GPU |
| `inference/tensorrt_l4_requirements.txt` | TensorRT 10.10 与 CUDA runtime 精确版本锁定 |
| `inference/validate_tensorrt_l4_conda.py` | 在 Conda env 中执行全部 28 个 engine，并检查 parity 与重复执行确定性 |
| `inference/analyze_trt_engine_inspectors.py` | 汇总 Q/DQ、Reformat、datatype 和低精度 MatMul/Gemm 命中率 |
| `inference/benchmark_l4_runtime_acceleration.py` | 固定 16s TensorRT enqueue、预分配、pinned I/O 与直接 CUDA Runtime Graph 配对 benchmark |
| `inference/tensorrt_cuda_graph_runner.py` | 不依赖 PyTorch 的完整 H2D→TRT→D2H CUDA Graph 生产参考 runner |
| `inference/benchmark_trt_builder_search.py` | O3/O4/O5、workspace、auxiliary stream 的 16s builder/tactic 搜索 |
| `inference/benchmark_trt_online_burst_acceleration.py` | 50 请求、单/双 context 的 enqueue/graph/pinned/hybrid 交错 benchmark |
| `inference/benchmark_ort_cuda_graph_fixed16s.py` | ORT session.run、IOBinding 与 CUDA Graph 固定 16s 对照 |
| `inference/benchmark_pytorch_compile_fixed16s.py` | PyTorch eager/AMP/SDPA、三种 compile mode 及两种真实 compile+SDPA 组合的固定 16s 对照 |
| `inference/benchmark_attention_sdpa_fixed16s.py` | WavLM/Conformer attention manual/auto/forced-Flash 微基准 |
| `inference/tests/test_benchmark_l4_runtime_acceleration.py` | runtime benchmark 的统计、graph 判定和参数单元测试 |
| `inference/quantize_segmentation_onnx_static.py` | Static INT8 QDQ 量化 |
| `inference/run_export_kaldi_merged_1219_all_ft_large_epoch_0002.sh` | ONNX 导出 |

### 12.2 报告与 JSON（纳入 git）

| 文件 | 说明 |
|------|------|
| `SEGMENTATION_BENCHMARK_REPORT.md` | 本文档 |
| `epoch_0016_precision_benchmark.json` | 速度 + FP32/FP16 精度 |
| `epoch_0016_int8_accuracy_report.json` | dynamic INT8 逐文件精度 |
| `epoch_0016_static_int8_accuracy_report.json` | static INT8 逐文件精度 |
| `epoch_0016_l4_baseline.json` | L4 PyTorch FP32/FP16 与 ORT FP32 固定 10s 基线 |
| `epoch_0016_l4_dynamic_int8_fixed10s.json` | L4 dynamic INT8 固定 10s 结果 |
| `epoch_0016_l4_static_int8_fixed10s.json` | L4 两种 static INT8 固定 10s 结果 |
| `epoch_0016_l4_trt1010_fp16_fixed10s.json` | L4 TensorRT 10.10 FP16 速度、构建和 synthetic parity 汇总 |
| `epoch_0016_l4_trt1010_all_precisions_fixed10s.json` | L4 TensorRT 全精度速度、inspector 证据与 synthetic parity 汇总 |
| `epoch_0016_l4_trt1010_conda_all_engines_validation.json` | Conda 迁移后 28-engine 执行、parity 与 repeat exact 回归 |
| `epoch_0016_l4_trt1010_engine_inspector_analysis.json` | 24 个多精度 engine inspector 的 Q/DQ/Reformat/GEMM 汇总 |
| `epoch_0016_l4_trt1010_fp16_fixed16s.json` | L4 TensorRT FP16 固定 16s、bs=1/2/4/8/16/32 结果 |
| `epoch_0016_l4_trt1010_fp16_fixed16s_parity.json` | 固定 16s TensorRT FP16 与 ORT FP32 synthetic parity |
| `epoch_0016_l4_trt1010_fp16_16s_burst50.json` | 50 个 16s 请求在 1/2/4/8 contexts 下的完成延迟与吞吐 |
| `epoch_0016_l4_trt1010_fp16_16s_burst50_memory.json` | 1/2 contexts 的短复测与设备内存占用 |
| `epoch_0016_l4_trt1010_fp16_dynamic_duration_bs1.json` | batch=1 动态 2–16s profile 的逐时长结果 |
| `epoch_0016_l4_ort_fp32_synthetic16s_reference.json` | 固定 16s ORT CPU FP32 synthetic reference 元数据 |
| `epoch_0016_l4_runtime_acceleration_fixed16s.json` | TensorRT 预分配、pinned I/O 与有效 CUDA Graph 的 200 次配对结果 |
| `epoch_0016_l4_trt_production_runner_fixed16s.json` | 独立 libcudart 完整数据链路 graph 的节点、延迟、host enqueue 与 parity |
| `epoch_0016_l4_trt_builder_search_fixed16s.json` | O3/O4/O5、8/16GiB workspace、aux stream 搜索与选型 |
| `epoch_0016_l4_trt1010_fp16_16s_burst50_cudagraph.json` | 1/2 context 的 enqueue/graph 与 pinned I/O 配对结果 |
| `epoch_0016_l4_trt1010_fp16_16s_burst50_hybrid.json` | 双 context 六模式同轮交错复核，排除 hybrid 偶然收益 |
| `epoch_0016_l4_trt1010_fp16_16s_burst50_final_capacity.json` | 带 engine SHA/环境元数据、由 `flock` 串行执行的最终双 context pinned 容量口径 |
| `epoch_0016_l4_ort_cuda_graph_fixed16s.json` | ORT session.run/IOBinding/CUDA Graph 固定 16s 对照及 EP profile |
| `diarizen_pytorch_{eager,sdpa,compile}_*_fixed16s.json` | PyTorch FP32/AMP/SDPA/compile 固定 16s 分项结果（含两种 compile+SDPA，共 9 个 JSON） |
| `diarizen_attention_fixed16s.json` | WavLM/Conformer SDPA kernel 微基准 |

### 12.3 本地模型文件（**.gitignore 排除，不入库**）

| 文件 | 大小 |
|------|------|
| `epoch_0016_multilabel_hard.onnx` | 266 MB |
| `epoch_0016_multilabel_hard.matmul-dynamic-int8.onnx` | 98 MB |
| `epoch_0016_multilabel_hard.static-int8.onnx` | 99 MB |
| `epoch_0016_multilabel_hard.static-int8-entropy-10s.onnx` | 99 MB |
| `epoch_0016_multilabel_hard.modelopt-{fp8,int8}.onnx` | 约 141 MB，显式 Q/DQ |
| `epoch_0016_multilabel_hard.modelopt-int4-trt.onnx` | 约 136 MB，128 个 INT4 DQ 权重 |
| `trt_l4_trt1010/segmentation_10s_bs{1,4,8,32}_fp16.plan` | 164–385 MB |
| `trt_l4_trt1010_16s/segmentation_16s_bs{1,2,4,8,16,32}_fp16.plan` | 163–502 MiB；生产只需 bs=1 |
| `trt_l4_trt1010_dynamic/segmentation_bs1_2s-16s_opt10s_fp16.plan` | 153.7 MiB；动态时长、固定 batch=1 |
| `trt_l4_precisions/segmentation_10s_bs{1,4,8,32}_{fp32_tf32,fp32,bf16,fp8,int8,int4}.plan` | 多精度 engines 与 inspector JSON |

---

## 13. 结论与后续工作

### 13.1 结论

1. **L4 固定 10s 性能首选**：TensorRT 10.10 FP16；ORT CUDA FP32 保留为生产回退。
2. **L4 固定 10s 实测**：bs=1/8/32/50 为 **20.5/163.0/748.7/1184.2 ms**；bs=50 约为 A800 的 5.1× 延迟。
3. **L4 ORT INT8 不可取**：dynamic/static 均慢于 FP32，且 bs=50 OOM；全静音/噪声 synthetic 100% 一致不代表真实音频精度可接受。
4. **TensorRT FP16 问题已解决**：10.0.1 的 builder 段错误通过独立升级到 10.10.0.31 解决；四个固定 shape engine 全部成功。
5. **TensorRT FP16 是全精度实测冠军**：FP8/INT8 确实命中低精度层，但分别慢约 11–26% / 3–13%；INT8 synthetic 精度严重失真且重复执行非完全确定。
6. **INT4/FP4 不适合 L4**：INT4 仅压缩部分权重、实际 Float 计算且更慢；FP4 需要 Blackwell，Ada L4 不支持。
7. **线上最长 16s 时仍选 FP16 batch=1**：10.741ms/条；batch=32 每条 17.021ms，吞吐反而下降 36.9%。
8. **并发 50 使用两个 contexts**：最终完整 pinned 链路 50 条总完成 mean/p95 约 543.9/547.3ms，请求完成 p95 约 522.2ms，吞吐约 91.9 req/s；4/8 contexts 更慢。
9. **一张 L4 是最低成本起点**：admission 向下取整为 73 个最坏 16s 请求/秒；更高持续到达率或更低突发 p95 SLA 才扩卡。
10. **时长路由避免无谓补零**：动态 2–16s plan 配合固定 10s/16s plan；不使用大 batch 或当前 INT8/FP8/INT4 路线。
11. **CUDA Graph 有小而稳定的单槽收益**：完整 16s 数据链路 mean/p95 改善约 2.6%/2.9%，host enqueue 降至约 8µs；双槽六模式差异约 0.21%，不采用复杂 hybrid 调度。
12. **builder 深搜没有更好 plan**：O4/O5、16GiB workspace、aux streams 均未同时改善 mean/p95；继续使用 O3/8GiB。
13. **非 TensorRT 路径也已测完**：ORT CUDA Graph 只改善 2.09% 且仍慢 3.54×；真正的 PyTorch reduce-overhead+SDPA+AMP 比 eager AMP 快 2.12×，但仍慢 1.72×。二者都是可选/实验性回退，不是主后端；全零 hard parity 不能替代真实验收。

### 13.2 收尾状态与外部约束

| 项目 | 状态 | 说明 |
|------|------|------|
| TensorRT Conda 环境与版本锁定 | ✅ 完成 | `setup_tensorrt_l4_conda.sh` 可幂等复现；`pip check` 通过 |
| 全部固定 10s engines 回归 | ✅ 完成 | 7 种路径 × 4 个 batch，共 28 个 plan 全部执行成功 |
| Q/DQ / Reformat / GEMM 分析 | ✅ 完成 | 使用 detailed engine inspector，不依赖缺失的 `trtexec/nsys` |
| CUDA stream 正确性 | ✅ 完成 | parity/validation/benchmark 均加入跨 stream 同步；验证增加 warmup 与 repeat exact |
| 固定 16s 容量与 batch 扫描 | ✅ 完成 | FP16 bs=1/2/4/8/16/32，均 50 次计时 |
| 并发 50 调度扫描 | ✅ 完成 | batch=1 的 1/2/4/8 contexts；2 contexts 最佳 |
| 动态时长与路由切点 | ✅ 完成 | 2/4/6/8/10/12/13/14/16s synthetic 测速 |
| TensorRT runtime / CUDA Graph | ✅ 完成 | 16s enqueue/pinned/graph、独立 libcudart runner、graph 节点与 exact parity |
| TensorRT builder/tactic 搜索 | ✅ 完成 | O3/O4/O5、8/16GiB、aux=0/2；没有候选替换现有 plan |
| ORT / PyTorch / SDPA 加速 | ✅ 完成 | 16s ORT CUDA Graph、PyTorch AMP/SDPA/compile 和 attention kernel 微基准 |
| 双 context graph/hybrid 复核 | ✅ 完成 | 6 种模式同轮正逆序交错；差异约 0.21%，不宣称 hybrid 收益 |
| 真实音频 DER/JER | 按约束不执行 | 用户最终口径为只跑固定 16 秒 synthetic，不跑真实音频 |
| SmoothQuant alpha 扫描 | 当前工具链不具备 | ModelOpt 0.46 ONNX 无 SmoothQuant；需要新增图变换或训练工具链，且普通 INT8 已精度失败并非确定 |
| 部署镜像/注册表推送 | 无目标可执行 | 尚未提供镜像仓库、服务入口或部署目标；环境脚本与 engines 已就绪 |

---

*报告更新：DiariZen inference benchmark，2026-08-27*

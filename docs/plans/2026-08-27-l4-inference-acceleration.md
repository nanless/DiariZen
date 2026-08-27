# L4 Inference Acceleration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在单张 L4 上找出并固化对当前固定 16 秒 segmentation 推理确有收益的无训练加速方案。

**Architecture:** 以 TensorRT FP16 固定 shape 为生产基线，分别优化 engine 构建搜索和运行时 CUDA Graph/内存传输；ORT 与 PyTorch 作为替代后端做隔离对照。每项实验统一 synthetic 输入、串行占用 GPU、输出机器可读 JSON，并用数值一致性门禁决定是否采用。

**Tech Stack:** Python 3.10/3.11, PyTorch CUDA, TensorRT 10.10, ONNX Runtime CUDA EP, NVIDIA L4.

---

### Task 1: 建立统一 16 秒实验门禁

**Files:**
- Create: `inference/benchmark_l4_runtime_acceleration.py`
- Create: `inference/tests/test_benchmark_l4_runtime_acceleration.py`

**Steps:**
1. 为 percentile、统计汇总、固定随机输入和结果一致性写 CPU 单元测试。
2. 运行测试，确认在实现前失败。
3. 实现共享参数校验、JSON schema、固定 seed 生成和统计函数。
4. 运行单元测试及 `--help` smoke test。

### Task 2: TensorRT CUDA Graph 与传输路径

**Files:**
- Modify: `inference/benchmark_l4_runtime_acceleration.py`
- Modify: `inference/tests/test_benchmark_l4_runtime_acceleration.py`
- Create after run: `inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_l4_runtime_acceleration_fixed16s.json`

**Steps:**
1. 实现固定 context/stream/device buffers 的普通异步 runner。
2. 实现 CUDA Graph capture/replay；capture 失败时返回结构化错误，不静默伪装成功。
3. 实现预分配 pinned host buffer 的 H2D + inference + D2H 测量。
4. 对普通 enqueue、Graph replay 和端到端路径做 warmup 及重复测量。
5. 用固定 synthetic 输入验证 graph 与普通路径 exact match。

### Task 3: TensorRT builder 搜索

**Files:**
- Create: `inference/benchmark_trt_builder_search.py`
- Create after run: `inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_l4_trt_builder_search_fixed16s.json`

**Steps:**
1. 复用同一 ONNX、FP16 和固定 shape，生成 optimization level 3/4/5 候选；支持可选 seed timing cache。
2. 记录构建配置、构建时间、engine 大小和显存。
3. 在 GPU 锁内用同一 runtime harness 各跑两轮。
4. 只将 mean/p95 均稳定改善的候选标为推荐。

### Task 4: ORT CUDA Graph / I/O Binding 对照

**Files:**
- Create: `inference/benchmark_ort_cuda_graph_fixed16s.py`
- Create after run: `inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_l4_ort_cuda_graph_fixed16s.json`

**Steps:**
1. 比较普通 `session.run`、I/O Binding 和 CUDA Graph。
2. 检查节点是否全部落在 CUDA EP；记录不支持原因。
3. 固定输入输出 device 地址并验证 replay 数值一致性。
4. 报告 wall mean/p50/p95 和相对 TensorRT 基线。

### Task 5: PyTorch compile/SDPA 可行性

**Files:**
- Create: `inference/benchmark_pytorch_compile_fixed16s.py`
- Create after run: `inference/models/kaldi_merged_1219_all_ft_large/diarizen_pytorch_*_fixed16s.json`

**Steps:**
1. 建立 eager FP32/FP16 的固定 16 秒基线。
2. 测试 `torch.compile` default、reduce-overhead、max-autotune，记录 graph breaks 和编译失败。
3. 检查 WavLM gated relative-position bias 与 fused SDPA 的兼容性；单独测 Conformer 和代表性 WavLM Attention。
4. 只在全模型数值一致且实测有收益时推荐替代路径。

### Task 6: 文档、回归和交付

**Files:**
- Modify: `inference/models/kaldi_merged_1219_all_ft_large/SEGMENTATION_BENCHMARK_REPORT.md`
- Modify: `inference/ADR_L4_ONLINE_SERVING_16S_CONCURRENCY50.md`
- Modify: `inference/README.md`

**Steps:**
1. 汇总采用和否决项，区分纯 GPU、CPU enqueue、端到端延迟。
2. 运行所有新增 CPU tests、脚本 `--help`、JSON parse 和 Python compile checks。
3. 在 L4 上复跑最终推荐命令，确认无其他 GPU 进程干扰。
4. 审查 `git diff`，只提交本任务文件。
5. commit 并 push 当前 `feature/nemo-ssl-nest-finetune` 分支。

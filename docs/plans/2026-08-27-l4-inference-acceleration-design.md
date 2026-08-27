# L4 推理加速实验设计

## 目标与约束

在 `dev_L4_1gpus` 的单张 NVIDIA L4 上，以当前 `epoch_0016` segmentation 模型和固定 16 秒、16 kHz、batch=1 合成输入为唯一性能测试对象，寻找相对现有 TensorRT FP16 基线有可复现收益的无训练优化。速度测试不使用真实音频；数值回归使用固定种子 Gaussian synthetic 输入。所有结论必须记录软件版本、warmup、重复次数、mean/p50/p95、吞吐及数值误差。

当前基线为固定 16 秒 TensorRT 10.10 FP16，约 10.741 ms。实验不重复已证明无收益的动态 batching、INT8/FP8/INT4、BF16 路径。模型蒸馏、进一步结构化剪枝和局部 Attention 可能带来更大收益，但需要训练和真实验证集确认准确率，不属于本轮“只测 16 秒 synthetic”的可上线代码改动。

## 候选方案

### 方案 A：生产执行路径优化（推荐）

保持 TensorRT FP16 engine 不变，在固定 GPU 地址上预分配输入输出，比较普通 `execute_async_v3` 与 CUDA Graph replay，并分别报告纯 GPU 时间和 CPU enqueue 墙钟时间。增加固定 pinned-host buffer、异步 H2D/D2H 的端到端测量。该方案改动最小、数值行为不变，最可能直接进入线上 runner。

### 方案 B：构建期搜索

用相同 ONNX、固定 `[1,1,256000]` profile 和 FP16 构建约束，比较 builder optimization level 3/4/5、可用 workspace 以及 auxiliary stream 设置；脚本支持可选 seed timing cache。只保留配对测试中稳定降低 mean/p95 的 engine 配置。构建时间不是线上成本，但 engine 文件大小、显存和可复现命令必须记录。

### 方案 C：替代后端优化

ORT 测试 I/O Binding、固定 device buffer 与 CUDA Graph；PyTorch 测 AMP、`torch.compile` 以及 SDPA/FlashAttention 可行性。这两组主要作为兼容路径和后续模型改造依据。除非实测超过 TensorRT FP16，否则不替换生产主后端。

## 数据流与安全边界

请求进入预分配的 pinned CPU 槽位，通过专属 CUDA stream 异步复制到固定 GPU 输入地址，replay 对应 TensorRT CUDA Graph，再把小型输出异步复制回固定 pinned 输出。每个并发执行槽拥有独立 context、stream、device buffer 和 graph；不能共享 context。异常时回退到同一 context 的普通 `execute_async_v3`，并记录 graph capture 失败原因。

测试期间通过 `/tmp/diarizen_l4_benchmark.lock` 串行使用 GPU，避免并行代理污染数字。候选若收益落在噪声范围内则进行同轮交错或独立进程复核；明显回退、p95 变差、数值不一致或引入不可接受维护风险的方案直接记录为“无收益/不采用”。

## 验收

- 所有 benchmark 脚本可从仓库命令行复现，输出 JSON。
- 固定 16 秒 synthetic 输出 shape 正确，与普通 TensorRT FP16 路径 exact match。
- 正收益方案的 mean 和 p95 都有稳定改善，且没有用不同精度偷换基线。
- 报告区分 GPU execution、CPU enqueue 和含传输端到端延迟。
- 最终更新中文 benchmark 报告与线上 ADR，工作树干净并推送当前分支。

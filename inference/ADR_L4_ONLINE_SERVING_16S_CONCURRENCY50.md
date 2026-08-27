# ADR: L4 上最长 16 秒、并发 50 的 segmentation serving

- 状态：Accepted
- 日期：2026-08-27
- 范围：`epoch_0016_multilabel_hard.onnx` segmentation-only
- 环境：NVIDIA L4 24GB；Conda `diarizen-trt1010`；TensorRT 10.10.0.31；FP16 mixed precision

## 背景与 NFR

线上单条音频最长 16 秒，同时在途请求最多 50。目标是在满足该边界时最小化 GPU 推理延迟和单请求 GPU 成本。当前没有给出硬性端到端 SLA 或持续峰值 RPS，因此容量判断同时报告 50 请求突发和可持续最坏请求速率。

边界条件：输入为 16kHz 单通道；只测 synthetic；GPU 数字不包含网络、解码、重采样、排队前端及后处理。服务必须拒绝超过 16 秒的请求，具有有界队列、背压、warmup、监控和独立回退进程。

## 决策

1. 使用 TensorRT 10.10 FP16 mixed precision，不使用当前 FP8/INT8/INT4 engine。
2. 所有线上执行均为 `batch=1`，关闭动态合批。engine 只反序列化一次；每个活跃 engine 预创建两个独立 execution contexts，每个 context 独占 pinned host buffer、device buffer 和 CUDA stream；不要增加到 4 或 8，也不要逐请求分配显存。
3. 一张 L4 作为默认实例；最终双 context pinned 链路为 91.923 RPS，最坏 16 秒请求按 **73 RPS**（略低于实测容量的 80%）做 admission control。队列满时返回 429/RESOURCE_EXHAUSTED，或路由到第二张 L4。
4. 为兼顾可变时长成本与固定 shape 速度，加载三套 bs=1 FP16 plan。动态路径向上取整到 1 秒 bucket，并按时长路由：
   - `<2s`：补到 2s，动态 plan；
   - `2–8s`：动态 plan；
   - `>8–10s`：固定 10s plan；
   - `>10–12s`：动态 plan；
   - `>12–16s`：固定 16s plan。
5. 服务入口优先使用 gRPC 二进制 PCM16。上游完成 mono/16kHz 变换；避免 JSON/base64。服务预分配 pinned host 和 CUDA buffers，启动时 warmup 2/3/4/5/6/7/8/10/11/12/16 秒 shape。
6. 固定 shape/地址完成 warmup 后，捕获完整 pinned H2D→TensorRT→pinned D2H CUDA Graph。只有一个槽位活跃时优先 replay graph；有积压时向两个预分配槽位轮转派发。双槽 graph/enqueue/hybrid 同轮差异约 0.21%，不实现固定 hybrid 状态机；最终双槽容量使用普通 pinned enqueue 路径。
7. ORT CUDA FP32 作为独立进程/容器的可选回退，运行在 `diarizen` 环境，可启用固定地址 IOBinding + CUDA Graph。不能放在 `diarizen-trt1010` 主进程里，因为该环境的 ORT CUDA provider 缺少 cuDNN 8 时会回退 CPU。PyTorch 实验性二级回退候选按“先安装 SDPA patch，再 `torch.compile(mode="reduce-overhead")` + AMP”执行，启动时预编译固定 16 秒 shape；因本轮不做真实音频验收，不自动启用该候选。

## 实测依据

固定 16 秒 FP16：

| Batch | 总 mean | 每条 mean | 吞吐 |
|-------|-----------|-----------|------|
| 1 | 10.741ms | 10.741ms | 93.10 req/s |
| 2 | 25.572ms | 12.786ms | 78.21 req/s |
| 4 | 54.980ms | 13.745ms | 72.75 req/s |
| 8 | 119.112ms | 14.889ms | 67.16 req/s |
| 16 | 256.401ms | 16.025ms | 62.40 req/s |
| 32 | 544.677ms | 17.021ms | 58.75 req/s |

50 条 16 秒请求同时到达的早期 device 执行基线（不含完整 pinned H2D/D2H）：

| Contexts | 总完成 mean | 请求完成 p95 mean | 吞吐 | 设备内存短复测 |
|----------|-------------|-------------------|------|----------------|
| 1 | 546.610ms | 519.827ms | 91.47 req/s | 约 601MiB |
| 2 | 534.770ms | 514.271ms | 93.50 req/s | 约 813MiB |
| 4 | 555.050ms | 535.477ms | 90.08 req/s | — |
| 8 | 571.648ms | 554.939ms | 87.47 req/s | — |

最终生产形态用 `flock` 独占 GPU，仅比较双 context 完整 pinned 链路，3 次 warmup、30 轮：普通 enqueue 总完成 mean/p95 为 **543.935/547.260ms**，请求完成 p95 mean 为 **522.152ms**，吞吐 **91.923 req/s**；双 graph 为 91.590 req/s。因此 93.5 req/s 只保留为历史 device 基线，容量和成本统一按 91.923 req/s 计算。

动态时长 plan 的 2/4/6/8/10/12/13/14/16 秒 mean 为 3.449/3.970/5.004/6.209/6.909/9.711/10.732/11.645/13.895ms。固定 10s 为 6.283ms，固定 16s 为 10.741ms；13s mean 接近但固定 16s p95 更低，由此得到上述路由切点。

固定 16 秒 bs=1/2/4/8/16/32 相对 ORT CPU FP32 synthetic reference 的 cell/frame exact 均为 100%；但普通 synthetic hard output 全零，只能证明图执行、shape 和静音边界一致。已有 10 秒 adversarial synthetic 不能替代 16 秒非零类别/阈值附近回归；没有真实音频 DER/JER 结论。

进一步加速统一只测固定 16 秒 synthetic：

| 路径 | 关键结果 | 决策 |
|------|----------|------|
| TensorRT 完整 CUDA Graph，单 context | burst mean/p95 改善 2.67%/2.92%；完整链路 host enqueue 约 1.425ms→0.0079ms | **采用** |
| TensorRT 双 context 六模式 | 50 请求总完成 mean 555.586–556.754ms，最大差约 0.21% | 保留双槽，不固定 hybrid |
| TRT builder O4/O5/16GiB/aux | 都未同时优于 O3/8GiB 的 mean/p95；O5 最长构建 645s | 保留 O3/8GiB |
| ORT IOBinding + CUDA Graph | 38.816→38.003ms，快 2.09%，但仍比 TRT 慢 3.54× | 实验性可选回退 |
| PyTorch compile + SDPA + AMP | 39.099→**18.464ms**，快 2.12×，但仍比 TRT 慢 1.72×；raw 未通过 1e-3 allclose | 实验性二级候选，需真实验收 |

有效 graph 由 CUDA Runtime stream capture 产生，主实现为 394 个节点，独立 runner 为 396 个节点，且 graph/enqueue 在当前全零 hard-output synthetic 输入上 bit-exact；这不能外推非零类别准确率。曾出现的 0.014ms `torch.cuda.CUDAGraph` 空捕获已作废；生产 readiness 必须检查节点数非零和 exact parity。

## 取舍与被拒方案

- 大 batch / Triton dynamic batching：拒绝。该 L4/模型组合中 batch 越大，每条时间越高；50 条拆 32+16+2 约 826.7ms，明显慢于 batch=1 队列约 537ms。
- 4/8 个 contexts：拒绝。资源争用使吞吐和 p95 都变差。
- 双槽固定 hybrid 调度：拒绝。同轮交错测试的总完成 mean 并未优于两个普通 enqueue；跨进程偶然优势不可作为生产依据。
- O4/O5、更大 workspace、auxiliary streams：拒绝。没有同时改善 mean/p95，且显著增加构建时间。
- 强制 Flash Attention：拒绝。WavLM 非空 mask 路径不可用；Conformer 强制 Flash 也慢于 SDPA auto kernel。
- 只用动态 duration plan：运维简单，但 16s 为 13.895ms，比固定 16s 慢约 29.4%。
- 只用固定 10s/16s：更简单，但 2s 请求 GPU 时间高约 82%（6.283ms vs 3.449ms）。如果短音频占比极低，可选择这一简化方案。
- 两张 L4 默认常驻：拒绝，除非真实峰值超过约 73 个最坏请求/秒，或 50 请求突发端到端 p95 SLA 明显低于约 0.55s。

## 故障模式与运维要求

- 队列增长：用有界队列和 admission control；监控 queue depth/wait p95 和 429 数。
- sticky CUDA error/OOM：该 worker 退出并由进程管理器重启；不要只捕获异常后继续复用 context。
- plan 不兼容：plan 与 GPU 架构和 TensorRT 版本绑定；镜像升级或换 GPU 时从 ONNX 重建并跑 synthetic parity。
- 首请求抖动：启动阶段完成 plan 反序列化、各 shape warmup 后才通过 readiness。
- CUDA Graph 捕获错误：readiness 验证 graph 节点数、输出 shape、有限值和 enqueue/graph exact parity；任何一项失败即禁用 graph，回到普通 enqueue。
- 数据错误：校验 mono、16kHz、非空、有限值和 `duration<=16s`；超限不静默截断。
- 可观测性：分别记录 decode、queue、H2D、GPU、D2H、postprocess、端到端 p50/p95/p99、GPU 利用率/显存及 active contexts。

## 成本与扩容规则

最终普通 pinned enqueue 双 context 在 30 轮 50 请求测试中实测 **91.923 req/s**，即满载理论约 **330,900 请求/GPU-hour**；按 80% 水位约 264,700 请求/GPU-hour。生产 admission 向下取整为 73 RPS。因此：

`单请求 GPU 成本 ≈ L4 每小时价格 / 262800`（按 73 RPS admission 的生产容量口径）。

只有在 5 分钟持续到达率超过 73 RPS、queue wait p95 超过 SLA，或突发 p95 SLA 无法满足时才增加副本。缩容前要求队列为空并完成连接排空。

## 验证与回滚

上线前以 synthetic 固定输入跑 2/8/10/12/16 秒 readiness；对固定 16 秒 graph 额外检查节点数非零、enqueue/graph bit-exact，并确认 p95 未较本 ADR 基线回退超过 10%。灰度期间同时保留 ORT FP32 独立回退实例；发生 CUDA 错误、graph 捕获/输出 shape 异常或延迟回退时，先禁用 graph 回到普通 TensorRT enqueue，仍失败再切回 ORT，而不是运行当前 INT8 engine。

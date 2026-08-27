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
2. 所有线上执行均为 `batch=1`，关闭动态合批。每个活跃 engine 使用两个独立 execution contexts/CUDA streams；不要增加到 4 或 8。
3. 一张 L4 作为默认实例；最坏 16 秒请求按约 74.8 RPS（实测容量的 80%）做 admission control。队列满时返回 429/RESOURCE_EXHAUSTED，或路由到第二张 L4。
4. 为兼顾可变时长成本与固定 shape 速度，加载三套 bs=1 FP16 plan。动态路径向上取整到 1 秒 bucket，并按时长路由：
   - `<2s`：补到 2s，动态 plan；
   - `2–8s`：动态 plan；
   - `>8–10s`：固定 10s plan；
   - `>10–12s`：动态 plan；
   - `>12–16s`：固定 16s plan。
5. 服务入口优先使用 gRPC 二进制 PCM16。上游完成 mono/16kHz 变换；避免 JSON/base64。服务预分配 pinned host 和 CUDA buffers，启动时 warmup 2/3/4/5/6/7/8/10/11/12/16 秒 shape。
6. ORT CUDA FP32 仅作独立进程/容器回退，运行在 `diarizen` 环境。不能放在 `diarizen-trt1010` 主进程里，因为该环境的 ORT CUDA provider 缺少 cuDNN 8 时会回退 CPU。

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

50 条 16 秒请求同时到达：

| Contexts | 总完成 mean | 请求完成 p95 mean | 吞吐 | 设备内存短复测 |
|----------|-------------|-------------------|------|----------------|
| 1 | 546.610ms | 519.827ms | 91.47 req/s | 约 601MiB |
| 2 | 534.770ms | 514.271ms | 93.50 req/s | 约 813MiB |
| 4 | 555.050ms | 535.477ms | 90.08 req/s | — |
| 8 | 571.648ms | 554.939ms | 87.47 req/s | — |

动态时长 plan 的 2/4/6/8/10/12/13/14/16 秒 mean 为 3.449/3.970/5.004/6.209/6.909/9.711/10.732/11.645/13.895ms。固定 10s 为 6.283ms，固定 16s 为 10.741ms；13s mean 接近但固定 16s p95 更低，由此得到上述路由切点。

固定 16 秒 bs=1/2/4/8/16/32 相对 ORT CPU FP32 synthetic reference 的 cell/frame exact 均为 100%；普通 synthetic 输出全零，非零类行为由已有 10 秒 adversarial synthetic parity 覆盖。没有真实音频 DER/JER 结论。

## 取舍与被拒方案

- 大 batch / Triton dynamic batching：拒绝。该 L4/模型组合中 batch 越大，每条时间越高；50 条拆 32+16+2 约 826.7ms，明显慢于 batch=1 队列约 537ms。
- 4/8 个 contexts：拒绝。资源争用使吞吐和 p95 都变差。
- 只用动态 duration plan：运维简单，但 16s 为 14.048ms，比固定 16s 慢约 30.8%。
- 只用固定 10s/16s：更简单，但 2s 请求 GPU 时间高约 82%（6.283ms vs 3.449ms）。如果短音频占比极低，可选择这一简化方案。
- 两张 L4 默认常驻：拒绝，除非真实峰值超过约 74.8 个最坏请求/秒，或 50 请求突发端到端 p95 SLA 明显低于 500ms。

## 故障模式与运维要求

- 队列增长：用有界队列和 admission control；监控 queue depth/wait p95 和 429 数。
- sticky CUDA error/OOM：该 worker 退出并由进程管理器重启；不要只捕获异常后继续复用 context。
- plan 不兼容：plan 与 GPU 架构和 TensorRT 版本绑定；镜像升级或换 GPU 时从 ONNX 重建并跑 synthetic parity。
- 首请求抖动：启动阶段完成 plan 反序列化、各 shape warmup 后才通过 readiness。
- 数据错误：校验 mono、16kHz、非空、有限值和 `duration<=16s`；超限不静默截断。
- 可观测性：分别记录 decode、queue、H2D、GPU、D2H、postprocess、端到端 p50/p95/p99、GPU 利用率/显存及 active contexts。

## 成本与扩容规则

最坏 16 秒输入在两个 contexts 下实测约 93.5 req/s，即满载理论约 336,600 请求/GPU-hour；按 80% 水位约 269,300 请求/GPU-hour。因此：

`单请求 GPU 成本 ≈ L4 每小时价格 / 269300`（生产容量口径）。

只有在 5 分钟持续到达率超过 74.8 RPS、queue wait p95 超过 SLA，或突发 p95 SLA 无法满足时才增加副本。缩容前要求队列为空并完成连接排空。

## 验证与回滚

上线前以 synthetic 固定输入跑 2/8/10/12/16 秒 readiness，确认 shape、确定性和 p95 未较本 ADR 基线回退超过 10%。灰度期间同时保留 ORT FP32 独立回退实例；发生 CUDA 错误、输出 shape 异常或延迟回退时，将流量切回 ORT，而不是运行当前 INT8 engine。

# DiariZen Segmentation 模型 L4 线上推理周报总结

> **模型**：`kaldi_merged_1219_all_ft_large/epoch_0016`<br>
> **测试服务器**：`dev_L4_1gpus`，NVIDIA L4 24GB<br>
> **测试日期**：2026-08-26～2026-08-27<br>
> **总结日期**：2026-08-28<br>
> **范围**：仅 segmentation 模型；不含 speaker embedding、VBx、VAD、音频解码、网络和后处理<br>
> **完整证据报告**：[SEGMENTATION_BENCHMARK_REPORT.md](./SEGMENTATION_BENCHMARK_REPORT.md)

---

## 1. 一页结论

| 项目 | 最终结论 |
|------|----------|
| GPU | 单张 NVIDIA L4 24GB |
| 主后端 | TensorRT 10.10.0.31 FP16 |
| Conda 环境 | `/root/miniforge3/envs/diarizen-trt1010` |
| Batch | 固定 `batch=1`，关闭动态合批 |
| 模型调用 | 每条音频只执行一次 segmentation forward |
| 长音频约束 | 不切窗、不拆分、不截断 |
| Engine 组织 | 同一个 ONNX 模型生成 5 个时长优化 engine/profile |
| Worker | 4 个持久 worker；每个 worker 独立 context、stream 和预分配 buffer |
| 调度 | 立即分发到 outstanding GPU work 最少的 worker；不等待凑 50 条 |
| 目标流量 | 平均 5～6s、中位数 4 秒多、约 5% `>16s`、99.9% `<=30s` |
| 代表性测试 | 20 个 50 请求 burst，共 1000 个 synthetic 请求；均值 5.68s、中位数 4.5s、5% `>16s` |
| 满载能力 | **199.952 req/s** |
| 生产 admission | **159 req/s**，同时限制 `inflight_requests <= 50` 和 GPU work units |
| 50 并发整批完成 | burst mean/p95：**250.060/265.993ms** |
| 单请求完成 | request p50/p95：**164.444/245.640ms** |
| 常驻显存 | 约 **6151 MiB** |
| 相对全补 16s | 吞吐 **2.175×**；burst mean 与单位 GPU 成本约下降 **54.03%** |
| 30s 单次 forward | mean/p95 **31.613/32.540ms**，输出 `[1,1499,4]` |
| Triton 判断 | 运维能力更强，但不会让同一 TensorRT plan 的 kernel 更快；当前最低延迟仍选自定义 runner |

**最终建议**：使用 **1 张 L4 + TensorRT FP16 + batch=1 + 5 个时长路由 + 4 个持久 worker + GPU work-unit admission**。主路由覆盖到 30s；极少数 `>30s` 请求保持完整时长，走独立 ORT CUDA/CPU 单次 forward，或在明确绝对上限后新增 TensorRT overflow profile。

---

## 2. 业务前提与测试边界

| 维度 | 约束或说明 |
|------|------------|
| 音频 | 16kHz、单通道 |
| 平均时长 | 5～6s |
| 中位数 | 4 秒多；代表性 synthetic manifest 为 4.5s |
| 长尾 | 约 5% 超过 16s，99.9% 不超过 30s |
| 最大并发 | 50 |
| 推理语义 | 同一个 segmentation 模型、每条请求一次 forward |
| 禁止行为 | 禁止静默截断、切窗或拆分 |
| 本轮新增速度输入 | Synthetic tensor，不使用真实音频 |
| 本轮不包含 | 网络、PCM 解码、重采样、业务排队、embedding、聚类、VAD 和后处理 |
| 准确率边界 | Synthetic parity 不能替代真实音频 DER/JER 验收 |
| 30s 含义 | `99.9% <= 30s` 是分布描述，不等于绝对最大时长为 30s |

代表性 A/B manifest 用于在没有真实线上直方图的前提下复现已知分布摘要：

| Manifest | 50 条时长组成 | 均值/中位数 | `>16s` |
|----------|---------------|-------------|--------|
| A | `1×4, 2×6, 3×8, 4.5×8, 5×7, 6×5, 8×5, 10×3, 12×2, 18×1, 22×1` | 5.50/4.5s | 4% |
| B | `1×4, 2×6, 3×8, 4.5×8, 5×7, 6×5, 8×4, 10×3, 12×2, 18×1, 22×1, 26×1` | 5.86/4.5s | 6% |
| A+B 聚合 | 各 10 轮、1000 请求 | **5.68/4.5s** | **5%** |

这些数字只代表当前 synthetic 场景。上线后的真实时长直方图、到达过程和 `>30s` 比例变化时，需要重新测量或重算 admission。

---

## 3. 最终时长路由

5 个 TensorRT engine 来自相同的 ONNX 文件和相同权重，不是 5 个不同模型。拆分 profile 的原因是 TensorRT tactic 会受到 min/opt/max shape 影响，一个覆盖 2～30s 的宽 profile 在中长输入上明显更慢。

| 原始时长 | 执行路径 | 实际输入 | 选择原因 |
|----------|----------|----------|----------|
| `(0,2s)` | short dynamic `2/6/16` | 补到 2s | profile 最小 shape；仍只执行一次 forward |
| `[2s,7s]` | short dynamic `2/6/16` | 原长或 1s bucket | opt=6 对短输入最快 |
| `(7s,8s]` | mid dynamic `2/10/16` | 原长或 1s bucket | 8s 点 mid profile 更快 |
| `(8s,10s]` | fixed10 | 补到 10s | 固定 10s mean 6.283ms，优于动态 10s |
| `(10s,12s]` | mid dynamic `2/10/16` | 原长或 1s bucket | 避免补到 16s |
| `(12s,16s]` | fixed16 | 补到 16s | 固定 16s mean 10.741ms，14～16s 明显更快且 p95 稳定 |
| `(16s,30s]` | long dynamic `16/24/30` | 原长或 1s bucket | 长区间专用 tactic；同模型单次 forward |
| `>30s` | ORT CUDA/CPU overflow，或未来 TRT overflow | 保持完整原长 | 主 TRT profile 有界到 30s，禁止错误送入或截断 |

### 3.1 为什么不用一个 2～30s 万能 engine

| 输入 | 宽 profile `2/6/30` | 专用 profile | 改善 |
|------|---------------------|--------------|------|
| 8s | 7.080ms | 6.334ms（mid） | 10.5% |
| 10s | 9.116ms | 6.283ms（fixed10） | 31.1% |
| 16s | 16.603ms | 10.741ms（fixed16） | 35.3% |
| 24s | 30.680ms | 19.326ms（long） | 37.0% |
| 30s | 44.072ms | 31.613ms（long） | 28.3% |

结论：多 profile 路由不是为了改变模型输出，而是为了让 TensorRT 在每个长度区间选到更合适的执行 tactic。

---

## 4. 四类延迟指标的区别

| 指标 | 统计对象 | 含义 | 当前结果 |
|------|----------|------|----------|
| Burst Mean | 20 个 50 并发 burst | 每批 50 条全部完成时间的平均值 | 250.060ms |
| Burst P95 | 20 个 50 并发 burst | 95% 的 burst 能在该时间内全部清空 | 265.993ms |
| Request P50 | 1000 个单独请求 | 50% 的请求从 burst 开始到完成不超过该时间 | 164.444ms |
| Request P95 | 1000 个单独请求 | 95% 的请求从 burst 开始到完成不超过该时间 | 245.640ms |

一次 50 并发可以理解为：

```text
0ms        50 条请求同时到达
164.444ms  约一半请求完成
245.640ms  约 95% 请求完成
250ms 左右 最后几个长请求完成，整批清空
265.993ms  95% 的整批测试都能在此时间内清空
```

Request P95 适合描述绝大多数单请求的服务延迟；Burst P95 适合描述系统清空一次 50 并发突发的能力。两者都不包含进入 benchmark 之前的网络、业务服务排队和音频解码。

---

## 5. 单请求关键时长结果

### 5.1 短音频

| 时长 | 最终路径 | Mean | P95 |
|------|----------|------|-----|
| 2s | short dynamic | 2.845ms | 2.861ms |
| 3s | short dynamic | 3.152ms | 3.190ms |
| 4s | short dynamic | 3.636ms | 3.669ms |
| 4.5s | short dynamic | 3.789ms | 3.830ms |
| 5s | short dynamic | 4.074ms | 4.138ms |
| 6s | short dynamic | 4.357ms | 4.390ms |
| 7s | short dynamic | 5.382ms | 5.404ms |
| 8s | mid dynamic | 6.334ms | 6.370ms |
| 10s | fixed10 | 6.283ms | 6.434ms |
| 12s | mid dynamic | 9.897ms | 10.178ms |
| 16s | fixed16 | 10.741ms | 10.823ms |

### 5.2 16～30s 原长单次 forward

| 时长 | Mean | P95 | 输出 shape |
|------|------|-----|------------|
| 18s | 15.079ms | 15.633ms | `[1,899,4]` |
| 20s | 17.028ms | 17.481ms | `[1,999,4]` |
| 22s | 19.482ms | 20.142ms | `[1,1099,4]` |
| 24s | 19.326ms | 19.842ms | `[1,1199,4]` |
| 26s | 25.270ms | 26.180ms | `[1,1299,4]` |
| 28s | 28.428ms | 29.209ms | `[1,1399,4]` |
| 30s | **31.613ms** | **32.540ms** | `[1,1499,4]` |

30s 结果证明输入 `[1,1,480000]` 可以由同一个 segmentation 模型单次执行；它不代表真实业务 P99.9 样本准确率已完成验收。

---

## 6. Worker 1～8 全扫描

同一 A/B manifest、每档 20 个 burst、1000 请求：

| Workers | RPS | Burst Mean | Burst P95 | Request P50 | Request P95 | 常驻显存 | 判断 |
|---------|-----|------------|-----------|-------------|-------------|----------|------|
| 1 | 146.001 | 342.463ms | 357.416ms | 228.477ms | 340.748ms | 2267MiB | 并行度不足 |
| 2 | 185.865 | 269.012ms | 280.653ms | 182.211ms | 264.762ms | 3561MiB | 明显改善 |
| 3 | **200.119** | **249.851ms** | 271.515ms | 170.689ms | 253.369ms | 4857MiB | RPS 小数点最高，但尾延迟较差 |
| **4** | **199.952** | **250.060ms** | **265.993ms** | **164.444ms** | **245.640ms** | **6151MiB** | **生产选择** |
| 5 | 193.444 | 258.472ms | 267.187ms | 167.539ms | 254.093ms | 7447MiB | 开始争用 |
| 6 | 185.196 | 269.985ms | 285.220ms | 175.391ms | 262.857ms | 8741MiB | 性能回退 |
| 7 | 180.318 | 277.288ms | 303.522ms | 180.647ms | 272.363ms | 10039MiB | p95 明显恶化 |
| 8 | 188.136 | 265.765ms | 281.671ms | 171.162ms | 258.133ms | 11333MiB | 资源争用、显存过高 |

3 workers 的吞吐只比 4 workers 高 0.084%，小于这类 GPU benchmark 的正常波动；4 workers 的 burst p95 低 5.522ms，请求 p95 低 7.729ms，请求 p50 也低 6.245ms。因此生产按“吞吐相同量级、尾延迟更低”选择 4 workers。5～8 workers 增加了 context 和显存，但没有继续提高容量，说明已进入 SM/L2/DRAM/context 调度争用区。

---

## 7. 50 并发最终结果与历史基线

| 方案 | Burst Mean | Burst P95 | Request P95 | 吞吐 |
|------|------------|-----------|-------------|------|
| 所有请求补到 16s（历史基线） | 543.935ms | 547.260ms | 522.152ms | 91.923 req/s |
| **按时长路由、4 workers** | **250.060ms** | **265.993ms** | **245.640ms** | **199.952 req/s** |
| 改善 | **54.03%** | **51.40%** | **52.96%** | **2.175×** |

| 成本口径 | 公式 |
|----------|------|
| 满载 199.952 RPS | `L4 每小时价格 / 719827` |
| 生产 159 RPS | `L4 每小时价格 / 572400` |

成本公式只覆盖 segmentation GPU 推理。相对全补 16s，当前代表性分布下的单位 GPU 成本约下降 54.03%。

---

## 8. Admission 与在线调度

### 8.1 推荐门槛

| 门槛 | 建议值 | 作用 |
|------|--------|------|
| 持续 admission | 159 req/s | 199.952 RPS 满载能力的 80%，向下取整 |
| Inflight 请求 | `<=50` | 控制连接、内存和请求数量 |
| Outstanding WU | `<=50` | 避免 50 条长请求被当成 50 条短请求 |
| Worker | 4 | 选择当前 outstanding WU 最低的 worker |
| Batch | 1 | 请求立即执行，不等待动态合批 |

定义 `1 WU = 10.741ms`，即固定 16s FP16 请求的 mean GPU 工作量。每条请求按选中 route 的 `p95_gpu_ms / 10.741` 计费；没有精确测点时向上取相邻时长 bucket。

### 8.2 在线请求流程

```text
接收二进制 PCM
  -> 解码/重采样到 16kHz 单通道
  -> 读取真实样本长度
  -> 按时长选择 short/mid/fixed10/fixed16/long
  -> 计算该请求 GPU work units
  -> 检查 RPS、inflight 和 outstanding WU
  -> 选择 outstanding WU 最低的 worker
  -> 复用 pinned/device buffer
  -> 单次 TensorRT forward
  -> 返回 multilabel
```

真实在线服务不知道未来完整 50 请求列表，因此不能照搬 benchmark 的离线 LPT 排序；在线等价实现是每次把新请求投给当前 outstanding WU 最低的 worker。禁止为了凑满 50 条而等待，否则会增加用户排队延迟。

---

## 9. 推荐生产配置

```yaml
hardware:
  gpu: NVIDIA L4 24GB
  replicas: 1

runtime:
  conda_env: /root/miniforge3/envs/diarizen-trt1010
  tensorrt: 10.10.0.31
  precision: fp16
  batch_size: 1
  dynamic_batching: false

routes:
  short_dynamic: 2s-16s_opt6s
  mid_dynamic: 2s-16s_opt10s
  fixed_10s: 10s
  fixed_16s: 16s
  long_dynamic: 16s-30s_opt24s

execution:
  persistent_workers: 4
  contexts_per_engine_per_worker: 1
  pinned_host_buffers: preallocated
  device_buffers: preallocated
  dispatch: least_outstanding_gpu_work
  wait_for_batch: false

admission:
  sustained_requests_per_second: 159
  max_inflight_requests: 50
  max_outstanding_work_units: 50

overflow:
  duration_gt_30s: preserve_original_length
  primary: independent_ort_cuda_single_forward
  cuda_failure: whole_request_ort_cpu_single_forward
  future: build_trt_profile_after_absolute_max_is_known
```

服务启动时应加载一次全部 plan，并预热 2/6/8/10/12/16/18/24/30s bucket。禁止逐请求反序列化 engine、创建 context 或执行 `cudaMalloc`。

---

## 10. 精度与风险边界

### 10.1 30s synthetic parity

| 输入 | ORT CUDA FP32 | TRT FP16 | Cell Exact | Frame Exact | Mismatch |
|------|-----------------|----------|------------|-------------|----------|
| Harmonic synthetic | 0/5996 正类 | 0/5996 正类 | 100% | 100% | 0 cell/0 frame |
| Adversarial synthetic | 921/5996 正类 | 929/5996 正类 | **99.766511%** | **99.199466%** | 14 cells/12 frames |

普通 harmonic 输入全零，只证明 shape、执行和静音边界一致。Adversarial 输入包含非零类别，但仍是人工输入，说明 FP16 与 ORT FP32 高度接近但不是严格无损。上线前仍需要真实音频 DER/JER 验收；本轮按要求没有运行真实音频。

### 10.2 `>30s` 请求

主 TensorRT long profile 的最大输入为 30s。由于当前只知道 99.9% 不超过 30s，而不知道绝对最大长度，不能构建一个有界且覆盖全部输入的 TensorRT profile。

| 情况 | 处理 |
|------|------|
| `<=30s` | 走 5 个主 TensorRT 路由 |
| `>30s` | 保留原始完整 shape，独立 ORT CUDA 单次 forward |
| ORT CUDA 对超长 shape 失败 | 整条请求回退 ORT CPU，仍只执行一次完整 forward |
| 后续获得绝对最大时长 | 构建低频 TensorRT overflow profile 并单独验证显存/延迟 |

不能把 `>30s` 错误送入 max=30s 的 plan，也不能通过截断或拆分掩盖错误。

---

## 11. 为什么最终选择 TensorRT FP16

| 路线 | 结论 |
|------|------|
| TensorRT FP16 | L4 上速度最快，作为主后端 |
| TensorRT FP32/TF32/BF16 | 已测，可用但速度不如 FP16 |
| TensorRT FP8 | 速度和精度综合不优于 FP16 |
| TensorRT INT8 | Synthetic 精度严重失败且运行时不稳定，不采用 |
| TensorRT INT4 weight-only | 仍以 Float 计算，精度下降且没有加速 |
| ORT CUDA | 作为独立 overflow/回退；主路径明显慢于 TensorRT |
| PyTorch compile+SDPA+AMP | 比 eager 快，但仍慢于 TensorRT；作为开发回退 |
| 动态合批 | 固定 16s 扫描显示 batch 增大后单位请求效率下降，生产关闭 |
| CUDA Graph | 固定 shape 单槽有约 2.6% 小收益；多 worker 混合动态 shape 不作为核心容量来源 |

---

## 12. Triton Inference Server 判断

| 维度 | 当前自定义 TensorRT runner | NVIDIA Triton |
|------|-----------------------------|---------------|
| TensorRT kernel | 相同 plan | 相同 plan，不会自动更快 |
| 最低单请求延迟 | 更有优势 | 增加协议、调度和队列层 |
| 混合时长调度 | 已实现按 GPU work unit 调度 | 默认 scheduler 不理解 3s 与 30s 的成本差异 |
| 动态 batching | 已关闭 | 也必须关闭 |
| 健康检查/指标/版本管理 | 需要自行维护 | 更完善 |
| Kubernetes/多模型/多GPU | 需要额外开发 | 更方便 |

当前目标是单张 L4 上的最低延迟和最低成本，因此继续使用自定义 TensorRT runner。若后续重点转向 Kubernetes、多模型、多 GPU、标准指标和模型版本治理，再用同一批 plans 和 A/B manifest 做 Triton 对照压测；不能在没有实测时假设 Triton 会超过 199.952 RPS。

---

## 13. 监控与扩容建议

| 指标 | 用途 |
|------|------|
| `duration_bucket` | 观察真实时长分布是否偏离 5.68s 代表性场景 |
| `route` | 统计五路由比例和 engine 热点 |
| `decode_ms` | 区分音频解码与 GPU 延迟 |
| `queue_ms` | 判断 admission 是否过高 |
| `gpu_ms` | 更新各时长 bucket 的 WU 权重 |
| `request_latency_p50/p95/p99` | 用户侧 SLA |
| `burst_clear_ms` | 突发流量清空能力 |
| `inflight_requests` | 请求数量保护 |
| `outstanding_work_units` | 实际 GPU 工作量保护 |
| `overflow_gt_30s` | 评估是否需要新建 TRT overflow profile |
| `rejected_requests` | 评估是否需要第二张 L4 |
| GPU 显存/利用率/温度 | 确认 4-worker 配置运行稳定 |

扩容触发建议：持续到达率接近 159 RPS、queue p95 持续上升、`>30s` overflow 比例上升，或端到端 p95 超过业务 SLA 时，增加 L4 副本，而不是继续增加单卡 worker。5～8 workers 已实测进入争用区。

---

## 14. 本周完成事项

| 事项 | 状态 | 结果 |
|------|------|------|
| TensorRT Conda 环境 | ✅ | `/root/miniforge3/envs/diarizen-trt1010`，TRT 10.10.0.31 |
| TensorRT 问题修复 | ✅ | 解决旧版 FP16 构建崩溃，engine 可构建和执行 |
| 多精度扫描 | ✅ | FP32/TF32/FP16/BF16/FP8/INT8/INT4 完成，FP16 最优 |
| 2～30s profile 扫描 | ✅ | 2/6/16、2/10/16、2/6/30、16/20/30、16/24/30 |
| 最终五路由 | ✅ | short/mid/fixed10/fixed16/long |
| 30s 单次 forward | ✅ | 31.613/32.540ms，输出 `[1,1499,4]` |
| 50 并发混合分布 | ✅ | 20 burst、1000 请求，5% `>16s` |
| Worker 1～8 扫描 | ✅ | 4 workers 生产最优 |
| Admission | ✅ | 159 RPS、50 inflight、WU 限流 |
| 30s parity | ✅ | 全零 100%；非零 cell/frame 99.766511%/99.199466% |
| 单元测试 | ✅ | 24 项通过 |
| JSON 结果 | ✅ | 全部可解析并入库 |
| 中文完整报告 | ✅ | `SEGMENTATION_BENCHMARK_REPORT.md` |
| 脚本/报告提交 | ✅ | 已在 `feature/nemo-ssl-nest-finetune` 分支持续维护 |

---

## 15. 下一步

| 优先级 | 事项 | 目的 |
|--------|------|------|
| P0 | 用真实线上时长直方图替换 A/B synthetic manifest | 收敛真实容量、成本和 admission |
| P0 | 明确业务绝对最大音频时长 | 决定 `>30s` 是拒绝、ORT overflow 还是新增 TRT profile |
| P0 | 上线前补真实音频 FP16 DER/JER | 验证 synthetic parity 不能覆盖的真实准确率 |
| P1 | 在真实服务链路测端到端 p50/p95/p99 | 加入网络、解码、排队和后处理 |
| P1 | 实现 duration route、WU admission 和最小 outstanding work 调度 | 将 benchmark 最优配置落实到服务 |
| P1 | 建立 159 RPS 压测、过载和第二张 L4 扩容测试 | 验证稳定性和降级策略 |
| P2 | 若需要平台化，再做 Triton A/B | 评估运维收益是否值得少量延迟开销 |

---

## 16. 产物位置

| 产物 | 路径 |
|------|------|
| 完整详细报告 | `inference/models/kaldi_merged_1219_all_ft_large/SEGMENTATION_BENCHMARK_REPORT.md` |
| 本周总结 | `inference/models/kaldi_merged_1219_all_ft_large/L4_ONLINE_INFERENCE_WEEKLY_SUMMARY.md` |
| 动态时长 benchmark | `inference/benchmark_trt_dynamic_duration.py` |
| 混合时长/worker benchmark | `inference/benchmark_trt_mixed_duration_burst.py` |
| 动态 TensorRT parity | `inference/benchmark_trt_fixed10s_parity.py` |
| Worker 1～8 JSON | `epoch_0016_l4_trt1010_fp16_mixed_duration_50concurrency_workers{1..8}_bs1.json` |
| Long profile JSON | `epoch_0016_l4_trt1010_fp16_dynamic_16s-30s_opt24s_bs1.json` |
| 30s parity JSON | `epoch_0016_l4_trt1010_fp16_dynamic30s*_parity.json` |

大型 TensorRT plan 保留在 L4 工作目录并由 `.gitignore` 排除；Git 中保存构建脚本、完整 profile、plan SHA256、性能 JSON 和复现命令。

---

*本文是周报和上线评审摘要；实验方法、历史 A800/L4 数据、多精度结果、完整复现命令和全部 JSON 索引以 `SEGMENTATION_BENCHMARK_REPORT.md` 为准。*

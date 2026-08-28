# 线上最优方案与容量决策

## 当前推荐结论

在已验证的 L4、TensorRT 10.10、epoch 16 ONNX、代表性混合时长和 50 请求 burst 条件下：

- 推理：TensorRT FP16，`batch=1`，不启用 dynamic batching。
- 模型调用：每条音频整段、单次 segmentation forward，不切片、不截断。
- 路由：short/mid/fixed10/fixed16/long 五类 engine/profile，按时长选择。
- 并发：4 个长期驻留 worker；每 worker/route 独立 context、stream 和 buffer。
- 调度：在线按最小 outstanding work units 分派，不等待凑 batch。
- 准入：`inflight_requests <= 50` 且 `outstanding_WU <= 50`；超过阈值排队或限流。
- 容量：混合分布实验满载 `199.952 RPS`，上线规划先按 80% 折扣约 `159 RPS/GPU`。
- 超长：`16–30s` 用 long dynamic 单次推理；`>30s` 不能截断，走独立 ORT CUDA/CPU 单次 forward，或在确认绝对上限后另建 overflow TRT profile。

这是“已验证条件下的推荐”，不是对任意线上流量的永久保证。

## 路由表

| 实际时长 | 使用路线 | 实际输入 | 原因 |
|---|---|---|---|
| `(0,2s)` | short dynamic `2/6/16` | pad 到 2s | 满足最小 shape |
| `[2,7s]` | short dynamic `2/6/16` | 原长 | opt=6 靠近主体短请求 |
| `(7,8s]` | mid dynamic `2/10/16` | 原长 | 避免 short profile 边缘退化 |
| `(8,10s]` | fixed10 | pad 到 10s | 固定 10 秒 tactic 更优 |
| `(10,12s]` | mid dynamic `2/10/16` | 原长 | 避免多余 pad |
| `(12,16s]` | fixed16 | pad 到 16s | 固定 16 秒性能更稳 |
| `(16,30s]` | long dynamic `16/24/30` | 原长 | 专门优化长请求 |
| `>30s` | overflow fallback | 保留全长 | 满足单次推理且不截断 |

服务启动时加载五个 TRT engine，不要按请求加载/卸载。它们共享同一 ONNX 权重语义，但 plan 独立。

## 代表性实测

流量构造：20 个 burst，每个 50 个请求，共 1000 个合成请求；平均时长 `5.68s`，中位数 `4.5s`，约 `5%` 超过 16 秒。

### 推荐 4 worker

| 指标 | 结果 |
|---|---:|
| 吞吐 | `199.952 RPS` |
| Burst Mean | `250.060 ms` |
| Burst P95 | `265.993 ms` |
| 请求 P50 | `164.444 ms` |
| 请求 P95 | `245.640 ms` |
| GPU 显存 | `6151 MiB` |

这些是 GPU 侧 benchmark 口径，不包含线上网络、音频下载/解码、排队、Python/服务框架开销和后处理。

### 相比全部 pad 到 16 秒

| 方案 | 吞吐 | Burst Mean | Burst P95 | 请求 P95 |
|---|---:|---:|---:|---:|
| 全部 pad16 基线 | `91.923 RPS` | `543.935 ms` | `547.260 ms` | `522.152 ms` |
| 时长路由 + 4 worker | `199.952 RPS` | `250.060 ms` | `265.993 ms` | `245.640 ms` |

路由方案吞吐约 `2.175×`，Burst Mean 约下降 `54.03%`。主要收益来自避免把平均 5–6 秒的音频全部按 16 秒计算，而不是来自更激进的量化。

## Work Unit 调度

以 fixed16 平均推理 `10.741 ms` 定义 `1 WU`。每条路线根据已测 route P95 换算 WU：

```text
request_WU = route_p95_ms / 10.741
```

时长未知或刚落在边界时，向更重的桶取整，避免低估。每个 worker 维护当前累计 outstanding WU；新请求派到最小者。完成后扣减对应 WU。

双阈值的作用：

- `inflight_requests` 防止大量短请求把请求数打爆。
- `outstanding_WU` 防止少量 30 秒请求把实际计算量打爆。

阈值 `50/50` 是当前起点，不是永远最佳。用真实端到端压测和 SLO 校准。

## 长音频处理

“5% 超过 16 秒、99.9% 不超过 30 秒”仍意味着存在 `>30s` 请求。正确处理：

1. `(16,30s]`：long dynamic `16/24/30`，整段一次 forward。
2. `>30s`：不截断、不切片；进入独立 overflow 队列。
3. 若业务能提供绝对最大值，构建覆盖该最大值的专用 TRT overflow profile，再做显存、P95 和一致性测试。
4. 最大值未知时，ORT CUDA 是兼容性优先回退；必要时 CPU 是最后保障。将 overflow 的容量和 SLO 单独核算，不能混入主路由的 `159 RPS`。

已测 30 秒 long profile：平均约 `31.613 ms`、P95 `32.540 ms`、输出 `[1,1499,4]`。

## 容量与成本

满载实验吞吐 `199.952 RPS` 不应直接作为售卖容量。当前规划值：

```text
production_rps = 199.952 × 0.8 ≈ 159 RPS/GPU
```

若一张 L4 每小时综合成本为 `C`：

```text
满载每百万请求成本 = C / 719827 × 1,000,000
159 RPS 规划容量下每百万请求成本 = C / 572400 × 1,000,000
```

这里 `719827 ≈ 199.952 × 3600`，`572400 = 159 × 3600`。实际成本还应加入空闲率、overflow、服务 CPU、网络、重试、监控和冗余副本。

## Triton 取舍

Triton 不会让同一个 TensorRT plan 的 kernel 自动更快，通常还会增加协议、调度和内存复制开销。它的价值在工程治理：模型版本、健康检查、指标、Kubernetes 集成、实例管理和标准化协议。

如果已有完善的 Python/C++ 常驻 worker，追求最低延迟和最低成本，优先保留轻量自研服务。若组织需要统一模型平台，可测试 Triton，但配置应为：

- 复用已验证 plan。
- `max_batch_size: 0`，不启用 dynamic batching。
- 客户端或前置路由器按时长选 model/instance。
- `instance_group` 数量与已测 worker 规模对应。
- 端到端 A/B：客户端网络、序列化、排队、GPU、错误率、显存一起测。

Triton 默认调度器不知道 WU，也不知道一个 30 秒请求比 4 秒请求重。若直接轮询，尾延迟可能退化；需要外置 duration-aware/WU router 或自定义调度层。

## 监控与回滚

至少监控：

- 各时长桶 QPS、P50/P95/P99、排队时间。
- inflight、outstanding WU、拒绝/超时/回退率。
- 每 route 错误率、shape/profile 越界、NaN/Inf。
- GPU 利用率、显存、功耗、温度、频率。
- `>30s` 比例和 overflow 延迟。
- 线上真实时长直方图与基准分布的偏移。

回滚条件包括：真实质量指标退化、P95/P99 超 SLO、显存逼近安全线、engine 反序列化失败、profile 越界或 overflow 堆积。保留 ORT CUDA 路线作为可解释的回退。

## 上线前仍需完成的验证

当前数字基于合成音频和 GPU 侧计时。正式生产前必须补：

1. 脱敏真实时长直方图回放。
2. 真实代表集的 segmentation/DER/JER 或业务质量门槛。
3. 包含网络、解码、路由、排队和后处理的端到端压测。
4. 持续流量、突发流量、长音频尖峰和故障回退测试。
5. 按实际 SLO 重新确定安全 RPS、准入阈值和副本数。

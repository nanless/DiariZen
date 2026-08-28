# L4/TensorRT 实验方法

## 一、先定义“测到了什么”

性能数字必须同时带上六类上下文：

1. 身份：主机、GPU、软件版本、分支、提交、ONNX SHA256、plan SHA256。
2. 输入：真实/合成、采样率、时长、分布、随机种子、是否 padding。
3. 执行：精度、profile、batch、worker、stream/context、调度器。
4. 计时：GPU kernel、H2D/D2H、客户端、网络、排队中包含哪些。
5. 统计：warmup、重复数、burst 数、每 burst 请求数、分位数定义。
6. 正确性：输出 shape、有限值、一致率、DER/JER 或业务指标。

少任一类，结论应标记为“探索性”。

## 二、GPU 互斥与污染控制

正式实验前：

- 查看 `nvidia-smi` 的 compute process、功耗、频率、显存和温度。
- 用 `flock /tmp/diarizen_l4_benchmark.lock` 包住完整 benchmark。
- 预热后再计时；不要把首次 engine 反序列化、CUDA 上下文初始化算进稳态推理。
- 对候选方案交错或重复测试，避免只按固定顺序造成温度/频率偏差。
- 记录异常值，不静默删除；若因外部进程污染重跑，要写明原因。

## 三、精度路线

固定 10 秒阶段已覆盖 FP32、TF32、FP16、BF16、FP8、INT8、INT4。结论是 FP16 在本模型与 L4 上综合最优。低位宽没有自动带来更低延迟：量化/反量化、图分割、算子支持和校准误差都会抵消理论收益。

选择顺序：

1. 构建/加载是否稳定。
2. 输出是否有限、shape 是否正确。
3. 与 ORT FP32 的数值一致性。
4. 真实代表集质量门槛。
5. 平均/尾延迟、吞吐、显存。

不要用全零输入 100% 一致证明 FP16/INT8 “无损”。全零只适合 smoke test；至少加入谐波、噪声、脉冲、幅度边界等对抗合成波形。

## 四、Profile 设计原则

`min/opt/max` 不是越宽越好：TensorRT 会围绕 opt shape 选择 tactic。已验证例子中，30 秒请求在宽 profile `2/6/30` 下均值约 `44.072 ms`，改用长音频 profile 后约 `31.613 ms`，改善约 `28.3%`。

当前五路方案：

| Route | 请求时长 | Engine/Profile | Padding |
|---|---|---|---|
| A | `(0, 2s)` | dynamic `2/6/16` | pad 到 2s |
| B | `[2, 7s]` | dynamic `2/6/16` | 不额外 pad |
| C | `(7, 8s]` | dynamic `2/10/16` | 不额外 pad |
| D | `(8, 10s]` | fixed 10s | pad 到 10s |
| E | `(10, 12s]` | dynamic `2/10/16` | 不额外 pad |
| F | `(12, 16s]` | fixed 16s | pad 到 16s |
| G | `(16, 30s]` | dynamic `16/24/30` | 不额外 pad |

表中逻辑上有七个时长区间，但复用五个 engine/profile 产物：short dynamic、mid dynamic、fixed10、fixed16、long dynamic。它们来自同一 ONNX 权重，不是七个模型。

每次 `--reuse-existing` 都要从 plan/engine inspector 或运行时 binding/profile 读取实际范围。不要相信命令行里准备使用的 profile 就等于已有 plan 的 profile。

## 五、Context/Stream/Buffer 所有权

线上每个 worker/route 至少需要独立：

- TensorRT execution context。
- CUDA stream。
- 输入/输出 device buffer。
- 必要的 pinned host buffer。

不要跨并发请求共享可变 context/buffer。动态 shape 变化只能在同一 context 的前序 stream 工作完成后安全设置。稳态阶段预分配最大需要的 buffer，不要每请求反序列化 plan、创建 context 或 `cudaMalloc`。

## 六、Worker 扫描与停止条件

从 1 worker 递增，直到：

- 吞吐进入平台或下降；
- 请求 P95/Burst P95 恶化；
- 显存余量不足；
- 失败/超时出现。

当前混合时长实测（20 个 burst × 50 请求，共 1000 个合成请求）：

| Worker | RPS | Burst Mean ms | Burst P95 ms | 请求 P50 ms | 请求 P95 ms | 显存 MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 146.001 | 342.463 | 357.416 | 228.477 | 340.748 | 2267 |
| 2 | 185.865 | 269.012 | 280.653 | 182.211 | 264.762 | 3561 |
| 3 | 200.119 | 249.851 | 271.515 | 170.689 | 253.369 | 4857 |
| 4 | 199.952 | 250.060 | 265.993 | 164.444 | 245.640 | 6151 |
| 5 | 193.444 | 258.472 | 267.187 | 167.539 | 254.093 | 7447 |
| 6 | 185.196 | 269.985 | 285.220 | 175.391 | 262.857 | 8741 |
| 7 | 180.318 | 277.288 | 303.522 | 180.647 | 272.363 | 10039 |
| 8 | 188.136 | 265.765 | 281.671 | 171.162 | 258.133 | 11333 |

4 worker 是联合指标最优点。3 worker 的 RPS 高 `0.167`、仅约 `0.084%`，但 4 worker 的请求 P50/P95 与 Burst P95 都更低。继续增加 worker 没有收益。

## 七、混合时长与并发语义

代表性分布基线：平均 `5.68s`、中位数 `4.5s`、约 `5%` 超过 16 秒、`99.9%` 不超过 30 秒。该分布只能代表当时构造，不等于任意满足同样均值/中位数的线上分布。

区分两种调度：

- 离线已知整批：可按预计工作量做 LPT 排序，得到更好的 burst 清空时间。
- 在线请求逐个到达：不能等凑齐 50 个；应将请求派到累计 outstanding work units 最小的 worker。

如果 benchmark 使用 LPT，报告必须注明它是离线/同时到达上界，不应直接宣称为在线 greedy 的实测延迟。

## 八、指标定义

在本项目 50 并发 burst 语境中：

- **Burst Mean**：每个 50 请求 burst 从共同开始到最后一个请求结束的耗时，跨 burst 取平均。
- **Burst P95**：上述每个 burst 完整清空耗时的 P95。
- **请求 P50**：全部单请求从 burst 开始到各自完成的延迟中位数。
- **请求 P95**：全部单请求完成延迟的 P95。

Burst 指标回答“这一波什么时候清空”，请求指标回答“用户通常/尾部等多久”。两组不能互相替代。

## 九、数值一致性已知证据

30 秒对抗合成输入的 TensorRT FP16 对 ORT FP32：

- cell 一致率：`99.766511%`
- frame 一致率：`99.199466%`
- 约 14 个 cell、12 个 frame 不一致
- 输出 shape：`[1, 1499, 4]`

30 秒谐波全零判决可达到 100%，但证据强度较弱。上线质量决策仍需真实代表集。

## 十、其他加速路线的结论

- ORT CUDA Graph 约有 `2.09%` 收益，但仍明显慢于 TensorRT。
- PyTorch compile + SDPA + AMP FP16 是更好的 PyTorch fallback，但仍慢于 TensorRT。
- TensorRT 单 context CUDA Graph 平均约 `2.6%` 收益；多 worker 场景仅约 `0.21%`，不足以支撑复杂 hybrid。
- builder O4/O5、16 GB workspace、额外 auxiliary stream 未带来稳定胜出；当前 O3/8 GB 足够。
- dynamic batching 对本模型、batch=1、低延迟目标不利；不能因为通用最佳实践就启用。

## 十一、回归触发条件

以下任一变化至少重跑正确性、关键 shape 和混合负载：

- ONNX 哈希或输入/输出契约变化。
- GPU 型号、驱动、CUDA、TensorRT 版本变化。
- profile、builder flag、workspace 或 tactic cache 变化。
- 时长直方图、并发模型、SLO 或调度器变化。
- worker/context/stream/buffer 实现变化。
- Triton/HTTP/gRPC 等服务层引入或版本变化。

# TensorRT 故障、反常结果与 Git 交付

## 一、TensorRT “用不了”的诊断顺序

### 1. 确认使用的是目标 Conda 环境

目标环境：`/root/miniforge3/envs/diarizen-trt1010`

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda env list
conda activate diarizen-trt1010
which python
python -c "import tensorrt as trt; print(trt.__version__)"
python inference/validate_tensorrt_l4_conda.py
```

常见误判：

- 非交互 shell 没加载 Conda hook，因此 `conda activate` 不工作。
- 当前连接的是另一台机器或另一用户，`conda env list` 自然不同。
- 环境按绝对路径存在但没有注册到当前 Conda 的 env 列表。
- `pip` 对应 base/system Python，而运行时使用另一个解释器。

检查 `which python`、`sys.executable` 和 `trt.__file__`，不要只看包名。

### 2. 区分 build、deserialize、execute 三阶段

- build 崩溃：查看 builder 日志、workspace、精度 flag、动态 shape 和 plugin。
- deserialize 失败：通常是 TensorRT 版本、GPU 架构、plugin 或 plan 损坏不兼容。
- execute 失败：检查 profile 范围、binding shape、buffer 大小、stream/context 并发所有权。

历史上 TensorRT `10.0.1` 构建 FP16 曾段错误；通过独立 Conda 环境升级到 `10.10.0.31` 解决。不要继续在旧环境里叠加随机 pip 包。

### 3. LayerNorm FP16 overflow 警告

TensorRT 会把易溢出的 LayerNorm 计算回退到更高精度。这是精度保护，不代表整个 FP16 engine 失败。不要强制所有层使用 FP16 来消除警告；先验证输出一致性和真实性能。

### 4. Dynamic profile 与已有 plan 不一致

`--reuse-existing` 只表示复用文件，不保证它是本轮想要的 `min/opt/max`。读取 engine inspector/profile/binding shape，必要时比较 plan SHA256。范围不符就重建，不要拿错误 plan 生成“新 profile”报告。

### 5. 并发随机错误或输出串扰

重点检查：

- 多 worker 是否共享 execution context。
- buffer 是否被下一个请求提前覆盖。
- 动态 shape 是否在前一 stream 未同步时改变。
- output shape/size 是否按当前输入更新。
- plan/context 是否每请求重复创建导致抖动。

安全基线是每 worker/route 独立 context、stream、device buffer，并预分配。

## 二、性能结果反常时

### 低精度反而更慢

检查图分割、reformat/quantize/dequantize、算子覆盖、校准和 tactic。FP8/INT8/INT4 不是自动胜出，本项目已实测 FP16 最优。

### Worker 越多吞吐越低

这是正常的饱和现象。更多 context 会增加资源竞争和显存。达到 3–4 worker 平台后应看尾延迟；本项目 5–8 worker 总体退化。

### CUDA Graph 收益很小

固定单 context 有约 `2.6%` 收益，多 worker 仅约 `0.21%`。当 kernel 执行而非 launch overhead 占主导时，不值得引入复杂 hybrid。

### Builder O4/O5 或更大 workspace 没更快

优化等级和 workspace 只扩大搜索空间，不保证找到稳定更优 tactic。重复测、看方差；当前 O3/8 GB 已足够。

### 30 秒在宽 profile 退化

过宽 profile 且 opt 偏短会为长 shape 选择不理想 tactic。把长音频单独放入 `16/24/30` profile；已测比 `2/6/30` 快约 `28.3%`。

### Triton 后延迟变高

检查 HTTP/gRPC、序列化、scheduler queue、instance group、输入拷贝和 dynamic batching。Triton 优势主要在运维，不是 kernel 自动加速；必须用端到端 A/B 说话。

## 三、结论防误用清单

- 不把 GPU-only 延迟写成 API 端到端延迟。
- 不把合成音频一致率写成真实 DER/JER。
- 不把 99.9% 不超过 30 秒写成“最大 30 秒”。
- 不把同一模型的多 engine/profile 误写成多个模型。
- 不把离线 LPT burst 结果直接当在线逐请求 greedy 实测。
- 不因平均时长/中位数相同就假定完整分布相同。
- 不只看平均 RPS；同时看 Burst P95、请求 P95 和显存。
- 不因“batch 通常提升吞吐”就开启 dynamic batching。
- 不提交 plan 并假定可跨 TensorRT/GPU 复用。
- 不在没记录模型哈希时比较跨轮结果。

## 四、文档与证据分层

推荐写法：

1. **实测事实**：直接来自 JSON/命令输出，附环境、参数和样本数。
2. **计算结果**：例如 `199.952 × 0.8 ≈ 159 RPS`，给公式。
3. **工程建议**：例如 4 worker、WU 调度、80% 容量折扣，说明依据。
4. **未验证项**：真实音频质量、端到端网络开销、持续流量等。

总报告保留技术细节；周报提炼结论、表格、收益、风险和下一步。更新结果时同步两者，避免周报仍引用旧数字。

## 五、提交前检查

```bash
cd /root/code/github_repos/DiariZen
git status --short --branch
git diff --check
git diff --stat
git diff -- .agents/skills/diarizen-l4-inference-optimization
```

并完成：

- Skill 结构验证通过。
- 所有相对链接存在。
- 报告引用的脚本/JSON 路径存在。
- 没有误加入 `.plan`、缓存、凭据或大文件。
- 没有覆盖用户原有未提交修改。
- 提交信息为中文、具体说明新增内容和价值。

只有用户要求或上下文明确授权时才提交/推送。

## 六、GitHub SSH 端口故障

该服务器曾因 GitHub 自定义 SSH 端口 `12222` 超时而推送失败；直连 `github.com:22` 已验证可用并以 `nanless` 身份认证。若再次出现同类超时：

1. 先只读检查 remote、当前分支和待推送提交。
2. 验证 22 端口 SSH 身份。
3. 使用一次性 Git 配置推送：

```bash
git -c core.sshCommand="ssh -p 22" push origin feature/nemo-ssl-nest-finetune
```

除非用户明确要求修改 SSH 配置，不要为了单次推送永久改 `~/.ssh/config` 或仓库 remote。

推送后验证：

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse origin/feature/nemo-ssl-nest-finetune
```

HEAD 与远端分支提交一致且工作区干净，才可报告“已推送完成”。

## 七、完成定义

一次 L4 推理优化任务只有同时满足以下条件才算完成：

- 环境可重建、可验证。
- engine/profile 与模型身份可追溯。
- 性能结果带测试契约和正确性门槛。
- 线上建议包含 overflow、准入、容量折扣和监控。
- 原始 JSON、脚本、中文文档在仓库内。
- 若用户要求 Git 交付，提交和远端同步已核验。

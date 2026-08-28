---
name: diarizen-l4-inference-optimization
description: "诊断、复现、压测并上线 DiariZen segmentation 模型在 NVIDIA L4 上的 TensorRT 推理，覆盖 Conda/TensorRT 修复、ONNX 与 engine 身份校验、精度/profile/worker 扫描、混合音频时长与 50 并发容量建模、单模型单次推理约束、Triton 取舍、中文报告及 Git 交付。用于 /root/code/github_repos/DiariZen 的 L4 推理工作；不用于训练、端到端说话人分离质量评估或没有实测证据的通用性能承诺。"
---

# DiariZen L4 推理优化

## 目标与边界

把一次性“跑得快”变成可复现、可审计、可上线的结论。默认目标是 DiariZen segmentation ONNX 在 `dev_L4_1gpus` 的 NVIDIA L4 上做 `batch=1`、整段音频、单次 forward 推理，并兼顾吞吐、尾延迟、显存和数值一致性。

开始前先明确：

- segmentation 只是完整 diarization 链路的一段；不要把 GPU kernel 时间写成端到端接口延迟。
- 合成音频只能证明运行正确性、性能和有限的数值一致性；不能代替真实音频的 DER/JER。
- “一个模型、单次推理”允许同一 ONNX 权重构建多个 shape profile/plan；不允许切片、多次 forward、截断长音频。
- 未明确授权时只做读取和诊断；构建 engine、安装依赖、改仓库、提交或推送前遵守用户给出的变更范围。

## 按任务读取参考手册

- 第一次进入项目、确认服务器/分支/模型/环境/文件位置：读 [项目地图](references/project-map.md)。
- 设计或复核基准、精度、profile、worker、混合负载：读 [实验方法](references/benchmark-methodology.md)。
- 回答线上方案、容量、延迟、成本、长音频和 Triton：读 [线上决策手册](references/production-decision-guide.md)。
- TensorRT 失败、结果反常、plan 不兼容、Git/SSH 交付：读 [故障与交付手册](references/troubleshooting-and-delivery.md)。

不要一次性加载所有 JSON。先读两份总报告，再根据问题打开对应结果文件。

## 标准工作流

### 1. 固定实验身份

先记录主机、GPU、驱动、CUDA、Conda 环境、TensorRT/PyTorch/ONNX Runtime 版本、分支、提交、ONNX 哈希和本轮约束。任何一项变化都视为新实验环境，不直接沿用旧 plan 或旧结论。

最低检查项：

```bash
cd /root/code/github_repos/DiariZen
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor origin/main HEAD
nvidia-smi
source /root/miniforge3/etc/profile.d/conda.sh
conda env list
conda activate diarizen-trt1010
python inference/validate_tensorrt_l4_conda.py
sha256sum inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx
```

如果版本、哈希或分支不一致，先更新实验说明，不能把历史数字包装成本轮实测。

### 2. 写清测试契约

在运行前声明以下条件：

- 音频是真实还是合成；长度分布、采样率、波形构造和随机种子。
- 单次 forward、是否 padding、禁止切片/截断。
- 并发语义：同时到达、持续流量还是离线整批。
- 延迟边界：只计 GPU、包含 H2D/D2H、还是接口端到端。
- 预热轮数、实测轮数、worker 数、调度算法、是否动态 batching。
- 正确性门槛：输出 shape、有限值、cell/frame 一致率、必要时真实集 DER/JER。

契约不清时，结果只能标记为探索性，不能直接给线上容量承诺。

### 3. 独占 GPU 并做预检

所有正式 GPU 实验使用 `/tmp/diarizen_l4_benchmark.lock`；确认没有未授权计算进程。锁只序列化遵守同一约定的任务，仍要检查 `nvidia-smi`。避免并行启动多个 benchmark 争抢同一张 L4。

### 4. 先修运行环境，再谈优化

固定使用 `/root/miniforge3/envs/diarizen-trt1010`。不要因为当前 shell 的 `conda env list` 没显示目标环境，就把 pip 装进 base 或系统 Python。若环境损坏，用仓库脚本重建并运行验证脚本；TensorRT 10.0.1 的 FP16 builder 段错误属于已知历史问题，当前基线是 TensorRT `10.10.0.31`。

### 5. 由粗到细缩小搜索空间

推荐顺序：

1. ORT CUDA FP32 建立数值与性能基线。
2. TensorRT 扫 FP32/TF32/FP16/BF16/FP8/INT8/INT4，先筛掉构建失败、明显变慢或数值不合格者。
3. 对胜出精度扫描固定 shape 与动态 profile；`min/opt/max` 必须反映真实长度分布。
4. 对候选路线扫描 worker 数，联合观察吞吐、请求 P95、burst P95 和显存。
5. 最后用真实或代表性长度直方图做混合并发测试，而不是只测 10 秒/16 秒等长输入。

不要因为低位宽“理论上更快”就跳过实测；本项目的最终选择是 FP16。

### 6. 正确性门槛先于性能排名

至少校验：

- engine 输入范围覆盖请求长度，输出为 `[B, frames, 4]`。
- 输出无 NaN/Inf，shape 与 ORT 基线一致。
- 对抗合成波形上的 cell/frame 一致率；全零输入只能做 smoke test。
- 量化或算子改写若要上线，补真实代表集的 DER/JER 或业务指标。

TensorRT 对 LayerNorm 自动回退到更高精度是安全行为；不要为了“纯 FP16”强制取消以换取不可靠数字。

### 7. 用联合指标选线上点

不要只按平均 RPS 排名。对每个 worker/profile 共同比较：

- 总吞吐 RPS。
- Burst Mean、Burst P95。
- 请求 P50、请求 P95。
- GPU 显存峰值。
- 正确性结果与失败率。

吞吐进入平台区后，优先尾延迟更稳、显存留量更大的点。当前 3 worker 的平均吞吐仅比 4 worker 高约 `0.084%`，但 4 worker 尾延迟更好，因此选 4 worker。

### 8. 形成可执行上线方案

结果必须包括：路由区间、每条 profile、padding 规则、worker/context/stream/buffer 所有权、调度器、准入阈值、超长回退、容量折扣、成本公式、监控指标和回滚条件。当前已验证方案见线上决策手册，但在模型、GPU、TensorRT 或时长分布变化后必须重测。

### 9. 固化证据并交付

保留：

- 可复现脚本和参数。
- JSON 原始结果。
- ONNX SHA256、plan SHA256、profile 元数据和构建命令。
- 中文总报告/周报，明确“实测”“推导”“建议”“未验证”。
- 分支状态、提交号和推送结果。

不要提交大体积 `.plan`；plan 与 TensorRT 版本、GPU 和 profile 强绑定。提交前检查用户已有修改，不覆盖不属于本任务的内容。

## 输出要求

向用户汇报时先给结论，再给证据和边界。至少回答：

1. 本轮在哪台机器、哪个环境、哪个模型哈希和提交上测得。
2. 测了哪些精度/profile/worker，哪些没有测。
3. 最优方案为什么胜出，吞吐与尾延迟的取舍是什么。
4. 对真实长度分布、超过 16 秒和超过 30 秒请求如何处理。
5. 数字是 GPU-only 还是端到端，合成还是真实音频。
6. 产物位置、测试状态、提交与推送状态。

遇到新事实时更新参考手册，不把过期基线继续写成“当前最优”。

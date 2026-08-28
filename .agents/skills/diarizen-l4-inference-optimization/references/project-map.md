# 项目地图与基线身份

## 适用仓库

- 服务器别名：`dev_L4_1gpus`
- 仓库：`/root/code/github_repos/DiariZen`
- 当前实验分支：`feature/nemo-ssl-nest-finetune`
- 已确认关系：该分支以 `main` 为祖先继续开发；每次交付仍应用 `git merge-base --is-ancestor origin/main HEAD` 复核。
- 本 Skill 沉淀时的基线提交：`441261e0da79e1a1bb0f61b4f303c7369e4e9b98`
- 核心实验提交：`fadf637e19745765cf0a3a569f1a93d004bcd565`

提交号只是追溯锚点，不是永远固定的运行前提。新提交可能改变脚本或结论。

## 硬件与软件基线

| 项目 | 已验证值 |
|---|---:|
| GPU | NVIDIA L4 24 GB |
| 驱动 | 535.129.03 |
| 驱动报告 CUDA | 12.4 |
| Conda 环境 | `/root/miniforge3/envs/diarizen-trt1010` |
| TensorRT | 10.10.0.31 |
| PyTorch | 2.3.1+cu121 |

不要混淆“驱动支持的 CUDA 版本”和 PyTorch wheel 使用的 CUDA runtime。任何版本变化都应重新验证 engine 构建、反序列化和性能。

## 模型身份与范围

主 ONNX：

`inference/models/kaldi_merged_1219_all_ft_large/epoch_0016_multilabel_hard.onnx`

已记录摘要：

- SHA256：`114ac3c37deb04b47deeeb50c2f5aff981cd4702d23b9df9b0bfe3ba07f85969`
- MD5：`0a2142c0874e553206633e65b8348dd1`
- 输出语义：hard multilabel segmentation
- 输出形状：`[B, frames, 4]`

本轮性能结论只覆盖 segmentation 模型，不覆盖聚类、embedding、VAD、音频解码、网络传输、排队和业务后处理。

## 入口文档

先读：

1. `inference/models/kaldi_merged_1219_all_ft_large/SEGMENTATION_BENCHMARK_REPORT.md`
2. `inference/models/kaldi_merged_1219_all_ft_large/L4_ONLINE_INFERENCE_WEEKLY_SUMMARY.md`
3. `inference/ADR_L4_ONLINE_SERVING_16S_CONCURRENCY50.md`

第一份是技术证据总表，第二份是中文周报式总结，ADR 是 16 秒/50 并发阶段的架构决策。若三者数字不一致，以更新日期、模型哈希、脚本参数和 JSON 证据共同判断，不按文件名猜测。

## 关键脚本

| 脚本 | 用途 |
|---|---|
| `inference/setup_tensorrt_l4_conda.sh` | 创建/修复独立 TensorRT Conda 环境 |
| `inference/validate_tensorrt_l4_conda.py` | 验证环境、CUDA、TensorRT 与 engine 可用性 |
| `inference/diagnose_tensorrt_build.py` | 定位 TensorRT builder 问题 |
| `inference/benchmark_segmentation_precision.py` | 精度路线比较 |
| `inference/benchmark_tensorrt_fixed10s.py` | 固定 10 秒 TensorRT 基准 |
| `inference/benchmark_trt_fixed10s_parity.py` | 固定 10 秒数值一致性 |
| `inference/benchmark_trt_dynamic_duration.py` | 动态时长/profile 基准 |
| `inference/benchmark_trt_builder_search.py` | builder 配置搜索 |
| `inference/benchmark_trt_online_burst.py` | 固定时长 burst/worker 容量 |
| `inference/benchmark_trt_mixed_duration_burst.py` | 混合时长、50 并发、worker 扫描 |
| `inference/benchmark_ort_cuda_graph_fixed16s.py` | ORT CUDA Graph 对照 |
| `inference/benchmark_pytorch_compile_fixed16s.py` | PyTorch compile/SDPA/AMP 对照 |
| `inference/tensorrt_cuda_graph_runner.py` | TensorRT CUDA Graph 运行封装 |

先运行脚本的 `--help` 并读参数默认值。不要从历史命令盲抄，因为脚本可能随提交更新。

## 结果文件定位

结果集中在：

`inference/models/kaldi_merged_1219_all_ft_large/`

常用命名：

- `epoch_0016_l4_trt1010_all_precisions_fixed10s.json`
- `epoch_0016_l4_trt1010_fp16_dynamic_*_bs1.json`
- `epoch_0016_l4_trt1010_fp16_fixed16s*.json`
- `epoch_0016_l4_trt1010_fp16_mixed_duration_50concurrency_workers*_bs1.json`
- `epoch_0016_l4_ort_*_reference.json`
- `epoch_0016_l4_trt_builder_search_fixed16s.json`

读取 JSON 时先确认内嵌的模型路径/哈希、profile、worker、样本数量和计时口径，不要只从文件名推断。

## Engine 产物规则

`.gitignore` 已忽略以下 engine 目录：

- `trt_l4_trt1010/`
- `trt_l4_precisions/`
- `trt_l4_trt1010_16s/`
- `trt_l4_trt1010_dynamic/`
- `trt_l4_trt1010_16s_builder_search/`

原因：plan 体积大且强绑定 TensorRT/GPU/profile。仓库应保存生成脚本、构建参数、plan 哈希和测试 JSON，而不是把二进制 plan 当作可跨机器发布的通用模型。

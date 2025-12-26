# Simple DiariZen 推理脚本

简洁的说话人分离推理脚本，直接使用训练好的 DiariZen 模型进行推理。

## 特点

- **简单直接**：不依赖复杂的 pipeline，只做纯粹的 segmentation 模型推理
- **与训练一致**：使用与训练时完全相同的推理逻辑
- **完整输出**：生成 RTTM 标注文件和可视化图表
- **支持 Powerset 和 Multilabel**：自动检测模型类型

## 使用方法

### 方式 1：使用包装脚本（推荐）

```bash
./run_simple_diarize.sh \
  /path/to/audios \
  --out-dir /path/to/output \
  --ckpt-dir /path/to/checkpoints/epoch_0004 \
  --config /path/to/config.toml \
  --device cuda
```

### 方式 2：直接使用 conda 环境

```bash
conda run -n diarizen python simple_diarize.py \
  /path/to/audios \
  --out-dir /path/to/output \
  --ckpt-dir /path/to/checkpoints/epoch_0004 \
  --config /path/to/config.toml \
  --device cuda
```

## 参数说明

### 必需参数

- `in_root`: 输入音频根目录（会递归扫描所有音频文件）
- `--out-dir`: 输出目录（保存 RTTM 和可视化图表）
- `--ckpt-dir`: checkpoint 目录（包含 pytorch_model.bin）
- `--config`: 训练配置文件路径（config.toml）

### 可选参数

- `--device`: 设备选择，默认 `cuda`
  - `cuda`: 使用 GPU
  - `cpu`: 使用 CPU
- `--sample-rate`: 采样率，默认 `16000`
- `--chunk-size`: 分块大小（秒），`None` 表示整句推理（默认）
- `--min-duration`: 最小段持续时间（秒），默认 `0.0`
- `--plot` / `--no-plot`: 是否生成可视化，默认开启

## 示例

### 示例 1：使用训练好的模型推理

```bash
./run_simple_diarize.sh \
  /root/code/own/download_gp_online_audios_for_speakerdetection_1113/original_audios \
  --out-dir ./output \
  --ckpt-dir /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_temp_ft_large/checkpoints/epoch_0004 \
  --config /root/code/github_repos/DiariZen/recipes/diar_ssl/exp/kaldi_merged_1219_all_temp_ft_large/config__2025_12_24--02_01_24.toml \
  --device cuda
```

### 示例 2：使用 CPU 推理（GPU 内存不足时）

```bash
./run_simple_diarize.sh \
  /path/to/audios \
  --out-dir ./output \
  --ckpt-dir /path/to/checkpoints/epoch_0004 \
  --config /path/to/config.toml \
  --device cpu
```

### 示例 3：设置最小段持续时间过滤短段

```bash
./run_simple_diarize.sh \
  /path/to/audios \
  --out-dir ./output \
  --ckpt-dir /path/to/checkpoints/epoch_0004 \
  --config /path/to/config.toml \
  --min-duration 0.3
```

## 输出

脚本会为每个音频文件生成：

1. **RTTM 文件**：`{session_name}.rttm`
   - 标准 RTTM 格式
   - 包含说话人 ID、开始时间、持续时间
   
2. **可视化图表**：`{session_name}.png`
   - 上方：音频波形
   - 下方：说话人分离结果（时间轴+说话人段）

3. **汇总文件**：`summary.json`
   - 包含所有处理文件的元信息
   - 音频路径、时长、说话人数、段数等

## RTTM 格式示例

```
SPEAKER session_name 1 1.240 0.360 <NA> <NA> speaker_00 <NA> <NA>
SPEAKER session_name 1 1.700 0.400 <NA> <NA> speaker_00 <NA> <NA>
SPEAKER session_name 1 2.220 0.280 <NA> <NA> speaker_01 <NA> <NA>
```

格式说明：
- `SPEAKER`: 标记类型
- `session_name`: 会话名称
- `1`: 通道号
- `1.240`: 开始时间（秒）
- `0.360`: 持续时间（秒）
- `speaker_00`: 说话人 ID

## 支持的音频格式

- WAV
- MP3
- M4A
- FLAC
- OGG
- AAC
- WMA

## 注意事项

1. **环境要求**：必须使用 `diarizen` conda 环境，因为该环境配置了本地修改的 pyannote-audio
2. **GPU 内存**：如果 GPU 内存不足，使用 `--device cpu`
3. **推理模式**：默认使用整句推理（不分块），速度快且效果好
4. **模型类型**：自动检测 powerset 或 multilabel 模式

## 与 batch_diarize_finetuned.py 的区别

| 特性 | simple_diarize.py | batch_diarize_finetuned.py |
|------|-------------------|----------------------------|
| 复杂度 | ✅ 简单直接 | ❌ 复杂（含 pipeline） |
| 依赖 | ✅ 最少 | ❌ 需要完整 pyannote pipeline |
| 速度 | ✅ 快（纯 segmentation） | ❌ 慢（含 embedding + clustering） |
| 训练一致性 | ✅ 与训练完全一致 | ❌ 增加了额外处理 |
| 输出 | RTTM + 可视化 | RTTM + 可视化 |
| 使用场景 | 快速推理、调试 | 需要 embedding 聚类时 |

## 故障排除

### 问题：找不到 pyannote.audio.core.model

**解决**：必须使用 `diarizen` conda 环境
```bash
conda activate diarizen
# 或使用包装脚本
./run_simple_diarize.sh ...
```

### 问题：CUDA out of memory

**解决**：使用 CPU 推理
```bash
./run_simple_diarize.sh ... --device cpu
```

### 问题：音频文件未找到

**解决**：检查音频路径，确保文件大小 > 1KB

## 技术细节

### 推理流程

1. 加载模型和配置
2. 扫描音频文件
3. 对每个音频：
   - 加载音频（单声道或多声道）
   - 模型前向推理
   - Powerset → Multilabel 转换（如需要）
   - 二值化（阈值 0.5）
   - 帧级标签 → 时间段
   - 生成 RTTM
   - 生成可视化

### 帧到段转换

- 使用模型的感受野参数（`model_rf_step`）计算时间
- 连续的活跃帧合并为一个段
- 支持最小段持续时间过滤

### Powerset vs Multilabel

- **Powerset 模式**：模型输出 → log_softmax → powerset.to_multilabel() → 二值化
- **Multilabel 模式**：模型输出 → sigmoid → 二值化

脚本自动检测模型类型并使用正确的转换方式。


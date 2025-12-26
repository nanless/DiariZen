# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

"""
说话人分离数据集模块

该模块提供了用于说话人分离任务的数据集类和相关工具函数。
主要功能：
1. 加载音频文件和标注（RTTM格式）
2. 将长音频切分为固定大小的chunk
3. 将标注转换为帧级别的多标签表示
4. 批处理时进行填充和对齐

数据集格式：
- SCP文件：记录ID到音频文件路径的映射
- RTTM文件：说话人标注（Rich Transcription Time Marked）
- UEM文件：有效音频段的时间范围
"""

import os
 
import torch
import numpy as np

import soundfile as sf  # 用于读取音频文件
from typing import Dict

from torch.utils.data import Dataset  # PyTorch数据集基类

def get_dtype(value: int) -> str:
    """根据值的大小返回最适合的numpy数据类型

    为了节省内存，根据值的大小选择最小的合适数据类型。
    这样可以减少内存占用，特别是在处理大量数据时。

    参数
    ----------
    value: int
        需要存储的整数值

    返回
    -------
    str
        numpy格式的数据类型字符串
        - "b": int8 (8位有符号整数，范围 -128 到 127)
        - "i2": int16 (16位有符号整数，范围 -32768 到 32767)
        - "i": int32 (32位有符号整数，范围 -2^31 到 2^31-1)
        - "i8": int64 (64位有符号整数)
        
    参考
    -----
    https://numpy.org/doc/stable/reference/arrays.dtypes.html
    """
    # 定义数据类型列表：(最大值, 类型字符串)
    # signed byte (8 bits), signed short (16 bits), signed int (32 bits)
    types_list = [(127, "b"), (32_768, "i2"), (2_147_483_648, "i")]
    # 筛选出能够容纳该值的数据类型
    filtered_list = [
        (max_val, type) for max_val, type in types_list if max_val > abs(value)
    ]
    # 如果没有找到合适的类型，使用int64
    if not filtered_list:
        return "i8"  # signed long (64 bits)
    # 返回第一个（最小的）合适类型
    return filtered_list[0][1]

def load_scp(scp_file: str) -> Dict[str, str]:
    """加载SCP文件，返回记录ID到音频文件路径的映射

    SCP（Script）文件格式：每行包含记录ID和对应的音频文件路径
    例如：
        rec001 /path/to/audio1.wav
        rec002 /path/to/audio2.wav

    参数
    ----------
    scp_file : str
        SCP文件路径

    返回
    -------
    Dict[str, str]
        字典，键为记录ID，值为音频文件路径
        {记录ID: 音频文件路径}
    """
    # 读取文件，每行按空格分割，第一个字段是记录ID，剩余部分是文件路径
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    # 构建字典：记录ID -> 文件路径
    return {x[0]: x[1] for x in lines}

def load_uem(uem_file: str) -> Dict[str, float]:
    """加载UEM文件，返回记录ID到有效时间段的映射

    UEM（Un-partitioned Evaluation Map）文件格式：
    每行包含记录ID、通道、开始时间和结束时间
    例如：
        rec001 1 0.0 10.5
        表示rec001的有效时间段为[0.0, 10.5]秒

    参数
    ----------
    uem_file : str
        UEM文件路径

    返回
    -------
    Dict[str, List[float, float]] 或 None
        字典，键为记录ID，值为[start_time, end_time]列表
        {记录ID: [开始时间(秒), 结束时间(秒)]}
        如果文件不存在，返回None
    """
    if not os.path.exists(uem_file):
        return None
    # 读取文件，每行按空格分割
    lines = [line.strip().split() for line in open(uem_file)]
    # 构建字典：记录ID -> [开始时间, 结束时间]
    # x[-2]是倒数第二个字段（开始时间），x[-1]是最后一个字段（结束时间）
    return {x[0]: [float(x[-2]), float(x[-1])] for x in lines}
    
def _gen_chunk_indices(
    init_posi: int,
    data_len: int,
    size: int,
    step: int,
) -> None:
    """生成chunk的起始和结束索引

    将一段数据切分为多个固定大小的chunk，支持重叠（step < size）。
    这是一个生成器函数，用于高效地生成chunk索引。

    参数
    ----------
    init_posi : int
        初始位置（起始位置，通常为0）
    data_len : int
        数据总长度
    size : int
        chunk大小（长度）
    step : int
        步长（相邻chunk之间的间隔）
        如果step < size，则chunk之间会有重叠
        如果step == size，则chunk之间无重叠

    生成
    -----
    tuple (start, end)
        每个chunk的起始和结束索引（左闭右开区间 [start, end)）

    示例
    -----
    init_posi=0, data_len=100, size=20, step=10:
    生成: (1, 21), (11, 31), (21, 41), ..., (81, 101)
    """
    # 调整索引：init_posi+1作为实际起始位置
    init_posi = int(init_posi + 1)
    # 调整数据长度：data_len-1作为实际结束位置
    data_len = int(data_len - 1)
    # 计算当前可用长度
    cur_len = data_len - init_posi
    # 确保可用长度大于chunk大小
    assert cur_len > size
    # 计算可以生成的chunk数量
    # 公式：(可用长度 - chunk大小 + 步长) / 步长
    num_chunks = int((cur_len - size + step) / step)
    
    # 生成每个chunk的起始和结束索引
    for i in range(num_chunks):
        start = init_posi + (i * step)
        end = start + size
        yield start, end

def _collate_fn(batch, max_speakers_per_chunk=4) -> torch.Tensor:
    """批处理函数：对齐变长波形和帧标签

    将不同长度的样本组合成一个批次，进行填充对齐。
    同时处理说话人数量不一致的情况，并生成掩码用于损失计算。

    参数
    ----------
    batch : list
        批次数据，每个元素为 (waveform, frame_labels, session_name)
        - waveform: numpy数组，形状为 (channels, samples)
        - frame_labels: numpy数组，形状为 (frames, num_speakers)
        - session_name: 字符串，会话名称
    max_speakers_per_chunk : int, 默认4
        每个chunk中最大说话人数量

    返回
    -------
    dict
        包含以下键的字典：
        - 'xs': torch.Tensor, 填充后的波形，形状为 (batch, channels, max_samples)
        - 'ts': torch.Tensor, 填充后的帧标签，形状为 (batch, max_frames, max_speakers)
        - 'mask': torch.Tensor, 有效帧掩码，形状为 (batch, max_frames)
        - 'names': list, 会话名称列表
    """
    xs, ys, names = [], [], []  # 存储波形、标签和名称
    wav_lengths = []  # 存储原始波形长度
    frame_lengths = []  # 存储原始帧长度

    # 第一步：处理说话人数量不一致的问题
    for x, y, name in batch:
        num_speakers = y.shape[-1]  # 当前chunk的说话人数量
        
        if num_speakers > max_speakers_per_chunk:
            # 如果说话人数量超过限制，保留最活跃的说话人
            # 按说话活跃度（总说话时间）降序排序
            indices = np.argsort(-np.sum(y, axis=0), axis=0)
            # 只保留前max_speakers_per_chunk个最活跃的说话人
            y = y[:, indices[: max_speakers_per_chunk]]
        elif num_speakers < max_speakers_per_chunk:
            # 如果说话人数量不足，用零填充创建不活跃的说话人
            y = np.pad(
                y,
                ((0, 0), (0, max_speakers_per_chunk - num_speakers)),
                mode="constant",  # 用0填充
            )
        
        # 记录填充前的长度（用于后续生成掩码）
        wav_lengths.append(x.shape[-1])
        frame_lengths.append(y.shape[0])

        xs.append(x)
        ys.append(y)
        names.append(name)

    # 找到批次中的最大长度
    max_wav_len = max(wav_lengths)  # 最大波形长度
    max_frame_len = max(frame_lengths)  # 最大帧长度

    padded_xs = []  # 填充后的波形
    padded_ys = []  # 填充后的标签
    masks = []  # 有效帧掩码

    # 第二步：对时间维度进行填充
    for x, y, flen in zip(xs, ys, frame_lengths):
        # 填充波形到最大长度
        # x形状: (channels, samples)
        if x.shape[-1] < max_wav_len:
            pad_width = ((0, 0), (0, max_wav_len - x.shape[-1]))
            x = np.pad(x, pad_width, mode="constant")
        
        # 填充帧标签到最大长度
        # y形状: (frames, speakers)
        if y.shape[0] < max_frame_len:
            y = np.pad(
                y,
                ((0, max_frame_len - y.shape[0]), (0, 0)),
                mode="constant",
            )
        
        padded_xs.append(x)
        padded_ys.append(y)

        # 生成有效帧掩码：1表示有效帧，0表示填充帧
        mask = np.zeros((max_frame_len,), dtype=np.float32)
        mask[:flen] = 1.0  # 前flen帧是有效的
        masks.append(mask)

    # 转换为torch.Tensor并返回
    return {
        'xs': torch.from_numpy(np.stack(padded_xs)).float(),  # 堆叠并转换为float32
        'ts': torch.from_numpy(np.stack(padded_ys)),  # 堆叠帧标签
        'mask': torch.from_numpy(np.stack(masks)),  # 堆叠掩码
        'names': names  # 会话名称列表
    }        
        
        
class DiarizationDataset(Dataset):
    """说话人分离数据集类

    该类实现了PyTorch的Dataset接口，用于加载和处理说话人分离任务的数据。
    主要功能：
    1. 从SCP、RTTM、UEM文件加载数据
    2. 将长音频切分为固定大小的chunk
    3. 将RTTM标注转换为帧级别的多标签表示
    4. 根据模型的感受野参数进行时间对齐

    参数
    ----------
    scp_file : str
        SCP文件路径，包含记录ID到音频文件路径的映射
    rttm_file : str
        RTTM文件路径，包含说话人标注信息
    uem_file : str
        UEM文件路径，包含有效音频段的时间范围
    model_num_frames : int
        模型输出的帧数（用于计算chunk对应的输出帧数）
    model_rf_duration : float
        模型感受野持续时间（秒）
    model_rf_step : float
        模型感受野步长（秒），即相邻输出帧之间的时间间隔
    chunk_size : int, 默认5
        chunk大小（秒）
    chunk_shift : int, 默认5
        chunk之间的步长（秒），如果小于chunk_size则会有重叠
    sample_rate : int, 默认16000
        音频采样率（Hz）
    full_utterance : bool, 默认False
        是否使用完整utterance作为单个chunk（不切分）
    max_sessions : int, 可选
        限制处理的会话数量，用于快速调试（None表示不限制）
    max_chunks : int, 可选
        限制生成的chunk数量，用于快速调试（None表示不限制）
    """
    def __init__(
        self, 
        scp_file: str, 
        rttm_file: str,
        uem_file: str,
        model_num_frames: int,    # default: wavlm_base
        model_rf_duration: float,  # model.receptive_field.duration, seconds
        model_rf_step: float,  # model.receptive_field.step, seconds
        chunk_size: int = 5,  # seconds
        chunk_shift: int = 5, # seconds
        sample_rate: int = 16000,
        full_utterance: bool = False,
        max_sessions: int = None,   # limit number of recordings for quick debug
        max_chunks: int = None,     # limit number of generated chunks
    ): 
        # 存储所有chunk的索引：(session, audio_path, start_sec, end_sec)
        self.chunk_indices = []
        
        # 保存配置参数
        self.sample_rate = sample_rate  # 采样率
        self.full_utterance = full_utterance  # 是否使用完整utterance
        
        # 保存模型感受野参数（用于时间对齐）
        self.model_rf_step = model_rf_step  # 感受野步长（秒）
        self.model_rf_duration = model_rf_duration  # 感受野持续时间（秒）
        self.model_num_frames = model_num_frames  # 模型输出帧数
        
        # 加载数据文件
        self.rec_scp = load_scp(scp_file)  # 记录ID -> 音频文件路径
        self.reco2dur = load_uem(uem_file)  # 记录ID -> [开始时间, 结束时间]
        
        # 创建会话到索引的映射，用于O(1)查找
        self.session_to_idx = {k: i for i, k in enumerate(self.rec_scp.keys())}

        # 可选：限制会话数量用于快速调试
        # 0或None表示不限制
        if (max_sessions is not None) and (max_sessions > 0):
            rec_items = list(self.reco2dur.items())[:max_sessions]
            self.reco2dur = {k: v for k, v in rec_items}
            self.rec_scp = {k: self.rec_scp[k] for k, _ in rec_items if k in self.rec_scp}
            self.session_to_idx = {k: i for i, k in enumerate(self.rec_scp.keys())}
        
        # 解析RTTM文件一次，按会话存储标注，避免重复扫描
        self.annotations_by_session = self.rttm2label(rttm_file)

        # 为每个记录生成chunk索引
        for rec, dur_info in self.reco2dur.items():
            # 跳过没有标注的会话
            if rec not in self.annotations_by_session:
                continue
            start_sec, end_sec = dur_info  # 有效时间段
            try:
                # 如果full_utterance为True，使用整个录音作为一个chunk
                if (not self.full_utterance) and chunk_size > 0:
                    # 生成重叠的chunk
                    for st, ed in _gen_chunk_indices(
                            start_sec,
                            end_sec,
                            chunk_size,
                            chunk_shift
                    ):
                        # 存储：(会话ID, 音频路径, 开始时间, 结束时间)
                        self.chunk_indices.append((rec, self.rec_scp[rec], st, ed))
                else:
                    # 使用整个有效时间段作为一个chunk
                    self.chunk_indices.append((rec, self.rec_scp[rec], start_sec, end_sec))
            except:
                print(f'Un-matched recording: {rec}')

            # 如果达到最大chunk数量限制，停止生成
            if (max_chunks is not None) and (max_chunks > 0) and len(self.chunk_indices) >= max_chunks:
                break

    def get_session_idx(self, session):
        """将会话ID转换为会话索引

        参数
        ----------
        session : str
            会话ID（记录ID）

        返回
        -------
        int
            会话索引（从0开始）
        """
        return self.session_to_idx[session]
            
    def rttm2label(self, rttm_file):
        """解析RTTM文件，转换为结构化标注

        RTTM（Rich Transcription Time Marked）文件格式：
        SPEAKER <recording_id> <channel> <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
        例如：
        SPEAKER train100_306 1 15.71 1.76 <NA> <NA> 5456 <NA> <NA>
        表示：录音train100_306，通道1，从15.71秒开始，持续1.76秒，说话人ID为5456

        参数
        ----------
        rttm_file : str
            RTTM文件路径

        返回
        -------
        Dict[str, np.ndarray]
            按会话组织的标注数组
            每个数组包含结构化记录：(session_idx, start, end, label_idx)
            - session_idx: 会话索引
            - start: 开始时间（秒）
            - end: 结束时间（秒）
            - label_idx: 说话人标签索引（在该会话内的索引）
        """
        annotations_by_session = {}  # 按会话存储标注
        label_lists = {}  # 按会话存储说话人ID列表（用于分配标签索引）

        with open(rttm_file, 'r') as file:
            for seg_idx, line in enumerate(file):   
                parts = line.split()  # 按空格分割
                # RTTM格式：SPEAKER <recording_id> <channel> <start_time> <duration> ...
                session, start, dur = parts[1], parts[3], parts[4]

                # 跳过不在选定子集中的会话
                if session not in self.session_to_idx:
                    continue

                # 计算开始和结束时间
                start = float(start)
                end = start + float(dur)
                
                # 提取说话人ID（可能是倒数第二个或倒数第三个字段）
                spk = parts[-2] if parts[-2] != "<NA>" else parts[-3]
                
                # 为每个会话维护说话人ID列表，并分配标签索引
                if session not in label_lists:
                    label_lists[session] = []
                if spk not in label_lists[session]:
                    label_lists[session].append(spk)
                label_idx = label_lists[session].index(spk)  # 获取说话人在该会话中的索引
                
                # 存储标注信息
                annotations_by_session.setdefault(session, []).append(
                    (
                        self.get_session_idx(session),  # 会话索引
                        start,  # 开始时间
                        end,  # 结束时间
                        label_idx  # 说话人标签索引
                    )
                )
                
        # 定义结构化数据类型，使用最节省内存的类型
        segment_dtype = [
            (
                "session_idx",
                get_dtype(len(self.session_to_idx)),  # 根据会话数量选择合适类型
            ),
            ("start", "f"),  # float32
            ("end", "f"),  # float32
            ("label_idx", get_dtype(max((len(v) for v in label_lists.values()), default=1))),  # 根据最大说话人数量选择类型
        ]
        
        # 将列表转换为numpy结构化数组，提高访问效率
        for session, segs in annotations_by_session.items():
            annotations_by_session[session] = np.array(segs, dtype=segment_dtype)
        
        return annotations_by_session

    def extract_wavforms(self, path, start, end, num_channels=8):
        """从音频文件中提取指定时间段的波形

        参数
        ----------
        path : str
            音频文件路径
        start : float
            开始时间（秒）
        end : float
            结束时间（秒）
        num_channels : int, 默认8
            返回的通道数（如果音频通道数更多，只返回前num_channels个）

        返回
        -------
        np.ndarray
            音频波形，形状为 (num_channels, samples)
        """
        # 将时间（秒）转换为样本索引
        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)
        
        # 使用soundfile读取指定范围的音频
        data, sample_rate = sf.read(path, start=start, stop=end)
        
        # 检查采样率是否匹配
        assert sample_rate == self.sample_rate
        
        # 处理单声道和多声道音频
        if data.ndim == 1:
            # 单声道：reshape为 (1, samples)
            data = data.reshape(1, -1)
        else:
            # 多声道：从 (time, channels) 转换为 (channels, time)
            data = np.einsum('tc->ct', data)
        
        # 只返回前num_channels个通道
        return data[:num_channels, :]

    def __len__(self):
        """返回数据集大小（chunk数量）

        返回
        -------
        int
            数据集中的chunk数量
        """
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        """获取指定索引的数据样本

        参数
        ----------
        idx : int
            样本索引

        返回
        -------
        tuple
            (data, mask_label, session)
            - data: np.ndarray, 音频波形，形状为 (channels, samples)
            - mask_label: np.ndarray, 帧级别的多标签掩码，形状为 (frames, num_speakers)
            - session: str, 会话ID
        """
        # 获取chunk信息
        session, path, chunk_start, chunk_end = self.chunk_indices[idx]
        
        # 提取音频波形 [chunk_start, chunk_end)
        data = self.extract_wavforms(path, chunk_start, chunk_end)
        
        # 获取该chunk对应的标注（与chunk有重叠的标注段）
        annotations_session = self.annotations_by_session.get(session, None)
        if annotations_session is None or len(annotations_session) == 0:
            # 如果没有标注，创建空数组
            chunked_annotations = np.zeros((0,), dtype=[("start","f"),("end","f"),("label_idx","i")])
            labels = []
            print(f'No annotations found for session: {session}')
        else:
            # 筛选与当前chunk有重叠的标注段
            # 条件：标注开始时间 < chunk结束时间 且 标注结束时间 > chunk开始时间
            chunked_annotations = annotations_session[
                (annotations_session["start"] < chunk_end) & (annotations_session["end"] > chunk_start)
            ]
            # 获取该chunk中出现的所有说话人标签
            labels = list(np.unique(chunked_annotations['label_idx']))
        
        # 将连续时间标注离散化为模型输出分辨率（帧级别）
        step = self.model_rf_step  # 感受野步长（秒）
        half = 0.5 * self.model_rf_duration  # 感受野的一半持续时间（用于对齐）
        
        # 计算标注段在chunk内的相对开始时间，并转换为帧索引
        # 考虑感受野中心对齐：start - chunk_start - half
        start = np.maximum(chunked_annotations["start"], chunk_start) - chunk_start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        # 计算标注段在chunk内的相对结束时间，并转换为帧索引
        end = np.minimum(chunked_annotations["end"], chunk_end) - chunk_start - half
        end_idx = np.round(end / step).astype(int)
        
        # 确定输出帧数和说话人数量
        num_labels = max(len(labels), 1)  # 至少要有1个说话人（即使没有标注）
        
        # 确定动态帧长度（考虑填充安全）
        max_end = int(np.max(end_idx)) if len(end_idx) > 0 else -1  # 最大结束帧索引
        chunk_len = chunk_end - chunk_start  # chunk长度（秒）
        est_frames = int(np.ceil(max(chunk_len / step, 1)))  # 估算的帧数
        # 最终帧数 = max(估算帧数, 最大结束索引+1, 1)
        num_frames = max(est_frames, max_end + 1, 1)
        
        # 创建帧级别的多标签掩码：形状为 (frames, num_speakers)
        mask_label = np.zeros((num_frames, num_labels), dtype=np.uint8)

        # 将说话人标签映射到索引（在该chunk中的索引）
        mapping = {label: idx for idx, label in enumerate(labels)}
        
        # 填充掩码：对于每个标注段，将对应帧的对应说话人位置设为1
        for start, end, label in zip(
            start_idx, end_idx, chunked_annotations['label_idx']
        ):
            mapped_label = mapping[label]  # 获取说话人在该chunk中的索引
            end_clipped = min(end, num_frames - 1)  # 确保不超出范围
            # 将[start, end_clipped+1]范围内的帧标记为该说话人活跃
            mask_label[start : end_clipped + 1, mapped_label] = 1
        
        return data, mask_label, session
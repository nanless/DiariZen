#!/usr/bin/env python3
"""
将 Kaldi 风格的 wav.scp / rttm / reco2dur 拆分为 train / dev 两份。
保持与原有内嵌脚本一致的划分逻辑：按比例随机打乱，确保 dev 至少含 1 个录音。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_wav_map(wav_scp: Path) -> Dict[str, str]:
    """读取 wav.scp，返回录音 ID 到音频路径的映射。"""
    wav_map: Dict[str, str] = {}
    with wav_scp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, path = line.split(None, 1)
            wav_map[utt] = path
    return wav_map


def load_durations(reco2dur: Path) -> Dict[str, float]:
    """读取 reco2dur，返回录音 ID 到时长（秒）的映射。"""
    dur_map: Dict[str, float] = {}
    with reco2dur.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                dur_map[parts[0]] = float(parts[1])
    return dur_map


def read_rttm_lines(rttm: Path) -> List[str]:
    """读取 RTTM，保留非空行，后续按录音 ID 过滤。"""
    return [line for line in rttm.open() if line.strip()]


def write_subset(
    out_dir: Path,
    name: str,
    ids: Iterable[str],
    wav_map: Dict[str, str],
    rttm_lines: Iterable[str],
    dur_map: Dict[str, float],
) -> None:
    """
    根据给定的 ID 集合写出子集 wav.scp / rttm / all.uem。
    - wav.scp：录音 ID 与路径映射
    - rttm：仅保留属于子集 ID 的标注行
    - all.uem：为每个录音写出全时长段，供裁切时使用
    """
    subset_dir = out_dir / name
    subset_dir.mkdir(parents=True, exist_ok=True)

    with (subset_dir / "wav.scp").open("w") as fw:
        for utt in ids:
            fw.write(f"{utt} {wav_map[utt]}\n")

    with (subset_dir / "rttm").open("w") as fr:
        for line in rttm_lines:
            utt = line.split()[1]
            if utt in wav_map and utt in ids:
                fr.write(line)

    with (subset_dir / "all.uem").open("w") as fu:
        for utt in ids:
            if utt not in dur_map:
                raise RuntimeError(f"Duration missing for {utt}")
            fu.write(f"{utt} 0 {dur_map[utt]:.3f}\n")


def split_dataset(
    src: Path,
    out: Path,
    val_ratio: float,
    seed: int,
) -> Tuple[int, int]:
    """
    按给定比例拆分数据集，返回 (train_count, dev_count)。
    - val_ratio 取值在 (0,1) 内正常生效，否则回退为 9:1。
    - split 经过上下限保护，避免 dev 为空；若极端情况仍为空，则强制从 train 借 1 条。
    """
    wav_map = load_wav_map(src / "wav.scp")
    rttm_lines = read_rttm_lines(src / "rttm")
    dur_map = load_durations(src / "reco2dur")

    recs = list(wav_map.keys())
    random.seed(seed)
    random.shuffle(recs)

    if val_ratio <= 0 or val_ratio >= 1:
        split = max(1, int(len(recs) * 0.9))
    else:
        split = int(len(recs) * (1 - val_ratio))
    split = min(max(1, split), len(recs) - 1)

    train_ids = set(recs[:split])
    dev_ids = set(recs[split:])
    if not dev_ids:
        dev_ids.add(train_ids.pop())

    write_subset(out, "train", train_ids, wav_map, rttm_lines, dur_map)
    write_subset(out, "dev", dev_ids, wav_map, rttm_lines, dur_map)
    return len(train_ids), len(dev_ids)


def main(argv: List[str]) -> None:
    if len(argv) != 5:
        raise SystemExit(
            "Usage: split_kaldi_data.py <data_src> <data_out> <val_ratio> <seed>"
        )

    src = Path(argv[1])
    out = Path(argv[2])
    val_ratio = float(argv[3])
    seed = int(argv[4])

    train_count, dev_count = split_dataset(src, out, val_ratio, seed)
    print(f"Prepared {train_count} train and {dev_count} dev recordings.")


if __name__ == "__main__":
    main(sys.argv)


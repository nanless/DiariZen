from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import soundfile as sf


AUDIO_EXTS = ("wav", "mp3", "m4a", "flac", "ogg", "aac", "wma")


def list_audio_files(root_dir: str) -> List[str]:
    root = Path(root_dir)
    assert root.is_dir(), f"in_root does not exist: {root}"
    files = []
    for ext in AUDIO_EXTS:
        files.extend(root.rglob(f"*.{ext}"))
    files = [p for p in files if p.is_file() and p.stat().st_size > 1024]
    files.sort()
    return [p.as_posix() for p in files]


def load_audio_mono_16k(audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, float]:
    data, sr = sf.read(audio_path, always_2d=False)
    assert sr == sample_rate, f"sample_rate mismatch: {sr} vs {sample_rate} for {audio_path}"

    if data.ndim == 1:
        mono = data
    else:
        # (samples, channels) -> mono
        mono = data.mean(axis=1)

    mono = np.asarray(mono, dtype=np.float32)
    duration = float(mono.shape[0]) / float(sample_rate)
    # return (channels=1, samples)
    return mono.reshape(1, -1), duration


def frames_to_segments(
    frame_labels: np.ndarray,
    frame_step: float,
    min_duration: float = 0.0,
) -> List[Tuple[float, float, int]]:
    """
    Convert (num_frames, num_speakers) {0,1} to segments.
    Returns List[(start, end, speaker_id)]
    """
    assert frame_labels.ndim == 2
    num_frames, num_speakers = frame_labels.shape
    segments: List[Tuple[float, float, int]] = []

    for spk_id in range(num_speakers):
        spk_frames = frame_labels[:, spk_id]
        in_segment = False
        start_frame = 0

        for frame_idx in range(num_frames):
            if spk_frames[frame_idx] > 0.5:
                if not in_segment:
                    start_frame = frame_idx
                    in_segment = True
            else:
                if in_segment:
                    end_frame = frame_idx
                    start_time = start_frame * frame_step
                    end_time = end_frame * frame_step
                    if (end_time - start_time) >= min_duration:
                        segments.append((start_time, end_time, spk_id))
                    in_segment = False

        if in_segment:
            end_frame = num_frames
            start_time = start_frame * frame_step
            end_time = end_frame * frame_step
            if (end_time - start_time) >= min_duration:
                segments.append((start_time, end_time, spk_id))

    segments.sort(key=lambda x: x[0])
    return segments


def write_rttm(
    segments: Iterable[Tuple[float, float, int]],
    session_name: str,
    output_path: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for start, end, spk_id in segments:
            duration = end - start
            f.write(
                f"SPEAKER {session_name} 1 {start:.3f} {duration:.3f} <NA> <NA> speaker_{spk_id:02d} <NA> <NA>\n"
            )


def dump_json(obj, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

